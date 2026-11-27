"""
Outside-Agent OpenAI API Tool-Calling Loop for SAE-Guided Direct Weight Editing.

Supports three clean experimental arms on standardized benchmarks (TruthfulQA, BLiMP):
  - `no_sae`:      Raw hidden-state contrastive direction baseline (arXiv:2611.21885 style)
  - `frozen_sae`:  Pretrained SAEs guide weight edits; downstream SAEs remain frozen
  - `patched_sae`: Pretrained SAEs guide weight edits + Local Subspace SAE Patching (<0.15% latents)
                   keeps downstream SAEs synchronized after every committed edit.
"""

from pathlib import Path
import copy
import json
import os
from typing import Any, Dict, List, Optional, Sequence
import torch
import torch.nn.functional as F

from .benchmarks import BenchmarkSplit, ContrastivePair, evaluate_benchmark
from .sae import SparseAutoencoder
from .sandbox import WeightEditingSandbox


def load_openai_config(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Load OpenAI API key and agent defaults from `.env` (falling back to environment variables)."""
    from dotenv import dotenv_values

    root = Path(repo_root or Path(__file__).resolve().parent.parent)
    env_path = root / ".env"
    vals = dotenv_values(env_path) if env_path.exists() else {}

    placeholder_prefixes = ("sk-paste-", "YOUR_", "REPLACE_")
    dot_key = (vals.get("OPENAI_API_KEY") or "").strip()
    if dot_key and not any(dot_key.startswith(p) for p in placeholder_prefixes):
        api_key = dot_key
        key_source = str(env_path)
    else:
        env_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if env_key and not any(env_key.startswith(p) for p in placeholder_prefixes):
            api_key = env_key
            key_source = "environment (OPENAI_API_KEY)"
        else:
            api_key = ""
            key_source = "missing"

    return {
        "api_key": api_key,
        "key_source": key_source,
        "env_path": str(env_path),
        "model": (vals.get("MWE_AGENT_MODEL") or os.environ.get("MWE_AGENT_MODEL") or "gpt-6.1-sol").strip(),
        "reasoning_effort": (vals.get("MWE_REASONING_EFFORT") or os.environ.get("MWE_REASONING_EFFORT") or "medium").strip(),
        "max_turns": int(vals.get("MWE_MAX_TURNS") or os.environ.get("MWE_MAX_TURNS") or 20),
        "max_edits": int(vals.get("MWE_MAX_EDITS") or os.environ.get("MWE_MAX_EDITS") or 6),
    }


class WeightEditingAgentSession:
    """
    Stateful sandboxed tool environment for an outside LLM agent (or heuristic baseline)
    performing closed-loop weight surgery on a target LLM.

    Strict anti-leakage guarantee: the held-out test split is NEVER exposed to any agent tool.
    """

    def __init__(
        self,
        sandbox: WeightEditingSandbox,
        saes: Dict[int, SparseAutoencoder],
        benchmark: BenchmarkSplit,
        arm: str = "patched_sae",
        layers: Sequence[int] = (6, 7, 8, 9, 10, 11),
        max_edits: int = 6,
        max_kl: float = 0.05,
        min_keeper_retention: float = 0.90,
        fetch_neuronpedia: bool = True,
        run_dir: Path = Path("/tmp/mwe_runs"),
        cache_dir: Path = Path("/tmp/mwe_cache"),
    ):
        assert arm in ("no_sae", "frozen_sae", "patched_sae"), f"Invalid arm: {arm}"
        self.sandbox = sandbox
        self.saes = saes
        self.benchmark = benchmark
        self.arm = arm
        self.layers = sorted(int(l) for l in layers)
        self.max_edits = max_edits
        self.max_kl = max_kl
        self.min_keeper_retention = min_keeper_retention
        self.fetch_neuronpedia = fetch_neuronpedia
        self.run_dir = Path(run_dir)
        self.ckpt_dir = self.run_dir / "checkpoints" / arm
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir)

        # Evaluate Base Model (Iteration 0) on both Practice and Held-Out splits
        self.base_prac = evaluate_benchmark(sandbox, benchmark, split="practice", use_staged=False)
        self.base_held = evaluate_benchmark(sandbox, benchmark, split="heldout", use_staged=False)
        self.cur_prac = self.base_prac

        # Precompute baseline residual activations on practice + wikitext for Local Subspace SAE Patching
        self._patch_texts, self._patch_spans = self._build_patch_calibration_corpus()
        self._baseline_layer_acts: Dict[int, torch.Tensor] = {
            L: self.sandbox.capture_span_activations(
                self._patch_texts, self._patch_spans, layer=L, hook_site="resid_pre", pool="mean", use_staged=False
            )
            for L in self.layers
        }

        self._last_staged_eval: Optional[Dict[str, Any]] = None
        self._last_staged_passed: bool = False
        self._raw_dir_cache: Dict[int, torch.Tensor] = {}
        self.tool_trace: List[Dict[str, Any]] = []
        self.sae_patch_logs: List[Dict[str, Any]] = []
        self.finished: bool = False
        self.finish_summary: str = ""

    def patch_downstream_saes_for_commit(self, edited_layer: int, edit_num: int, write_vectors: List[torch.Tensor]) -> List[Dict[str, Any]]:
        """Locally patch <0.15% latents on every loaded SAE strictly downstream of `edited_layer`."""
        downstream_patches: List[Dict[str, Any]] = []
        if not write_vectors:
            return downstream_patches
        write_vecs = torch.stack(write_vectors, dim=0)
        for L_next in sorted(self.saes.keys()):
            if L_next <= edited_layer:
                continue
            post_acts = self.sandbox.capture_span_activations(
                self._patch_texts, self._patch_spans, layer=L_next, hook_site="resid_pre", pool="mean", use_staged=False
            )
            patch_stats = self.saes[L_next].local_subspace_patch(
                pre_edit_acts=self._baseline_layer_acts[L_next],
                post_edit_acts=post_acts,
                write_vectors=write_vecs,
                max_latents=32,
                steps=40,
            )
            self.saes[L_next].save_patch_delta(self.ckpt_dir / f"sae_L{L_next}_delta_edit_{edit_num}.safetensors")
            self._baseline_layer_acts[L_next] = post_acts
            summary = {
                "downstream_layer": L_next,
                "patched_latents": patch_stats["patched_latents_count"],
                "patched_pct": patch_stats["patched_fraction_pct"],
                "mse_before": patch_stats["post_edit_before_sae_patch"]["mse"],
                "mse_after": patch_stats["post_edit_after_sae_patch"]["mse"],
            }
            downstream_patches.append(summary)
            self.sae_patch_logs.append({"edit_num": edit_num, "edited_layer": edited_layer, **summary})
        return downstream_patches

    def clone_branch(
        self,
        target_arm: str,
        new_layers: Sequence[int],
        new_max_edits: int,
        apply_initial_sae_patch: bool = False,
    ) -> "WeightEditingAgentSession":
        """
        Fork this session from its exact committed weight state into an isolated downstream branch
        (`frozen_sae` vs `patched_sae`) for a true apples-to-apples paired-branch ablation.
        """
        assert target_arm in ("frozen_sae", "patched_sae"), f"Invalid branch arm: {target_arm}"
        branch = object.__new__(WeightEditingAgentSession)

        # Deep-copy model + SAEs so branches cannot mutate each other
        cloned_model = copy.deepcopy(self.sandbox.model).eval()
        cloned_model.requires_grad_(False)
        cloned_sb = WeightEditingSandbox(cloned_model, self.sandbox.tokenizer)
        cloned_sb.committed_deltas = copy.deepcopy(self.sandbox.committed_deltas)
        cloned_sb.committed_write_vectors = [v.clone() for v in self.sandbox.committed_write_vectors]
        cloned_sb.committed_history = copy.deepcopy(self.sandbox.committed_history)
        cloned_sb.used_mlp_units = {k: set(v) for k, v in self.sandbox.used_mlp_units.items()}
        cloned_sb._last_committed_write_vectors = [
            v.clone() for v in getattr(self.sandbox, "_last_committed_write_vectors", [])
        ]
        cloned_sb._last_committed_layer = getattr(self.sandbox, "_last_committed_layer", -1)

        branch.sandbox = cloned_sb
        branch.saes = {L: copy.deepcopy(s) for L, s in self.saes.items()}
        branch.benchmark = self.benchmark
        branch.arm = target_arm
        branch.layers = sorted(int(l) for l in new_layers)
        branch.max_edits = int(new_max_edits)
        branch.max_kl = self.max_kl
        branch.min_keeper_retention = self.min_keeper_retention
        branch.fetch_neuronpedia = self.fetch_neuronpedia
        branch.run_dir = self.run_dir
        branch.ckpt_dir = self.run_dir / "checkpoints" / f"branch_{target_arm}"
        branch.ckpt_dir.mkdir(parents=True, exist_ok=True)
        branch.cache_dir = self.cache_dir

        branch.base_prac = self.base_prac
        branch.base_held = self.base_held
        branch.cur_prac = copy.deepcopy(self.cur_prac)
        branch._patch_texts = list(self._patch_texts)
        branch._patch_spans = list(self._patch_spans)
        branch._baseline_layer_acts = {L: t.clone() for L, t in self._baseline_layer_acts.items()}

        branch._last_staged_eval = None
        branch._last_staged_passed = False
        branch._raw_dir_cache = {}
        branch.tool_trace = []
        branch.sae_patch_logs = []
        branch.finished = False
        branch.finish_summary = ""

        if apply_initial_sae_patch and target_arm == "patched_sae" and cloned_sb.committed_history:
            last_rec = cloned_sb.committed_history[-1]
            branch.patch_downstream_saes_for_commit(
                edited_layer=int(last_rec["layer"]),
                edit_num=int(last_rec["edit_num"]),
                write_vectors=cloned_sb._last_committed_write_vectors,
            )

        return branch

    def _build_patch_calibration_corpus(self):
        texts: List[str] = []
        spans: List[int] = []
        pairs: List[ContrastivePair] = (
            self.base_prac["_raw"]["failing_pairs"][:24] + self.base_prac["_raw"]["passing_pairs"][:12]
        )
        for cp in pairs:
            texts.append(cp.bad_text)
            spans.append(cp.bad_span_start)
            texts.append(cp.good_text)
            spans.append(cp.good_span_start)
        for wt in self.benchmark.wikitext_passages[:8]:
            texts.append(wt)
            spans.append(0)
        return texts, spans

    def _public_metrics(self, ev: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for k, v in ev.items() if k != "_raw"}

    def _select_pairs(self, subset: Optional[str] = None, max_pairs: int = 48) -> List[ContrastivePair]:
        failing: List[ContrastivePair] = self.cur_prac["_raw"]["failing_pairs"]
        if subset:
            filtered = [p for p in failing if p.category == subset]
            if filtered:
                failing = filtered
        if not failing:
            # Fallback to weakest passing pairs so agent can still improve margins
            passing: List[ContrastivePair] = sorted(
                self.cur_prac["_raw"]["passing_pairs"], key=lambda c: c.margin
            )
            failing = passing
        return failing[:max_pairs]

    # ------------------------------------------------------------------
    # Tool implementations callable by the outside GPT agent
    # ------------------------------------------------------------------

    def tool_inspect_benchmark_status(self, subset: Optional[str] = None) -> Dict[str, Any]:
        failing = self._select_pairs(subset=subset, max_pairs=5)
        samples = [
            {
                "item_id": cp.item_id,
                "category": cp.category,
                "margin": cp.margin,
                "bad_text": cp.bad_text[:220],
                "good_text": cp.good_text[:220],
            }
            for cp in failing
        ]
        return {
            "arm": self.arm,
            "available_layers": self.layers,
            "edits_committed": len(self.sandbox.committed_history),
            "edits_remaining": self.max_edits - len(self.sandbox.committed_history),
            "guards": {
                "max_wikitext2_kl": self.max_kl,
                "min_keeper_retention": self.min_keeper_retention,
            },
            "base_practice_metrics": self._public_metrics(self.base_prac),
            "current_practice_metrics": self._public_metrics(self.cur_prac),
            "committed_history": self.sandbox.committed_history[-6:],
            "worst_failing_samples": samples,
        }

    def tool_localize_sae_features(
        self,
        layer: int,
        subset: Optional[str] = None,
        top_k: int = 8,
        pool: str = "mean",
        fetch_labels: bool = True,
    ) -> Dict[str, Any]:
        if self.arm == "no_sae":
            return {"error": "SAE localization is disabled in the `no_sae` baseline arm. Use localize_raw_contrastive_direction."}
        if layer not in self.saes:
            return {"error": f"Layer {layer} has no loaded SAE. Available layers: {self.layers}"}

        pairs = self._select_pairs(subset=subset, max_pairs=48)
        pos_texts = [p.bad_text for p in pairs]
        pos_spans = [p.bad_span_start for p in pairs]
        neg_texts = [p.good_text for p in pairs]
        neg_spans = [p.good_span_start for p in pairs]

        pos_acts = self.sandbox.capture_span_activations(pos_texts, pos_spans, layer=layer, pool=pool, use_staged=False)
        neg_acts = self.sandbox.capture_span_activations(neg_texts, neg_spans, layer=layer, pool=pool, use_staged=False)

        hits = self.saes[layer].localize_features(
            pos_acts,
            neg_acts,
            top_k=min(max(int(top_k), 1), 16),
            fetch_labels=bool(fetch_labels and self.fetch_neuronpedia),
            cache_dir=self.cache_dir,
        )
        return {
            "layer": layer,
            "hook_point": self.saes[layer].hook_point,
            "num_pairs_analyzed": len(pairs),
            "pool": pool,
            "interpretation_hint": (
                "diff > 0 means the latent fires MORE on failing/wrong spans (good target to suppress via negative alpha). "
                "diff < 0 means the latent fires MORE on truthful/grammatical spans (good target to amplify via positive alpha)."
            ),
            "top_features": [
                {
                    "latent_idx": h.latent_idx,
                    "auc": h.auc,
                    "diff": h.diff,
                    "pos_mean": h.pos_mean,
                    "neg_mean": h.neg_mean,
                    "fire_rate_pos": h.fire_rate_pos,
                    "fire_rate_neg": h.fire_rate_neg,
                    "label": h.label,
                }
                for h in hits
            ],
        }

    def tool_localize_raw_contrastive_direction(
        self,
        layer: int,
        subset: Optional[str] = None,
        pool: str = "mean",
    ) -> Dict[str, Any]:
        if layer not in self.layers:
            return {"error": f"Layer {layer} not in available layers {self.layers}"}
        pairs = self._select_pairs(subset=subset, max_pairs=48)
        pos_acts = self.sandbox.capture_span_activations(
            [p.bad_text for p in pairs], [p.bad_span_start for p in pairs], layer=layer, pool=pool, use_staged=False
        )
        neg_acts = self.sandbox.capture_span_activations(
            [p.good_text for p in pairs], [p.good_span_start for p in pairs], layer=layer, pool=pool, use_staged=False
        )
        raw_diff = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)
        d_norm = F.normalize(raw_diff, dim=-1)
        self._raw_dir_cache[layer] = d_norm

        p_proj = pos_acts @ d_norm
        n_proj = neg_acts @ d_norm
        comp = (p_proj[:, None] > n_proj[None, :]).float() + 0.5 * (p_proj[:, None] == n_proj[None, :]).float()
        return {
            "layer": layer,
            "num_pairs_analyzed": len(pairs),
            "raw_diff_norm": round(float(raw_diff.norm().item()), 4),
            "separation_auc": round(float(comp.mean().item()), 4),
            "pos_proj_mean": round(float(p_proj.mean().item()), 4),
            "neg_proj_mean": round(float(n_proj.mean().item()), 4),
            "note": "Direction points from good spans -> bad/failing spans. Negative alpha suppresses failure direction.",
        }

    def _evaluate_staged_with_guards(self, stage_meta: Dict[str, Any]) -> Dict[str, Any]:
        staged_prac = evaluate_benchmark(
            self.sandbox,
            self.benchmark,
            split="practice",
            use_staged=True,
            base_cache=self.base_prac["_raw"],
        )
        kl_ok = staged_prac["wikitext2_kl"] <= self.max_kl
        keeper_ok = staged_prac["keeper_retention"] >= self.min_keeper_retention
        score_delta = round(staged_prac["primary_score"] - self.cur_prac["primary_score"], 4)
        margin_delta = round(staged_prac["mean_margin"] - self.cur_prac["mean_margin"], 4)
        progressed = (score_delta > 0.0) or (score_delta >= 0.0 and margin_delta > 0.01)
        passed = bool(kl_ok and keeper_ok and progressed)

        self._last_staged_eval = staged_prac
        self._last_staged_passed = passed

        return {
            "stage_meta": stage_meta,
            "before_primary_score": self.cur_prac["primary_score"],
            "staged_primary_score": staged_prac["primary_score"],
            "primary_score_delta": score_delta,
            "before_mean_margin": self.cur_prac["mean_margin"],
            "staged_mean_margin": staged_prac["mean_margin"],
            "mean_margin_delta": margin_delta,
            "staged_metrics": self._public_metrics(staged_prac),
            "guard_checks": {
                "kl_ok": kl_ok,
                "keeper_ok": keeper_ok,
                "progressed": progressed,
                "all_guards_passed": passed,
            },
            "next_action_hint": (
                "All guards PASSED! Call commit_staged_edit if you want to permanently commit this edit, "
                "or stage another candidate to compare."
                if passed
                else "Guards REJECTED this candidate. Try a smaller |alpha|/gain, a different layer, or a different latent."
            ),
        }

    def tool_stage_and_test_projection_edit(
        self,
        layer: int,
        component: str,
        source_latent_idx: int,
        alpha: float,
        target_latent_idx: Optional[int] = None,
    ) -> Dict[str, Any]:
        if self.arm == "no_sae":
            return {"error": "SAE projection edit is disabled in `no_sae` arm. Use stage_and_test_raw_direction_edit."}
        if layer not in self.saes:
            return {"error": f"Invalid layer {layer}. Available: {self.layers}"}
        sae = self.saes[layer]
        if not (0 <= source_latent_idx < sae.d_sae):
            return {"error": f"source_latent_idx {source_latent_idx} out of range [0, {sae.d_sae - 1}]"}
        alpha = max(-1.5, min(1.5, float(alpha)))

        src_dir = sae.W_dec[source_latent_idx]
        tgt_dir = None
        desc = f"L{layer}.{component} SAE proj #{source_latent_idx} (alpha={alpha:+.3f})"
        if target_latent_idx is not None:
            if not (0 <= target_latent_idx < sae.d_sae):
                return {"error": f"target_latent_idx {target_latent_idx} out of range [0, {sae.d_sae - 1}]"}
            tgt_dir = sae.W_dec[target_latent_idx]
            desc = f"L{layer}.{component} SAE redirect #{source_latent_idx}->#{target_latent_idx} (alpha={alpha:+.3f})"

        stage_meta = self.sandbox.stage_rank1_feature_projection(
            layer=layer,
            component=component,
            source_dir=src_dir,
            alpha=alpha,
            target_dir=tgt_dir,
            description=desc,
        )
        return self._evaluate_staged_with_guards(stage_meta)

    def tool_stage_and_test_neuron_gate_edit(
        self,
        layer: int,
        read_latent_indices: List[int],
        write_latent_indices: List[int],
        write_sign: float = -1.0,
        gain: float = 4.0,
        steepness: float = 6.0,
        margin: float = 0.15,
        subset: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.arm == "no_sae":
            return {"error": "SAE neuron gate is disabled in `no_sae` arm. Use stage_and_test_raw_direction_edit."}
        if layer not in self.saes:
            return {"error": f"Invalid layer {layer}. Available: {self.layers}"}
        sae = self.saes[layer]
        if not read_latent_indices or not write_latent_indices:
            return {"error": "read_latent_indices and write_latent_indices must be non-empty lists."}

        read_dir = torch.stack([sae.W_dec[int(i)] for i in read_latent_indices]).mean(dim=0)
        sign = -1.0 if float(write_sign) < 0 else 1.0
        write_dir = sign * torch.stack([sae.W_dec[int(i)] for i in write_latent_indices]).mean(dim=0)

        pairs = self._select_pairs(subset=subset, max_pairs=24)
        stage_meta = self.sandbox.stage_sae_gated_concept_edit(
            layer=layer,
            read_dir=read_dir,
            write_dir=write_dir,
            pos_texts=[p.bad_text for p in pairs],
            pos_spans=[p.bad_span_start for p in pairs],
            neg_texts=[p.good_text for p in pairs],
            neg_spans=[p.good_span_start for p in pairs],
            bg_texts=self.benchmark.wikitext_passages,
            gain=max(0.2, min(20.0, float(gain))),
            steepness=max(1.0, min(20.0, float(steepness))),
            margin=max(-0.5, min(2.0, float(margin))),
            description=(
                f"L{layer} dormant MLP gate read={read_latent_indices} write={sign:+.0f}*{write_latent_indices} "
                f"(gain={gain:.2f})"
            ),
        )
        return self._evaluate_staged_with_guards(stage_meta)

    def tool_stage_and_test_raw_direction_edit(
        self,
        layer: int,
        mode: str = "projection",
        component: str = "mlp",
        alpha: float = -0.35,
        gain: float = 4.0,
        steepness: float = 6.0,
        margin: float = 0.15,
        subset: Optional[str] = None,
    ) -> Dict[str, Any]:
        if layer not in self._raw_dir_cache:
            self.tool_localize_raw_contrastive_direction(layer=layer, subset=subset)
        d_raw = self._raw_dir_cache[layer]

        if mode == "projection":
            alpha = max(-1.5, min(1.5, float(alpha)))
            stage_meta = self.sandbox.stage_rank1_feature_projection(
                layer=layer,
                component=component,
                source_dir=d_raw,
                alpha=alpha,
                description=f"L{layer}.{component} raw diff_dir projection (alpha={alpha:+.3f})",
            )
        elif mode == "neuron_gate":
            pairs = self._select_pairs(subset=subset, max_pairs=24)
            stage_meta = self.sandbox.stage_sae_gated_concept_edit(
                layer=layer,
                read_dir=d_raw,
                write_dir=-d_raw,
                pos_texts=[p.bad_text for p in pairs],
                pos_spans=[p.bad_span_start for p in pairs],
                neg_texts=[p.good_text for p in pairs],
                neg_spans=[p.good_span_start for p in pairs],
                bg_texts=self.benchmark.wikitext_passages,
                gain=max(0.2, min(20.0, float(gain))),
                steepness=max(1.0, min(20.0, float(steepness))),
                margin=max(-0.5, min(2.0, float(margin))),
                description=f"L{layer} dormant MLP gate raw diff_dir (gain={gain:.2f})",
            )
        else:
            return {"error": f"mode must be 'projection' or 'neuron_gate', got {mode}"}
        return self._evaluate_staged_with_guards(stage_meta)

    def tool_commit_staged_edit(self, rationale: str = "") -> Dict[str, Any]:
        if len(self.sandbox.committed_history) >= self.max_edits:
            return {"error": f"Edit budget exhausted ({self.max_edits}/{self.max_edits}). Call finish_session."}
        if not self.sandbox.staged.deltas or self._last_staged_eval is None:
            return {"error": "No staged edit ready to commit. Call a stage_and_test_* tool first."}
        if not self._last_staged_passed:
            return {"error": "Refusing commit: the currently staged edit FAILED sandbox guards."}

        edit_type = self.sandbox.staged.edit_type
        rec = self.sandbox.commit_staged(rationale=rationale)
        if edit_type == "dormant_neuron_gate":
            self.sandbox.mark_last_unit_used()

        self.cur_prac = self._last_staged_eval
        self._last_staged_eval = None
        self._last_staged_passed = False

        edit_num = rec["edit_num"]
        edited_layer = rec["layer"]
        self.sandbox.save_committed_deltas(self.ckpt_dir / f"model_delta_edit_{edit_num}_L{edited_layer}.safetensors")

        # Co-evolution step: if arm == "patched_sae", locally patch downstream SAEs (<0.15% latents)!
        downstream_patches: List[Dict[str, Any]] = []
        if self.arm == "patched_sae" and self.sandbox._last_committed_write_vectors:
            downstream_patches = self.patch_downstream_saes_for_commit(
                edited_layer=edited_layer,
                edit_num=edit_num,
                write_vectors=self.sandbox._last_committed_write_vectors,
            )

        return {
            "committed": rec,
            "current_practice_metrics": self._public_metrics(self.cur_prac),
            "downstream_sae_patches": downstream_patches,
            "edits_remaining": self.max_edits - len(self.sandbox.committed_history),
        }

    def tool_finish_session(self, summary: str = "") -> Dict[str, Any]:
        self.finished = True
        self.finish_summary = summary
        return {"status": "session_finished", "edits_committed": len(self.sandbox.committed_history)}

    # ------------------------------------------------------------------
    # OpenAI Tool Schema + Dispatch
    # ------------------------------------------------------------------

    def openai_tool_schemas(self) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = [
            {
                "type": "function",
                "function": {
                    "name": "inspect_benchmark_status",
                    "description": (
                        "Inspect current practice benchmark scores, collateral WikiText-2 perplexity/KL, "
                        "remaining edit budget, committed edit history, and worst failing practice examples."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subset": {
                                "type": "string",
                                "description": "Optional subset/category filter (for BLiMP subsets).",
                            }
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "commit_staged_edit",
                    "description": (
                        "Permanently commit the currently staged weight edit (only allowed if all sandbox guards passed). "
                        "In `patched_sae` mode, also locally patches <0.15% of downstream SAE latents."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "rationale": {"type": "string", "description": "Short explanation of why this edit helps."}
                        },
                        "required": ["rationale"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "finish_session",
                    "description": "End the editing session and proceed to final strictly held-out evaluation.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "summary": {"type": "string", "description": "Brief summary of edits made and findings."}
                        },
                        "required": ["summary"],
                    },
                },
            },
        ]

        if self.arm == "no_sae":
            tools.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "localize_raw_contrastive_direction",
                            "description": "Compute raw hidden-state mean-difference vector (bad minus good spans) at a layer (no SAE).",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "layer": {"type": "integer", "description": f"Layer index in {self.layers}."},
                                    "subset": {"type": "string"},
                                    "pool": {"type": "string", "enum": ["mean", "max", "first", "last"]},
                                },
                                "required": ["layer"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "stage_and_test_raw_direction_edit",
                            "description": "Stage a rank-1 projection or dormant MLP neuron gate using the raw contrastive direction and test in sandbox.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "layer": {"type": "integer"},
                                    "mode": {"type": "string", "enum": ["projection", "neuron_gate"]},
                                    "component": {"type": "string", "enum": ["mlp", "attn"]},
                                    "alpha": {"type": "number", "description": "Projection scale in [-1.5, 1.5]. Negative suppresses failure dir."},
                                    "gain": {"type": "number", "description": "Dormant neuron output gain in [0.2, 20.0]."},
                                    "steepness": {"type": "number"},
                                    "margin": {"type": "number"},
                                    "subset": {"type": "string"},
                                },
                                "required": ["layer", "mode"],
                            },
                        },
                    },
                ]
            )
        else:
            tools.extend(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "localize_sae_features",
                            "description": (
                                "Run the layer's Sparse Autoencoder on failing vs. desired answer spans to find top "
                                "monosemantic SAE latents separating failures from correct answers (with Neuronpedia labels if available)."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "layer": {"type": "integer", "description": f"Layer index in {self.layers}."},
                                    "subset": {"type": "string", "description": "Optional BLiMP subset to focus on."},
                                    "top_k": {"type": "integer", "description": "Number of top SAE features (1-16)."},
                                    "pool": {"type": "string", "enum": ["mean", "max", "first", "last"]},
                                },
                                "required": ["layer"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "stage_and_test_projection_edit",
                            "description": (
                                "Stage an exact rank-1 weight projection on layer's MLP or Attention c_proj along an SAE decoder vector "
                                "(or redirect source_latent -> target_latent) and test in the reversible sandbox."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "layer": {"type": "integer"},
                                    "component": {"type": "string", "enum": ["mlp", "attn"]},
                                    "source_latent_idx": {"type": "integer", "description": "SAE latent index to suppress/amplify/redirect."},
                                    "alpha": {"type": "number", "description": "Scale in [-1.5, 1.5]. Negative suppresses, positive amplifies."},
                                    "target_latent_idx": {
                                        "type": "integer",
                                        "description": "Optional target SAE latent index for cross-feature redirection.",
                                    },
                                },
                                "required": ["layer", "component", "source_latent_idx", "alpha"],
                            },
                        },
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "stage_and_test_neuron_gate_edit",
                            "description": (
                                "Compile an SAE-gated concept edit into 1 dormant MLP neuron (1,537 weights): "
                                "reads `read_latent_indices`, fires `write_sign * gain * write_latent_indices` above background threshold."
                            ),
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "layer": {"type": "integer"},
                                    "read_latent_indices": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "description": "1-3 SAE latent indices that detect the failure pattern.",
                                    },
                                    "write_latent_indices": {
                                        "type": "array",
                                        "items": {"type": "integer"},
                                        "description": "1-3 SAE latent indices defining the concept direction to write.",
                                    },
                                    "write_sign": {
                                        "type": "number",
                                        "description": "-1.0 to suppress write_latent_indices, +1.0 to inject them.",
                                    },
                                    "gain": {"type": "number", "description": "Output magnitude (try 1.5 to 10.0)."},
                                    "steepness": {"type": "number", "description": "Gate sharpness (default 6.0)."},
                                    "margin": {"type": "number", "description": "Threshold margin above background (default 0.15)."},
                                    "subset": {"type": "string"},
                                },
                                "required": ["layer", "read_latent_indices", "write_latent_indices", "write_sign", "gain"],
                            },
                        },
                    },
                ]
            )

        return tools

    def dispatch_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        fn_map = {
            "inspect_benchmark_status": self.tool_inspect_benchmark_status,
            "localize_sae_features": self.tool_localize_sae_features,
            "localize_raw_contrastive_direction": self.tool_localize_raw_contrastive_direction,
            "stage_and_test_projection_edit": self.tool_stage_and_test_projection_edit,
            "stage_and_test_neuron_gate_edit": self.tool_stage_and_test_neuron_gate_edit,
            "stage_and_test_raw_direction_edit": self.tool_stage_and_test_raw_direction_edit,
            "commit_staged_edit": self.tool_commit_staged_edit,
            "finish_session": self.tool_finish_session,
        }
        if name not in fn_map:
            res = {"error": f"Unknown tool: {name}"}
        else:
            try:
                res = fn_map[name](**arguments)
            except Exception as e:
                res = {"error": f"{type(e).__name__}: {e}"}

        self.tool_trace.append({"tool": name, "args": arguments, "result": res})
        return res

    def finalize_and_evaluate_heldout(self) -> Dict[str, Any]:
        """Run strictly unseen held-out test split evaluation and save final checkpoint + report."""
        self.sandbox.staged.clear()
        final_held = evaluate_benchmark(
            self.sandbox,
            self.benchmark,
            split="heldout",
            use_staged=False,
            base_cache=self.base_held["_raw"],
        )
        self.sandbox.save_committed_deltas(self.ckpt_dir / "model_delta_final.safetensors")

        return {
            "arm": self.arm,
            "benchmark": self.benchmark.name,
            "benchmark_meta": self.benchmark.metadata,
            "edits_committed": len(self.sandbox.committed_history),
            "committed_history": self.sandbox.committed_history,
            "sae_patch_logs": self.sae_patch_logs,
            "base_practice": self._public_metrics(self.base_prac),
            "final_practice": self._public_metrics(self.cur_prac),
            "base_heldout": self._public_metrics(self.base_held),
            "final_heldout": self._public_metrics(final_held),
            "heldout_primary_delta": round(
                final_held["primary_score"] - self.base_held["primary_score"], 4
            ),
            "heldout_margin_delta": round(
                final_held["mean_margin"] - self.base_held["mean_margin"], 4
            ),
            "finish_summary": self.finish_summary,
        }


def run_openai_agent_loop(
    session: WeightEditingAgentSession,
    api_key: str,
    model_name: str = "gpt-6.1-sol",
    max_turns: int = 16,
    reasoning_effort: Optional[str] = "medium",
    verbose: bool = True,
    extra_user_context: str = "",
    stop_when_edits_reached: bool = False,
) -> Dict[str, Any]:
    """Run an outside OpenAI API reasoning/coding agent via standard function-calling tools."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    tools = session.openai_tool_schemas()

    benchmark_scoring_guide = (
        "Benchmark scoring (`blimp`):\n"
        "- Each item is a grammatical minimal pair (`sentence_good` vs. `sentence_bad`).\n"
        "- Scored via teacher-forced log-likelihood: `margin = log P(sentence_good) - log P(sentence_bad)`. Pass iff `margin > 0`.\n"
        "- `per_subset_acc` breaks down accuracy across 4 distinct grammar subsets. Attack ONE weak subset at a time using "
        "`subset=<name>` in `inspect_benchmark_status` / `localize_*` / `stage_and_test_neuron_gate_edit` so you don't average "
        "unrelated grammatical phenomena into a noisy direction.\n"
        "- IMPORTANT ANTI-SHORTCUT GUARD: `keeper_retention` strictly enforces >= 90% retention on EVERY individual keeper subset "
        "(including grammatical fronted-question siblings `left_branch_island_simple_question` and `matrix_question_npi_licensor_present`!). "
        "Blanket suppression of generic question-opening features will break those keepers and be REJECTED — look for finer-grained syntactic features!\n"
        if session.benchmark.name == "blimp"
        else "Benchmark scoring (`truthfulqa`):\n"
        "- Scored via teacher-forced continuation log-likelihood `log P(choice | Q: ...\\nA:)`.\n"
        "- `mc1_acc`: top-1 choice is true; `mc2_acc` (primary metric): total normalized probability mass on truthful choices; "
        "`mean_margin = log P(true) - log P(best_false)`.\n"
        "- Prefer edits that target generalizable semantic misconception/hallucination features rather than narrow question-specific quirks.\n"
    )

    if session.arm != "no_sae":
        arm_guide = (
            f"SAE tools & exact weight-surgery equations (arm=`{session.arm}`):\n"
            "- `localize_sae_features(layer, subset=..., top_k=8)` runs the layer's pretrained 24,576-latent SAE on failing vs. desired "
            "answer/diverging spans and returns top separating latents + Neuronpedia human-readable labels.\n"
            "  * `diff > 0` (AUC > 0.5): fires MORE on BAD/wrong spans -> good target to SUPPRESS (`alpha < 0` or `write_sign = -1.0`).\n"
            "  * `diff < 0` (AUC < 0.5): fires MORE on GOOD/correct spans -> good target to AMPLIFY (`alpha > 0` or `write_sign = +1.0`).\n"
            "  * Strongly prefer features with high |AUC - 0.5| AND causally sensible Neuronpedia labels (avoid spurious token/punctuation artifacts that overfit practice!).\n"
            "- Operator A — `stage_and_test_projection_edit(layer, component, source_latent_idx, alpha, target_latent_idx=None)`:\n"
            "  Exact rank-1 update on `mlp.c_proj` or `attn.c_proj`: `W' = W + alpha * (W d_src) d_tgt^T`, `b' = b + alpha * (b·d_src) d_tgt`.\n"
            "  * Same-feature (`target_latent_idx=None`): `alpha in [-0.3, -1.0]` dampens/ablates `d_src`; `alpha in [0.2, 0.8]` amplifies `d_src`.\n"
            "  * Cross-feature (`target_latent_idx` set): redirects activation along bad `d_src` into good `d_tgt` (`alpha in [0.2, 0.8]`).\n"
            "- Operator B — `stage_and_test_neuron_gate_edit(layer, read_latent_indices, write_latent_indices, write_sign, gain, steepness, margin)`:\n"
            "  Compiles a 1-neuron dormant MLP gate (1,537 weights): `GELU(steepness * (ln2(x)·r - theta)) * (write_sign * gain * w)`.\n"
            "  Stays silent on WikiText-2 background (`theta` auto-set above p95 background + `margin`), fires only when `read` latents activate. Try `gain` in `2.0..8.0`.\n"
            + (
                "- Co-evolution (`patched_sae`): Every `commit_staged_edit` at layer L automatically locally patches (<0.15% latents) "
                "all downstream SAEs (L+1..11) to stay synced with your weight change. Always re-call `localize_sae_features` on downstream layers AFTER committing to see newly disentangled residual failures!\n"
                if session.arm == "patched_sae"
                else "- Note (`frozen_sae`): Downstream SAEs remain frozen after commits. Re-localize on downstream layers after commits to find remaining failures.\n"
            )
        )
    else:
        arm_guide = (
            "Baseline arm (`no_sae` — no SAEs available):\n"
            "- `localize_raw_contrastive_direction(layer, subset=...)` computes `d_raw = normalize(mean(bad_acts) - mean(good_acts))`.\n"
            "- `stage_and_test_raw_direction_edit(layer, mode='projection'|'neuron_gate', component='mlp'|'attn', alpha=..., gain=...)`:\n"
            "  * `projection`: `W' = W + alpha * (W d_raw) d_raw^T` (try `alpha` in `-0.15` to `-0.8` to suppress failure direction without blowing KL).\n"
            "  * `neuron_gate`: dormant MLP neuron firing `-gain * d_raw` above background threshold (try `gain` in `1.5..6.0`).\n"
        )

    system_prompt = (
        "You are an expert mechanistic interpretability and Weight-Space Programming researcher. "
        "Your mission is to directly edit the weights of a frozen GPT-2 Small (124M) model — WITHOUT gradient descent "
        "or fine-tuning — to improve its held-out generalization on a standardized NLP benchmark while strictly preserving "
        f"general language fluency (WikiText-2 KL <= {session.max_kl}) and keeper retention (>= {session.min_keeper_retention}).\n\n"
        f"Active experimental arm: `{session.arm}`. Available layers: {session.layers}. Max committed edits: {session.max_edits}. Max turns: {max_turns}.\n\n"
        + benchmark_scoring_guide
        + "\n"
        + arm_guide
        + "\nExecution Protocol:\n"
        "1. First call `inspect_benchmark_status` to check baseline scores, per-subset weaknesses, and failing examples.\n"
        "2. Work sequentially from earlier to later layers (start at L6–L8 for your first edits, then move downstream L8–L11 for subsequent subsets/residual failures). Avoid jumping straight to L11 on your first edit — early/mid-layer edits compound better and let downstream SAEs adapt.\n"
        "3. Test surgical edits in the reversible sandbox. If an edit is REJECTED for high KL or keeper drop, cut |alpha| or gain in half or switch component (`mlp` <-> `attn`) / layer. Try both Rank-1 Projection and Dormant MLP Neuron Gate.\n"
        "4. Commit an edit via `commit_staged_edit` as soon as you find one that passes all guards with a clear practice gain, then move to the next weak subset / downstream layer to stack multiple complementary edits (aim for 3–6 clean edits if they genuinely help!).\n"
        "5. Never try to overfit tiny noise — prefer clean, low-KL edits that generalize to unseen held-out test data. Call `finish_session` once you exhaust the edit budget or hit diminishing returns."
    )

    # Convert tool schemas to OpenAI Responses API flat FunctionTool format (/v1/responses)
    responses_tools = [
        {
            "type": "function",
            "name": t["function"]["name"],
            "description": t["function"]["description"],
            "parameters": t["function"]["parameters"],
            "strict": False,
        }
        for t in tools
    ]

    transcript_log: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
    ]
    initial_user_msg = (
        f"Begin optimizing GPT-2 Small on the `{session.benchmark.name}` benchmark "
        f"(arm=`{session.arm}`, max_edits={session.max_edits}, max_turns={max_turns})."
        + (f"\n\n{extra_user_context}" if extra_user_context else "")
    )
    transcript_log.append({"role": "user", "content": initial_user_msg})

    usage_totals = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
        "turns": 0,
    }
    is_reasoning_model = any(model_name.startswith(p) for p in ("o1", "o3", "o4", "gpt-5", "gpt-6"))

    prev_response_id: Optional[str] = None
    next_input: Any = initial_user_msg

    for turn in range(1, max_turns + 1):
        if session.finished:
            break

        kwargs: Dict[str, Any] = {
            "model": model_name,
            "instructions": system_prompt,
            "input": next_input,
            "tools": responses_tools,
            "tool_choice": "auto",
        }
        if prev_response_id is not None:
            kwargs["previous_response_id"] = prev_response_id
        if is_reasoning_model:
            if reasoning_effort and reasoning_effort.lower() not in ("", "default"):
                kwargs["reasoning"] = {"effort": reasoning_effort.lower()}
        else:
            kwargs["temperature"] = 0.2

        resp = client.responses.create(**kwargs)
        prev_response_id = resp.id
        usage_totals["turns"] = turn

        if resp.usage:
            usage_totals["input_tokens"] += int(getattr(resp.usage, "input_tokens", 0) or 0)
            usage_totals["output_tokens"] += int(getattr(resp.usage, "output_tokens", 0) or 0)
            usage_totals["total_tokens"] += int(getattr(resp.usage, "total_tokens", 0) or 0)
            in_det = getattr(resp.usage, "input_tokens_details", None)
            if in_det is not None:
                usage_totals["cached_input_tokens"] += int(getattr(in_det, "cached_tokens", 0) or 0)
            out_det = getattr(resp.usage, "output_tokens_details", None)
            if out_det is not None:
                usage_totals["reasoning_tokens"] += int(getattr(out_det, "reasoning_tokens", 0) or 0)

        out_text = (getattr(resp, "output_text", None) or "").strip()
        fn_calls = [item for item in (resp.output or []) if getattr(item, "type", None) == "function_call"]

        transcript_log.append(
            {
                "turn": turn,
                "role": "assistant",
                "response_id": resp.id,
                "content": out_text,
                "tool_calls": [
                    {"call_id": fc.call_id, "name": fc.name, "arguments": fc.arguments}
                    for fc in fn_calls
                ],
            }
        )

        if verbose and out_text:
            print(f"\n[Agent Turn {turn}/{max_turns} | {model_name}] {out_text}")

        if not fn_calls:
            nudge = "Please call one of the provided tools to continue editing, or call `finish_session` to conclude."
            transcript_log.append({"turn": turn, "role": "user", "content": nudge})
            next_input = nudge
            continue

        tool_outputs: List[Dict[str, Any]] = []
        for fc in fn_calls:
            fn_name = fc.name
            try:
                fn_args = json.loads(fc.arguments or "{}")
            except json.JSONDecodeError:
                fn_args = {}

            if verbose:
                print(f"  -> Tool `{fn_name}`({json.dumps(fn_args)})")
            result = session.dispatch_tool(fn_name, fn_args)
            if verbose:
                brief = {
                    k: result[k]
                    for k in (
                        "primary_score_delta",
                        "mean_margin_delta",
                        "guard_checks",
                        "downstream_sae_patches",
                        "error",
                        "status",
                    )
                    if k in result
                }
                if brief:
                    print(f"     <- {json.dumps(brief)}")

            out_json = json.dumps(result)
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": fc.call_id,
                    "output": out_json,
                }
            )
            transcript_log.append(
                {
                    "turn": turn,
                    "role": "tool",
                    "call_id": fc.call_id,
                    "name": fn_name,
                    "result": result,
                }
            )

        next_input = tool_outputs
        if stop_when_edits_reached and len(session.sandbox.committed_history) >= session.max_edits:
            session.tool_finish_session(summary="Reached target edit count for this phase.")
            break

    report = session.finalize_and_evaluate_heldout()
    report["agent_model"] = model_name
    report["reasoning_effort"] = reasoning_effort
    report["api_usage"] = usage_totals
    report["tool_trace_len"] = len(session.tool_trace)
    report["_transcript"] = {
        "messages": transcript_log,
        "tool_trace": session.tool_trace,
    }
    return report


def run_heuristic_agent_loop(
    session: WeightEditingAgentSession,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Zero-API-cost deterministic controller using the exact same tool interface.
    Ideal for smoke-testing the full benchmark + SAE + sandbox + local SAE patching pipeline
    and serving as a non-LLM baseline against the outside GPT agent.
    """
    session.tool_inspect_benchmark_status()

    for layer in session.layers:
        if len(session.sandbox.committed_history) >= session.max_edits:
            break

        best_call = None
        best_score = (0.0, 0.0)  # (primary_score_delta, margin_delta)

        if session.arm == "no_sae":
            session.tool_localize_raw_contrastive_direction(layer=layer)
            candidates = [
                ("stage_and_test_raw_direction_edit", {"layer": layer, "mode": "projection", "component": "mlp", "alpha": a})
                for a in (-0.3, -0.6, -0.9)
            ] + [
                ("stage_and_test_raw_direction_edit", {"layer": layer, "mode": "projection", "component": "attn", "alpha": a})
                for a in (-0.4, -0.8)
            ]
        else:
            loc = session.tool_localize_sae_features(layer=layer, top_k=6, fetch_labels=False)
            feats = loc.get("top_features", [])
            pos_feats = [f for f in feats if f["diff"] > 0][:3]
            neg_feats = [f for f in feats if f["diff"] < 0][:2]
            candidates = []
            for f in pos_feats:
                for comp in ("mlp", "attn"):
                    for a in (-0.5, -0.9):
                        candidates.append(
                            (
                                "stage_and_test_projection_edit",
                                {"layer": layer, "component": comp, "source_latent_idx": f["latent_idx"], "alpha": a},
                            )
                        )
            for f in neg_feats:
                candidates.append(
                    (
                        "stage_and_test_projection_edit",
                        {"layer": layer, "component": "mlp", "source_latent_idx": f["latent_idx"], "alpha": 0.5},
                    )
                )
            if pos_feats and neg_feats:
                candidates.append(
                    (
                        "stage_and_test_projection_edit",
                        {
                            "layer": layer,
                            "component": "mlp",
                            "source_latent_idx": pos_feats[0]["latent_idx"],
                            "target_latent_idx": neg_feats[0]["latent_idx"],
                            "alpha": 0.6,
                        },
                    )
                )
            if pos_feats:
                candidates.append(
                    (
                        "stage_and_test_neuron_gate_edit",
                        {
                            "layer": layer,
                            "read_latent_indices": [pos_feats[0]["latent_idx"]],
                            "write_latent_indices": [neg_feats[0]["latent_idx"]] if neg_feats else [pos_feats[0]["latent_idx"]],
                            "write_sign": 1.0 if neg_feats else -1.0,
                            "gain": 3.0,
                            "steepness": 6.0,
                            "margin": 0.15,
                        },
                    )
                )

        for fn_name, fn_args in candidates:
            res = session.dispatch_tool(fn_name, fn_args)
            if res.get("guard_checks", {}).get("all_guards_passed"):
                key = (res["primary_score_delta"], res["mean_margin_delta"])
                if key > best_score:
                    best_score = key
                    best_call = (fn_name, fn_args)

        if best_call is not None:
            # Re-stage the winning candidate and commit it
            session.dispatch_tool(best_call[0], best_call[1])
            c_res = session.tool_commit_staged_edit(
                rationale=f"Heuristic best edit at L{layer}: {best_call[0]} {best_call[1]}"
            )
            if verbose:
                m = c_res["current_practice_metrics"]
                print(
                    f"  [Heuristic COMMIT L{layer} | {session.arm}] {best_call[0]} -> "
                    f"Practice {m['primary_metric_name']}={m['primary_score']:.4f}, "
                    f"Margin={m['mean_margin']:+.4f}, KL={m['wikitext2_kl']:.5f}"
                )
                if c_res["downstream_sae_patches"]:
                    p0 = c_res["downstream_sae_patches"][0]
                    print(
                        f"    -> Local SAE Patch L{p0['downstream_layer']}: "
                        f"MSE {p0['mse_before']:.2f} -> {p0['mse_after']:.2f} ({p0['patched_pct']}% latents)"
                    )
        else:
            session.sandbox.staged.clear()
            if verbose:
                print(f"  [Heuristic L{layer} | {session.arm}] No candidate beat practice guards; skipping.")

    session.tool_finish_session(summary="Heuristic sweep complete across layers.")
    report = session.finalize_and_evaluate_heldout()
    report["agent_model"] = "deterministic_heuristic"
    report["api_usage"] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "turns": 0}
    report["tool_trace_len"] = len(session.tool_trace)
    return report
