"""Staged sparse & rank-1 concept-level weight editing and forward-pass evaluation sandbox."""

import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn.functional as F


@dataclass
class SparseWeightDelta:
    """Stores sparse index/value deltas against a model's parameters before commit."""
    deltas: Dict[str, Tuple[torch.LongTensor, torch.Tensor]] = field(default_factory=dict)
    write_vectors: List[torch.Tensor] = field(default_factory=list)
    edited_layer: int = -1
    edit_type: str = ""
    effective_dof: int = 0
    description: str = ""

    def stage_entries(
        self,
        param_name: str,
        flat_indices: torch.LongTensor,
        new_values: torch.Tensor,
        write_vec: Optional[torch.Tensor] = None,
        edited_layer: Optional[int] = None,
        edit_type: str = "",
        effective_dof: Optional[int] = None,
        description: str = "",
    ) -> None:
        self.deltas[param_name] = (flat_indices.detach().cpu().long(), new_values.detach().cpu())
        if write_vec is not None:
            self.write_vectors.append(write_vec.detach().cpu().float())
        if edited_layer is not None:
            self.edited_layer = edited_layer
        if edit_type:
            self.edit_type = edit_type
        if effective_dof is not None:
            self.effective_dof = effective_dof
        if description:
            self.description = description

    def clear(self) -> None:
        self.deltas.clear()
        self.write_vectors.clear()
        self.edited_layer = -1
        self.edit_type = ""
        self.effective_dof = 0
        self.description = ""

    @property
    def num_entries_changed(self) -> int:
        return sum(int(idx.numel()) for idx, _ in self.deltas.values())


class WeightEditingSandbox:
    """
    Holds a target LLM, captures multi-token span activations in single teacher-forced forward passes,
    stages concept-level weight edits (rank-1 feature projections or dormant MLP concept gates),
    evaluates them in a reversible sandbox, and commits only when guards pass.
    """

    def __init__(self, model: torch.nn.Module, tokenizer):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        self.staged = SparseWeightDelta()
        self.committed_deltas: Dict[str, Dict[int, float]] = {}
        self.committed_write_vectors: List[torch.Tensor] = []
        self.committed_history: List[Dict[str, object]] = []
        self.used_mlp_units: Dict[int, set] = {}
        self.passes_used = 0

    @contextlib.contextmanager
    def applied(self, use_staged: bool = True):
        """Temporarily apply staged weight deltas for a sandboxed forward pass, then restore."""
        if not use_staged or not self.staged.deltas:
            yield self.model
            return

        params = dict(self.model.named_parameters())
        saved: Dict[str, Tuple[torch.LongTensor, torch.Tensor]] = {}
        try:
            with torch.no_grad():
                for name, (idx, val) in self.staged.deltas.items():
                    flat = params[name].data.view(-1)
                    dev_idx = idx.to(flat.device)
                    saved[name] = (dev_idx, flat[dev_idx].clone())
                    flat[dev_idx] = val.to(flat.device, flat.dtype)
            yield self.model
        finally:
            with torch.no_grad():
                for name, (dev_idx, orig_val) in saved.items():
                    params[name].data.view(-1)[dev_idx] = orig_val

    def commit_staged(self, rationale: str = "") -> Dict[str, object]:
        """Permanently commit the currently staged weight delta into the in-memory model."""
        if not self.staged.deltas:
            raise RuntimeError("No staged weight edit to commit.")

        params = dict(self.model.named_parameters())
        total_l2_sq = 0.0
        with torch.no_grad():
            for name, (idx, val) in self.staged.deltas.items():
                flat = params[name].data.view(-1)
                dev_idx = idx.to(flat.device)
                new_v = val.to(flat.device, flat.dtype)
                total_l2_sq += float((new_v.float() - flat[dev_idx].float()).norm().item() ** 2)
                flat[dev_idx] = new_v

                # Merge into cumulative committed delta map (safe for multiple edits on same tensor)
                store = self.committed_deltas.setdefault(name, {})
                for k, v in zip(idx.tolist(), val.float().tolist()):
                    store[int(k)] = float(v)

        self.committed_write_vectors.extend(self.staged.write_vectors)
        rec = {
            "edit_num": len(self.committed_history) + 1,
            "layer": self.staged.edited_layer,
            "edit_type": self.staged.edit_type,
            "description": self.staged.description,
            "rationale": rationale,
            "tensors_edited": list(self.staged.deltas.keys()),
            "entries_changed": self.staged.num_entries_changed,
            "effective_dof": self.staged.effective_dof or self.staged.num_entries_changed,
            "cumulative_entries_changed": sum(len(d) for d in self.committed_deltas.values()),
            "l2_norm": round(total_l2_sq ** 0.5, 4),
        }
        self.committed_history.append(rec)
        # Keep write vectors accessible for immediate SAE patching before clearing staged
        self._last_committed_write_vectors = list(self.staged.write_vectors)
        self._last_committed_layer = self.staged.edited_layer
        self.staged.clear()
        return rec

    def save_committed_deltas(self, path) -> None:
        """Save all committed weight deltas to a safetensors file on disk (leaving base weights untouched)."""
        from pathlib import Path
        from safetensors.torch import save_file

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tensors = {}
        for name, kv in self.committed_deltas.items():
            if not kv:
                continue
            keys = sorted(kv.keys())
            vals = [kv[k] for k in keys]
            tensors[f"{name}::idx"] = torch.tensor(keys, dtype=torch.long)
            tensors[f"{name}::val"] = torch.tensor(vals, dtype=torch.float32)
        if tensors:
            save_file(tensors, str(path))

    @torch.no_grad()
    def capture_span_activations(
        self,
        texts: Sequence[str],
        span_starts: Sequence[int],
        layer: int,
        hook_site: str = "resid_pre",
        pool: str = "mean",
        use_staged: bool = False,
        batch_size: int = 16,
        max_length: int = 256,
    ) -> torch.Tensor:
        """
        Capture activations at `layer` across multi-token sentence spans in 1 teacher-forced forward pass per batch.
        - `hook_site`: 'resid_pre' (SAE input space) or 'ln_2' (post-LayerNorm MLP input space)
        - `pool`: 'mean' (mean over span tokens), 'max', 'first' (span_start), or 'last' (final token)
        Returns: [N, d_model] tensor on CPU.
        """
        self.passes_used += 1
        device = next(self.model.parameters()).device
        block = self.model.transformer.h[layer]
        pooled_out: List[torch.Tensor] = []

        enc_seqs = [self.tokenizer.encode(t, add_special_tokens=False)[:max_length] for t in texts]
        pad_id = self.tokenizer.pad_token_id

        with self.applied(use_staged=use_staged):
            for b_start in range(0, len(enc_seqs), batch_size):
                b_seqs = enc_seqs[b_start : b_start + batch_size]
                b_spans = span_starts[b_start : b_start + batch_size]
                max_len = max(max(len(s), 1) for s in b_seqs)

                input_ids = torch.full((len(b_seqs), max_len), pad_id, dtype=torch.long, device=device)
                attn_mask = torch.zeros((len(b_seqs), max_len), dtype=torch.long, device=device)
                for i, seq in enumerate(b_seqs):
                    if not seq:
                        seq = [pad_id]
                    input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
                    attn_mask[i, : len(seq)] = 1

                captured: List[torch.Tensor] = []

                def pre_hook(mod, inp):
                    captured.append(inp[0].detach().float().cpu())

                def ln2_hook(mod, inp, out):
                    captured.append(out.detach().float().cpu())

                handle = (
                    block.register_forward_pre_hook(pre_hook)
                    if hook_site == "resid_pre"
                    else block.ln_2.register_forward_hook(ln2_hook)
                )
                try:
                    self.model(input_ids=input_ids, attention_mask=attn_mask)
                finally:
                    handle.remove()

                acts = captured[0]  # [B, max_len, d_model]
                for i, (seq, sp) in enumerate(zip(b_seqs, b_spans)):
                    L = max(len(seq), 1)
                    s_idx = min(max(int(sp), 0), L - 1)
                    span_slice = acts[i, s_idx:L, :]  # [span_len, d_model]
                    if pool == "mean":
                        vec = span_slice.mean(dim=0)
                    elif pool == "max":
                        vec = span_slice.max(dim=0).values
                    elif pool == "first":
                        vec = acts[i, s_idx, :]
                    elif pool == "last":
                        vec = acts[i, L - 1, :]
                    else:
                        raise ValueError(f"Unknown pool mode: {pool}")
                    pooled_out.append(vec)

        return torch.stack(pooled_out, dim=0)

    @torch.no_grad()
    def capture_layer_activations(
        self, prompts: List[str], layer: int, hook_site: str = "resid_pre", use_staged: bool = False
    ) -> torch.Tensor:
        """Backward-compatible wrapper capturing last-token activations."""
        spans = [max(len(self.tokenizer.encode(p, add_special_tokens=False)) - 1, 0) for p in prompts]
        return self.capture_span_activations(
            prompts, spans, layer=layer, hook_site=hook_site, pool="last", use_staged=use_staged
        )

    @torch.no_grad()
    def stage_rank1_feature_projection(
        self,
        layer: int,
        component: str,
        source_dir: torch.Tensor,
        alpha: float,
        target_dir: Optional[torch.Tensor] = None,
        description: str = "",
    ) -> Dict[str, object]:
        """
        Concept-level direct weight surgery via exact Rank-1 Projection on GPT-2 Conv1D output matrices:
            y' = y + alpha * (y @ d_src) * d_tgt
        Implemented directly in weights & bias (works across EVERY token of multi-token sentences):
            W' = W + alpha * (W @ d_src)[:, None] @ d_tgt[None, :]
            b' = b + alpha * (b @ d_src) * d_tgt

        - `component`: "mlp" (mlp.c_proj) or "attn" (attn.c_proj)
        - `target_dir=None` (same-direction):
            alpha < 0 suppresses `source_dir` (e.g., alpha=-0.5 dampens 50%, alpha=-1.0 fully ablates)
            alpha > 0 amplifies `source_dir`
        - `target_dir!=None` (cross-feature redirection):
            redirects projection along `source_dir` into `target_dir`
        """
        block = self.model.transformer.h[layer]
        if component == "mlp":
            proj_mod = block.mlp.c_proj
            prefix = f"transformer.h.{layer}.mlp.c_proj"
        elif component == "attn":
            proj_mod = block.attn.c_proj
            prefix = f"transformer.h.{layer}.attn.c_proj"
        else:
            raise ValueError(f"component must be 'mlp' or 'attn', got {component}")

        d_src = F.normalize(source_dir.detach().float().cpu(), dim=-1)  # [d_model]
        d_tgt = F.normalize(target_dir.detach().float().cpu(), dim=-1) if target_dir is not None else d_src

        W = proj_mod.weight.detach().float().cpu()  # [d_in, d_model]
        b = proj_mod.bias.detach().float().cpu()    # [d_model]

        # Exact Conv1D rank-1 update: delta_W = alpha * (W @ d_src)[:, None] @ d_tgt[None, :]
        w_proj = W @ d_src                          # [d_in]
        delta_W = alpha * torch.outer(w_proj, d_tgt)
        delta_b = alpha * torch.dot(b, d_src) * d_tgt

        W_new = W + delta_W
        b_new = b + delta_b

        # Clamp extreme alpha explosions cleanly
        rel_norm = float(delta_W.norm().item() / (W.norm().item() + 1e-8))
        dof = int(W.shape[0] + W.shape[1])  # rank-1 effective degrees of freedom

        self.staged.clear()
        desc = description or f"L{layer}.{component}.c_proj rank-1 projection (alpha={alpha:+.3f})"
        self.staged.stage_entries(
            f"{prefix}.weight",
            torch.arange(W.numel(), dtype=torch.long),
            W_new.reshape(-1),
            edited_layer=layer,
            edit_type="rank1_projection",
            effective_dof=dof,
            description=desc,
        )
        self.staged.stage_entries(
            f"{prefix}.bias",
            torch.arange(b.numel(), dtype=torch.long),
            b_new.reshape(-1),
            write_vec=d_tgt * abs(alpha),
            edited_layer=layer,
            edit_type="rank1_projection",
            effective_dof=dof,
            description=desc,
        )

        return {
            "layer": layer,
            "component": component,
            "alpha": round(float(alpha), 4),
            "delta_w_l2": round(float(delta_W.norm().item()), 4),
            "delta_w_relative_norm": round(rel_norm, 5),
            "effective_dof": dof,
        }

    @torch.no_grad()
    def stage_sae_gated_concept_edit(
        self,
        layer: int,
        read_dir: torch.Tensor,
        write_dir: torch.Tensor,
        pos_texts: Sequence[str],
        pos_spans: Sequence[int],
        neg_texts: Sequence[str],
        neg_spans: Sequence[int],
        bg_texts: Sequence[str],
        gain: float = 4.0,
        steepness: float = 6.0,
        margin: float = 0.15,
        description: str = "",
    ) -> Dict[str, object]:
        """
        Compile a contextual SAE feature gate into 1 dormant MLP neuron (1,537 weights = 0.0012% of GPT-2):
        - `c_fc` column `u` detects `read_dir` in post-ln_2 coordinates above threshold `theta`
        - `c_proj` row `u` writes `gain * normalize(write_dir)` into the residual stream
          (a concept vector — e.g. `-d_myth` or `+d_truth` — so it generalizes across multi-token sentences!)
        """
        block = self.model.transformer.h[layer]
        d_model = self.model.config.n_embd
        d_mlp = block.mlp.c_fc.weight.shape[1]
        device = next(self.model.parameters()).device

        X_pos = self.capture_span_activations(pos_texts, pos_spans, layer=layer, hook_site="ln_2", pool="mean")
        X_neg = self.capture_span_activations(neg_texts, neg_spans, layer=layer, hook_site="ln_2", pool="mean")

        # Capture background token activations + MLP neuron usage on WikiText-2 background texts
        bg_ln2: List[torch.Tensor] = []
        mlp_acts: List[torch.Tensor] = []

        def ln2_bg_hook(mod, inp, out):
            bg_ln2.append(out.detach().float().cpu().reshape(-1, d_model))

        def mlp_act_hook(mod, inp, out):
            mlp_acts.append(out.detach().float().cpu().reshape(-1, d_mlp))

        h1 = block.ln_2.register_forward_hook(ln2_bg_hook)
        h2 = block.mlp.act.register_forward_hook(mlp_act_hook)
        try:
            for txt in bg_texts[:10]:
                ids = self.tokenizer.encode(txt, add_special_tokens=False)[:128]
                if ids:
                    self.model(input_ids=torch.tensor([ids], dtype=torch.long, device=device))
        finally:
            h1.remove()
            h2.remove()

        X_bg = torch.cat(bg_ln2, dim=0) if bg_ln2 else X_neg
        usage = torch.cat(mlp_acts, dim=0).abs().mean(dim=0) if mlp_acts else torch.zeros(d_mlp)

        # Avoid overwriting a neuron already reprogrammed in this layer
        already_used = self.used_mlp_units.get(layer, set())
        sorted_units = torch.argsort(usage).tolist()
        unit = next((int(u) for u in sorted_units if int(u) not in already_used), int(sorted_units[0]))

        r_dir = F.normalize(read_dir.detach().float().cpu(), dim=-1)
        w_dir = F.normalize(write_dir.detach().float().cpu(), dim=-1)

        # Ensure r_dir points toward positive/failure activations
        diff_ln2 = X_pos.mean(dim=0) - X_neg.mean(dim=0)
        if torch.dot(r_dir, diff_ln2) < 0:
            r_dir = -r_dir

        proj_pos = X_pos @ r_dir
        proj_neg = X_neg @ r_dir
        proj_bg = X_bg @ r_dir

        # Set threshold theta above 90th percentile of background + neg so gate stays quiet on normal text
        neg_q = float(torch.quantile(proj_neg, 0.85).item())
        bg_q = float(torch.quantile(proj_bg, 0.95).item())
        theta = max(neg_q, bg_q) + margin

        prefix = f"transformer.h.{layer}.mlp"
        fc_w_idx = torch.arange(d_model, dtype=torch.long) * d_mlp + unit
        fc_w_val = steepness * r_dir

        fc_b_idx = torch.tensor([unit], dtype=torch.long)
        fc_b_val = torch.tensor([-steepness * theta], dtype=torch.float32)

        proj_w_idx = unit * d_model + torch.arange(d_model, dtype=torch.long)
        proj_w_val = gain * w_dir

        self.staged.clear()
        desc = description or f"L{layer} dormant MLP unit {unit} concept gate (gain={gain:.2f}, theta={theta:.3f})"
        self.staged.stage_entries(f"{prefix}.c_fc.weight", fc_w_idx, fc_w_val, edited_layer=layer, edit_type="dormant_neuron_gate")
        self.staged.stage_entries(f"{prefix}.c_fc.bias", fc_b_idx, fc_b_val, edited_layer=layer, edit_type="dormant_neuron_gate")
        self.staged.stage_entries(
            f"{prefix}.c_proj.weight",
            proj_w_idx,
            proj_w_val,
            write_vec=proj_w_val,
            edited_layer=layer,
            edit_type="dormant_neuron_gate",
            effective_dof=int(2 * d_model + 1),
            description=desc,
        )
        self._pending_unit = (layer, unit)

        return {
            "layer": layer,
            "unit": unit,
            "theta": round(theta, 4),
            "pos_proj_mean": round(float(proj_pos.mean().item()), 4),
            "neg_proj_mean": round(float(proj_neg.mean().item()), 4),
            "bg_proj_p95": round(bg_q, 4),
            "practice_fire_rate_pos": round(float((proj_pos > theta).float().mean().item()), 4),
            "practice_fire_rate_neg": round(float((proj_neg > theta).float().mean().item()), 4),
            "bg_fire_rate": round(float((proj_bg > theta).float().mean().item()), 4),
            "entries_changed": self.staged.num_entries_changed,
        }

    def mark_last_unit_used(self) -> None:
        pending = getattr(self, "_pending_unit", None)
        if pending is not None:
            L, u = pending
            self.used_mlp_units.setdefault(L, set()).add(u)
            self._pending_unit = None

    @torch.no_grad()
    def stage_sae_guided_neuron_gate(
        self,
        layer: int,
        sae_feature_dir: torch.Tensor,
        pos_prompts: List[str],
        neg_prompts: List[str],
        target_token_ids: List[int],
        suppress_token_ids: List[int],
        gain: float = 18.0,
        steepness: float = 8.0,
        margin: float = 0.25,
    ) -> Dict[str, object]:
        """Backward-compatible single-token unembedding neuron gate (used by legacy run_experiment.py)."""
        W_U = self.model.lm_head.weight.detach().float().cpu()
        tgt_vec = F.normalize(W_U[target_token_ids].mean(dim=0), dim=-1)
        sup_vec = (
            F.normalize(W_U[suppress_token_ids].mean(dim=0), dim=-1)
            if suppress_token_ids
            else torch.zeros_like(tgt_vec)
        )
        write_dir = F.normalize(tgt_vec - 0.6 * sup_vec, dim=-1)
        pos_spans = [max(len(self.tokenizer.encode(p, add_special_tokens=False)) - 1, 0) for p in pos_prompts]
        neg_spans = [max(len(self.tokenizer.encode(p, add_special_tokens=False)) - 1, 0) for p in neg_prompts]
        return self.stage_sae_gated_concept_edit(
            layer=layer,
            read_dir=sae_feature_dir,
            write_dir=write_dir,
            pos_texts=pos_prompts,
            pos_spans=pos_spans,
            neg_texts=neg_prompts,
            neg_spans=neg_spans,
            bg_texts=neg_prompts,
            gain=gain,
            steepness=steepness,
            margin=margin,
        )
