"""
Main runner for Outside-Agent SAE-Guided Weight Editing on Standardized LLM Benchmarks
(TruthfulQA MC1/MC2 and BLiMP, with WikiText-2 perplexity/KL guardrails & optional EleutherAI `lm-eval`).

Usage:
  1. Put your OpenAI API key in `.env` (see `.env.example`)
  2. Run with OpenAI API agent:
       python3 run_agent_benchmark.py --benchmark truthfulqa --arm patched_sae
       python3 run_agent_benchmark.py --benchmark blimp --arm patched_sae
  3. Run full 3-arm ablation (`no_sae` vs `frozen_sae` vs `patched_sae`):
       python3 run_agent_benchmark.py --benchmark blimp --ablation-all
  4. Zero-cost dry run without spending OpenAI API credits:
       python3 run_agent_benchmark.py --benchmark blimp --heuristic --ablation-all
  5. Optional official EleutherAI `lm-eval` check on Base vs. Final Edited model:
       python3 run_agent_benchmark.py --benchmark truthfulqa --official-lm-eval --lm-eval-limit 100
"""

import importlib.metadata
_orig_version = importlib.metadata.version
def _patched_version(pkg):
    v = _orig_version(pkg)
    return v.split(".dev")[0] if pkg == "torch" and ".dev" in v else v
importlib.metadata.version = _patched_version

import argparse
import json
from pathlib import Path
import time
import torch

torch.__version__ = torch.__version__.split(".dev")[0]
from transformers import AutoModelForCausalLM, AutoTokenizer

from mwe import (
    SparseAutoencoder,
    WeightEditingAgentSession,
    WeightEditingSandbox,
    load_blimp_benchmark,
    load_openai_config,
    load_truthfulqa_benchmark,
    run_heuristic_agent_loop,
    run_official_lm_eval,
    run_openai_agent_loop,
)

CACHE_DIR = Path("/tmp/mwe_cache")
RUN_DIR = Path("/tmp/mwe_runs")
LAYERS = [6, 7, 8, 9, 10, 11]


def _resolve_local_gpt2_path(cache_dir: Path) -> str:
    snap_root = cache_dir / "models--gpt2" / "snapshots"
    if snap_root.exists():
        snaps = sorted(p for p in snap_root.iterdir() if p.is_dir() and (p / "config.json").exists())
        if snaps:
            return str(snaps[-1])
    return "gpt2"


def load_fresh_gpt2_and_saes(load_saes: bool = True):
    model_ref = _resolve_local_gpt2_path(CACHE_DIR)
    is_local = Path(model_ref).exists()
    tok = AutoTokenizer.from_pretrained(model_ref, cache_dir=str(CACHE_DIR), local_files_only=is_local)
    model = AutoModelForCausalLM.from_pretrained(model_ref, cache_dir=str(CACHE_DIR), local_files_only=is_local).eval()
    saes = {L: SparseAutoencoder.from_pretrained_gpt2(layer=L, cache_dir=CACHE_DIR) for L in LAYERS} if load_saes else {}
    return model, tok, saes


def run_single_arm(args, benchmark, arm: str, cfg: dict) -> dict:
    t0 = time.time()
    print("\n" + "=" * 88)
    print(f"ARM: {arm.upper()} | BENCHMARK: {benchmark.name} | CONTROLLER: {'HEURISTIC' if args.heuristic else cfg['model']}")
    print("=" * 88)

    model, tok, saes = load_fresh_gpt2_and_saes(load_saes=(arm != "no_sae"))
    sandbox = WeightEditingSandbox(model, tok)

    session = WeightEditingAgentSession(
        sandbox=sandbox,
        saes=saes,
        benchmark=benchmark,
        arm=arm,
        layers=LAYERS,
        max_edits=args.max_edits or cfg["max_edits"],
        max_kl=args.max_kl,
        min_keeper_retention=args.min_keeper,
        fetch_neuronpedia=not args.no_neuronpedia,
        run_dir=RUN_DIR / f"{benchmark.name}_seed{args.seed}",
        cache_dir=CACHE_DIR,
    )

    bp = session.base_prac
    bh = session.base_held
    print(
        f"[Base GPT-2] Practice {bp['primary_metric_name']}={bp['primary_score']:.4f} (margin={bp['mean_margin']:+.4f}) | "
        f"Held-Out {bh['primary_metric_name']}={bh['primary_score']:.4f} (margin={bh['mean_margin']:+.4f}) | "
        f"WikiText-2 PPL={bh['wikitext2_ppl']:.2f}"
    )

    if args.heuristic:
        report = run_heuristic_agent_loop(session, verbose=True)
    else:
        api_key = cfg["api_key"]
        if not api_key:
            raise SystemExit(
                f"\nERROR: No valid OPENAI_API_KEY found!\n"
                f"  -> Paste your key into: {cfg['env_path']}\n"
                f"  -> Or run with `--heuristic` for a $0 local dry-run.\n"
            )
        model_name = args.model or cfg["model"]
        reasoning_effort = args.reasoning_effort or cfg["reasoning_effort"]
        max_turns = args.max_turns or cfg["max_turns"]
        print(
            f"Using OpenAI API key from {cfg['key_source']} | Model: {model_name} "
            f"(reasoning_effort={reasoning_effort}) | Max turns: {max_turns}"
        )
        report = run_openai_agent_loop(
            session=session,
            api_key=api_key,
            model_name=model_name,
            max_turns=max_turns,
            reasoning_effort=reasoning_effort,
            verbose=True,
        )

    if args.official_lm_eval:
        lm_tasks = ["truthfulqa_mc1", "truthfulqa_mc2"] if benchmark.name == "truthfulqa" else [
            f"blimp_{s}" for s in benchmark.metadata["subsets"][:2]
        ]
        print(f"\nRunning official EleutherAI `lm-eval` verification on edited model ({lm_tasks})...")
        report["official_lm_eval_edited"] = run_official_lm_eval(
            sandbox.model, tok, tasks=lm_tasks, limit=args.lm_eval_limit
        )

    report["elapsed_sec"] = round(time.time() - t0, 2)
    fh = report["final_heldout"]
    print(
        f"\n[FINAL HELD-OUT | {arm}] "
        f"{bh['primary_metric_name']}: {bh['primary_score']:.4f} -> {fh['primary_score']:.4f} "
        f"({report['heldout_primary_delta']:+.4f}) | "
        f"Margin: {bh['mean_margin']:+.4f} -> {fh['mean_margin']:+.4f} ({report['heldout_margin_delta']:+.4f}) | "
        f"Keeper={fh['keeper_retention']*100:.1f}% | "
        f"WikiText-2 PPL={fh['wikitext2_ppl']:.2f} (KL={fh['wikitext2_kl']:.5f}) | "
        f"Edits={report['edits_committed']} | Time={report['elapsed_sec']}s"
    )
    return report


def run_paired_branch_ablation(args, benchmark, cfg: dict) -> dict:
    """
    True apples-to-apples causal test of Local Subspace SAE Patching:
      1. Shared Phase 1: Run 1 early/mid-layer edit (Layers 6..9) starting from Base GPT-2.
      2. Fork the EXACT post-Edit-1 model weights into two isolated downstream branches (L > edit1_layer):
         - Branch A (`frozen_sae`):  Downstream SAEs remain untouched/frozen.
         - Branch B (`patched_sae`): Downstream SAEs are immediately locally patched (<0.15% latents)
                                     for Edit #1 and after every subsequent commit.
      3. Evaluate Base, Shared Edit #1, Branch A Final, and Branch B Final on the identical Held-Out split.
    """
    t0 = time.time()
    model_name = "deterministic_heuristic" if args.heuristic else (args.model or cfg["model"])
    reasoning_effort = args.reasoning_effort or cfg["reasoning_effort"]
    total_max_edits = args.max_edits or cfg["max_edits"]
    phase2_turns = args.max_turns or cfg["max_turns"]
    phase1_turns = min(12, phase2_turns)

    if not args.heuristic and not cfg["api_key"]:
        raise SystemExit(
            f"\nERROR: No valid OPENAI_API_KEY found!\n"
            f"  -> Paste your key into: {cfg['env_path']}\n"
            f"  -> Or run with `--heuristic` for a $0 local dry-run.\n"
        )

    print("\n" + "=" * 96)
    print(f"PAIRED-BRANCH CONTROLLED ABLATION | BENCHMARK: {benchmark.name} | CONTROLLER: {model_name}")
    print("=" * 96)
    print("BASELINE ARM: NO_SAE (raw contrastive diff_dir from Base GPT-2, full layers [6..11])...")
    no_sae_report = run_single_arm(args, benchmark, "no_sae", cfg)

    print("\n" + "-" * 96)
    print("PHASE 1 (SHARED SAE): Discovering & committing Edit #1 on early/mid layers [6, 7, 8, 9]...")
    print("-" * 96)

    model, tok, saes = load_fresh_gpt2_and_saes(load_saes=True)
    sandbox = WeightEditingSandbox(model, tok)
    phase1_session = WeightEditingAgentSession(
        sandbox=sandbox,
        saes=saes,
        benchmark=benchmark,
        arm="frozen_sae",  # Keep SAEs untouched in Phase 1 so Branch B applies patching cleanly at fork time
        layers=[6, 7, 8, 9],
        max_edits=1,
        max_kl=args.max_kl,
        min_keeper_retention=args.min_keeper,
        fetch_neuronpedia=not args.no_neuronpedia,
        run_dir=RUN_DIR / f"{benchmark.name}_paired_seed{args.seed}",
        cache_dir=CACHE_DIR,
    )
    # Also precompute baseline layer acts for L10 and L11 so Branch B can locally patch them right after fork!
    for L_extra in (10, 11):
        if L_extra not in phase1_session._baseline_layer_acts:
            phase1_session._baseline_layer_acts[L_extra] = phase1_session.sandbox.capture_span_activations(
                phase1_session._patch_texts,
                phase1_session._patch_spans,
                layer=L_extra,
                hook_site="resid_pre",
                pool="mean",
                use_staged=False,
            )

    bp = phase1_session.base_prac
    bh = phase1_session.base_held
    print(
        f"[Base GPT-2] Practice {bp['primary_metric_name']}={bp['primary_score']:.4f} (margin={bp['mean_margin']:+.4f}) | "
        f"Held-Out {bh['primary_metric_name']}={bh['primary_score']:.4f} (margin={bh['mean_margin']:+.4f}) | "
        f"WikiText-2 PPL={bh['wikitext2_ppl']:.2f}"
    )

    if args.heuristic:
        p1_report = run_heuristic_agent_loop(phase1_session, verbose=True)
    else:
        p1_report = run_openai_agent_loop(
            session=phase1_session,
            api_key=cfg["api_key"],
            model_name=model_name,
            max_turns=phase1_turns,
            reasoning_effort=reasoning_effort,
            verbose=True,
            extra_user_context=(
                "PHASE 1 (SHARED BASELINE EDIT): Commit EXACTLY ONE strong, clean edit on early/mid layers [6, 7, 8, 9] "
                "(prefer Layer 6 or 7 first so maximum downstream layers remain available!). As soon as you commit Edit #1, "
                "this phase ends and your exact post-Edit-1 model will be forked into `frozen_sae` vs `patched_sae` branches."
            ),
            stop_when_edits_reached=True,
        )

    if not phase1_session.sandbox.committed_history:
        raise RuntimeError("Phase 1 failed to commit Edit #1 on layers [6..9]; cannot fork paired branches.")

    edit1_rec = phase1_session.sandbox.committed_history[0]
    edit1_layer = int(edit1_rec["layer"])
    downstream_layers = [L for L in LAYERS if L > edit1_layer]
    e1_held = p1_report["final_heldout"]
    print(
        f"\n[SHARED EDIT #1 COMMITTED @ L{edit1_layer}] "
        f"Held-Out {bh['primary_metric_name']}: {bh['primary_score']:.4f} -> {e1_held['primary_score']:.4f} "
        f"({p1_report['heldout_primary_delta']:+.4f}) | Margin: {e1_held['mean_margin']:+.4f} "
        f"({p1_report['heldout_margin_delta']:+.4f}) | Downstream layers to fork: {downstream_layers}"
    )

    branch_reports = {}
    for branch_arm in ("frozen_sae", "patched_sae"):
        t_b = time.time()
        print("\n" + "-" * 96)
        print(f"PHASE 2 BRANCH: {branch_arm.upper()} (forked from IDENTICAL post-Edit-1 L{edit1_layer} weights | layers={downstream_layers})")
        print("-" * 96)

        b_session = phase1_session.clone_branch(
            target_arm=branch_arm,
            new_layers=downstream_layers,
            new_max_edits=total_max_edits,
            apply_initial_sae_patch=(branch_arm == "patched_sae"),
        )
        if b_session.sae_patch_logs:
            print("  [Initial Fork Local SAE Patching after Shared Edit #1]:")
            for p in b_session.sae_patch_logs:
                print(
                    f"    -> Patched L{p['downstream_layer']} SAE ({p['patched_pct']}% latents): "
                    f"MSE {p['mse_before']:.2f} -> {p['mse_after']:.2f}"
                )

        if args.heuristic:
            b_rep = run_heuristic_agent_loop(b_session, verbose=True)
        else:
            b_rep = run_openai_agent_loop(
                session=b_session,
                api_key=cfg["api_key"],
                model_name=model_name,
                max_turns=phase2_turns,
                reasoning_effort=reasoning_effort,
                verbose=True,
                extra_user_context=(
                    f"PHASE 2 (FORKED DOWNSTREAM BRANCH `{branch_arm}`): Edit #1 at Layer {edit1_layer} "
                    f"({edit1_rec['description']}) is ALREADY committed into the model. "
                    f"Your available downstream layers are {downstream_layers}. You may commit up to "
                    f"{total_max_edits - 1} additional downstream edits to maximize held-out generalization while passing all guards."
                ),
            )

        b_rep["elapsed_sec"] = round(time.time() - t_b, 2)
        b_rep["phase2_heldout_delta"] = round(
            b_rep["final_heldout"]["primary_score"] - e1_held["primary_score"], 4
        )
        b_rep["phase2_heldout_margin_delta"] = round(
            b_rep["final_heldout"]["mean_margin"] - e1_held["mean_margin"], 4
        )
        b_rep["phase2_edits_added"] = int(b_rep["edits_committed"] - 1)
        branch_reports[branch_arm] = b_rep

        fh = b_rep["final_heldout"]
        print(
            f"\n[BRANCH {branch_arm.upper()} HELD-OUT] "
            f"Total: {bh['primary_score']:.4f} -> {fh['primary_score']:.4f} ({b_rep['heldout_primary_delta']:+.4f}) | "
            f"Phase-2 Only (vs Edit #1): {b_rep['phase2_heldout_delta']:+.4f} "
            f"(Margin {b_rep['phase2_heldout_margin_delta']:+.4f}) | "
            f"Keeper={fh['keeper_retention']*100:.1f}% | KL={fh['wikitext2_kl']:.5f} | Phase-2 Edits={b_rep['phase2_edits_added']}"
        )

    return {
        "mode": "paired_branch",
        "benchmark": benchmark.name,
        "seed": args.seed,
        "controller": model_name,
        "downstream_layers": downstream_layers,
        "elapsed_total_sec": round(time.time() - t0, 2),
        "no_sae": no_sae_report,
        "phase1_shared": p1_report,
        "frozen_sae": branch_reports["frozen_sae"],
        "patched_sae": branch_reports["patched_sae"],
    }


def main():
    parser = argparse.ArgumentParser(description="Outside-Agent SAE-Guided Direct Weight Editing Benchmark Harness")
    parser.add_argument("--benchmark", choices=["truthfulqa", "blimp"], default="truthfulqa")
    parser.add_argument("--arm", choices=["no_sae", "frozen_sae", "patched_sae"], default="patched_sae")
    parser.add_argument("--ablation-all", action="store_true", help="Run all 3 arms (no_sae, frozen_sae, patched_sae) back-to-back")
    parser.add_argument("--paired-branch", action="store_true", help="True apples-to-apples causal test: share Edit #1, then fork into frozen_sae vs patched_sae")
    parser.add_argument("--heuristic", action="store_true", help="Use zero-cost deterministic controller instead of OpenAI API")
    parser.add_argument("--model", type=str, default=None, help="Override OpenAI model from .env (e.g. gpt-6.1-sol, gpt-6.2-astra, gpt-6.1-terra)")
    parser.add_argument("--reasoning-effort", type=str, default=None, choices=["low", "medium", "high"], help="Reasoning effort for GPT-6 / o-series models (default from .env: medium)")
    parser.add_argument("--max-turns", type=int, default=None, help="Max OpenAI tool-calling turns")
    parser.add_argument("--max-edits", type=int, default=None, help="Max committed weight edits")
    parser.add_argument("--max-kl", type=float, default=0.05, help="Max allowed WikiText-2 KL divergence")
    parser.add_argument("--min-keeper", type=float, default=0.90, help="Min required keeper retention rate")
    parser.add_argument("--practice-size", type=int, default=40, help="Practice items (TruthfulQA total, or per-subset for BLiMP)")
    parser.add_argument("--heldout-size", type=int, default=60, help="Held-out test items (TruthfulQA total, or per-subset for BLiMP)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-neuronpedia", action="store_true", help="Skip fetching Neuronpedia SAE feature labels")
    parser.add_argument("--official-lm-eval", action="store_true", help="Run EleutherAI lm-eval verification at the end")
    parser.add_argument("--lm-eval-limit", type=int, default=100, help="Optional item limit for official lm-eval")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for JSON report (default: ./results)")
    args = parser.parse_args()

    cfg = load_openai_config(Path(__file__).resolve().parent)

    print(f"Loading standardized `{args.benchmark}` benchmark + WikiText-2 collateral guardrail...")
    if args.benchmark == "truthfulqa":
        benchmark = load_truthfulqa_benchmark(
            practice_size=args.practice_size,
            heldout_size=args.heldout_size,
            seed=args.seed,
            cache_dir=CACHE_DIR,
        )
    else:
        benchmark = load_blimp_benchmark(
            practice_per_subset=args.practice_size,
            heldout_per_subset=args.heldout_size,
            seed=args.seed,
            cache_dir=CACHE_DIR,
        )

    out_dir = Path(args.out_dir) if args.out_dir else (Path(__file__).resolve().parent / "results")
    ctrl = "heuristic" if args.heuristic else (args.model or cfg["model"]).replace("/", "_")

    if args.paired_branch:
        paired = run_paired_branch_ablation(args, benchmark, cfg)
        stem = f"{args.benchmark}_paired_branch_{ctrl}_seed{args.seed}"

        transcripts = {}
        clean_paired = dict(paired)
        for key in ("no_sae", "phase1_shared", "frozen_sae", "patched_sae"):
            sub = dict(paired[key])
            if "_transcript" in sub:
                transcripts[key] = sub.pop("_transcript")
            clean_paired[key] = sub

        def _write_paired(base_dir: Path) -> Path:
            base_dir.mkdir(parents=True, exist_ok=True)
            out_path = base_dir / f"{stem}.json"
            out_path.write_text(json.dumps(clean_paired, indent=2))
            if transcripts:
                t_dir = base_dir / "transcripts"
                t_dir.mkdir(parents=True, exist_ok=True)
                for k, tr in transcripts.items():
                    (t_dir / f"{args.benchmark}_paired_{k}_{ctrl}_seed{args.seed}_transcript.json").write_text(
                        json.dumps(tr, indent=2)
                    )

            p1 = clean_paired["phase1_shared"]
            ns = clean_paired["no_sae"]
            bh = p1["base_heldout"]
            e1h = p1["final_heldout"]
            e1_rec = p1["committed_history"][0]
            md = [
                f"# {args.benchmark.upper()} Paired-Branch Controlled Ablation (`{ctrl}`, seed={args.seed})",
                "",
                f"**Shared SAE Edit #1 (L{e1_rec['layer']}, `{e1_rec['edit_type']}`):** {e1_rec['description']}",
                f"- Held-Out Base -> Edit #1: `{bh['primary_score']:.4f}` -> `{e1h['primary_score']:.4f}` (`{p1['heldout_primary_delta']:+.4f}`), Margin `{bh['mean_margin']:+.4f}` -> `{e1h['mean_margin']:+.4f}` (`{p1['heldout_margin_delta']:+.4f}`)",
                "",
                "| Arm | Base Held-Out | Final Held-Out | **Total Δ** | **Phase-2 Δ (vs Edit #1)** | Margin Δ | Keeper | WT2 PPL | WT2 KL | Total Edits |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
            ns_fh = ns["final_heldout"]
            md.append(
                f"| `no_sae` *(baseline)* | {bh['primary_score']:.4f} | {ns_fh['primary_score']:.4f} | "
                f"**{ns['heldout_primary_delta']:+.4f}** | — | {ns['heldout_margin_delta']:+.4f} | "
                f"{ns_fh['keeper_retention']*100:.1f}% | {ns_fh['wikitext2_ppl']:.2f} | "
                f"{ns_fh['wikitext2_kl']:.5f} | {ns['edits_committed']} |"
            )
            for b_arm in ("frozen_sae", "patched_sae"):
                r = clean_paired[b_arm]
                fh = r["final_heldout"]
                md.append(
                    f"| `{b_arm}` | {bh['primary_score']:.4f} | {fh['primary_score']:.4f} | "
                    f"**{r['heldout_primary_delta']:+.4f}** | **{r['phase2_heldout_delta']:+.4f}** | "
                    f"{r['heldout_margin_delta']:+.4f} | {fh['keeper_retention']*100:.1f}% | "
                    f"{fh['wikitext2_ppl']:.2f} | {fh['wikitext2_kl']:.5f} | {r['edits_committed']} |"
                )
            md.append("")
            md.append("## Baseline Arm: `no_sae`")
            if ns.get("finish_summary"):
                md.append(f"**Agent Summary:** {ns['finish_summary']}")
            for c in ns.get("committed_history", []):
                md.append(f"- **Edit #{c['edit_num']} (L{c['layer']}, {c['edit_type']}):** {c['description']} — _{c.get('rationale', '')}_")
            md.append("")
            for b_arm in ("frozen_sae", "patched_sae"):
                r = clean_paired[b_arm]
                md.append(f"## Phase-2 Branch: `{b_arm}`")
                if r.get("finish_summary"):
                    md.append(f"**Agent Summary:** {r['finish_summary']}")
                for c in r.get("committed_history", [])[1:]:
                    md.append(f"- **Edit #{c['edit_num']} (L{c['layer']}, {c['edit_type']}):** {c['description']} — _{c.get('rationale', '')}_")
                if r.get("sae_patch_logs"):
                    md.append("")
                    md.append("**Local Subspace SAE Patching Logs:**")
                    for p in r["sae_patch_logs"]:
                        md.append(
                            f"- After Edit #{p['edit_num']} (L{p['edited_layer']}) -> Patched L{p['downstream_layer']} SAE "
                            f"({p['patched_pct']}% latents): MSE `{p['mse_before']:.2f}` -> `{p['mse_after']:.2f}`"
                        )
                md.append("")
            (base_dir / f"{stem}.md").write_text("\n".join(md))
            return out_path

        try:
            out_path = _write_paired(out_dir)
        except PermissionError:
            out_path = _write_paired(RUN_DIR / "results")

        p1 = clean_paired["phase1_shared"]
        ns = clean_paired["no_sae"]
        ns_fh = ns["final_heldout"]
        bh = p1["base_heldout"]
        e1h = p1["final_heldout"]
        print(f"\nSaved paired-branch report: {out_path}")
        print(f"Saved markdown summary:     {out_path.with_suffix('.md')}")
        print("\n" + "=" * 112)
        print(f"PAIRED-BRANCH SUMMARY ({args.benchmark.upper()} | HELD-OUT TEST SPLIT | seed={args.seed})")
        print("=" * 112)
        print(f"Shared SAE Base -> Edit #1 (L{p1['committed_history'][0]['layer']}): {bh['primary_score']:.4f} -> {e1h['primary_score']:.4f} ({p1['heldout_primary_delta']:+.4f}) | Margin {bh['mean_margin']:+.4f} -> {e1h['mean_margin']:+.4f}")
        print("-" * 112)
        print(f"{'Arm':<14} | {'Base':>7} | {'Final':>7} | {'Total Δ':>8} | {'Phase-2 Δ':>10} | {'Margin Δ':>9} | {'Keeper':>7} | {'WT2 PPL':>8} | {'WT2 KL':>8} | {'Edits':>5}")
        print("-" * 112)
        print(
            f"{'no_sae':<14} | {bh['primary_score']:7.4f} | {ns_fh['primary_score']:7.4f} | "
            f"{ns['heldout_primary_delta']:+8.4f} | {'—':>10} | {ns['heldout_margin_delta']:+9.4f} | "
            f"{ns_fh['keeper_retention']*100:6.1f}% | {ns_fh['wikitext2_ppl']:8.2f} | "
            f"{ns_fh['wikitext2_kl']:8.5f} | {ns['edits_committed']:5d}"
        )
        for b_arm in ("frozen_sae", "patched_sae"):
            r = clean_paired[b_arm]
            fh = r["final_heldout"]
            print(
                f"{b_arm:<14} | {bh['primary_score']:7.4f} | {fh['primary_score']:7.4f} | "
                f"{r['heldout_primary_delta']:+8.4f} | {r['phase2_heldout_delta']:+10.4f} | "
                f"{r['heldout_margin_delta']:+9.4f} | {fh['keeper_retention']*100:6.1f}% | "
                f"{fh['wikitext2_ppl']:8.2f} | {fh['wikitext2_kl']:8.5f} | {r['edits_committed']:5d}"
            )
        return

    arms = ["no_sae", "frozen_sae", "patched_sae"] if args.ablation_all else [args.arm]
    all_reports = {}
    for arm in arms:
        all_reports[arm] = run_single_arm(args, benchmark, arm, cfg)

    tag = "ablation_all" if args.ablation_all else args.arm
    stem = f"{args.benchmark}_{tag}_{ctrl}_seed{args.seed}"

    # Separate verbose transcripts into results/transcripts/ so the main report stays clean and easy to inspect
    transcripts = {}
    clean_reports = {}
    for arm, rep in all_reports.items():
        rep_copy = dict(rep)
        if "_transcript" in rep_copy:
            transcripts[arm] = rep_copy.pop("_transcript")
        clean_reports[arm] = rep_copy

    def _write_outputs(base_dir: Path) -> Path:
        base_dir.mkdir(parents=True, exist_ok=True)
        out_path = base_dir / f"{stem}.json"
        out_path.write_text(json.dumps(clean_reports, indent=2))

        if transcripts:
            t_dir = base_dir / "transcripts"
            t_dir.mkdir(parents=True, exist_ok=True)
            for arm, tr in transcripts.items():
                (t_dir / f"{args.benchmark}_{arm}_{ctrl}_seed{args.seed}_transcript.json").write_text(
                    json.dumps(tr, indent=2)
                )

        # Write a human-readable Markdown summary table + committed edit log
        md_lines = [
            f"# {args.benchmark.upper()} Benchmark Findings (`{ctrl}`, seed={args.seed})",
            "",
            "| Arm | Base Held-Out | Final Held-Out | Delta | Margin Δ | Keeper | WikiText-2 PPL | WikiText-2 KL | Edits | API Tokens |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for arm in arms:
            r = clean_reports[arm]
            b, f = r["base_heldout"], r["final_heldout"]
            toks = r.get("api_usage", {}).get("total_tokens", 0)
            md_lines.append(
                f"| `{arm}` | {b['primary_score']:.4f} | {f['primary_score']:.4f} | {r['heldout_primary_delta']:+.4f} | "
                f"{r['heldout_margin_delta']:+.4f} | {f['keeper_retention']*100:.1f}% | {f['wikitext2_ppl']:.2f} | "
                f"{f['wikitext2_kl']:.5f} | {r['edits_committed']} | {toks:,} |"
            )
        md_lines.append("")
        for arm in arms:
            r = clean_reports[arm]
            md_lines.append(f"## Arm: `{arm}`")
            if r.get("finish_summary"):
                md_lines.append(f"**Agent Summary:** {r['finish_summary']}")
            if r.get("committed_history"):
                md_lines.append("")
                for c in r["committed_history"]:
                    md_lines.append(f"- **Edit #{c['edit_num']} (L{c['layer']}, {c['edit_type']}):** {c['description']} — _{c.get('rationale', '')}_")
            if r.get("sae_patch_logs"):
                md_lines.append("")
                md_lines.append("**Local Subspace SAE Patching Logs:**")
                for p in r["sae_patch_logs"]:
                    md_lines.append(
                        f"- After Edit #{p['edit_num']} (L{p['edited_layer']}) -> Patched L{p['downstream_layer']} SAE "
                        f"({p['patched_pct']}% latents): MSE `{p['mse_before']:.2f}` -> `{p['mse_after']:.2f}`"
                    )
            md_lines.append("")
        md_path = base_dir / f"{stem}.md"
        md_path.write_text("\n".join(md_lines))
        return out_path

    try:
        out_path = _write_outputs(out_dir)
    except PermissionError:
        out_path = _write_outputs(RUN_DIR / "results")
    print(f"\nSaved benchmark report: {out_path}")
    print(f"Saved markdown summary: {out_path.with_suffix('.md')}")
    if transcripts:
        print(f"Saved full agent transcripts in: {out_path.parent / 'transcripts'}")

    if len(arms) > 1:
        print("\n" + "=" * 88)
        print(f"SUMMARY TABLE ({args.benchmark.upper()} | HELD-OUT TEST SPLIT)")
        print("=" * 88)
        print(f"{'Arm':<14} | {'Base':>8} | {'Final':>8} | {'Delta':>8} | {'Margin Δ':>9} | {'Keeper':>7} | {'WT2 PPL':>8} | {'WT2 KL':>8} | {'Edits':>5}")
        print("-" * 88)
        for arm in arms:
            r = all_reports[arm]
            b = r["base_heldout"]
            f = r["final_heldout"]
            print(
                f"{arm:<14} | {b['primary_score']:8.4f} | {f['primary_score']:8.4f} | "
                f"{r['heldout_primary_delta']:+8.4f} | {r['heldout_margin_delta']:+9.4f} | "
                f"{f['keeper_retention']*100:6.1f}% | {f['wikitext2_ppl']:8.2f} | "
                f"{f['wikitext2_kl']:8.5f} | {r['edits_committed']:5d}"
            )


if __name__ == "__main__":
    main()
