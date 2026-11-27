"""
Official EleutherAI `lm-evaluation-harness` (`lm-eval`) Verification & Adversarial Shortcut Audit.

Compares Base GPT-2 Small (124M) against one or more saved MWE `.safetensors` checkpoints
across three distinct benchmark groups:
  1. TARGET (4,000 items):          The 4 BLiMP-Hard tasks targeted by weight surgery
  2. ADVERSARIAL_WH (5,000 items):  Grammatical Wh-question & island tasks designed specifically
                                    to catch "fronted-Wh penalty" shortcuts!
  3. COLLATERAL:                    Unrelated OOD BLiMP tasks + LAMBADA + HellaSwag

Usage:
  # Compare all 3 Seed 44 checkpoints (no_sae vs. frozen_sae vs. patched_sae) in one run:
  python3 verify_checkpoint_lm_eval.py \
      --compare \
        no_sae=/tmp/mwe_runs/blimp_seed44/checkpoints/no_sae/model_delta_final.safetensors \
        frozen_sae=/tmp/mwe_runs/blimp_paired_seed44/checkpoints/branch_frozen_sae/model_delta_final.safetensors \
        patched_sae=checkpoints/seed44_paired_patched_sae.safetensors \
      --tag seed44_adversarial_audit

  # Or single checkpoint mode:
  python3 verify_checkpoint_lm_eval.py \
      --delta checkpoints/seed44_paired_patched_sae.safetensors \
      --tag seed44_paired_patched_sae
"""

import importlib.metadata

_orig_version = importlib.metadata.version


def _patched_version(pkg):
    v = _orig_version(pkg)
    return v.split(".dev")[0] if pkg == "torch" and ".dev" in v else v


importlib.metadata.version = _patched_version

import argparse
import copy
import json
import os
from pathlib import Path
import time
import torch

os.environ.setdefault("HF_DATASETS_CACHE", "/tmp/mwe_cache")
torch.__version__ = torch.__version__.split(".dev")[0]
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import mwe  # noqa: F401
from mwe.benchmarks import run_official_lm_eval

CACHE_DIR = Path("/tmp/mwe_cache")
DEFAULT_DELTA = Path("checkpoints/seed44_paired_patched_sae.safetensors")

TARGET_TASKS = [
    "blimp_left_branch_island_echo_question",
    "blimp_principle_A_reconstruction",
    "blimp_npi_present_1",
    "blimp_tough_vs_raising_1",
]

# Adversarial Wh-question shortcut-detector tasks:
# If an edit merely penalized fronted Wh-words ("Which/Whose/What") indiscriminately,
# these tasks will drop sharply or fail to benefit!
ADVERSARIAL_WH_TASKS = [
    "blimp_left_branch_island_simple_question",
    "blimp_wh_questions_object_gap",
    "blimp_wh_questions_subject_gap",
    "blimp_wh_vs_that_with_gap",
    "blimp_matrix_question_npi_licensor_present",
]

COLLATERAL_BLIMP_TASKS = [
    "blimp_determiner_noun_agreement_1",
    "blimp_animate_subject_passive",
]

COLLATERAL_GENERAL_TASKS = [
    "lambada_openai",
    "hellaswag",
]


def _resolve_local_gpt2_path(cache_dir: Path) -> str:
    snap_root = cache_dir / "models--gpt2" / "snapshots"
    if snap_root.exists():
        snaps = sorted(p for p in snap_root.iterdir() if p.is_dir() and (p / "config.json").exists())
        if snaps:
            return str(snaps[-1])
    return "gpt2"


def apply_safetensors_delta(model: torch.nn.Module, delta_path: Path) -> dict:
    """Apply a saved MWE sparse/dense `.safetensors` delta file onto a fresh GPT-2 model."""
    tensors = load_file(str(delta_path))
    params = dict(model.named_parameters())

    param_names = sorted({k.rsplit("::", 1)[0] for k in tensors.keys()})
    total_entries = 0
    total_l2_sq = 0.0

    with torch.no_grad():
        for name in param_names:
            idx = tensors[f"{name}::idx"].long()
            val = tensors[f"{name}::val"]
            flat = params[name].data.view(-1)
            dev_idx = idx.to(flat.device)
            new_v = val.to(flat.device, flat.dtype)
            total_l2_sq += float((new_v.float() - flat[dev_idx].float()).norm().item() ** 2)
            flat[dev_idx] = new_v
            total_entries += int(idx.numel())

    total_params = sum(p.numel() for p in model.parameters())
    return {
        "delta_path": str(delta_path),
        "tensors_modified": param_names,
        "entries_modified": total_entries,
        "total_model_params": total_params,
        "pct_entries_modified": round(100.0 * total_entries / max(total_params, 1), 5),
        "delta_l2_norm": round(total_l2_sq ** 0.5, 4),
    }


def _extract_acc(task_res: dict) -> float:
    for key in ("acc,none", "acc_norm,none", "acc"):
        if key in task_res and task_res[key] is not None:
            return float(task_res[key])
    return float("nan")


def _eval_model_suite(model, tok, blimp_tasks, blimp_limit, collateral_limit, skip_general, batch_size):
    res = run_official_lm_eval(model, tok, tasks=blimp_tasks, limit=blimp_limit, batch_size=batch_size)
    if not skip_general:
        gen = run_official_lm_eval(
            model, tok, tasks=COLLATERAL_GENERAL_TASKS, limit=collateral_limit, batch_size=batch_size
        )
        res.update(gen)
    return res


def main():
    parser = argparse.ArgumentParser(description="Official EleutherAI lm-eval verification & adversarial shortcut audit")
    parser.add_argument("--delta", type=str, default=None, help="Single checkpoint path (model_delta_final.safetensors)")
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="Compare multiple checkpoints as name=path pairs (e.g. frozen_sae=... patched_sae=...)",
    )
    parser.add_argument("--tag", type=str, default="seed44_adversarial_audit", help="Output report filename tag")
    parser.add_argument("--blimp-limit", type=int, default=None, help="Optional item cap per BLiMP task (default: ALL 1,000 items/task)")
    parser.add_argument("--collateral-limit", type=int, default=200, help="Item cap for lambada_openai & hellaswag")
    parser.add_argument("--skip-general-collateral", action="store_true", help="Skip lambada_openai & hellaswag")
    parser.add_argument("--skip-adversarial-wh", action="store_true", help="Skip adversarial Wh-question tasks")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    # Build list of (label, Path) checkpoints to evaluate
    checkpoints = []
    if args.compare:
        for spec in args.compare:
            if "=" not in spec:
                raise SystemExit(f"ERROR: --compare items must be `name=path`, got: {spec}")
            lbl, p_str = spec.split("=", 1)
            p = Path(p_str)
            if not p.exists():
                raise SystemExit(f"ERROR: Checkpoint `{lbl}` not found: {p}")
            checkpoints.append((lbl, p))
    else:
        p = Path(args.delta or DEFAULT_DELTA)
        if not p.exists():
            raise SystemExit(f"ERROR: Delta checkpoint not found: {p}")
        checkpoints.append(("edited", p))

    t0 = time.time()
    model_ref = _resolve_local_gpt2_path(CACHE_DIR)
    is_local = Path(model_ref).exists()
    print(f"Loading Base GPT-2 Small from {model_ref}...")
    tok = AutoTokenizer.from_pretrained(model_ref, cache_dir=str(CACHE_DIR), local_files_only=is_local)
    base_model = AutoModelForCausalLM.from_pretrained(model_ref, cache_dir=str(CACHE_DIR), local_files_only=is_local).eval()

    adv_tasks = [] if args.skip_adversarial_wh else list(ADVERSARIAL_WH_TASKS)
    blimp_tasks = TARGET_TASKS + adv_tasks + COLLATERAL_BLIMP_TASKS
    all_tasks = blimp_tasks + ([] if args.skip_general_collateral else COLLATERAL_GENERAL_TASKS)

    total_steps = 1 + len(checkpoints)
    print(f"\n[1/{total_steps}] Evaluating BASE GPT-2 ({len(blimp_tasks)} BLiMP tasks, limit={args.blimp_limit})...")
    base_res = _eval_model_suite(
        base_model, tok, blimp_tasks, args.blimp_limit, args.collateral_limit, args.skip_general_collateral, args.batch_size
    )

    ckpt_results = {}
    ckpt_stats = {}
    for idx, (lbl, p) in enumerate(checkpoints, start=2):
        edited_model = copy.deepcopy(base_model).eval()
        st = apply_safetensors_delta(edited_model, p)
        ckpt_stats[lbl] = st
        print(
            f"\n[{idx}/{total_steps}] Evaluating `{lbl}` ({p.name}: {st['entries_modified']:,} entries, L2={st['delta_l2_norm']})..."
        )
        ckpt_results[lbl] = _eval_model_suite(
            edited_model, tok, blimp_tasks, args.blimp_limit, args.collateral_limit, args.skip_general_collateral, args.batch_size
        )

    def role_of(task: str) -> str:
        if task in TARGET_TASKS:
            return "TARGET"
        if task in adv_tasks:
            return "ADVERSARIAL_WH"
        return "COLLATERAL"

    # Print side-by-side comparison table
    labels = [lbl for lbl, _ in checkpoints]
    col_header = " | ".join(f"{lbl:>18}" for lbl in labels)
    width = 68 + 21 * len(labels)
    print("\n" + "=" * width)
    print(f"OFFICIAL ELEUTHERAI `lm-eval` ADVERSARIAL AUDIT ({args.tag})")
    print("=" * width)
    print(f"{'Task':<42} | {'Role':<14} | {'Base':>7} | {col_header}")
    print("-" * width)

    rows = []
    group_scores = {
        role: {"base": [], **{lbl: [] for lbl in labels}}
        for role in ("TARGET", "ADVERSARIAL_WH", "COLLATERAL")
    }

    for task in all_tasks:
        role = role_of(task)
        b_acc = _extract_acc(base_res.get(task, {}))
        group_scores[role]["base"].append(b_acc)
        row = {"task": task, "role": role, "base": round(b_acc, 4), "checkpoints": {}}
        cells = []
        for lbl in labels:
            e_acc = _extract_acc(ckpt_results[lbl].get(task, {}))
            delta = e_acc - b_acc
            group_scores[role][lbl].append(e_acc)
            row["checkpoints"][lbl] = {"acc": round(e_acc, 4), "delta": round(delta, 4)}
            cells.append(f"{e_acc*100:6.2f}% ({delta*100:+6.2f}%)")
        rows.append(row)
        print(f"{task:<42} | {role:<14} | {b_acc*100:6.2f}% | " + " | ".join(cells))

    print("-" * width)
    summary_groups = {}
    for role, nice_name in (
        ("TARGET", "TARGET BLiMP-HARD MEAN"),
        ("ADVERSARIAL_WH", "ADVERSARIAL WH-QUESTION MEAN (Shortcut?)"),
        ("COLLATERAL", "GENERAL COLLATERAL MEAN"),
    ):
        b_list = group_scores[role]["base"]
        if not b_list:
            continue
        b_mean = sum(b_list) / len(b_list)
        summary_groups[role] = {"base": round(b_mean, 4), "checkpoints": {}}
        cells = []
        for lbl in labels:
            e_mean = sum(group_scores[role][lbl]) / len(group_scores[role][lbl])
            d_mean = e_mean - b_mean
            summary_groups[role]["checkpoints"][lbl] = {"acc": round(e_mean, 4), "delta": round(d_mean, 4)}
            cells.append(f"{e_mean*100:6.2f}% ({d_mean*100:+6.2f}%)")
        print(f"{nice_name:<42} | {role:<14} | {b_mean*100:6.2f}% | " + " | ".join(cells))
    print("=" * width)

    elapsed = round(time.time() - t0, 2)
    payload = {
        "tag": args.tag,
        "elapsed_sec": elapsed,
        "blimp_limit": args.blimp_limit,
        "collateral_limit": None if args.skip_general_collateral else args.collateral_limit,
        "checkpoint_stats": ckpt_stats,
        "group_means": summary_groups,
        "rows": rows,
        "raw_base_results": base_res,
        "raw_checkpoint_results": ckpt_results,
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"lm_eval_{args.tag}.json"
    md_path = out_dir / f"lm_eval_{args.tag}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    md = [
        f"# Official EleutherAI `lm-eval` Adversarial Audit (`{args.tag}`)",
        "",
        "| Task | Role | Base GPT-2 | " + " | ".join(f"`{lbl}`" for lbl in labels) + " |",
        "| :--- | :---: | :---: | " + " | ".join(":---:" for _ in labels) + " |",
    ]
    for r in rows:
        vals = [f"{r['checkpoints'][lbl]['acc']*100:.2f}% (**{r['checkpoints'][lbl]['delta']*100:+.2f}%**)" for lbl in labels]
        md.append(f"| `{r['task']}` | {r['role']} | {r['base']*100:.2f}% | " + " | ".join(vals) + " |")
    md.append("")
    md.append("## Group Means")
    md.append("| Group | Base GPT-2 | " + " | ".join(f"`{lbl}`" for lbl in labels) + " |")
    md.append("| :--- | :---: | " + " | ".join(":---:" for _ in labels) + " |")
    for role, g in summary_groups.items():
        vals = [f"{g['checkpoints'][lbl]['acc']*100:.2f}% (**{g['checkpoints'][lbl]['delta']*100:+.2f}%**)" for lbl in labels]
        md.append(f"| **{role}** | {g['base']*100:.2f}% | " + " | ".join(vals) + " |")
    md_path.write_text("\n".join(md))

    print(f"\nSaved official lm-eval audit: {json_path}")
    print(f"Saved markdown summary:       {md_path}")
    print(f"Total runtime: {elapsed}s")


if __name__ == "__main__":
    main()
