"""
Official EleutherAI `lm-evaluation-harness` (`lm-eval`) Verification Script.

Loads untouched Base GPT-2 Small (124M) and applies any saved MWE `.safetensors` weight delta
checkpoint (e.g., Seed 44 `branch_patched_sae`), then runs unmodified `lm_eval.simple_evaluate`
on both Base and Edited GPT-2 across:
  - Target BLiMP-Hard tasks (all 1,000 items per task by default = 4,000 official questions!)
  - Collateral OOD BLiMP keeper tasks (2,000 official questions) + optional LAMBADA / HellaSwag

Usage:
  # Verify Seed 44 paired-branch patched_sae (+14.08% checkpoint) on full official BLiMP + LAMBADA:
  python3 verify_checkpoint_lm_eval.py \
      --delta /tmp/mwe_runs/blimp_paired_seed44/checkpoints/branch_patched_sae/model_delta_final.safetensors

  # Also test the ultra-sparse 37KB (3,074-weight, 0.0025% of GPT-2!) Seed 44 checkpoint:
  python3 verify_checkpoint_lm_eval.py \
      --delta /tmp/mwe_runs/blimp_seed44/checkpoints/patched_sae/model_delta_final.safetensors \
      --tag seed44_sparse_3074w
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
DEFAULT_DELTA = Path("/tmp/mwe_runs/blimp_paired_seed44/checkpoints/branch_patched_sae/model_delta_final.safetensors")

TARGET_TASKS = [
    "blimp_left_branch_island_echo_question",
    "blimp_principle_A_reconstruction",
    "blimp_npi_present_1",
    "blimp_tough_vs_raising_1",
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


def main():
    parser = argparse.ArgumentParser(description="Official EleutherAI lm-eval verification of MWE weight deltas")
    parser.add_argument("--delta", type=str, default=str(DEFAULT_DELTA), help="Path to model_delta_final.safetensors")
    parser.add_argument("--tag", type=str, default="seed44_paired_patched_sae", help="Output report filename tag")
    parser.add_argument("--blimp-limit", type=int, default=None, help="Optional item cap per BLiMP task (default: None = ALL 1,000 items/task)")
    parser.add_argument("--collateral-limit", type=int, default=200, help="Item cap for general collateral tasks (lambada_openai, hellaswag)")
    parser.add_argument("--skip-general-collateral", action="store_true", help="Skip lambada_openai & hellaswag (faster BLiMP-only run)")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    delta_path = Path(args.delta)
    if not delta_path.exists():
        raise SystemExit(f"ERROR: Delta checkpoint not found: {delta_path}")

    t0 = time.time()
    model_ref = _resolve_local_gpt2_path(CACHE_DIR)
    is_local = Path(model_ref).exists()
    print(f"Loading Base GPT-2 Small from {model_ref}...")
    tok = AutoTokenizer.from_pretrained(model_ref, cache_dir=str(CACHE_DIR), local_files_only=is_local)
    base_model = AutoModelForCausalLM.from_pretrained(model_ref, cache_dir=str(CACHE_DIR), local_files_only=is_local).eval()

    edited_model = copy.deepcopy(base_model).eval()
    delta_stats = apply_safetensors_delta(edited_model, delta_path)
    print(
        f"Applied checkpoint `{delta_path.name}`: "
        f"{delta_stats['entries_modified']:,} entries across {len(delta_stats['tensors_modified'])} tensors "
        f"({delta_stats['pct_entries_modified']}% of GPT-2, L2={delta_stats['delta_l2_norm']})"
    )

    blimp_tasks = TARGET_TASKS + COLLATERAL_BLIMP_TASKS

    print(f"\n[1/2] Running Official EleutherAI `lm-eval` on BASE GPT-2 ({len(blimp_tasks)} BLiMP tasks, limit={args.blimp_limit})...")
    base_res = run_official_lm_eval(base_model, tok, tasks=blimp_tasks, limit=args.blimp_limit, batch_size=args.batch_size)
    if not args.skip_general_collateral:
        print(f"      Running general collateral tasks on BASE GPT-2 ({COLLATERAL_GENERAL_TASKS}, limit={args.collateral_limit})...")
        base_gen = run_official_lm_eval(
            base_model, tok, tasks=COLLATERAL_GENERAL_TASKS, limit=args.collateral_limit, batch_size=args.batch_size
        )
        base_res.update(base_gen)

    print(f"\n[2/2] Running Official EleutherAI `lm-eval` on EDITED GPT-2 ({len(blimp_tasks)} BLiMP tasks, limit={args.blimp_limit})...")
    edit_res = run_official_lm_eval(edited_model, tok, tasks=blimp_tasks, limit=args.blimp_limit, batch_size=args.batch_size)
    if not args.skip_general_collateral:
        print(f"      Running general collateral tasks on EDITED GPT-2 ({COLLATERAL_GENERAL_TASKS}, limit={args.collateral_limit})...")
        edit_gen = run_official_lm_eval(
            edited_model, tok, tasks=COLLATERAL_GENERAL_TASKS, limit=args.collateral_limit, batch_size=args.batch_size
        )
        edit_res.update(edit_gen)

    all_tasks = blimp_tasks + ([] if args.skip_general_collateral else COLLATERAL_GENERAL_TASKS)

    print("\n" + "=" * 88)
    print(f"OFFICIAL ELEUTHERAI `lm-eval` VERIFICATION ({args.tag})")
    print("=" * 88)
    print(f"{'Task':<42} | {'Role':<10} | {'Base GPT-2':>10} | {'Edited':>10} | {'Delta':>9}")
    print("-" * 88)

    rows = []
    target_base, target_edit = [], []
    collat_base, collat_edit = [], []

    for task in all_tasks:
        role = "TARGET" if task in TARGET_TASKS else "COLLATERAL"
        b_acc = _extract_acc(base_res.get(task, {}))
        e_acc = _extract_acc(edit_res.get(task, {}))
        delta = e_acc - b_acc
        if role == "TARGET":
            target_base.append(b_acc)
            target_edit.append(e_acc)
        else:
            collat_base.append(b_acc)
            collat_edit.append(e_acc)
        rows.append({"task": task, "role": role, "base": round(b_acc, 4), "edited": round(e_acc, 4), "delta": round(delta, 4)})
        print(f"{task:<42} | {role:<10} | {b_acc*100:9.2f}% | {e_acc*100:9.2f}% | {delta*100:+8.2f}%")

    t_b_mean = sum(target_base) / len(target_base)
    t_e_mean = sum(target_edit) / len(target_edit)
    c_b_mean = sum(collat_base) / len(collat_base)
    c_e_mean = sum(collat_edit) / len(collat_edit)

    print("-" * 88)
    print(f"{'TARGET BLiMP-HARD MEAN (4,000 items)':<42} | {'TARGET':<10} | {t_b_mean*100:9.2f}% | {t_e_mean*100:9.2f}% | {(t_e_mean - t_b_mean)*100:+8.2f}%")
    print(f"{'COLLATERAL MEAN (OOD + General)':<42} | {'COLLATERAL':<10} | {c_b_mean*100:9.2f}% | {c_e_mean*100:9.2f}% | {(c_e_mean - c_b_mean)*100:+8.2f}%")
    print("=" * 88)

    elapsed = round(time.time() - t0, 2)
    payload = {
        "tag": args.tag,
        "elapsed_sec": elapsed,
        "blimp_limit": args.blimp_limit,
        "collateral_limit": None if args.skip_general_collateral else args.collateral_limit,
        "delta_stats": delta_stats,
        "target_mean_base": round(t_b_mean, 4),
        "target_mean_edited": round(t_e_mean, 4),
        "target_mean_delta": round(t_e_mean - t_b_mean, 4),
        "collateral_mean_base": round(c_b_mean, 4),
        "collateral_mean_edited": round(c_e_mean, 4),
        "collateral_mean_delta": round(c_e_mean - c_b_mean, 4),
        "rows": rows,
        "raw_base_results": base_res,
        "raw_edited_results": edit_res,
    }

    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"lm_eval_{args.tag}.json"
    md_path = out_dir / f"lm_eval_{args.tag}.md"
    json_path.write_text(json.dumps(payload, indent=2, default=str))

    md = [
        f"# Official EleutherAI `lm-eval` Verification (`{args.tag}`)",
        "",
        f"- **Checkpoint:** `{delta_path}`",
        f"- **Modified tensors:** `{len(delta_stats['tensors_modified'])}` (`{delta_stats['entries_modified']:,}` entries, L2=`{delta_stats['delta_l2_norm']}`)",
        f"- **Target BLiMP-Hard Mean:** `{t_b_mean*100:.2f}%` -> **`{t_e_mean*100:.2f}%`** (**`{(t_e_mean - t_b_mean)*100:+.2f}%`**)",
        f"- **Collateral Mean:** `{c_b_mean*100:.2f}%` -> `{c_e_mean*100:.2f}%` (`{(c_e_mean - c_b_mean)*100:+.2f}%`)",
        "",
        "| Task | Role | Base GPT-2 | Edited GPT-2 | Delta |",
        "| :--- | :---: | :---: | :---: | :---: |",
    ]
    for r in rows:
        md.append(f"| `{r['task']}` | {r['role']} | {r['base']*100:.2f}% | {r['edited']*100:.2f}% | **{r['delta']*100:+.2f}%** |")
    md_path.write_text("\n".join(md))

    print(f"\nSaved official lm-eval report: {json_path}")
    print(f"Saved markdown summary:        {md_path}")
    print(f"Total runtime: {elapsed}s")


if __name__ == "__main__":
    main()
