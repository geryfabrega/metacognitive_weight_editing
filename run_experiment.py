"""
End-to-End Metacognitive Weight Editing Loop:
1. Download GPT-2 Small + Pretrained SAEs (Layer 6 & Layer 9) from Hugging Face
2. Localize failure vs. keeper features using the Layer 6 SAE
3. Stage an SAE-guided 1-neuron weight edit in Layer 6 MLP
4. Test inside the Forward-Pass Sandbox (Target Fix Rate, Keeper Retention, Collateral KL)
5. Commit weight edit if guards pass
6. Locally patch ONLY the affected subset (<0.15%) of the downstream Layer 9 SAE dictionary
"""

import importlib.metadata
_orig_version = importlib.metadata.version
def _patched_version(pkg):
    v = _orig_version(pkg)
    return v.split(".dev")[0] if pkg == "torch" and ".dev" in v else v
importlib.metadata.version = _patched_version

import json
from pathlib import Path
import time
import torch
torch.__version__ = torch.__version__.split(".dev")[0]
from transformers import AutoModelForCausalLM, AutoTokenizer

from mwe import (
    SparseAutoencoder,
    WeightEditingSandbox,
    evaluate_model,
    get_unverified_entity_task,
)

CACHE_DIR = Path("/tmp/mwe_cache")
RESULTS_DIR = Path("/tmp/mwe_runs")


def main():
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("STEP 0: Downloading / Loading GPT-2 Small & Pretrained SAEs (Layers 6 & 9)")
    print("=" * 78)
    tok = AutoTokenizer.from_pretrained("gpt2", cache_dir=str(CACHE_DIR), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=str(CACHE_DIR), local_files_only=True).eval()

    sae_l6 = SparseAutoencoder.from_pretrained_gpt2(layer=6, cache_dir=CACHE_DIR)
    sae_l9 = SparseAutoencoder.from_pretrained_gpt2(layer=9, cache_dir=CACHE_DIR)
    print(f"Loaded GPT-2 ({sum(p.numel() for p in model.parameters()):,} params)")
    print(f"Loaded Layer 6 SAE ({sae_l6.hook_point}): d_in={sae_l6.d_in}, d_sae={sae_l6.d_sae:,} latents")
    print(f"Loaded Layer 9 SAE ({sae_l9.hook_point}): d_in={sae_l9.d_in}, d_sae={sae_l9.d_sae:,} latents\n")

    sandbox = WeightEditingSandbox(model, tok)
    task = get_unverified_entity_task()

    # Baseline evaluations (practice & heldout)
    base_prac = evaluate_model(sandbox, task, split="practice", use_staged=False)
    base_held = evaluate_model(sandbox, task, split="heldout", use_staged=False)

    print("=" * 78)
    print("STEP 1: Localize Failure Mode via Layer 6 Pretrained SAE")
    print("=" * 78)
    pos_acts_l6 = sandbox.capture_layer_activations(task.practice_pos, layer=6, use_staged=False)
    neg_acts_l6 = sandbox.capture_layer_activations(task.practice_neg, layer=6, use_staged=False)

    # Also cache pre-edit downstream activations at Layer 9 for the local SAE patch step
    pre_edit_acts_l9 = sandbox.capture_layer_activations(
        task.practice_pos + task.practice_neg, layer=9, use_staged=False
    )

    hits = sae_l6.localize_features(pos_acts_l6, neg_acts_l6, top_k=5)
    print("Top differential SAE latents at Layer 6 (Failure label=1 vs Keeper label=0):")
    for h in hits:
        print(
            f"  Latent #{h.latent_idx:5d} | AUC={h.auc:.3f} | "
            f"pos_mean={h.pos_mean:6.3f} (fire={h.fire_rate_pos*100:4.0f}%) | "
            f"neg_mean={h.neg_mean:6.3f} (fire={h.fire_rate_neg*100:4.0f}%)"
        )

    # Combine top 3 positive-separating SAE decoder vectors as the semantic feature direction
    pos_hits = [h for h in hits if h.diff > 0][:3] or hits[:1]
    sae_feature_dir = torch.stack([sae_l6.W_dec[h.latent_idx] for h in pos_hits]).mean(dim=0)

    print("\n" + "=" * 78)
    print("STEPS 2 & 3: Iterative Sandbox Loop (Stage ΔW -> Forward Pass -> Rollback/Commit)")
    print("=" * 78)
    target_ids = [tok.encode(t, add_special_tokens=False)[0] for t in task.target_tokens]
    suppress_ids = [tok.encode(t, add_special_tokens=False)[0] for t in task.suppress_tokens]

    # Include full keeper continuations in background calibration so the gate stays silent during multi-token replies
    keeper_full_texts = [p + g for p, g in zip(task.practice_neg, base_prac["_raw"]["neg_generations"])]

    candidates = [
        {"margin": 0.10, "steepness": 8.0, "gain": 28.0},
        {"margin": 0.25, "steepness": 12.0, "gain": 32.0},
        {"margin": 0.30, "steepness": 14.0, "gain": 36.0},
    ]

    gate_meta = None
    staged_prac = None
    guards_passed = False

    for attempt, cfg in enumerate(candidates, 1):
        gate_meta = sandbox.stage_sae_guided_neuron_gate(
            layer=6,
            sae_feature_dir=sae_feature_dir,
            pos_prompts=task.practice_pos,
            neg_prompts=task.practice_neg + task.calib_texts,
            target_token_ids=target_ids,
            suppress_token_ids=suppress_ids,
            gain=cfg["gain"],
            steepness=cfg["steepness"],
            margin=cfg["margin"],
        )
        staged_prac = evaluate_model(sandbox, task, split="practice", use_staged=True, base_cache=base_prac["_raw"])
        print(
            f"[Sandbox Attempt {attempt}] (margin={cfg['margin']}, g={cfg['steepness']}, gain={cfg['gain']}, "
            f"theta={gate_meta['theta']:.3f}):"
        )
        print(
            f"  Practice Failure Rate : {base_prac['failure_rate']*100:5.1f}% -> {staged_prac['failure_rate']*100:5.1f}% "
            f"(Abstention Fix: {staged_prac['abstention_fix_rate']*100:5.1f}%)"
        )
        print(f"  Practice Keeper Ident.: {staged_prac['keeper_identical_rate']*100:5.1f}% byte-identical")
        print(f"  Collateral Chat KL    : {staged_prac['collateral_kl']:.6f}")

        guards_passed = (
            staged_prac["failure_rate"] <= 0.5 * base_prac["failure_rate"]
            and staged_prac["keeper_identical_rate"] >= 0.95
            and staged_prac["collateral_kl"] < 0.01
        )
        if guards_passed:
            print(f"  -> PASS: All sandbox guards satisfied on Attempt {attempt}!")
            break
        else:
            print(f"  -> REJECTED by sandbox guard (keeper_identical={staged_prac['keeper_identical_rate']*100:.1f}% < 95%). Rolling back ΔW!\n")
            sandbox.staged.clear()

    print("\n" + "=" * 78)
    print("STEP 4: Commit Weight Edit & Evaluate on Held-Out Prompts")
    print("=" * 78)
    if not guards_passed:
        print("Sandbox guards failed! Rolling back staged weight edit.")
        sandbox.staged.clear()
        return

    commit_info = sandbox.commit_staged()
    print(f"COMMITTED weight edit: {commit_info['description']}")
    print(f"Changed {commit_info['entries_changed']:,} / {sum(p.numel() for p in model.parameters()):,} weights "
          f"({100.0 * commit_info['entries_changed'] / sum(p.numel() for p in model.parameters()):.5f}%)")

    held_after = evaluate_model(sandbox, task, split="heldout", use_staged=False, base_cache=base_held["_raw"])
    print(
        f"\n[HELD-OUT] Failure Rate     : {base_held['failure_rate']*100:5.1f}% -> {held_after['failure_rate']*100:5.1f}%"
    )
    print(f"[HELD-OUT] Keeper Identical : {held_after['keeper_identical_rate']*100:5.1f}% byte-identical")
    print(f"[HELD-OUT] Collateral KL    : {held_after['collateral_kl']:.6f}")

    print("\nSample Held-Out Failure Prompts (Base vs. Patched):")
    for (p, b_gen), (_, e_gen) in zip(base_held["pos_samples"], held_after["pos_samples"]):
        print(f"  Prompt : {p}")
        print(f"    Base   -> {b_gen!r}")
        print(f"    Patched-> {e_gen!r}")

    print("\nSample Held-Out Keeper Prompts (Verified Factual Recall):")
    for (p, b_gen), (_, e_gen) in zip(base_held["neg_samples"], held_after["neg_samples"]):
        print(f"  Prompt : {p}")
        print(f"    Base   -> {b_gen!r}")
        print(f"    Patched-> {e_gen!r}")

    print("\n" + "=" * 78)
    print("STEP 5: Local Subspace SAE Patch on Downstream Layer 9 SAE")
    print("=" * 78)
    post_edit_acts_l9 = sandbox.capture_layer_activations(
        task.practice_pos + task.practice_neg, layer=9, use_staged=False
    )
    write_vecs = torch.stack(sandbox.staged.write_vectors)
    sae_patch_report = sae_l9.local_subspace_patch(
        pre_edit_acts=pre_edit_acts_l9,
        post_edit_acts=post_edit_acts_l9,
        write_vectors=write_vecs,
        max_latents=32,
        steps=40,
    )
    print(json.dumps(sae_patch_report, indent=2))

    summary = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "top_sae_latents_l6": [h.__dict__ for h in hits],
        "gate_meta": gate_meta,
        "commit_info": commit_info,
        "heldout_base": {k: v for k, v in base_held.items() if k != "_raw"},
        "heldout_patched": {k: v for k, v in held_after.items() if k != "_raw"},
        "downstream_sae_patch": sae_patch_report,
    }
    out_file = RESULTS_DIR / "latest_run.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved full run summary to {out_file} (completed in {summary['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
