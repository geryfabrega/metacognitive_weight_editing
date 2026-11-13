"""
Multi-Iteration Metacognitive Weight Editing Loop (Layers 6 -> 11):
- Keeps base GPT-2 and base pretrained SAEs (Layers 6..11) 100% untouched in /tmp/mwe_cache
- Iteratively loops across layers:
    1. Localize remaining failure signal using the current layer's (locally synced!) SAE
    2. Stage sparse 1-neuron weight edits in the sandbox and test candidate thresholds
    3. Commit ONLY if target abstention improves while keeper retention >= 95% and KL < 0.01
    4. Locally patch the next layer's SAE (<0.15% of latents) on the post-edit activations
    5. Save both model weight deltas and SAE patch deltas to disk at every committed iteration
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
CKPT_DIR = RESULTS_DIR / "checkpoints"


def main():
    t0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    layers = [6, 7, 8, 9, 10, 11]
    print("=" * 80)
    print("STEP 0: Loading Frozen Base GPT-2 & Pretrained SAEs for Layers 6..11")
    print("=" * 80)
    tok = AutoTokenizer.from_pretrained("gpt2", cache_dir=str(CACHE_DIR), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=str(CACHE_DIR), local_files_only=True).eval()
    total_params = sum(p.numel() for p in model.parameters())

    saes = {L: SparseAutoencoder.from_pretrained_gpt2(layer=L, cache_dir=CACHE_DIR) for L in layers}
    print(f"Loaded GPT-2 ({total_params:,} params) + {len(saes)} Pretrained SAEs (24,576 latents each)\n")

    sandbox = WeightEditingSandbox(model, tok)
    task = get_unverified_entity_task()
    target_ids = [tok.encode(t, add_special_tokens=False)[0] for t in task.target_tokens]
    suppress_ids = [tok.encode(t, add_special_tokens=False)[0] for t in task.suppress_tokens]

    base_prac = evaluate_model(sandbox, task, split="practice", use_staged=False)
    base_held = evaluate_model(sandbox, task, split="heldout", use_staged=False)

    cur_prac = base_prac
    cur_held = base_held

    print(
        f"[Iteration 0 | Unedited Base Model]\n"
        f"  Practice Failure: {cur_prac['failure_rate']*100:5.1f}% | Target Prob: {cur_prac['mean_target_token_prob']*100:5.2f}%\n"
        f"  Held-Out Failure: {cur_held['failure_rate']*100:5.1f}% | Target Prob: {cur_held['mean_target_token_prob']*100:5.2f}%\n"
    )

    all_prompts = task.practice_pos + task.practice_neg
    # Cache baseline pre-edit activations at each layer before any edits
    baseline_layer_acts = {
        L: sandbox.capture_layer_activations(all_prompts, layer=L, use_staged=False) for L in layers
    }

    iterations_log = []
    committed_count = 0

    for idx, L in enumerate(layers):
        print("=" * 80)
        print(f"LOOP ITERATION {idx + 1} (Layer {L} | SAE: {saes[L].hook_point})")
        print("=" * 80)

        # 1. Focus on residual (unsolved) positive prompts first, then weakest-confidence positive prompts!
        active_pos_prompts = (
            cur_prac["_raw"]["unsolved_pos_prompts"]
            or cur_prac["_raw"]["weakest_pos_prompts"]
            or task.practice_pos
        )
        print(f"  [Boosting Focus] Targeting {len(active_pos_prompts)} / {len(task.practice_pos)} residual/weakest practice prompt(s)")

        pos_acts = sandbox.capture_layer_activations(active_pos_prompts, layer=L, use_staged=False)
        neg_acts = sandbox.capture_layer_activations(task.practice_neg, layer=L, use_staged=False)

        hits = saes[L].localize_features(pos_acts, neg_acts, top_k=5)
        top_hit = hits[0]
        print(
            f"  [SAE Localizer] Top Latent #{top_hit.latent_idx} (AUC={top_hit.auc:.3f}, "
            f"pos_fire={top_hit.fire_rate_pos*100:.0f}%, neg_fire={top_hit.fire_rate_neg*100:.0f}%)"
        )

        pos_hits = [h for h in hits if h.diff > 0][:3] or hits[:1]
        sae_feature_dir = torch.stack([saes[L].W_dec[h.latent_idx] for h in pos_hits]).mean(dim=0)

        # 2. Sweep candidate threshold/gain settings in the reversible sandbox
        candidates = [
            {"margin": 0.05, "steepness": 10.0, "gain": 28.0},
            {"margin": 0.12, "steepness": 12.0, "gain": 30.0},
            {"margin": 0.22, "steepness": 12.0, "gain": 28.0},
            {"margin": 0.30, "steepness": 14.0, "gain": 32.0},
        ]

        accepted_cfg = None
        accepted_meta = None
        accepted_prac = None

        for attempt, cfg in enumerate(candidates, 1):
            gate_meta = sandbox.stage_sae_guided_neuron_gate(
                layer=L,
                sae_feature_dir=sae_feature_dir,
                pos_prompts=active_pos_prompts,
                neg_prompts=task.practice_neg + task.calib_texts,
                target_token_ids=target_ids,
                suppress_token_ids=suppress_ids,
                gain=cfg["gain"],
                steepness=cfg["steepness"],
                margin=cfg["margin"],
            )
            staged_prac = evaluate_model(
                sandbox, task, split="practice", use_staged=True, base_cache=base_prac["_raw"]
            )

            # Check progression vs regression against current committed state
            keeper_ok = staged_prac["keeper_identical_rate"] >= 0.95
            kl_ok = staged_prac["collateral_kl"] < 0.01
            progressed = (
                staged_prac["failure_rate"] < cur_prac["failure_rate"]
                or (
                    staged_prac["failure_rate"] == cur_prac["failure_rate"]
                    and staged_prac["mean_target_token_prob"] > cur_prac["mean_target_token_prob"] + 0.005
                )
            )

            status = "PASS" if (keeper_ok and kl_ok and progressed) else "REJECT"
            print(
                f"  [Sandbox Try {attempt}] margin={cfg['margin']:.2f}, theta={gate_meta['theta']:.3f} -> "
                f"Prac Fail: {staged_prac['failure_rate']*100:5.1f}%, "
                f"TgtProb: {staged_prac['mean_target_token_prob']*100:5.1f}%, "
                f"Keeper: {staged_prac['keeper_identical_rate']*100:5.1f}%, "
                f"KL: {staged_prac['collateral_kl']:.5f} [{status}]"
            )

            if keeper_ok and kl_ok and progressed:
                accepted_cfg = cfg
                accepted_meta = gate_meta
                accepted_prac = staged_prac
                break
            else:
                sandbox.staged.clear()

        if accepted_prac is None:
            print(f"  -> No candidate at Layer {L} improved over current state without violating guards. Skipping Layer {L}.\n")
            iterations_log.append({"iteration": idx + 1, "layer": L, "committed": False})
            continue

        # 3. Commit the accepted weight delta and save checkpoint to disk
        committed_count += 1
        commit_info = sandbox.commit_staged()
        model_ckpt_path = CKPT_DIR / f"model_delta_iter_{committed_count}_L{L}.safetensors"
        sandbox.save_committed_deltas(model_ckpt_path)

        cur_prac = accepted_prac
        cur_held = evaluate_model(sandbox, task, split="heldout", use_staged=False, base_cache=base_held["_raw"])

        print(
            f"  -> COMMITTED Iteration #{committed_count} at Layer {L} (unit {accepted_meta['unit']})!\n"
            f"     Saved model delta checkpoint: {model_ckpt_path.name}\n"
            f"     Cumulative weights changed  : {commit_info['cumulative_entries_changed']:,} / {total_params:,}\n"
            f"     Held-Out Failure Rate       : {base_held['failure_rate']*100:5.1f}% -> {cur_held['failure_rate']*100:5.1f}% "
            f"(Target Prob: {cur_held['mean_target_token_prob']*100:5.1f}%)\n"
            f"     Held-Out Keeper Identical   : {cur_held['keeper_identical_rate']*100:5.1f}% | KL: {cur_held['collateral_kl']:.5f}"
        )

        # 4. Locally patch the NEXT downstream SAE (L_next) so the next iteration's SAE microscope is synced!
        sae_patch_report = None
        if idx + 1 < len(layers):
            next_L = layers[idx + 1]
            post_edit_acts_next = sandbox.capture_layer_activations(all_prompts, layer=next_L, use_staged=False)
            write_vecs = torch.stack(sandbox.committed_write_vectors)
            sae_patch_report = saes[next_L].local_subspace_patch(
                pre_edit_acts=baseline_layer_acts[next_L],
                post_edit_acts=post_edit_acts_next,
                write_vectors=write_vecs,
                max_latents=32,
                steps=40,
            )
            sae_ckpt_path = CKPT_DIR / f"sae_L{next_L}_delta_iter_{committed_count}.safetensors"
            saes[next_L].save_patch_delta(sae_ckpt_path)
            mse_before = sae_patch_report["post_edit_Before_SAE_patch"]["mse"]
            mse_after = sae_patch_report["post_edit_After_SAE_patch"]["mse"]
            print(
                f"  -> SYNCED Downstream Layer {next_L} SAE (32 / 24,576 latents patched): "
                f"MSE {mse_before:.2f} -> {mse_after:.2f} (saved {sae_ckpt_path.name})\n"
            )

        iterations_log.append(
            {
                "iteration": idx + 1,
                "committed_step": committed_count,
                "layer": L,
                "committed": True,
                "top_sae_latent": top_hit.__dict__,
                "gate_meta": accepted_meta,
                "commit_info": commit_info,
                "practice_metrics": {k: v for k, v in cur_prac.items() if k != "_raw"},
                "heldout_metrics": {k: v for k, v in cur_held.items() if k != "_raw"},
                "downstream_sae_patch": sae_patch_report,
            }
        )

        # Stop early if practice failure is 0% and target token confidence exceeds 99%
        if cur_prac["failure_rate"] == 0.0 and cur_prac["mean_target_token_prob"] >= 0.99:
            print("  -> Goal state reached (0.0% practice failure, >=99% target confidence). Stopping loop!")
            break

    # Save final cumulative model checkpoint
    final_ckpt = CKPT_DIR / "model_delta_final.safetensors"
    sandbox.save_committed_deltas(final_ckpt)

    print("=" * 80)
    print(f"MULTI-ITERATION SUMMARY ({committed_count} Committed Edits across {len(layers)} Layers)")
    print("=" * 80)
    print(f"Base Held-Out Failure Rate  : {base_held['failure_rate']*100:5.1f}% (Target Prob: {base_held['mean_target_token_prob']*100:5.2f}%)")
    print(f"Final Held-Out Failure Rate : {cur_held['failure_rate']*100:5.1f}% (Target Prob: {cur_held['mean_target_token_prob']*100:5.2f}%)")
    print(f"Final Held-Out Keeper Ident.: {cur_held['keeper_identical_rate']*100:5.1f}% byte-identical")
    print(f"Final Collateral Chat KL    : {cur_held['collateral_kl']:.6f}")
    print(f"Checkpoints written to      : {CKPT_DIR}")

    summary = {
        "elapsed_seconds": round(time.time() - t0, 2),
        "layers_visited": layers,
        "committed_edits_count": committed_count,
        "heldout_base": {k: v for k, v in base_held.items() if k != "_raw"},
        "heldout_final": {k: v for k, v in cur_held.items() if k != "_raw"},
        "iterations": iterations_log,
    }
    out_file = RESULTS_DIR / "multi_loop_run.json"
    out_file.write_text(json.dumps(summary, indent=2))
    print(f"Saved multi-iteration run log to {out_file}")


if __name__ == "__main__":
    main()
