"""
Ablation 1: Multi-Layer Weight Editing WITH Local SAE Patching (ON) vs. WITHOUT Local SAE Patching (OFF).

Tests two regimes:
  Regime A (Blended SAE + Contrastive Read Direction, sae_blend=0.5):
    - Exact setting from run_experiment.py, comparing patch_sae=True vs patch_sae=False.
  Regime B (Pure SAE Dictionary Read Direction, sae_blend=1.0 — no contrastive diff_dir crutch):
    - The MLP gate is wired strictly using the SAE's own decoder vectors W_dec[latent_idx].
    - Tests whether local SAE patching on downstream layers changes latent selection, reconstruction MSE,
      and final held-out accuracy when relying purely on the SAE coordinate system.
"""

import importlib.metadata
_orig_version = importlib.metadata.version
def _patched_version(pkg):
    v = _orig_version(pkg)
    return v.split(".dev")[0] if pkg == "torch" and ".dev" in v else v
importlib.metadata.version = _patched_version

import copy
import json
from pathlib import Path
import time
import torch
import torch.nn.functional as F
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


def run_arm(base_model, tok, base_saes, task, patch_sae: bool, sae_blend: float, max_Latents_patch: int = 128):
    """Run the 6-layer iterative loop with Local SAE Patching turned ON or OFF."""
    model = copy.deepcopy(base_model).eval()
    saes = {L: copy.deepcopy(sae) for L, sae in base_saes.items()}
    layers = sorted(saes.keys())

    sandbox = WeightEditingSandbox(model, tok)
    target_ids = [tok.encode(t, add_special_tokens=False)[0] for t in task.target_tokens]
    suppress_ids = [tok.encode(t, add_special_tokens=False)[0] for t in task.suppress_tokens]

    base_prac = evaluate_model(sandbox, task, split="practice", use_staged=False)
    base_held = evaluate_model(sandbox, task, split="heldout", use_staged=False)
    cur_prac = base_prac
    cur_held = base_held

    all_prompts = task.practice_pos + task.practice_neg
    baseline_layer_acts = {
        L: sandbox.capture_layer_activations(all_prompts, layer=L, use_staged=False) for L in layers
    }

    trace = []
    committed_count = 0

    for idx, L in enumerate(layers):
        active_pos_prompts = (
            cur_prac["_raw"]["unsolved_pos_prompts"]
            or cur_prac["_raw"]["weakest_pos_prompts"]
            or task.practice_pos
        )

        pos_acts = sandbox.capture_layer_activations(active_pos_prompts, layer=L, use_staged=False)
        neg_acts = sandbox.capture_layer_activations(task.practice_neg, layer=L, use_staged=False)
        all_cur_acts = sandbox.capture_layer_activations(all_prompts, layer=L, use_staged=False)
        sae_recon_at_L = saes[L].reconstruction_metrics(all_cur_acts)

        hits = saes[L].localize_features(pos_acts, neg_acts, top_k=5)
        pos_hits = [h for h in hits if h.diff > 0][:3] or hits[:1]

        # Construct SAE feature direction from the SAE's ENCODER vectors W_enc[:, latent_idx]
        # mapped into post-LayerNorm (ln_2) space: (w_enc - mean(w_enc)) / ln_2.weight
        raw_w_enc = torch.stack([saes[L].W_enc[:, h.latent_idx] for h in pos_hits]).mean(dim=0)
        raw_w_enc = raw_w_enc - raw_w_enc.mean()
        ln2_gamma = model.transformer.h[L].ln_2.weight.detach().float().cpu()
        sae_vec = F.normalize(raw_w_enc / (ln2_gamma + 1e-6), dim=-1)

        # Candidates for threshold margin in post-ln_2 SAE coordinates
        candidates = [
            {"margin": 0.02, "steepness": 12.0, "gain": 28.0},
            {"margin": 0.08, "steepness": 12.0, "gain": 30.0},
            {"margin": 0.15, "steepness": 14.0, "gain": 32.0},
            {"margin": 0.25, "steepness": 14.0, "gain": 32.0},
        ]

        accepted_meta = None
        accepted_prac = None

        for cfg in candidates:
            # Stage using specified sae_blend (1.0 = pure SAE decoder vector, 0.5 = 50/50 blend with diff_dir)
            gate_meta = _stage_with_blend(
                sandbox=sandbox,
                layer=L,
                sae_feature_dir=sae_vec,
                sae_blend=sae_blend,
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
            keeper_ok = staged_prac["keeper_identical_rate"] >= 0.95
            kl_ok = staged_prac["collateral_kl"] < 0.01
            progressed = (
                staged_prac["failure_rate"] < cur_prac["failure_rate"]
                or (
                    staged_prac["failure_rate"] == cur_prac["failure_rate"]
                    and staged_prac["mean_target_token_prob"] > cur_prac["mean_target_token_prob"] + 0.005
                )
            )
            if keeper_ok and kl_ok and progressed:
                accepted_meta = gate_meta
                accepted_prac = staged_prac
                break
            else:
                sandbox.staged.clear()

        if accepted_prac is None:
            trace.append(
                {
                    "layer": L,
                    "committed": False,
                    "sae_mse_at_entry": sae_recon_at_L["mse"],
                    "top_latents": [h.latent_idx for h in pos_hits],
                    "top_auc": hits[0].auc,
                }
            )
            continue

        committed_count += 1
        sandbox.commit_staged()
        cur_prac = accepted_prac
        cur_held = evaluate_model(sandbox, task, split="heldout", use_staged=False, base_cache=base_held["_raw"])

        # If patch_sae is True, locally patch downstream SAEs (L' > L) on the post-edit activations
        # AND orthogonalize their encoder columns against the committed write vectors so downstream SAEs
        # don't merely re-detect upstream edits!
        sae_patch_info = None
        if patch_sae and idx + 1 < len(layers):
            write_vecs = torch.stack(sandbox.committed_write_vectors)
            wv_basis, _ = torch.linalg.qr(write_vecs.float().T)  # [d_model, k] orthonormal basis
            for downstream_L in layers[idx + 1:]:
                post_acts = sandbox.capture_layer_activations(all_prompts, layer=downstream_L, use_staged=False)
                info = saes[downstream_L].local_subspace_patch(
                    pre_edit_acts=baseline_layer_acts[downstream_L],
                    post_edit_acts=post_acts,
                    write_vectors=write_vecs,
                    max_latents=max_Latents_patch,
                    steps=40,
                )
                # Orthogonalize encoder columns of downstream SAE against the already-written subspace wv_basis
                with torch.no_grad():
                    W_e = saes[downstream_L].W_enc.data
                    proj = wv_basis @ (wv_basis.T @ W_e)
                    saes[downstream_L].W_enc.data = W_e - proj
                if downstream_L == layers[idx + 1]:
                    sae_patch_info = info

        trace.append(
            {
                "layer": L,
                "committed": True,
                "unit": accepted_meta["unit"],
                "sae_mse_at_entry": sae_recon_at_L["mse"],
                "top_latents": [h.latent_idx for h in pos_hits],
                "top_auc": hits[0].auc,
                "prac_fail": cur_prac["failure_rate"],
                "prac_prob": cur_prac["mean_target_token_prob"],
                "held_fail": cur_held["failure_rate"],
                "held_prob": cur_held["mean_target_token_prob"],
                "held_keeper": cur_held["keeper_identical_rate"],
                "kl": cur_held["collateral_kl"],
                "next_sae_patched_mse": (
                    f"{sae_patch_info['post_edit_Before_SAE_patch']['mse']:.1f} -> "
                    f"{sae_patch_info['post_edit_After_SAE_patch']['mse']:.1f}"
                    if sae_patch_info
                    else "UNPATCHED (Frozen)"
                ),
            }
        )

        if cur_prac["failure_rate"] == 0.0 and cur_prac["mean_target_token_prob"] >= 0.99:
            break

    return {
        "patch_sae": patch_sae,
        "sae_blend": sae_blend,
        "committed_edits": committed_count,
        "final_prac_fail": cur_prac["failure_rate"],
        "final_prac_prob": cur_prac["mean_target_token_prob"],
        "final_held_fail": cur_held["failure_rate"],
        "final_held_prob": cur_held["mean_target_token_prob"],
        "final_held_keeper": cur_held["keeper_identical_rate"],
        "final_kl": cur_held["collateral_kl"],
        "trace": trace,
    }


@torch.no_grad()
def _stage_with_blend(
    sandbox,
    layer: int,
    sae_feature_dir: torch.Tensor,
    sae_blend: float,
    pos_prompts,
    neg_prompts,
    target_token_ids,
    suppress_token_ids,
    gain: float,
    steepness: float,
    margin: float,
):
    """Same as stage_sae_guided_neuron_gate but with configurable sae_blend in [0.0, 1.0]."""
    block = sandbox.model.transformer.h[layer]
    d_model = sandbox.model.config.n_embd
    ln2_pos, ln2_neg, mlp_acts, all_neg_tokens = [], [], [], []

    h1 = block.ln_2.register_forward_hook(lambda m, i, o: ln2_pos.append(o[:, -1, :].detach().float().cpu()))
    for p in pos_prompts:
        sandbox.model(**sandbox.tokenizer(p, return_tensors="pt"))
    h1.remove()

    def ln2_neg_hook(m, i, o):
        ln2_neg.append(o[:, -1, :].detach().float().cpu())
        all_neg_tokens.append(o.detach().float().cpu().reshape(-1, d_model))

    h2 = block.ln_2.register_forward_hook(ln2_neg_hook)
    h3 = block.mlp.act.register_forward_hook(
        lambda m, i, o: mlp_acts.append(o.detach().float().cpu().reshape(-1, o.shape[-1]))
    )
    for p in neg_prompts:
        sandbox.model(**sandbox.tokenizer(p, return_tensors="pt"))
    h2.remove()
    h3.remove()

    X_pos = torch.cat(ln2_pos, dim=0)
    X_neg = torch.cat(ln2_neg, dim=0)
    X_bg = torch.cat(all_neg_tokens, dim=0)
    unit = int(torch.argmin(torch.cat(mlp_acts, dim=0).abs().mean(dim=0)).item())

    sae_dir = F.normalize(sae_feature_dir.float().cpu(), dim=-1)
    diff_dir = F.normalize(X_pos.mean(dim=0) - X_neg.mean(dim=0), dim=-1)
    if torch.dot(sae_dir, diff_dir) < 0:
        sae_dir = -sae_dir
    read_dir = F.normalize(sae_blend * sae_dir + (1.0 - sae_blend) * diff_dir, dim=-1)

    proj_pos = X_pos @ read_dir
    proj_neg = X_neg @ read_dir
    proj_bg = X_bg @ read_dir
    theta = max(proj_neg.max().item(), proj_bg.max().item()) + margin

    W_U = sandbox.model.lm_head.weight.detach().float().cpu()
    tgt_vec = F.normalize(W_U[target_token_ids].mean(dim=0), dim=-1)
    sup_vec = F.normalize(W_U[suppress_token_ids].mean(dim=0), dim=-1)
    write_vec = F.normalize(tgt_vec - 0.6 * sup_vec, dim=-1) * gain

    d_mlp = block.mlp.c_fc.weight.shape[1]
    prefix = f"transformer.h.{layer}.mlp"
    sandbox.staged.clear()
    sandbox.staged.stage_entries(f"{prefix}.c_fc.weight", torch.arange(d_model) * d_mlp + unit, steepness * read_dir)
    sandbox.staged.stage_entries(f"{prefix}.c_fc.bias", torch.tensor([unit]), torch.tensor([-steepness * theta]))
    sandbox.staged.stage_entries(
        f"{prefix}.c_proj.weight",
        unit * d_model + torch.arange(d_model),
        write_vec,
        write_vec=write_vec,
        description=f"L{layer} u{unit} (blend={sae_blend}, patch_sae)",
    )
    return {"layer": layer, "unit": unit, "theta": round(theta, 4), "pos_proj_mean": round(proj_pos.mean().item(), 4)}


def main():
    t0 = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    layers = [6, 7, 8, 9, 10, 11]

    print("Loading base GPT-2 & Pretrained SAEs (Layers 6..11)...")
    tok = AutoTokenizer.from_pretrained("gpt2", cache_dir=str(CACHE_DIR), local_files_only=True)
    base_model = AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=str(CACHE_DIR), local_files_only=True).eval()
    base_saes = {L: SparseAutoencoder.from_pretrained_gpt2(layer=L, cache_dir=CACHE_DIR) for L in layers}
    task = get_unverified_entity_task()

    results = {}
    for blend_name, blend_val in [("blended_0.5", 0.5), ("pure_sae_1.0", 1.0)]:
        print("\n" + "=" * 84)
        print(f"ABLATION REGIME: {blend_name} (sae_blend={blend_val})")
        print("=" * 84)
        for patch_flag in (False, True):
            label = f"{blend_name}__patch_sae_{'ON' if patch_flag else 'OFF'}"
            res = run_arm(
                base_model=base_model,
                tok=tok,
                base_saes=base_saes,
                task=task,
                patch_sae=patch_flag,
                sae_blend=blend_val,
                max_Latents_patch=128,
            )
            results[label] = res
            print(f"\n--- Arm: {label} ---")
            print(
                f"  Final Held-Out Failure : {res['final_held_fail']*100:5.1f}% | "
                f"Held-Out Target Prob: {res['final_held_prob']*100:5.2f}% | "
                f"Keepers Identical: {res['final_held_keeper']*100:5.1f}% | "
                f"KL: {res['final_kl']:.6f}"
            )
            for step in res["trace"]:
                if step["committed"]:
                    print(
                        f"    L{step['layer']} (COMMIT u{step['unit']}): "
                        f"top_latents={step['top_latents']} (AUC={step['top_auc']:.3f}, entry_MSE={step['sae_mse_at_entry']:.1f}) -> "
                        f"HeldFail={step['held_fail']*100:4.1f}%, HeldProb={step['held_prob']*100:5.1f}% | "
                        f"NextSAE: {step['next_sae_patched_mse']}"
                    )
                else:
                    print(
                        f"    L{step['layer']} (SKIP): "
                        f"top_latents={step['top_latents']} (AUC={step['top_auc']:.3f}, entry_MSE={step['sae_mse_at_entry']:.1f})"
                    )

    out_path = RESULTS_DIR / "ablation_1_patch_on_vs_off.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved Ablation 1 comparison to {out_path} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
