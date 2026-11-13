"""Pretrained SAE loader, feature localizer, and local dictionary patcher."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import torch
if ".dev" in torch.__version__:
    torch.__version__ = torch.__version__.split(".dev")[0]
import torch.nn as nn
import torch.nn.functional as F

CACHE_DIR = Path("/tmp/mwe_cache")


@dataclass
class FeatureHit:
    latent_idx: int
    auc: float
    pos_mean: float
    neg_mean: float
    diff: float
    fire_rate_pos: float
    fire_rate_neg: float


class SparseAutoencoder(nn.Module):
    """Pretrained Sparse Autoencoder with support for local subspace-masked patching."""

    def __init__(self, W_enc: torch.Tensor, b_enc: torch.Tensor, W_dec: torch.Tensor, b_dec: torch.Tensor, hook_point: str = ""):
        super().__init__()
        # Convention (sae-lens / GPT2-Small-SAEs):
        # W_enc: [d_in, d_sae], b_enc: [d_sae], W_dec: [d_sae, d_in], b_dec: [d_in]
        if W_enc.shape[0] > W_enc.shape[1]:
            W_enc = W_enc.T.contiguous()
        if W_dec.shape[0] < W_dec.shape[1]:
            W_dec = W_dec.T.contiguous()
        self.d_in, self.d_sae = W_enc.shape
        self.hook_point = hook_point

        self.W_enc = nn.Parameter(W_enc.float().clone())
        self.b_enc = nn.Parameter(b_enc.float().clone())
        self.W_dec = nn.Parameter(W_dec.float().clone())
        self.b_dec = nn.Parameter(b_dec.float().clone())

    @classmethod
    def from_pretrained_gpt2(cls, layer: int = 6, cache_dir: Optional[Path] = None) -> "SparseAutoencoder":
        """Download a pretrained GPT-2 Small residual-stream SAE from jbloom/GPT2-Small-SAEs-Reformatted."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cdir = Path(cache_dir or CACHE_DIR)
        cdir.mkdir(parents=True, exist_ok=True)
        hook = f"blocks.{layer}.hook_resid_pre"
        weights_path = hf_hub_download(
            repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
            filename=f"{hook}/sae_weights.safetensors",
            cache_dir=str(cdir),
            local_files_only=True,
        )
        tensors = load_file(weights_path)
        return cls(
            W_enc=tensors["W_enc"],
            b_enc=tensors["b_enc"],
            W_dec=tensors["W_dec"],
            b_dec=tensors["b_dec"],
            hook_point=hook,
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: [..., d_in] -> z: [..., d_sae] (ReLU activations)."""
        x_centered = x.float() - self.b_dec
        return F.relu(x_centered @ self.W_enc + self.b_enc)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: [..., d_sae] -> x_hat: [..., d_in]."""
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def reconstruction_metrics(self, x: torch.Tensor) -> Dict[str, float]:
        x_f = x.float()
        x_hat, z = self.forward(x_f)
        mse = F.mse_loss(x_hat, x_f).item()
        total_var = (x_f - x_f.mean(dim=0, keepdim=True)).pow(2).mean().item()
        fvu = mse / max(total_var, 1e-8)
        cos = F.cosine_similarity(x_hat, x_f, dim=-1).mean().item()
        l0 = (z > 0).float().sum(dim=-1).mean().item()
        return {
            "mse": round(mse, 6),
            "explained_variance": round(1.0 - fvu, 5),
            "cosine_sim": round(cos, 5),
            "mean_l0": round(l0, 2),
        }

    @torch.no_grad()
    def localize_features(self, pos_acts: torch.Tensor, neg_acts: torch.Tensor, top_k: int = 8) -> List[FeatureHit]:
        """Find the SAE latents that best separate failure activations (pos_acts) from keeper activations (neg_acts)."""
        z_pos = self.encode(pos_acts)  # [N_pos, d_sae]
        z_neg = self.encode(neg_acts)  # [N_neg, d_sae]

        pos_mean = z_pos.mean(dim=0)
        neg_mean = z_neg.mean(dim=0)
        diff = pos_mean - neg_mean

        # Pre-screen top 256 candidates by mean activation difference for fast exact AUC
        cand_indices = torch.argsort(diff.abs(), descending=True)[:256].tolist()
        hits: List[FeatureHit] = []
        for idx in cand_indices:
            p = z_pos[:, idx]
            n = z_neg[:, idx]
            # Pairwise Mann-Whitney U / ROC-AUC
            comp = (p[:, None] > n[None, :]).float() + 0.5 * (p[:, None] == n[None, :]).float()
            auc = comp.mean().item()
            hits.append(
                FeatureHit(
                    latent_idx=int(idx),
                    auc=round(auc, 4),
                    pos_mean=round(pos_mean[idx].item(), 4),
                    neg_mean=round(neg_mean[idx].item(), 4),
                    diff=round(diff[idx].item(), 4),
                    fire_rate_pos=round((p > 0).float().mean().item(), 4),
                    fire_rate_neg=round((n > 0).float().mean().item(), 4),
                )
            )
        hits.sort(key=lambda h: (abs(h.auc - 0.5), abs(h.diff)), reverse=True)
        return hits[:top_k]

    def local_subspace_patch(
        self,
        pre_edit_acts: torch.Tensor,
        post_edit_acts: torch.Tensor,
        write_vectors: torch.Tensor,
        max_latents: int = 32,
        steps: int = 40,
        lr: float = 2e-2,
        l1_coeff: float = 1e-3,
    ) -> Dict[str, object]:
        """
        Locally patch ONLY the small subset of SAE latents affected by the model weight edit.
        99.8%+ of the dictionary remains strictly frozen.
        """
        with torch.no_grad():
            before_metrics = self.reconstruction_metrics(post_edit_acts)
            z_pre = self.encode(pre_edit_acts)
            z_post = self.encode(post_edit_acts)
            act_shift = (z_post - z_pre).abs().mean(dim=0)

            # 1. Subspace overlap with the weight edit's write vectors [k, d_in]
            wv = write_vectors.float()
            if wv.dim() == 1:
                wv = wv.unsqueeze(0)
            wv_norm = F.normalize(wv, dim=-1)
            enc_norm = F.normalize(self.W_enc.T, dim=-1)  # [d_sae, d_in]
            dec_norm = F.normalize(self.W_dec, dim=-1)    # [d_sae, d_in]
            subspace_score = (enc_norm @ wv_norm.T).abs().max(dim=-1).values + (dec_norm @ wv_norm.T).abs().max(dim=-1).values

            # 2. Reconstruction residual alignment on post-edit activations
            x_hat_init, _ = self.forward(post_edit_acts.float())
            residual = post_edit_acts.float() - x_hat_init  # [N, d_in]
            mean_res_dir = F.normalize(residual.mean(dim=0, keepdim=True), dim=-1)
            res_score = (dec_norm @ mean_res_dir.T).abs().squeeze(-1)

            combined = subspace_score + 0.5 * act_shift / (act_shift.max() + 1e-8) + res_score
            selected = torch.argsort(combined, descending=True)[:max_latents].sort().values

            # Extract ONLY the selected subset of columns/rows for optimization
            sub_W_enc = self.W_enc[:, selected].clone().detach().requires_grad_(True)  # [d_in, k]
            sub_b_enc = self.b_enc[selected].clone().detach().requires_grad_(True)     # [k]
            sub_W_dec = self.W_dec[selected, :].clone().detach().requires_grad_(True)  # [k, d_in]

            # Precompute the frozen contribution of the other 99.8%+ latents!
            frozen_mask = torch.ones(self.d_sae, dtype=torch.bool, device=self.W_enc.device)
            frozen_mask[selected] = False
            x_centered = post_edit_acts.float() - self.b_dec
            z_frozen = F.relu(x_centered @ self.W_enc[:, frozen_mask] + self.b_enc[frozen_mask])
            x_hat_frozen = z_frozen @ self.W_dec[frozen_mask, :] + self.b_dec

        opt = torch.optim.Adam([sub_W_enc, sub_b_enc, sub_W_dec], lr=lr)
        target_residual = (post_edit_acts.float() - x_hat_frozen).detach()

        for _ in range(steps):
            opt.zero_grad()
            z_sub = F.relu(x_centered.detach() @ sub_W_enc + sub_b_enc)
            pred_res = z_sub @ sub_W_dec
            mse_loss = F.mse_loss(pred_res, target_residual)
            l1_loss = l1_coeff * z_sub.abs().mean()
            loss = mse_loss + l1_loss
            loss.backward()
            opt.step()
            with torch.no_grad():
                # Keep decoder rows unit-normed as in standard SAE training
                sub_W_dec.data = F.normalize(sub_W_dec.data, dim=-1)

        with torch.no_grad():
            self.W_enc[:, selected] = sub_W_enc.detach()
            self.b_enc[selected] = sub_b_enc.detach()
            self.W_dec[selected, :] = sub_W_dec.detach()
            after_metrics = self.reconstruction_metrics(post_edit_acts)
            keeper_metrics = self.reconstruction_metrics(pre_edit_acts)

        return {
            "hook_point": self.hook_point,
            "total_latents": self.d_sae,
            "patched_latents_count": int(selected.numel()),
            "patched_fraction_pct": round(100.0 * selected.numel() / self.d_sae, 4),
            "patched_latent_indices": selected.tolist()[:12],
            "post_edit_Before_SAE_patch": before_metrics,
            "post_edit_After_SAE_patch": after_metrics,
            "pre_edit_distribution_retention": keeper_metrics,
        }
