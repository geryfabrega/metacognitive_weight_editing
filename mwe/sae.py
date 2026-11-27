"""Pretrained SAE loader, feature localizer, Neuronpedia label fetcher, and local dictionary patcher."""

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
    label: str = ""


def fetch_neuronpedia_labels(
    layer: int,
    latent_indices: List[int],
    cache_dir: Optional[Path] = None,
    timeout: float = 2.5,
) -> Dict[int, str]:
    """
    Fetch and locally cache human-readable auto-interp descriptions for `jbloom/GPT2-Small-SAEs-Reformatted`
    from Neuronpedia (`gpt2-small` / `{layer}-res-jb`). Fails gracefully offline.
    """
    cdir = Path(cache_dir or CACHE_DIR)
    cdir.mkdir(parents=True, exist_ok=True)
    cache_path = cdir / f"neuronpedia_gpt2_L{layer}.json"

    cached: Dict[str, str] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except Exception:
            cached = {}

    out: Dict[int, str] = {}
    missing = [idx for idx in latent_indices if str(idx) not in cached]

    if missing:
        try:
            import requests
            changed = False
            for idx in missing:
                url = f"https://www.neuronpedia.org/api/feature/gpt2-small/{layer}-res-jb/{idx}"
                try:
                    resp = requests.get(url, timeout=timeout)
                    if resp.status_code == 200:
                        data = resp.json()
                        exps = data.get("explanations") or []
                        desc = exps[0].get("description", "").strip() if exps else ""
                        cached[str(idx)] = desc or "(no auto-interp label)"
                        changed = True
                    else:
                        cached[str(idx)] = "(label unavailable)"
                except Exception:
                    # Stop trying network on first timeout/offline error so loop stays fast
                    break
            if changed:
                cache_path.write_text(json.dumps(cached, indent=2))
        except Exception:
            pass

    for idx in latent_indices:
        out[idx] = cached.get(str(idx), "")
    return out


class SparseAutoencoder(nn.Module):
    """Pretrained Sparse Autoencoder with support for local subspace-masked patching."""

    def __init__(
        self,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        hook_point: str = "",
        layer: int = -1,
    ):
        super().__init__()
        # Convention (sae-lens / GPT2-Small-SAEs):
        # W_enc: [d_in, d_sae], b_enc: [d_sae], W_dec: [d_sae, d_in], b_dec: [d_in]
        if W_enc.shape[0] > W_enc.shape[1]:
            W_enc = W_enc.T.contiguous()
        if W_dec.shape[0] < W_dec.shape[1]:
            W_dec = W_dec.T.contiguous()
        self.d_in, self.d_sae = W_enc.shape
        self.hook_point = hook_point
        self.layer = layer

        self.W_enc = nn.Parameter(W_enc.float().clone())
        self.b_enc = nn.Parameter(b_enc.float().clone())
        self.W_dec = nn.Parameter(W_dec.float().clone())
        self.b_dec = nn.Parameter(b_dec.float().clone())

    @classmethod
    def from_pretrained_gpt2(cls, layer: int = 6, cache_dir: Optional[Path] = None) -> "SparseAutoencoder":
        """Load a pretrained GPT-2 Small residual-stream SAE from jbloom/GPT2-Small-SAEs-Reformatted."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        cdir = Path(cache_dir or CACHE_DIR)
        cdir.mkdir(parents=True, exist_ok=True)
        hook = f"blocks.{layer}.hook_resid_pre"

        # Fast path: resolve directly from local HF snapshot directory if cached
        weights_path = None
        snap_root = cdir / "models--jbloom--GPT2-Small-SAEs-Reformatted" / "snapshots"
        if snap_root.exists():
            for snap in sorted(snap_root.iterdir()):
                cand = snap / hook / "sae_weights.safetensors"
                if cand.exists():
                    weights_path = str(cand)
                    break

        if weights_path is None:
            try:
                weights_path = hf_hub_download(
                    repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
                    filename=f"{hook}/sae_weights.safetensors",
                    cache_dir=str(cdir),
                    local_files_only=True,
                )
            except Exception:
                weights_path = hf_hub_download(
                    repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
                    filename=f"{hook}/sae_weights.safetensors",
                    cache_dir=str(cdir),
                )
        tensors = load_file(weights_path)
        return cls(
            W_enc=tensors["W_enc"],
            b_enc=tensors["b_enc"],
            W_dec=tensors["W_dec"],
            b_dec=tensors["b_dec"],
            hook_point=hook,
            layer=layer,
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
    def localize_features(
        self,
        pos_acts: torch.Tensor,
        neg_acts: torch.Tensor,
        top_k: int = 8,
        fetch_labels: bool = False,
        cache_dir: Optional[Path] = None,
    ) -> List[FeatureHit]:
        """
        Find the SAE latents that best separate failure/bad span activations (`pos_acts`)
        from desired/good span activations (`neg_acts`).
        Returns both positive-diff (fires more on failures/myths) and negative-diff (fires more on truthful/good) features.
        """
        z_pos = self.encode(pos_acts)  # [N_pos, d_sae]
        z_neg = self.encode(neg_acts)  # [N_neg, d_sae]

        pos_mean = z_pos.mean(dim=0)
        neg_mean = z_neg.mean(dim=0)
        diff = pos_mean - neg_mean

        # Pre-screen top 256 candidates by mean activation difference for fast exact ROC-AUC
        cand_indices = torch.argsort(diff.abs(), descending=True)[:256].tolist()
        hits: List[FeatureHit] = []
        for idx in cand_indices:
            p = z_pos[:, idx]
            n = z_neg[:, idx]
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
        top_hits = hits[:top_k]

        if fetch_labels and self.layer >= 0:
            labels = fetch_neuronpedia_labels(
                layer=self.layer,
                latent_indices=[h.latent_idx for h in top_hits],
                cache_dir=cache_dir,
            )
            for h in top_hits:
                h.label = labels.get(h.latent_idx, "")

        return top_hits

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

            wv = write_vectors.float()
            if wv.dim() == 1:
                wv = wv.unsqueeze(0)
            wv_norm = F.normalize(wv, dim=-1)
            enc_norm = F.normalize(self.W_enc.T, dim=-1)  # [d_sae, d_in]
            dec_norm = F.normalize(self.W_dec, dim=-1)    # [d_sae, d_in]
            subspace_score = (
                (enc_norm @ wv_norm.T).abs().max(dim=-1).values
                + (dec_norm @ wv_norm.T).abs().max(dim=-1).values
            )

            x_hat_init, _ = self.forward(post_edit_acts.float())
            residual = post_edit_acts.float() - x_hat_init
            mean_res_dir = F.normalize(residual.mean(dim=0, keepdim=True), dim=-1)
            res_score = (dec_norm @ mean_res_dir.T).abs().squeeze(-1)

            combined = subspace_score + 0.5 * act_shift / (act_shift.max() + 1e-8) + res_score
            selected = torch.argsort(combined, descending=True)[:max_latents].sort().values

            sub_W_enc = self.W_enc[:, selected].clone().detach().requires_grad_(True)
            sub_b_enc = self.b_enc[selected].clone().detach().requires_grad_(True)
            sub_W_dec = self.W_dec[selected, :].clone().detach().requires_grad_(True)

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
                sub_W_dec.data = F.normalize(sub_W_dec.data, dim=-1)

        with torch.no_grad():
            self.W_enc[:, selected] = sub_W_enc.detach()
            self.b_enc[selected] = sub_b_enc.detach()
            self.W_dec[selected, :] = sub_W_dec.detach()
            after_metrics = self.reconstruction_metrics(post_edit_acts)
            keeper_metrics = self.reconstruction_metrics(pre_edit_acts)

        self._last_patched_indices = selected.detach().cpu()

        return {
            "hook_point": self.hook_point,
            "total_latents": self.d_sae,
            "patched_latents_count": int(selected.numel()),
            "patched_fraction_pct": round(100.0 * selected.numel() / self.d_sae, 4),
            "patched_latent_indices": selected.tolist()[:12],
            "post_edit_before_sae_patch": before_metrics,
            "post_edit_after_sae_patch": after_metrics,
            "pre_edit_distribution_retention": keeper_metrics,
        }

    def save_patch_delta(self, path: Path) -> None:
        """Save only the locally patched SAE latents to disk as safetensors (leaving base SAE untouched)."""
        from safetensors.torch import save_file

        idx = getattr(self, "_last_patched_indices", torch.tensor([], dtype=torch.long))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        save_file(
            {
                "patched_indices": idx,
                "W_enc_cols": self.W_enc[:, idx].detach().cpu().contiguous(),
                "b_enc_vals": self.b_enc[idx].detach().cpu().contiguous(),
                "W_dec_rows": self.W_dec[idx, :].detach().cpu().contiguous(),
            },
            str(path),
        )
