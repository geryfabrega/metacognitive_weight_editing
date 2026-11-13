"""Minimal skeleton for the SAE + Sandbox + Local SAE Update loop."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch


@dataclass
class SparseWeightDelta:
    """Tracks sparse index/value deltas against a frozen base model before commit."""
    deltas: Dict[str, Tuple[torch.LongTensor, torch.Tensor]] = field(default_factory=dict)

    def stage(self, param_name: str, flat_indices: torch.LongTensor, new_values: torch.Tensor) -> None:
        self.deltas[param_name] = (flat_indices.detach().cpu().long(), new_values.detach().cpu())

    def clear(self, param_name: Optional[str] = None) -> None:
        if param_name is None:
            self.deltas.clear()
        else:
            self.deltas.pop(param_name, None)

    @property
    def num_entries_changed(self) -> int:
        return sum(int(idx.numel()) for idx, _ in self.deltas.values())


class LocalSAEPatcher:
    """Locally updates only SAE latents intersecting a sparse weight edit or its residual."""

    def __init__(self, W_enc: torch.Tensor, W_dec: torch.Tensor, b_enc: torch.Tensor, b_dec: torch.Tensor):
        self.W_enc = W_enc  # [n_latents, d_model]
        self.W_dec = W_dec  # [d_model, n_latents]
        self.b_enc = b_enc  # [n_latents]
        self.b_dec = b_dec  # [d_model]

    @torch.no_grad()
    def affected_latents(self, write_subspace: torch.Tensor, threshold: float = 0.05) -> torch.LongTensor:
        """Return indices of SAE latents whose encoder weights have cosine overlap > threshold with write_subspace."""
        # write_subspace: [d_model, k] orthonormal basis of the weight edit's output column space
        q, _ = torch.linalg.qr(write_subspace.float())
        enc_norm = self.W_enc.float() / (self.W_enc.float().norm(dim=-1, keepdim=True) + 1e-8)
        overlap = (enc_norm @ q).norm(dim=-1)
        return (overlap > threshold).nonzero(as_tuple=False).squeeze(-1)


class WeightEditingSandbox:
    """Applies staged weight deltas inside a reversible forward-pass context and commits only when accepted."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.staged = SparseWeightDelta()
        self.committed: List[SparseWeightDelta] = []
