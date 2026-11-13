"""Staged sparse weight editing and forward-pass evaluation sandbox."""

import contextlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class SparseWeightDelta:
    """Stores sparse index/value deltas against a frozen model's parameters."""
    deltas: Dict[str, Tuple[torch.LongTensor, torch.Tensor]] = field(default_factory=dict)
    write_vectors: List[torch.Tensor] = field(default_factory=list)
    description: str = ""

    def stage_entries(
        self,
        param_name: str,
        flat_indices: torch.LongTensor,
        new_values: torch.Tensor,
        write_vec: Optional[torch.Tensor] = None,
        description: str = "",
    ) -> None:
        self.deltas[param_name] = (flat_indices.detach().cpu().long(), new_values.detach().cpu())
        if write_vec is not None:
            self.write_vectors.append(write_vec.detach().cpu().float())
        if description:
            self.description = description

    def clear(self) -> None:
        self.deltas.clear()
        self.write_vectors.clear()
        self.description = ""

    @property
    def num_entries_changed(self) -> int:
        return sum(int(idx.numel()) for idx, _ in self.deltas.values())


class WeightEditingSandbox:
    """
    Holds a frozen LLM, stages sparse weight deltas derived from SAE features,
    evaluates them in a reversible forward-pass sandbox, and commits only if guards pass.
    """

    def __init__(self, model: torch.nn.Module, tokenizer):
        self.model = model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.staged = SparseWeightDelta()
        self.committed_history: List[Dict[str, object]] = []
        self.passes_used = 0

    @contextlib.contextmanager
    def applied(self, use_staged: bool = True):
        """Temporarily apply staged sparse weight deltas for a sandboxed forward pass, then restore."""
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

    def commit_staged(self) -> Dict[str, object]:
        """Permanently commit the currently staged sparse weight delta into the model."""
        params = dict(self.model.named_parameters())
        total_l2 = 0.0
        with torch.no_grad():
            for name, (idx, val) in self.staged.deltas.items():
                flat = params[name].data.view(-1)
                dev_idx = idx.to(flat.device)
                new_v = val.to(flat.device, flat.dtype)
                total_l2 += float((new_v.float() - flat[dev_idx].float()).norm().item() ** 2)
                flat[dev_idx] = new_v
        rec = {
            "description": self.staged.description,
            "tensors_edited": list(self.staged.deltas.keys()),
            "entries_changed": self.staged.num_entries_changed,
            "l2_norm": round(total_l2 ** 0.5, 4),
        }
        self.committed_history.append(rec)
        return rec

    @torch.no_grad()
    def capture_layer_activations(
        self, prompts: List[str], layer: int, hook_site: str = "resid_pre", use_staged: bool = False
    ) -> torch.Tensor:
        """Capture last-token activations at `layer` ('resid_pre' or 'mlp_mid') inside the sandbox."""
        self.passes_used += 1
        acts: List[torch.Tensor] = []
        block = self.model.transformer.h[layer]

        def pre_hook(module, inputs):
            # inputs[0]: [batch, seq, d_model]
            acts.append(inputs[0][:, -1, :].detach().float().cpu())

        def mlp_hook(module, inputs, output):
            acts.append(output[:, -1, :].detach().float().cpu())

        with self.applied(use_staged=use_staged):
            handle = (
                block.register_forward_pre_hook(pre_hook)
                if hook_site == "resid_pre"
                else block.mlp.act.register_forward_hook(mlp_hook)
            )
            try:
                for p in prompts:
                    enc = self.tokenizer(p, return_tensors="pt")
                    self.model(**enc)
            finally:
                handle.remove()
        return torch.cat(acts, dim=0)

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
        """
        Compile an SAE feature direction + threshold into a single dormant MLP neuron in GPT-2:
        - Finds the least-active neuron in `transformer.h[layer].mlp`
        - Wires `c_fc` column `u` to read the normalized `ln_2(x)` projection along the SAE feature direction
        - Sets `c_fc.bias[u]` so all keeper/negative positions sit well below threshold (GELU negative tail ~ 0)
        - Wires `c_proj` row `u` to push logits toward `target_token_ids` and away from `suppress_token_ids`
        """
        block = self.model.transformer.h[layer]
        d_model = self.model.config.n_embd

        # Collect post-ln_2 MLP inputs and mid-MLP activations on practice prompts
        ln2_pos, ln2_neg, mlp_acts = [], [], []

        def ln2_hook_pos(mod, inp, out):
            ln2_pos.append(out[:, -1, :].detach().float().cpu())

        def ln2_hook_neg(mod, inp, out):
            ln2_neg.append(out[:, -1, :].detach().float().cpu())

        def act_hook(mod, inp, out):
            # Capture across ALL token positions to find truly least-used neuron & background projections
            mlp_acts.append(out.detach().float().cpu().reshape(-1, out.shape[-1]))

        h1 = block.ln_2.register_forward_hook(ln2_hook_pos)
        for p in pos_prompts:
            self.model(**self.tokenizer(p, return_tensors="pt"))
        h1.remove()

        all_neg_tokens = []

        def ln2_hook_all_neg(mod, inp, out):
            ln2_neg.append(out[:, -1, :].detach().float().cpu())
            all_neg_tokens.append(out.detach().float().cpu().reshape(-1, d_model))

        h2 = block.ln_2.register_forward_hook(ln2_hook_all_neg)
        h3 = block.mlp.act.register_forward_hook(act_hook)
        for p in neg_prompts:
            self.model(**self.tokenizer(p, return_tensors="pt"))
        h2.remove()
        h3.remove()

        X_pos = torch.cat(ln2_pos, dim=0)           # [N_pos, d_model]
        X_neg = torch.cat(ln2_neg, dim=0)           # [N_neg, d_model]
        X_bg = torch.cat(all_neg_tokens, dim=0)     # [N_all_tokens, d_model]
        usage = torch.cat(mlp_acts, dim=0).abs().mean(dim=0)
        unit = int(torch.argmin(usage).item())

        # Refine the SAE feature direction in post-ln_2 coordinates so it cleanly separates pos from neg+bg
        sae_dir = F.normalize(sae_feature_dir.float().cpu(), dim=-1)
        diff_dir = F.normalize(X_pos.mean(dim=0) - X_neg.mean(dim=0), dim=-1)
        if torch.dot(sae_dir, diff_dir) < 0:
            sae_dir = -sae_dir
        # Blend SAE monosemantic direction with empirical post-ln_2 contrastive direction
        read_dir = F.normalize(0.5 * sae_dir + 0.5 * diff_dir, dim=-1)

        proj_pos = X_pos @ read_dir
        proj_neg = X_neg @ read_dir
        proj_bg = X_bg @ read_dir
        theta = max(proj_neg.max().item(), proj_bg.max().item()) + margin

        # Build output write vector in residual space using the model's unembedding matrix
        W_U = self.model.lm_head.weight.detach().float().cpu()  # [vocab, d_model]
        tgt_vec = F.normalize(W_U[target_token_ids].mean(dim=0), dim=-1)
        sup_vec = F.normalize(W_U[suppress_token_ids].mean(dim=0), dim=-1) if suppress_token_ids else torch.zeros_like(tgt_vec)
        write_vec = F.normalize(tgt_vec - 0.6 * sup_vec, dim=-1) * gain

        # Stage sparse edits into GPT-2 Conv1D weights:
        # c_fc.weight is [d_model, 4*d_model] -> column `unit` has flat indices: row * (4*d_model) + unit
        # c_fc.bias is [4*d_model] -> index `unit`
        # c_proj.weight is [4*d_model, d_model] -> row `unit` has flat indices: unit * d_model + col
        d_mlp = block.mlp.c_fc.weight.shape[1]
        prefix = f"transformer.h.{layer}.mlp"

        fc_w_idx = torch.arange(d_model, dtype=torch.long) * d_mlp + unit
        fc_w_val = steepness * read_dir

        fc_b_idx = torch.tensor([unit], dtype=torch.long)
        fc_b_val = torch.tensor([-steepness * theta], dtype=torch.float32)

        proj_w_idx = unit * d_model + torch.arange(d_model, dtype=torch.long)
        proj_w_val = write_vec

        self.staged.clear()
        self.staged.stage_entries(f"{prefix}.c_fc.weight", fc_w_idx, fc_w_val)
        self.staged.stage_entries(f"{prefix}.c_fc.bias", fc_b_idx, fc_b_val)
        self.staged.stage_entries(
            f"{prefix}.c_proj.weight",
            proj_w_idx,
            proj_w_val,
            write_vec=write_vec,
            description=f"Layer {layer} MLP unit {unit} SAE-gated patch (theta={theta:.3f}, g={steepness})",
        )

        return {
            "layer": layer,
            "unit": unit,
            "theta": round(theta, 4),
            "pos_proj_mean": round(proj_pos.mean().item(), 4),
            "pos_proj_min": round(proj_pos.min().item(), 4),
            "neg_proj_max": round(proj_neg.max().item(), 4),
            "bg_proj_max": round(proj_bg.max().item(), 4),
            "practice_fire_rate_pos": round((proj_pos > theta).float().mean().item(), 4),
            "practice_fire_rate_neg": round((proj_neg > theta).float().mean().item(), 4),
            "entries_changed": self.staged.num_entries_changed,
        }
