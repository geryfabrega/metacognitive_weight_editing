"""Metacognitive Weight Editing: SAE-guided weight patching with local SAE updates."""

from .loop import LocalSAEPatcher, SparseWeightDelta, WeightEditingSandbox

__all__ = ["SparseWeightDelta", "LocalSAEPatcher", "WeightEditingSandbox"]
