"""Metacognitive Weight Editing: SAE-guided weight patching with local SAE updates."""

from .sae import FeatureHit, SparseAutoencoder
from .sandbox import SparseWeightDelta, WeightEditingSandbox
from .tasks import BenchmarkTask, evaluate_model, get_unverified_entity_task

__all__ = [
    "FeatureHit",
    "SparseAutoencoder",
    "SparseWeightDelta",
    "WeightEditingSandbox",
    "BenchmarkTask",
    "evaluate_model",
    "get_unverified_entity_task",
]
