"""Metacognitive Weight Editing: Outside-Agent SAE-guided direct weight surgery with local SAE patching."""

import importlib.metadata

_orig_version = importlib.metadata.version


def _patched_version(pkg):
    v = _orig_version(pkg)
    return v.split(".dev")[0] if pkg == "torch" and ".dev" in v else v


importlib.metadata.version = _patched_version

from .sae import FeatureHit, SparseAutoencoder, fetch_neuronpedia_labels
from .sandbox import SparseWeightDelta, WeightEditingSandbox
from .benchmarks import (
    BLiMPItem,
    BenchmarkSplit,
    ContrastivePair,
    DEFAULT_BLIMP_SUBSETS,
    TruthfulQAItem,
    evaluate_benchmark,
    load_blimp_benchmark,
    load_truthfulqa_benchmark,
    run_official_lm_eval,
)
from .agent import (
    WeightEditingAgentSession,
    load_openai_config,
    run_heuristic_agent_loop,
    run_openai_agent_loop,
)
from .tasks import BenchmarkTask, evaluate_model, get_unverified_entity_task

__all__ = [
    "FeatureHit",
    "SparseAutoencoder",
    "fetch_neuronpedia_labels",
    "SparseWeightDelta",
    "WeightEditingSandbox",
    "BLiMPItem",
    "BenchmarkSplit",
    "ContrastivePair",
    "DEFAULT_BLIMP_SUBSETS",
    "TruthfulQAItem",
    "evaluate_benchmark",
    "load_blimp_benchmark",
    "load_truthfulqa_benchmark",
    "run_official_lm_eval",
    "WeightEditingAgentSession",
    "load_openai_config",
    "run_heuristic_agent_loop",
    "run_openai_agent_loop",
    "BenchmarkTask",
    "evaluate_model",
    "get_unverified_entity_task",
]
