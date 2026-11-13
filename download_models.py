"""Download GPT-2 Small and Pretrained SAEs (Layers 6..11) to /tmp/mwe_cache."""

import importlib.metadata
_orig_version = importlib.metadata.version
def _patched_version(pkg):
    v = _orig_version(pkg)
    return v.split(".dev")[0] if pkg == "torch" and ".dev" in v else v
importlib.metadata.version = _patched_version

from pathlib import Path
import torch
torch.__version__ = torch.__version__.split(".dev")[0]
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download

CACHE_DIR = Path("/tmp/mwe_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

print("Ensuring GPT-2 tokenizer and model are cached...")
AutoTokenizer.from_pretrained("gpt2", cache_dir=str(CACHE_DIR))
AutoModelForCausalLM.from_pretrained("gpt2", cache_dir=str(CACHE_DIR))

for layer in (6, 7, 8, 9, 10, 11):
    hook = f"blocks.{layer}.hook_resid_pre"
    print(f"Ensuring pretrained SAE for {hook} is cached...")
    hf_hub_download(
        repo_id="jbloom/GPT2-Small-SAEs-Reformatted",
        filename=f"{hook}/sae_weights.safetensors",
        cache_dir=str(CACHE_DIR),
    )

print("All models and pretrained SAEs (Layers 6-11) cached in /tmp/mwe_cache!")
