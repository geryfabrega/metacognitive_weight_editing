# Official EleutherAI `lm-eval` Verification (`seed44_paired_patched_sae`)

- **Checkpoint:** `/tmp/mwe_runs/blimp_paired_seed44/checkpoints/branch_patched_sae/model_delta_final.safetensors`
- **Modified tensors:** `11` (`2,364,675` entries, L2=`26.9689`)
- **Target BLiMP-Hard Mean:** `53.17%` -> **`66.53%`** (**`+13.35%`**)
- **Collateral Mean:** `63.15%` -> `62.72%` (`-0.43%`)

| Task | Role | Base GPT-2 | Edited GPT-2 | Delta |
| :--- | :---: | :---: | :---: | :---: |
| `blimp_left_branch_island_echo_question` | TARGET | 48.10% | 95.10% | **+47.00%** |
| `blimp_principle_A_reconstruction` | TARGET | 39.70% | 39.80% | **+0.10%** |
| `blimp_npi_present_1` | TARGET | 59.60% | 64.10% | **+4.50%** |
| `blimp_tough_vs_raising_1` | TARGET | 65.30% | 67.10% | **+1.80%** |
| `blimp_determiner_noun_agreement_1` | COLLATERAL | 99.10% | 98.90% | **-0.20%** |
| `blimp_animate_subject_passive` | COLLATERAL | 79.50% | 79.00% | **-0.50%** |
| `lambada_openai` | COLLATERAL | 37.00% | 36.00% | **-1.00%** |
| `hellaswag` | COLLATERAL | 37.00% | 37.00% | **+0.00%** |