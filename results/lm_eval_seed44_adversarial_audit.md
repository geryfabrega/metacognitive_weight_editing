# Official EleutherAI `lm-eval` Adversarial Audit (`seed44_adversarial_audit`)

| Task | Role | Base GPT-2 | `no_sae` | `frozen_sae` | `patched_sae` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `blimp_left_branch_island_echo_question` | TARGET | 48.10% | 48.10% (**+0.00%**) | 93.40% (**+45.30%**) | 95.10% (**+47.00%**) |
| `blimp_principle_A_reconstruction` | TARGET | 39.70% | 40.10% (**+0.40%**) | 42.40% (**+2.70%**) | 39.80% (**+0.10%**) |
| `blimp_npi_present_1` | TARGET | 59.60% | 59.80% (**+0.20%**) | 59.70% (**+0.10%**) | 64.10% (**+4.50%**) |
| `blimp_tough_vs_raising_1` | TARGET | 65.30% | 64.60% (**-0.70%**) | 66.90% (**+1.60%**) | 67.10% (**+1.80%**) |
| `blimp_left_branch_island_simple_question` | ADVERSARIAL_WH | 79.10% | 79.40% (**+0.30%**) | 62.90% (**-16.20%**) | 64.40% (**-14.70%**) |
| `blimp_wh_questions_object_gap` | ADVERSARIAL_WH | 79.00% | 79.90% (**+0.90%**) | 79.00% (**+0.00%**) | 78.00% (**-1.00%**) |
| `blimp_wh_questions_subject_gap` | ADVERSARIAL_WH | 94.90% | 95.50% (**+0.60%**) | 94.90% (**+0.00%**) | 94.70% (**-0.20%**) |
| `blimp_wh_vs_that_with_gap` | ADVERSARIAL_WH | 63.10% | 61.10% (**-2.00%**) | 65.10% (**+2.00%**) | 65.00% (**+1.90%**) |
| `blimp_matrix_question_npi_licensor_present` | ADVERSARIAL_WH | 64.60% | 63.90% (**-0.70%**) | 0.40% (**-64.20%**) | 0.10% (**-64.50%**) |
| `blimp_determiner_noun_agreement_1` | COLLATERAL | 99.10% | 99.20% (**+0.10%**) | 99.00% (**-0.10%**) | 98.90% (**-0.20%**) |
| `blimp_animate_subject_passive` | COLLATERAL | 79.50% | 79.30% (**-0.20%**) | 79.30% (**-0.20%**) | 79.00% (**-0.50%**) |
| `lambada_openai` | COLLATERAL | 37.00% | 37.00% (**+0.00%**) | 36.50% (**-0.50%**) | 36.00% (**-1.00%**) |
| `hellaswag` | COLLATERAL | 37.00% | 37.00% (**+0.00%**) | 37.00% (**+0.00%**) | 37.00% (**+0.00%**) |

## Group Means
| Group | Base GPT-2 | `no_sae` | `frozen_sae` | `patched_sae` |
| :--- | :---: | :---: | :---: | :---: |
| **TARGET** | 53.17% | 53.15% (**-0.02%**) | 65.60% (**+12.43%**) | 66.53% (**+13.35%**) |
| **ADVERSARIAL_WH** | 76.14% | 75.96% (**-0.18%**) | 60.46% (**-15.68%**) | 60.44% (**-15.70%**) |
| **COLLATERAL** | 63.15% | 63.13% (**-0.02%**) | 62.95% (**-0.20%**) | 62.72% (**-0.43%**) |