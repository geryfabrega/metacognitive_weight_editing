# Metacognitive Weight Editing (`mwe`)

Closed-loop **Outside-Agent Weight-Space Programming** guided by **pretrained Sparse Autoencoders (SAEs)** and **Local Subspace SAE Patching (`<0.15%` latents)** — directly editing GPT-2 Small (124M) weights **without gradient descent or fine-tuning**, verified on **official EleutherAI `lm-eval`**.

## Architecture

```mermaid
flowchart LR
    Agent["Outside Reasoning Agent (OpenAI Responses API: gpt-6.1-sol / gpt-6.2-astra)"] -->|1. Inspect failures & localize span-pooled SAE latents + Neuronpedia labels| SAE["Synced Pretrained SAEs (L6..L11)"]
    SAE -->|2. Stage Rank-1 Feature Projection or 1-Neuron Dormant MLP Concept Gate| SB["Reversible Forward-Pass Sandbox"]
    SB -->|Practice + WikiText-2 KL / Keeper Guard Failed| Agent
    SB -->|Guards Passed: Commit ΔW| Model["Edited GPT-2 Small (124M)"]
    Model -->|3. Local Subspace SAE Patch (32/24,576 = 0.13% latents)| SAE
    Model -->|Strictly Unseen Held-Out Split & Official lm-eval| Eval["BLiMP-Hard + LAMBADA + HellaSwag + WikiText-2"]
```

---

## Verified Benchmark Results (GPT-2 Small 124M, Zero Gradients)

### 1. Official EleutherAI `lm-eval` Verification & Adversarial Shortcut Audit
Unmodified `lm_eval.simple_evaluate` comparing Base GPT-2 Small vs. all three Seed 44 checkpoints (`no_sae`, `frozen_sae`, `patched_sae`) across target tasks, **5 adversarial grammatical Wh-question shortcut traps**, and unrelated collateral benchmarks *(note: `lm-eval` evaluates all 1,000 BLiMP items/task, which includes 40/1,000 practice items + 960 unseen items)*:

| Official `lm-eval` Task | Role | Base GPT-2 | `no_sae` | `frozen_sae` | `patched_sae` |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `blimp_left_branch_island_echo_question` | **TARGET** | `48.10%` | `48.10% (+0.00%)` | `93.40% (+45.30%)` | `95.10% (+47.00%)` |
| `blimp_principle_A_reconstruction` | **TARGET** | `39.70%` | `40.10% (+0.40%)` | `42.40% (+2.70%)` | `39.80% (+0.10%)` |
| `blimp_npi_present_1` | **TARGET** | `59.60%` | `59.80% (+0.20%)` | `59.70% (+0.10%)` | **`64.10% (+4.50%)`** |
| `blimp_tough_vs_raising_1` | **TARGET** | `65.30%` | `64.60% (-0.70%)` | `66.90% (+1.60%)` | **`67.10% (+1.80%)`** |
| `blimp_left_branch_island_simple_question` | **ADVERSARIAL_WH** | `79.10%` | `79.40% (+0.30%)` | `62.90% (-16.20%)` | `64.40% (-14.70%)` |
| `blimp_wh_questions_object_gap` | **ADVERSARIAL_WH** | `79.00%` | `79.90% (+0.90%)` | `79.00% (+0.00%)` | `78.00% (-1.00%)` |
| `blimp_wh_questions_subject_gap` | **ADVERSARIAL_WH** | `94.90%` | `95.50% (+0.60%)` | `94.90% (+0.00%)` | `94.70% (-0.20%)` |
| `blimp_wh_vs_that_with_gap` | **ADVERSARIAL_WH** | `63.10%` | `61.10% (-2.00%)` | `65.10% (+2.00%)` | `65.00% (+1.90%)` |
| `blimp_matrix_question_npi_licensor_present` | **ADVERSARIAL_WH** | `64.60%` | `63.90% (-0.70%)` | **`0.40% (-64.20%)`** | **`0.10% (-64.50%)`** |
| `blimp_determiner_noun_agreement_1` | COLLATERAL | `99.10%` | `99.20% (+0.10%)` | `99.00% (-0.10%)` | `98.90% (-0.20%)` |
| `blimp_animate_subject_passive` | COLLATERAL | `79.50%` | `79.30% (-0.20%)` | `79.30% (-0.20%)` | `79.00% (-0.50%)` |
| `lambada_openai` (200) | COLLATERAL | `37.00%` | `37.00% (+0.00%)` | `36.50% (-0.50%)` | `36.00% (-1.00%)` |
| `hellaswag` (200) | COLLATERAL | `37.00%` | `37.00% (+0.00%)` | `37.00% (+0.00%)` | `37.00% (+0.00%)` |
| **Target BLiMP-Hard Mean** | **TARGET** | `53.17%` | `53.15% (-0.02%)` | `65.60% (+12.43%)` | `66.53% (+13.35%)` |
| **Adversarial Wh-Question Mean** | **ADVERSARIAL_WH** | `76.14%` | `75.96% (-0.18%)` | `60.46% (-15.68%)` | `60.44% (-15.70%)` |
| **General Collateral Mean** | COLLATERAL | `63.15%` | `63.13% (-0.02%)` | `62.95% (-0.20%)` | `62.72% (-0.43%)` |

> [!WARNING]
> **Critical Self-Audit Finding (Weight-Space Reward Hacking):** The adversarial Wh-question audit exposed that the headline `+47%` jump on `left_branch_island_echo_question` came largely from **suppressing sentence-initial question SAE features (`L7 #11434`, `L11 #22295`)**, which passed general fluency guardrails (`hellaswag +0.00%`, `WikiText-2 KL = 0.0056`) while collapsing the grammatical matrix-question sibling (`matrix_question_npi_licensor_present: 64.6% → 0.1%`). On the non-shortcut targets (`npi_present_1`, `tough_vs_raising_1`), `patched_sae` still yielded clean, honest gains (`+4.5%`, `+1.8%`).
>
> **Fix Implemented (v2 Guardrails):** `mwe/benchmarks.py` now enforces **strict per-keeper-subset retention (`>= 90%`)** on disjoint practice/held-out splits of both grammatical Wh-question siblings (`left_branch_island_simple_question` and `matrix_question_npi_licensor_present`), preventing the outside agent from exploiting fronted-question shortcuts.

### 2. Controlled Paired-Branch Ablation (3 Seeds × 3,840 Held-Out = 11,520 Unseen Items)
`frozen_sae` and `patched_sae` share the **exact same Phase-1 Edit #1**, then fork from identical post-Edit-1 weights for downstream Phase-2 edits (`gpt-6.1-sol`, `reasoning_effort=medium`):

| Arm | Seed 42 (Total / P2) | Seed 43 (Total / P2) | Seed 44 (Total / P2) | **Mean Total Δ** | **Mean Phase-2 Δ** | **Mean Margin Δ** | Mean Keeper |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`no_sae`** *(raw `diff_dir` baseline)* | `-0.62%` / — | `+3.05%` / — | `-0.39%` / — | `+0.68%` | — | `+0.032` | `97.9%` |
| **`frozen_sae`** *(static SAEs)* | `+0.63%` / `+0.26%` | `+3.02%` / `+0.83%` | `+12.57%` / `+11.95%` | `+5.41%` | `+4.35%` | `+0.845` | `97.9%` |
| **`patched_sae`** *(Local Subspace SAE Patching)* | `+0.37%` / `+0.00%` | **`+4.27%`** / **`+2.08%`** | **`+14.08%`** / **`+13.46%`** | **`+6.24%`** | **`+5.18%`** | **`+0.989`** | **`98.3%`** |

* Updating just **`32 / 24,576` (`0.13%`)** of downstream SAE latents after each commit cut downstream L11 SAE reconstruction MSE by **`-98.8%` to `-99.1%`** across stacked edits (e.g., Seed 44 L11 MSE: `1,016,486 → 416,099 → 91,518 → 8,970`).

### 3. Independent 5-Seed BLiMP-Hard Sweep (Seeds 42–46, 10,680 Unseen Items)
* **`patched_sae` beat `no_sae` on 5/5 seeds (8/8 across all independent + paired-branch runs):**
  * `no_sae`: Mean **`+0.70%`** (Median `+0.60%`), Margin `+0.069`, Keeper `98.4%`
  * `frozen_sae`: Mean **`+1.79%`** (Median `+1.98%`), Margin `+0.261`, Keeper `98.9%`
  * `patched_sae`: Mean **`+3.94%`** (Median **`+2.10%`**), Margin **`+0.559`**, Keeper **`99.1%`**

---

## Quick Start

### 1. Reproduce the Official `lm-eval` Result ($0 API Cost)
```bash
python3 download_models.py
python3 verify_checkpoint_lm_eval.py --delta checkpoints/seed44_paired_patched_sae.safetensors --tag seed44_paired_patched_sae
```

### 2. Run the Outside-Agent Closed Loop (OpenAI Responses API)
Copy `.env.example` to `.env` (gitignored) and add your `OPENAI_API_KEY`:
```env
OPENAI_API_KEY=sk-...
MWE_AGENT_MODEL=gpt-6.1-sol
MWE_REASONING_EFFORT=medium
MWE_MAX_TURNS=20
MWE_MAX_EDITS=6
```

Run a controlled paired-branch ablation (`no_sae` + shared Edit #1 → `frozen_sae` vs. `patched_sae`):
```bash
python3 run_agent_benchmark.py \
  --benchmark blimp --paired-branch \
  --practice-size 40 --heldout-size 960 --seed 44 \
  --model gpt-6.1-sol --reasoning-effort medium
```

Run an independent 3-arm ablation or zero-cost `--heuristic` dry run:
```bash
python3 run_agent_benchmark.py --benchmark blimp --ablation-all --model gpt-6.1-sol --reasoning-effort medium
python3 run_agent_benchmark.py --benchmark truthfulqa --ablation-all --practice-size 100 --heldout-size 200 --model gpt-6.1-sol
python3 run_agent_benchmark.py --benchmark blimp --paired-branch --heuristic
```
