# BLIMP Paired-Branch Controlled Ablation (`gpt-6.1-sol`, seed=42)

**Shared SAE Edit #1 (L6, `dormant_neuron_gate`):** L6 dormant MLP gate read=[16129] write=-1*[16129] (gain=6.00)
- Held-Out Base -> Edit #1: `0.4807` -> `0.4844` (`+0.0037`), Margin `-0.2793` -> `-0.2711` (`+0.0082`)

| Arm | Base Held-Out | Final Held-Out | **Total Δ** | **Phase-2 Δ (vs Edit #1)** | Margin Δ | Keeper | WT2 PPL | WT2 KL | Total Edits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_sae` *(baseline)* | 0.4807 | 0.4745 | **-0.0062** | — | -0.1099 | 96.4% | 47.96 | 0.00461 | 3 |
| `frozen_sae` | 0.4807 | 0.4870 | **+0.0063** | **+0.0026** | +0.0328 | 96.5% | 48.25 | 0.00063 | 2 |
| `patched_sae` | 0.4807 | 0.4844 | **+0.0037** | **+0.0000** | +0.0082 | 98.9% | 48.20 | 0.00018 | 1 |

## Baseline Arm: `no_sae`
**Agent Summary:** Committed three no-SAE edits, progressing downstream: L9 MLP raw projection alpha=-1.2, L10 MLP raw projection alpha=-0.8, and L11 dormant MLP neuron gate gain=3. Earlier L7/L8 candidates were not committed because they failed to improve overall accuracy. Practice BLiMP accuracy improved from 0.5188 to 0.5375 (83 to 86 correct of 160). Final WikiText-2 KL 0.00461 <= 0.05, keeper retention 0.988 >= 0.9, perplexity 47.9571 versus baseline 48.1774. Gains are modest and mixed: left-branch 0.35->0.40, NPI 0.60->0.65, reconstruction 0.40->0.375, tough/raising unchanged 0.725; mean margin declined 0.2949->0.2035. Stopping at turn budget, with held-out generalization not yet established.
- **Edit #1 (L9, rank1_projection):** L9.mlp raw diff_dir projection (alpha=-1.200) — _Layer-9 MLP projection improves overall practice accuracy from 0.5188 to 0.525 and mean margin by 0.0464; KL 0.000494 and keeper retention 0.988 pass both collateral guards. Gains arise in left-branch questions despite a small reconstruction regression._
- **Edit #2 (L10, rank1_projection):** L10.mlp raw diff_dir projection (alpha=-0.800) — _Adds a guarded practice-accuracy improvement from 0.525 to 0.5312, primarily through NPI accuracy increasing from 0.60 to 0.65. Cumulative KL is 0.004422 and keeper retention 0.988. Mean margin declines, so further edits should be approached conservatively._
- **Edit #3 (L11, dormant_neuron_gate):** L11 dormant MLP gate raw diff_dir (gain=3.00) — _Sparse layer-11 dormant-neuron gate increases left-branch accuracy from 0.375 to 0.4 without reducing other subset accuracies. Overall practice accuracy reaches 0.5375, cumulative KL 0.00461, keeper retention 0.988; only 1537 entries changed by this edit._

## Phase-2 Branch: `frozen_sae`
**Agent Summary:** Preserved inherited L6 gate and committed one additional downstream edit: L9 dormant MLP gated suppression read/write latent 22703 (negation/refutation), gain 8, targeting npi_present_1. Practice accuracy increased from branch-entry 53.75% to 55.00% (original 51.88%); NPI increased 60% to 67.5%, with reconstruction declining 40% to 37.5%. Final cumulative WikiText-2 KL 0.000632, perplexity 48.2453, keeper retention 0.9759, OOD keeper accuracy unchanged at 0.875. Tested both projection and gated operators; rejected non-improving L8-L10 candidates rather than commit noisy edits. Frozen downstream SAEs remained unchanged. Stopping at diminishing returns; held-out improvement remains to be evaluated.
- **Edit #2 (L9, dormant_neuron_gate):** L9 dormant MLP gate read=[22703] write=-1*[22703] (gain=8.00) — _Sparse L9 gated suppression of negation/refutation latent 22703 improves NPI accuracy from 60% to 67.5% and total practice accuracy from 53.75% to 55%, despite one reconstruction regression. Cumulative KL 0.000632 and keeper retention 0.9759 safely pass guards._

## Phase-2 Branch: `patched_sae`
**Agent Summary:** Preserved inherited L6 gate; no additional downstream edit demonstrated a clear overall practice gain. Tested projection and dormant-neuron gates across layers 7–11, focusing on principle_A_reconstruction, then NPI and left-branch residuals. Candidates met fluency bounds but failed progress checks, so none were committed. Retained practice accuracy 0.5375 versus original 0.5188, keeper retention 1.0, WikiText-2 KL 0.000182.

**Local Subspace SAE Patching Logs:**
- After Edit #1 (L6) -> Patched L7 SAE (0.1302% latents): MSE `923227.06` -> `567316.25`
- After Edit #1 (L6) -> Patched L8 SAE (0.1302% latents): MSE `872342.25` -> `299958.78`
- After Edit #1 (L6) -> Patched L9 SAE (0.1302% latents): MSE `772281.19` -> `235321.06`
- After Edit #1 (L6) -> Patched L10 SAE (0.1302% latents): MSE `619591.44` -> `320740.38`
- After Edit #1 (L6) -> Patched L11 SAE (0.1302% latents): MSE `1071942.62` -> `393221.81`
