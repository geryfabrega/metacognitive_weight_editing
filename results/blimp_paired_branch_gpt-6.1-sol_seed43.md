# BLIMP Paired-Branch Controlled Ablation (`gpt-6.1-sol`, seed=43)

**Shared SAE Edit #1 (L6, `dormant_neuron_gate`):** L6 dormant MLP gate read=[20588] write=-1*[20588] (gain=4.00)
- Held-Out Base -> Edit #1: `0.4820` -> `0.5039` (`+0.0219`), Margin `-0.2530` -> `+0.0665` (`+0.3195`)

| Arm | Base Held-Out | Final Held-Out | **Total Δ** | **Phase-2 Δ (vs Edit #1)** | Margin Δ | Keeper | WT2 PPL | WT2 KL | Total Edits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_sae` *(baseline)* | 0.4820 | 0.5125 | **+0.0305** | — | +0.2407 | 98.6% | 48.55 | 0.00256 | 3 |
| `frozen_sae` | 0.4820 | 0.5122 | **+0.0302** | **+0.0083** | +0.4498 | 99.8% | 48.27 | 0.00041 | 3 |
| `patched_sae` | 0.4820 | 0.5247 | **+0.0427** | **+0.0208** | +0.5720 | 97.4% | 48.62 | 0.00711 | 3 |

## Baseline Arm: `no_sae`
**Agent Summary:** Committed three raw-direction MLP rank-1 edits, without SAEs or gradient training: L8 Principle A alpha +0.6, L9 left-branch alpha +0.6, L10 NPI alpha +1.2. Tested both projection and dormant-neuron approaches; negative suppression and neuron gates did not improve practice performance. Practice accuracy improved from 0.4875 to 0.50625 (three additional correct pairs of 160), mean margin from -0.337 to -0.1121. Final cumulative WikiText-2 KL 0.002556 <= 0.05, keeper retention 0.9857 >= 0.9, perplexity 48.546 versus 48.1774 baseline. Final subset accuracies: left-branch .275, Principle A .4, NPI .625, tough/raising .725. Gains are modest and practice-only; proceed to strictly held-out evaluation without claiming established generalization.
- **Edit #1 (L8, rank1_projection):** L8.mlp raw diff_dir projection (alpha=+0.600) — _L8 MLP positive rank-1 edit improves Principle A accuracy from 37.5% to 40% and overall mean margin by 0.0596, with no subset regressions, full keeper retention, and KL 0.000222. Suppression had moved margins in the wrong direction._
- **Edit #2 (L9, rank1_projection):** L9.mlp raw diff_dir projection (alpha=+0.600) — _Complementary L9 left-branch rank-1 edit raises target accuracy to 30% and overall accuracy to 50%, improves mean margin, and preserves other subset accuracies and all keepers at cumulative KL 0.000614._
- **Edit #3 (L10, rank1_projection):** L10.mlp raw diff_dir projection (alpha=+1.200) — _L10 NPI-focused projection improves aggregate practice accuracy to 50.625% and mean margin by 0.1104; NPI and tough/raising each gain 2.5 points, offset by a 2.5-point left-branch regression. Cumulative KL remains low at 0.002556 and keeper retention 0.9857 exceeds the 0.9 guard._

## Phase-2 Branch: `frozen_sae`
**Agent Summary:** Retained inherited L6 gate and committed two complementary downstream edits: L7 dormant gate suppressing question-opening latent 11434 (gain 4), and L9 rank-one MLP amplification of pronoun/personal-action latent 16914 (alpha +0.8). Practice accuracy increased from branch-entry 50.62% to 51.88% (original baseline 48.75%); left-branch 37.5%, reconstruction 40%, NPI 60%, tough/raising 70%. Keeper retention 1.0, WikiText-2 KL 0.000413, perplexity 48.2736. Additional L10/L11 gated and projection candidates yielded no accuracy gains or regressions and were not committed. Stopping at diminishing returns; downstream SAEs remained frozen.
- **Edit #2 (L7, dormant_neuron_gate):** L7 dormant MLP gate read=[11434] write=-1*[11434] (gain=4.00) — _Selectively suppress L7 question-opening latent 11434 (AUC 0.8905) with a dormant gate, improving left-branch accuracy 35% to 37.5% with other subsets unchanged, keeper retention 1.0 and cumulative KL 0.000111._
- **Edit #3 (L9, rank1_projection):** L9.mlp SAE proj #16914 (alpha=+0.800) — _Amplify L9 pronoun/personal-action feature 16914, enriched on grammatical reconstruction spans (AUC 0.14). Rank-one MLP edit improves reconstruction accuracy 37.5% to 40%, preserves all other subsets and keepers, with cumulative KL only 0.000413._

## Phase-2 Branch: `patched_sae`
**Agent Summary:** Added two downstream gated edits to inherited L6 edit: L8 suppresses question-opening latents 11933/23037 (gain 8, margin .05), L9 suppresses bad-associated actions/reactions latent 14660 (gain 8, margin .05). Overall practice accuracy increased from fork baseline 50.62% to 53.12%; left-branch 35% to 40%, reconstruction 37.5% to 42.5%, other subsets unchanged. Final cumulative WikiText-2 KL .007115, keeper retention .9857. Tested projection and gate alternatives at L7-L11; remaining candidates lacked accuracy gains or degraded performance, so stopped at diminishing returns without committing them. Held-out generalization remains to be evaluated.
- **Edit #2 (L8, dormant_neuron_gate):** L8 dormant MLP gate read=[11933, 23037] write=-1*[11933, 23037] (gain=8.00) — _Suppress question-opening features selectively with a gated L8 neuron, improving left-branch accuracy 35% to 40% and overall 50.62% to 51.88%; keeper retention 1.0 and cumulative KL 0.001268._
- **Edit #3 (L9, dormant_neuron_gate):** L9 dormant MLP gate read=[14660] write=-1*[14660] (gain=8.00) — _Gate suppression of strongly bad-associated actions/reactions latent 14660 (AUC .926) improves reconstruction accuracy 37.5% to 42.5%, overall 53.12%; cumulative KL .007115 and keeper retention .9857 remain within guards._

**Local Subspace SAE Patching Logs:**
- After Edit #1 (L6) -> Patched L7 SAE (0.1302% latents): MSE `1079702.88` -> `787814.25`
- After Edit #1 (L6) -> Patched L8 SAE (0.1302% latents): MSE `1022662.62` -> `437562.53`
- After Edit #1 (L6) -> Patched L9 SAE (0.1302% latents): MSE `901342.81` -> `286641.56`
- After Edit #1 (L6) -> Patched L10 SAE (0.1302% latents): MSE `721662.00` -> `388541.59`
- After Edit #1 (L6) -> Patched L11 SAE (0.1302% latents): MSE `1243097.38` -> `577687.81`
- After Edit #2 (L8) -> Patched L9 SAE (0.1302% latents): MSE `335630.06` -> `22218.54`
- After Edit #2 (L8) -> Patched L10 SAE (0.1302% latents): MSE `443105.66` -> `176464.30`
- After Edit #2 (L8) -> Patched L11 SAE (0.1302% latents): MSE `658513.00` -> `198943.94`
- After Edit #3 (L9) -> Patched L10 SAE (0.1302% latents): MSE `184154.34` -> `19864.93`
- After Edit #3 (L9) -> Patched L11 SAE (0.1302% latents): MSE `206892.86` -> `14377.52`
