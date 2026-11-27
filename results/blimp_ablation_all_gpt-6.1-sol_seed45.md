# BLIMP Benchmark Findings (`gpt-6.1-sol`, seed=45)

| Arm | Base Held-Out | Final Held-Out | Delta | Margin Δ | Keeper | WikiText-2 PPL | WikiText-2 KL | Edits | API Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_sae` | 0.4839 | 0.4815 | -0.0024 | -0.0424 | 95.5% | 48.26 | 0.00083 | 2 | 89,996 |
| `frozen_sae` | 0.4839 | 0.5086 | +0.0247 | +0.4562 | 97.5% | 48.15 | 0.00023 | 2 | 146,463 |
| `patched_sae` | 0.4839 | 0.4914 | +0.0075 | +0.0588 | 97.5% | 48.29 | 0.00053 | 2 | 168,305 |

## Arm: `no_sae`
**Agent Summary:** Committed two raw-direction dormant MLP gates without SAEs or gradient training: L9 NPI gain 6 and L10 reconstruction gain 6, both margin 0. Practice BLiMP rose from 0.4437 to 0.475 (+5 net items/160); NPI 0.45→0.575 and left-branch 0.40→0.425. Final committed WikiText-2 KL 0.00083, keeper retention 0.9859, PPL 48.2625 versus baseline 48.1774. Earlier layer 6–8 projections/gates and final L10 projection did not improve accuracy and were not committed. Stopping with only the two guard-passing edits; held-out generalization remains to be evaluated.

- **Edit #1 (L9, dormant_neuron_gate):** L9 dormant MLP gate raw diff_dir (gain=6.00) — _Layer-9 NPI-targeted dormant neuron gate raises practice accuracy from 44.37% to 46.25% and NPI accuracy from 45% to 57.5%, with KL 0.000633 and keeper retention 0.9718._
- **Edit #2 (L10, dormant_neuron_gate):** L10 dormant MLP gate raw diff_dir (gain=6.00) — _Complementary layer-10 reconstruction gate adds two net practice successes, lifting accuracy to 47.5%, while cumulative KL remains 0.00083 and keeper retention 0.9859._

## Arm: `frozen_sae`
**Agent Summary:** Committed two dormant-neuron gates (L8 question-feature suppression; L9 personal-pronoun/action injection), changing 3,074 weight entries without gradient training or SAE updates. Practice BLiMP improved from 0.4437 to 0.4813; mean margin -0.4022 to 0.0492. Final WikiText-2 KL 0.000230, perplexity 48.1521 vs baseline 48.1774, keeper retention 0.9859. Rank-1 projections did not produce clean gains. Downstream NPI gate trials produced mixed subset effects and were not committed; stopping rather than fitting small practice fluctuations. Held-out generalization remains to be evaluated.

- **Edit #1 (L8, dormant_neuron_gate):** L8 dormant MLP gate read=[4249, 19318] write=-1*[4249] (gain=8.00) — _Suppress an overactive fronted-question direction only under joint question/uncertainty feature gating; improves echo-question accuracy by 10 points and total accuracy by 2.51 points with negligible KL and 95.77% keeper retention._
- **Edit #2 (L9, dormant_neuron_gate):** L9 dormant MLP gate read=[14660] write=+1*[16914] (gain=6.00) — _Gated injection of a personal-pronoun/action direction recovers reconstruction and tough/raising examples, increasing practice accuracy by 1.25 points while preserving keeper retention and negligible KL._

## Arm: `patched_sae`
**Agent Summary:** Committed two gradient-free edits: L8 attention rank-1 amplification of SAE latent 22110 (+0.8), improving mean margin without accuracy loss; and L9 one-neuron context gate suppressing negation latent 22703 (gain 8), improving NPI accuracy 0.45->0.475. Downstream SAEs patched after each commit. Final practice accuracy 0.45 vs baseline 0.4437, mean margin -0.3031 vs -0.4022, WikiText-2 KL 0.000531, keeper retention 0.9859, PPL 48.2867 vs 48.1774. Other tested projections/gates did not clearly improve accuracy and were not committed. Stopping at turn budget with modest practice improvement; held-out generalization remains to be evaluated.

- **Edit #1 (L8, rank1_projection):** L8.attn SAE proj #22110 (alpha=+0.800) — _Small attention projection amplification improves mean grammatical-pair margin by 0.0383 without losing any practice passes, with full keeper retention and KL 0.000083. Accuracy is unchanged; commit as a low-collateral margin improvement before probing patched downstream features._
- **Edit #2 (L9, dormant_neuron_gate):** L9 dormant MLP gate read=[22703] write=-1*[22703] (gain=8.00) — _Context-gated suppression of the negation/refutation latent improves NPI practice accuracy from 45% to 47.5% and overall accuracy to 45%, with improved mean margin. Cumulative KL 0.000531 and keeper retention 0.9859 comfortably satisfy guards; only one dormant MLP neuron is changed._

**Local Subspace SAE Patching Logs:**
- After Edit #1 (L8) -> Patched L9 SAE (0.1302% latents): MSE `825507.38` -> `394459.88`
- After Edit #1 (L8) -> Patched L10 SAE (0.1302% latents): MSE `669768.12` -> `573997.75`
- After Edit #1 (L8) -> Patched L11 SAE (0.1302% latents): MSE `1134832.50` -> `628768.06`
- After Edit #2 (L9) -> Patched L10 SAE (0.1302% latents): MSE `574198.81` -> `434276.28`
- After Edit #2 (L9) -> Patched L11 SAE (0.1302% latents): MSE `629090.44` -> `194512.66`
