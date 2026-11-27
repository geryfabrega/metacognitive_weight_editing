# BLIMP Paired-Branch Controlled Ablation (`gpt-6.1-sol`, seed=44)

**Shared SAE Edit #1 (L7, `dormant_neuron_gate`):** L7 dormant MLP gate read=[11434] write=-1*[11434] (gain=8.00)
- Held-Out Base -> Edit #1: `0.4852` -> `0.4914` (`+0.0062`), Margin `-0.2462` -> `-0.1612` (`+0.0850`)

| Arm | Base Held-Out | Final Held-Out | **Total Δ** | **Phase-2 Δ (vs Edit #1)** | Margin Δ | Keeper | WT2 PPL | WT2 KL | Total Edits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_sae` *(baseline)* | 0.4852 | 0.4813 | **-0.0039** | — | -0.0356 | 98.6% | 48.10 | 0.00051 | 2 |
| `frozen_sae` | 0.4852 | 0.6109 | **+0.1257** | **+0.1195** | +2.0527 | 97.3% | 48.35 | 0.00153 | 4 |
| `patched_sae` | 0.4852 | 0.6260 | **+0.1408** | **+0.1346** | +2.3870 | 98.5% | 48.41 | 0.00564 | 4 |

## Baseline Arm: `no_sae`
**Agent Summary:** Stopping at diminishing returns with 2 committed raw rank-1 MLP projections: L8 alpha=-0.8 localized on last-token left_branch_island_echo_question contrasts; L9 alpha=-0.8 localized on mean principle_A_reconstruction contrasts. Both projection and dormant-neuron approaches tested across L6-L10; gates and final L10 projection gave no meaningful progress and were not committed. Practice accuracy 0.4125 -> 0.4188 (one additional correct pair), cumulative WikiText-2 KL 0.000512, keeper retention 1.0, PPL 48.1774 -> 48.1021. Mean margin worsened -0.4989 -> -0.5305; gains are modest and held-out improvement remains unestablished.
- **Edit #1 (L8, rank1_projection):** L8.mlp raw diff_dir projection (alpha=-0.800) — _Layer-8 last-token contrast projection improves practice mean margin by 0.0227 with unchanged accuracy, perfect keeper retention, and KL 0.000065; retain this low-collateral margin improvement rather than the broader layer-7 trade-offs._
- **Edit #2 (L9, rank1_projection):** L9.mlp raw diff_dir projection (alpha=-0.800) — _Layer-9 MLP projection increases aggregate practice accuracy from 0.4125 to 0.4188 without losing keepers; cumulative KL remains 0.000512. Target subset is unchanged; observed gain is in NPI, while mean margin worsens._

## Phase-2 Branch: `frozen_sae`
**Agent Summary:** Preserved inherited L7 gate and committed three complementary downstream edits with frozen SAEs: L9 NPI-negation gate suppressing 22703 gain 8; L10 MLP rank-one amplification of adjective feature 10474 alpha .8; L11 echo-question gate suppressing question feature 22295 gain 8. Branch-entry practice accuracy .425 improved to .5625; subsets echo .30->.775, reconstruction .375->.40, NPI .50->.525, tough/raising .525->.55. Mean margin -.4436->1.4756. Final cumulative WikiText-2 KL .001535, keeper retention .9848, PPL 48.3478. Four total committed edits including inherited edit. Stopping within turn budget with three clean downstream gains; held-out generalization remains to be evaluated.
- **Edit #2 (L9, dormant_neuron_gate):** L9 dormant MLP gate read=[22703] write=-1*[22703] (gain=8.00) — _Suppress the negation/refutation latent 22703 selectively on NPI failure contexts (AUC .925; gate fires on 90% bad spans versus 0% good and 0.98% background). Gain-8 sparse gate improves NPI accuracy .50 to .525 and overall .425 to .4313, with mean margin +.0527, cumulative KL .001199 and keeper retention .9848. Projection and lower-gain gate failed to improve accuracy._
- **Edit #3 (L10, rank1_projection):** L10.mlp SAE proj #10474 (alpha=+0.800) — _Amplify L10 adjective/person-quality feature 10474, strongly enriched in grammatical tough/raising spans (AUC .1316), via MLP rank-one projection alpha .8. Improves tough/raising .525 to .55 and reconstruction .375 to .40, overall .4313 to .4437 and mean margin +.0403; cumulative KL .001523, retention .9848. Lower projection strength improved margins only; gated version was inactive on target spans._
- **Edit #4 (L11, dormant_neuron_gate):** L11 dormant MLP gate read=[22295] write=-1*[22295] (gain=8.00) — _L11 question feature 22295 has AUC .9923 on remaining echo-question failures. Sparse suppression gate cleanly separates bad versus good spans (100% versus 0% firing; background .82%). Gain 8 improves echo-question accuracy .30 to .775 and overall .4437 to .5625, mean margin +1.8262, without losses in other subsets. Cumulative KL .001535 and keeper retention .9848 pass comfortably. Downstream SAEs remain frozen._

## Phase-2 Branch: `patched_sae`
**Agent Summary:** Retained inherited L7 gate and added three downstream edits: L8 MLP rank-1 amplification of latent 22110 (+0.8), L9 gated suppression of question/scenario latent 12505 (gain 8), and L11 gated suppression of question latent 22295 (gain 8). Both projection and gate alternatives were tested; unhelpful L10 candidates were not committed. Practice accuracy improved from fork baseline .425 to .5563 (unmodified baseline .4125), echo subset .300 to .775, mean margin -.4436 to 1.7047. Final cumulative WikiText-2 KL .005642, PPL 48.4058, keeper retention .9394, OOD keeper accuracy .8625. Stop with four total committed edits; remaining subset features lacked compelling causal labels or sandbox gains. Proceed to strictly held-out evaluation; practice gains are not held-out claims.
- **Edit #2 (L8, rank1_projection):** L8.mlp SAE proj #22110 (alpha=+0.800) — _Amplify L8 latent 22110, which separates grammatical from ungrammatical reconstruction spans (AUC .04). Projection improves overall accuracy .425 to .4313 and mean margin .0256, with retention 1.0 and cumulative KL .000972. Gain is modest and occurs in NPI rather than reconstruction; treat as preliminary, not evidence of a resolved binding mechanism._
- **Edit #3 (L9, dormant_neuron_gate):** L9 dormant MLP gate read=[12505] write=-1*[12505] (gain=8.00) — _Suppress question/scenario latent 12505 only above background threshold. This targets excessive fronted-question preference; echo accuracy rises .300 to .325 and NPI .525 to .600, overall .4313 to .4562 with mean margin +.3673. Cumulative KL .005632 and keeper retention .9697 pass guards._
- **Edit #4 (L11, dormant_neuron_gate):** L11 dormant MLP gate read=[22295] write=-1*[22295] (gain=8.00) — _L11 question latent 22295 has AUC .9931 on residual echo-question failures. Background-thresholded suppression raises echo accuracy .325 to .775 and overall .4562 to .5563, mean margin +1.7554; NPI drops .600 to .550 but net gain is substantial. Cumulative WikiText KL .005642 and keeper retention .9394 pass both guards._

**Local Subspace SAE Patching Logs:**
- After Edit #1 (L7) -> Patched L8 SAE (0.1302% latents): MSE `816873.75` -> `270619.50`
- After Edit #1 (L7) -> Patched L9 SAE (0.1302% latents): MSE `718834.12` -> `173209.89`
- After Edit #1 (L7) -> Patched L10 SAE (0.1302% latents): MSE `573713.31` -> `327508.91`
- After Edit #1 (L7) -> Patched L11 SAE (0.1302% latents): MSE `1016486.38` -> `416099.12`
- After Edit #2 (L8) -> Patched L9 SAE (0.1302% latents): MSE `174717.30` -> `12018.04`
- After Edit #2 (L8) -> Patched L10 SAE (0.1302% latents): MSE `330505.34` -> `145737.09`
- After Edit #2 (L8) -> Patched L11 SAE (0.1302% latents): MSE `418674.03` -> `91518.02`
- After Edit #3 (L9) -> Patched L10 SAE (0.1302% latents): MSE `151230.33` -> `20972.61`
- After Edit #3 (L9) -> Patched L11 SAE (0.1302% latents): MSE `96691.80` -> `8970.50`
