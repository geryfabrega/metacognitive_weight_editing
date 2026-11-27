# BLIMP Paired-Branch Controlled Ablation (`gpt-6.1-sol`, seed=46)

**Shared SAE Edit #1 (L6, `dormant_neuron_gate`):** L6 dormant MLP gate read=[20588] write=-1*[20588] (gain=4.00)
- Held-Out Base -> Edit #1: `0.4807` -> `0.5023` (`+0.0216`), Margin `-0.2663` -> `+0.0576` (`+0.3239`)

| Arm | Base Held-Out | Final Held-Out | **Total Δ** | **Phase-2 Δ (vs Edit #1)** | Margin Δ | Keeper | WT2 PPL | WT2 KL | Total Edits |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_sae` *(baseline)* | 0.4807 | 0.4883 | **+0.0076** | — | +0.0541 | 99.1% | 48.45 | 0.00109 | 1 |
| `frozen_sae` | 0.4807 | 0.5109 | **+0.0302** | **+0.0086** | +0.4717 | 99.8% | 48.19 | 0.00018 | 3 |
| `patched_sae` | 0.4807 | 0.5234 | **+0.0427** | **+0.0211** | +0.6713 | 99.7% | 48.34 | 0.00167 | 3 |

## Baseline Arm: `no_sae`
**Agent Summary:** Committed one clean raw-direction rank-1 projection at L10 MLP, alpha +0.4, localized on principle_A_reconstruction. Practice BLiMP improved from 0.5188 to 0.5312 (83 to 85 of 160); Principle A improved 0.525 to 0.55 and tough/raising 0.575 to 0.6. Mean margin improved -0.0171 to 0.0579. Final committed WikiText-2 KL 0.001085, perplexity 48.448 versus baseline 48.1774, keeper retention 1.0. Tested both projections and dormant-neuron gates across L6/L8 and downstream projections L10/L11. Did not commit no-gain gates, incidental one-item gains with worse margins, or final L11 left-branch edit that reduced accuracy. Stopped at diminishing returns; held-out generalization remains to be evaluated.
- **Edit #1 (L10, rank1_projection):** L10.mlp raw diff_dir projection (alpha=+0.400) — _Layer-10 MLP positive rank-1 edit improves Principle A and tough/raising by one item each, raising total practice accuracy from 0.5188 to 0.5312 and mean margin by 0.075 with full keeper retention and KL 0.001085. Negative suppression was consistently unhelpful; the smaller positive scale avoids losses seen at 0.8._

## Phase-2 Branch: `frozen_sae`
**Agent Summary:** Retained inherited L6 gate and committed two downstream frozen-SAE dormant-neuron gates: L8 suppress 4790 gain 4 targeting tough/raising, and L9 suppress 10681 gain 8 targeting malformed fronted questions. Practice accuracy improved from branch-start 0.5437 to 0.5563 (original baseline 0.5188); mean margin 0.5027. Final committed WikiText-2 KL 0.000182, PPL 48.1945, keeper retention 1.0. L7 edits and L11 amplification failed to improve accuracy and were not committed. Reconstruction/NPI localizations lacked sufficiently grammar-specific labels; stopped at diminishing returns rather than force additional edits. Held-out generalization remains to be evaluated.
- **Edit #2 (L8, dormant_neuron_gate):** L8 dormant MLP gate read=[4790] write=-1*[4790] (gain=4.00) — _Selectively suppress the ability/decision-to-proceed feature implicated in tough-versus-raising errors. Target accuracy improves from 57.5% to 60%, with other subsets unchanged, KL 0.000098 and full keeper retention._
- **Edit #3 (L9, dormant_neuron_gate):** L9 dormant MLP gate read=[10681] write=-1*[10681] (gain=8.00) — _Question-selective L9 suppression improves the weakest subset from 40% to 45% and overall accuracy from 55% to 55.63%; mean margin rises by 0.1838. One tough/raising item regresses, but net accuracy improves with KL 0.000182 and full keeper retention._

## Phase-2 Branch: `patched_sae`
**Agent Summary:** Added two downstream edits to inherited L6 gate: L7 gated suppression of question latent 19164 (gain 4), then L10 attention amplification of adjective latent 8553 (alpha +0.4). Practice accuracy improved from fork-start 54.37% to 56.87% (original baseline 51.88%); left-branch subset 40% to 50%, tough-versus-raising 57.5% to 60%, principle A 52.5% to 50%, NPI unchanged 67.5%. Final WikiText-2 KL 0.001674 and keeper retention 0.988 satisfy guards. Explored projection and neuron gates on reconstruction-related pronoun features without reliable gains; rejected those and stopped at diminishing returns. Three total committed edits including inherited L6; no held-out results yet.
- **Edit #2 (L7, dormant_neuron_gate):** L7 dormant MLP gate read=[19164] write=-1*[19164] (gain=4.00) — _Suppress the causally relevant fronted 'What is' question feature using a selective layer-7 gate. Target accuracy rises from 40% to 50% and total accuracy to 56.25%, with KL 0.000815 and keeper retention 0.988._
- **Edit #3 (L10, rank1_projection):** L10.attn SAE proj #8553 (alpha=+0.400) — _Amplify the positive-adjective feature through layer-10 attention, improving tough-versus-raising accuracy from 57.5% to 60% without other subset regressions; overall accuracy 56.87%, KL 0.001674, keeper retention 0.988. MLP amplification caused collateral regressions and was not committed._

**Local Subspace SAE Patching Logs:**
- After Edit #1 (L6) -> Patched L7 SAE (0.1302% latents): MSE `925560.94` -> `671959.12`
- After Edit #1 (L6) -> Patched L8 SAE (0.1302% latents): MSE `873660.06` -> `401627.19`
- After Edit #1 (L6) -> Patched L9 SAE (0.1302% latents): MSE `772962.56` -> `242694.83`
- After Edit #1 (L6) -> Patched L10 SAE (0.1302% latents): MSE `608040.56` -> `309376.00`
- After Edit #1 (L6) -> Patched L11 SAE (0.1302% latents): MSE `1072174.62` -> `507924.00`
- After Edit #2 (L7) -> Patched L8 SAE (0.1302% latents): MSE `441093.28` -> `83682.07`
- After Edit #2 (L7) -> Patched L9 SAE (0.1302% latents): MSE `272735.53` -> `15491.71`
- After Edit #2 (L7) -> Patched L10 SAE (0.1302% latents): MSE `351166.06` -> `125488.25`
- After Edit #2 (L7) -> Patched L11 SAE (0.1302% latents): MSE `569882.56` -> `175177.72`
- After Edit #3 (L10) -> Patched L11 SAE (0.1302% latents): MSE `168643.02` -> `48178.27`
