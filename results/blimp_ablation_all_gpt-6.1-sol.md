# BLIMP Benchmark Findings (`gpt-6.1-sol`)

| Arm | Base Held-Out | Final Held-Out | Delta | Margin Δ | Keeper | WikiText-2 PPL | WikiText-2 KL | Edits | API Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_sae` | 0.4833 | 0.5083 | +0.0250 | +0.2056 | 98.3% | 48.60 | 0.00341 | 3 | 89,946 |
| `frozen_sae` | 0.4833 | 0.4958 | +0.0125 | +0.2805 | 100.0% | 48.19 | 0.00052 | 3 | 139,057 |
| `patched_sae` | 0.4833 | 0.5417 | +0.0584 | +0.6219 | 99.1% | 48.31 | 0.00235 | 3 | 151,457 |

## Arm: `no_sae`
**Agent Summary:** Committed three no-SAE rank-1 MLP projections, each alpha +0.6: L8 principle_A_reconstruction, L9 left_branch_island_echo_question, L10 principle_A_reconstruction. Negative projections and dormant neuron gates failed to improve practice and were not committed. Practice accuracy improved 0.5188 -> 0.5437, mean margin 0.2949 -> 0.4802. Final cumulative WikiText-2 KL 0.003411, keeper retention 0.988, PPL 48.6028 versus baseline 48.1774. Ending within the 20-turn budget for strictly held-out evaluation; practice improvements do not establish held-out gains.

- **Edit #1 (L8, rank1_projection):** L8.mlp raw diff_dir projection (alpha=+0.600) — _L8 MLP positive rank-1 edit improves mean grammatical margin by 0.064 with unchanged accuracy, full keeper retention, and KL 0.00019; suppression and gating were counterproductive._
- **Edit #2 (L9, rank1_projection):** L9.mlp raw diff_dir projection (alpha=+0.600) — _L9 MLP positive projection increases practice accuracy to 0.525 and mean margin to 0.4186, including an improvement on the targeted left-branch subset, with KL 0.000926 and keeper retention 0.988._
- **Edit #3 (L10, rank1_projection):** L10.mlp raw diff_dir projection (alpha=+0.600) — _L10 reconstruction-specific MLP projection improves targeted accuracy from 0.375 to 0.45 and overall accuracy to 0.5437 without losing accuracy on other subsets; cumulative KL 0.003411 and keeper retention 0.988 pass both guards._

## Arm: `frozen_sae`
**Agent Summary:** Committed three frozen-SAE weight edits sequentially at L7, L9, L10: question-feature suppression using a dormant neuron gate, pronoun/action attention-feature amplification, and downstream question/action attention suppression. Practice accuracy improved 0.5188 -> 0.5375; mean margin 0.2949 -> 0.5395; left-branch echo 0.35 -> 0.40 and Principle A 0.40 -> 0.425. Keeper retention 1.0, WikiText-2 KL 0.000518 and PPL 48.1938 versus baseline 48.1774. Both projection and neuron-gate operators tested; rejected candidates not committed. L11 trials regressed, so stopped at diminishing returns rather than spending remaining edit budget. No downstream SAE patches.

- **Edit #1 (L7, dormant_neuron_gate):** L7 dormant MLP gate read=[15444, 11434] write=-1*[15444] (gain=6.00) — _Context-gated suppression of question features improves left-branch echo accuracy from 35% to 40%, with full keeper retention and negligible KL; ungated projection did not improve accuracy._
- **Edit #2 (L9, rank1_projection):** L9.attn SAE proj #16914 (alpha=+0.800) — _Amplify the pronoun-and-action feature through L9 attention to improve Principle A reconstruction from 40% to 42.5%, preserving all keepers with cumulative KL 0.000157._
- **Edit #3 (L10, rank1_projection):** L10.attn SAE proj #23349 (alpha=-0.800) — _Downstream attention suppression of question/action feature increases mean grammatical margin by 0.0745 without accuracy loss, with full keeper retention and cumulative KL 0.000518._

## Arm: `patched_sae`
**Agent Summary:** Committed three sparse dormant-neuron gates at layers 7, 9, and 11 targeting question, negation, and likelihood features, respectively. Practice BLiMP accuracy improved from 0.5188 to 0.5563; left-branch echo 0.35→0.40, NPI 0.60→0.65, tough/raising 0.725→0.775; reconstruction unchanged at 0.40. Final WikiText-2 KL 0.002355, keeper retention 0.988, OOD keeper accuracy unchanged 0.875. Total 4,611 weight entries edited without gradient descent. Non-improving projection and gate candidates were not committed. Ending within turn budget for strictly held-out evaluation.

- **Edit #1 (L7, dormant_neuron_gate):** L7 dormant MLP gate read=[15444] write=-1*[15444] (gain=6.00) — _Sparse question-feature suppression improves left-branch echo discrimination by 5 points with negligible KL and full keeper retention._
- **Edit #2 (L9, dormant_neuron_gate):** L9 dormant MLP gate read=[22703] write=-1*[22703] (gain=6.00) — _Gated suppression of the separating negation feature improves NPI accuracy from 60% to 65%; aggregate accuracy rises with KL 0.000282 and keeper retention 0.988._
- **Edit #3 (L11, dormant_neuron_gate):** L11 dormant MLP gate read=[24468] write=-1*[24468] (gain=6.00) — _Context-gated suppression of the likelihood/probability feature improves tough-versus-raising accuracy by 5 points, with aggregate accuracy 55.63%, KL 0.002355 and keeper retention 0.988._

**Local Subspace SAE Patching Logs:**
- After Edit #1 (L7) -> Patched L8 SAE (0.1302% latents): MSE `877397.06` -> `379187.66`
- After Edit #1 (L7) -> Patched L9 SAE (0.1302% latents): MSE `779102.25` -> `266890.16`
- After Edit #1 (L7) -> Patched L10 SAE (0.1302% latents): MSE `626802.94` -> `323117.59`
- After Edit #1 (L7) -> Patched L11 SAE (0.1302% latents): MSE `1076687.12` -> `474246.12`
- After Edit #2 (L9) -> Patched L10 SAE (0.1302% latents): MSE `323429.00` -> `86481.90`
- After Edit #2 (L9) -> Patched L11 SAE (0.1302% latents): MSE `474529.53` -> `80482.64`
