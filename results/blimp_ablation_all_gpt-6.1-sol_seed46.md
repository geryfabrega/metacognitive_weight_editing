# BLIMP Benchmark Findings (`gpt-6.1-sol`, seed=46)

| Arm | Base Held-Out | Final Held-Out | Delta | Margin Δ | Keeper | WikiText-2 PPL | WikiText-2 KL | Edits | API Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_sae` | 0.4807 | 0.4922 | +0.0115 | +0.1210 | 99.6% | 48.21 | 0.00076 | 2 | 91,489 |
| `frozen_sae` | 0.4807 | 0.5005 | +0.0198 | +0.2695 | 98.9% | 48.23 | 0.00067 | 4 | 141,833 |
| `patched_sae` | 0.4807 | 0.5174 | +0.0367 | +0.4672 | 99.6% | 49.59 | 0.03221 | 3 | 122,334 |

## Arm: `no_sae`
**Agent Summary:** Committed two raw rank-1 MLP projections without SAEs or gradient training: L8 alpha=+0.6 targeting principle_A_reconstruction and L9 alpha=+0.6 targeting left_branch_island_echo_question. Practice BLiMP accuracy improved 0.5188 -> 0.5375 (three additional correct pairs), mean margin -0.0171 -> 0.1149, keeper retention remains 1.0, WikiText-2 KL 0.000765 and PPL 48.2122 versus baseline 48.1774. Tested both projection and dormant neuron gates; suppression/gates were unhelpful. Final L10 trials offered no aggregate accuracy gain or regressed and remain uncommitted. Stopping at diminishing returns rather than stacking noisy edits; proceed to held-out evaluation using only the two committed edits.

- **Edit #1 (L8, rank1_projection):** L8.mlp raw diff_dir projection (alpha=+0.600) — _L8 MLP projection with positive scale improves principle-A and tough/raising practice accuracy, raises aggregate mean margin by 0.0668, and preserves all keepers with KL 0.000267. Suppression and neuron gates did not yield comparable gains._
- **Edit #2 (L9, rank1_projection):** L9.mlp raw diff_dir projection (alpha=+0.600) — _Complementary L9 MLP edit improves the targeted echo-question subset from 30% to 32.5%, increases mean margin by 0.0652 without subset regressions, and maintains full keeper retention at cumulative KL 0.000765._

## Arm: `frozen_sae`
**Agent Summary:** Committed four frozen-SAE weight edits: L7 question-feature suppression gate, L9 personal-pronoun amplification projection, L10 residual question/action suppression gate, and L11 pronoun amplification projection. Practice BLiMP improved from 0.5188 to 0.5500; echo questions 0.30 to 0.40, reconstruction 0.525 to 0.55, other subsets unchanged from baseline. Mean margin improved -0.0171 to 0.3798. Cumulative WikiText-2 KL 0.000671, keeper retention 1.0, perplexity 48.2327 versus baseline 48.1774. Final tough-feature amplification candidate was rejected and not committed. Stopping with four clean edits rather than forcing remaining budget; held-out generalization remains to be evaluated.

- **Edit #1 (L7, dormant_neuron_gate):** L7 dormant MLP gate read=[15444, 11434] write=-1*[15444] (gain=6.00) — _Question-feature gated suppression improves left-branch echo-question accuracy by 5 points with negligible KL and full keeper retention._
- **Edit #2 (L9, rank1_projection):** L9.mlp SAE proj #16914 (alpha=+0.800) — _Amplifying a personal-pronoun/action feature provides a small complementary tough-versus-raising gain without regressions, full keeper retention and KL 0.000351._
- **Edit #3 (L10, dormant_neuron_gate):** L10 dormant MLP gate read=[23349, 14633] write=-1*[23349] (gain=10.00) — _Downstream question/action gate raises echo-question accuracy to 40%, improving overall accuracy and margins with KL 0.000427 and no keeper loss._
- **Edit #4 (L11, rank1_projection):** L11.mlp SAE proj #19203 (alpha=+0.800) — _Amplifying the downstream personal-pronoun feature improves reconstruction accuracy by 2.5 points without subset regressions or keeper loss; cumulative KL remains 0.000671._

## Arm: `patched_sae`
**Agent Summary:** Committed three gradient-free edits: L7 sparse question-feature suppression gate (gain 6), L9 MLP pronoun/action-feature amplification (alpha +0.6), and L11 attention prepositional-phrase-feature suppression (alpha -0.5). Tried both projection and dormant-neuron operators; discarded candidates without accuracy gains. Practice accuracy improved 0.5188 to 0.575, echo-question subset 0.30 to 0.50, tough-vs-raising 0.575 to 0.60, other subsets unchanged. Final keeper retention 1.0, WikiText-2 KL 0.032207 <= 0.05, perplexity 49.5897 versus baseline 48.1774. Stopping after three clean edits to preserve fluency headroom and avoid chasing practice noise; held-out performance remains to be evaluated.

- **Edit #1 (L7, dormant_neuron_gate):** L7 dormant MLP gate read=[15444, 11434] write=-1*[15444] (gain=6.00) — _A sparse layer-7 question-feature suppression gate improves the weakest echo-question subset from 30% to 35%, with full keeper retention and negligible WikiText KL._
- **Edit #2 (L9, rank1_projection):** L9.mlp SAE proj #16914 (alpha=+0.600) — _Amplifying a pronoun/action feature at layer 9 yields a small complementary gain on tough-versus-raising pairs without reducing any subset accuracy or keeper retention; cumulative KL remains very low._
- **Edit #3 (L11, rank1_projection):** L11.attn SAE proj #6936 (alpha=-0.500) — _Layer-11 attention suppression of the prepositional-phrase feature yields six additional correct echo-question pairs with no subset losses and full keeper retention. Cumulative WikiText KL 0.032207 remains below 0.05._

**Local Subspace SAE Patching Logs:**
- After Edit #1 (L7) -> Patched L8 SAE (0.1302% latents): MSE `896724.81` -> `414755.53`
- After Edit #1 (L7) -> Patched L9 SAE (0.1302% latents): MSE `795124.81` -> `271142.44`
- After Edit #1 (L7) -> Patched L10 SAE (0.1302% latents): MSE `633806.69` -> `304803.03`
- After Edit #1 (L7) -> Patched L11 SAE (0.1302% latents): MSE `1106368.75` -> `469506.66`
- After Edit #2 (L9) -> Patched L10 SAE (0.1302% latents): MSE `309906.91` -> `82370.10`
- After Edit #2 (L9) -> Patched L11 SAE (0.1302% latents): MSE `475010.53` -> `78294.70`
