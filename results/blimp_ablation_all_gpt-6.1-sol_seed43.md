# BLIMP Benchmark Findings (`gpt-6.1-sol`, seed=43)

| Arm | Base Held-Out | Final Held-Out | Delta | Margin Δ | Keeper | WikiText-2 PPL | WikiText-2 KL | Edits | API Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_sae` | 0.4950 | 0.5010 | +0.0060 | +0.1541 | 98.6% | 48.28 | 0.00106 | 2 | 97,885 |
| `frozen_sae` | 0.4950 | 0.5140 | +0.0190 | +0.2537 | 99.6% | 48.17 | 0.00021 | 2 | 143,475 |
| `patched_sae` | 0.4950 | 0.5160 | +0.0210 | +0.1111 | 100.0% | 48.39 | 0.00183 | 2 | 159,226 |

## Arm: `no_sae`
**Agent Summary:** Committed two no-SAE rank-1 MLP projections, L9 and L11, each alpha=-0.8. Tested earlier layers and both projection/neuron-gate modes; rejected candidates lacking gains. Practice accuracy improved 0.4875 to 0.5000 (two additional correct pairs); mean margin -0.3370 to -0.1728. Left-branch accuracy 0.275 to 0.300; principle-A 0.375 to 0.400; other subsets unchanged. Keeper retention 1.0, OOD keeper accuracy unchanged at 0.875, WikiText-2 KL 0.00106 and PPL 48.279 versus baseline 48.1774. Stopping at turn budget with modest practice gains; held-out generalization remains to be evaluated.

- **Edit #1 (L9, rank1_projection):** L9.mlp raw diff_dir projection (alpha=-0.800) — _Layer-9 MLP rank-1 suppression improves practice accuracy from 48.75% to 49.38% and mean margin by 0.0189, with full keeper retention and KL 0.000191. The gain is small and held-out generalization remains unproven._
- **Edit #2 (L11, rank1_projection):** L11.mlp raw diff_dir projection (alpha=-0.800) — _Complementary L11 MLP projection raises overall practice accuracy to 50.0% and mean margin by 0.1453 relative to the committed L9 model. Cumulative KL remains 0.00106 with 100% keeper retention; all four subset accuracies are preserved or improved._

## Arm: `frozen_sae`
**Agent Summary:** Committed two low-KL dormant-neuron gates at layers 7 and 10 suppressing question-fronting features. Practice BLiMP accuracy improved from 0.4875 to 0.5125; left-branch island accuracy rose from 0.275 to 0.400. Final committed WikiText-2 KL 0.000213, keeper retention 0.9872, PPL 48.1732 versus baseline 48.1774. Tested both rank-1 projections and gates; reconstruction and NPI candidates did not yield clean net gains and were not committed. Stopped at diminishing returns; downstream SAEs stayed frozen throughout.

- **Edit #1 (L7, dormant_neuron_gate):** L7 dormant MLP gate read=[15444, 11434] write=-1*[15444] (gain=6.00) — _Gate suppression of question-fronting features improves left-branch island accuracy by 5 points while preserving keepers and negligible background KL._
- **Edit #2 (L10, dormant_neuron_gate):** L10 dormant MLP gate read=[23349, 19832] write=-1*[23349] (gain=8.00) — _Downstream gated question-feature suppression adds two net correct pairs, raising left-branch accuracy to 40%, with KL 0.000213 and 98.72% keeper retention._

## Arm: `patched_sae`
**Agent Summary:** Committed two gradient-free rank-1 MLP edits: L7 SAE #11434 alpha=-1.5 and L11 SAE #9209 alpha=-0.8. Practice BLiMP improved from 78/160 (48.75%) to 81/160 (50.625%); principle_A_reconstruction rose 37.5% to 45%, other subsets unchanged. Mean margin improved -0.337 to -0.2425. Final WikiText-2 KL 0.001829 <=0.05, keeper retention 1.0 >=0.9, PPL 48.3946 versus 48.1774 baseline; OOD keeper accuracy 87.5% to 88.75%. Tested projection and dormant-neuron gates; unhelpful candidates were not committed. Downstream SAEs patched after first edit and re-localized. Stopping at turn budget with remaining edit capacity rather than forcing noisy edits; held-out generalization awaits final evaluation.

- **Edit #1 (L7, rank1_projection):** L7.mlp SAE proj #11434 (alpha=-1.500) — _Suppress an overactive initial-question feature at layer 7: improves overall practice accuracy and mean grammatical margin with KL 0.000483 and full keeper retention._
- **Edit #2 (L11, rank1_projection):** L11.mlp SAE proj #9209 (alpha=-0.800) — _Suppress layer-11 descriptive-attribute feature: sandbox improves aggregate practice accuracy by two items, improves mean margin and OOD keeper accuracy, while retaining all keepers and KL 0.001829. Gain occurs in reconstruction rather than targeted tough/raising subset; held-out benefit remains unverified._

**Local Subspace SAE Patching Logs:**
- After Edit #1 (L7) -> Patched L8 SAE (0.1302% latents): MSE `989524.69` -> `336265.72`
- After Edit #1 (L7) -> Patched L9 SAE (0.1302% latents): MSE `865780.81` -> `228974.23`
- After Edit #1 (L7) -> Patched L10 SAE (0.1302% latents): MSE `683220.44` -> `432283.59`
- After Edit #1 (L7) -> Patched L11 SAE (0.1302% latents): MSE `1190740.88` -> `415871.06`
