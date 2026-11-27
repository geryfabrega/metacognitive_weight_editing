# TRUTHFULQA Benchmark Findings (`gpt-6.1-sol`, seed=42)

| Arm | Base Held-Out | Final Held-Out | Delta | Margin Δ | Keeper | WikiText-2 PPL | WikiText-2 KL | Edits | API Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_sae` | 0.4436 | 0.4478 | +0.0042 | +0.0649 | 100.0% | 48.18 | 0.00040 | 3 | 77,285 |
| `frozen_sae` | 0.4436 | 0.4392 | -0.0044 | +0.0494 | 98.0% | 48.65 | 0.00266 | 3 | 127,789 |
| `patched_sae` | 0.4436 | 0.4470 | +0.0034 | +0.1075 | 100.0% | 48.18 | 0.00009 | 1 | 119,808 |

## Arm: `no_sae`
**Agent Summary:** Committed three no-SAE edits: L6 MLP raw-direction projection alpha=-0.4, L8 dormant MLP neuron gate gain=3, and L9 attention projection alpha=-0.6. Practice MC2 improved from 0.4078 to 0.4128 (+0.0050), mean margin from -8.2798 to -8.1833; MC1 stayed 0.28. Committed-model WikiText-2 KL=0.000405, perplexity=48.1816 versus baseline 48.1774, keeper retention=1.0. L10 gate/projection and L11 gate/attention projections did not improve MC2 and were not committed. Halving the KL-rejected L11 projection restored KL compliance but still reduced MC2. Stopped with three unused edit slots due to diminishing downstream returns; proceed to strictly held-out evaluation.

- **Edit #1 (L6, rank1_projection):** L6.mlp raw diff_dir projection (alpha=-0.400) — _Suppress the practice-derived L6 failure direction with a low-distortion MLP rank-1 edit: MC2 rises by 0.0015, margin improves, KL is 0.000178, and all keepers are retained._
- **Edit #2 (L8, dormant_neuron_gate):** L8 dormant MLP gate raw diff_dir (gain=3.00) — _Complement L6 projection with a sparse L8 dormant-neuron gate: failure-span firing 58.3% versus good-span 4.17% and background 1.88%; MC2 improves 0.0025 with full keeper retention and cumulative KL 0.000198._
- **Edit #3 (L9, rank1_projection):** L9.attn raw diff_dir projection (alpha=-0.600) — _A downstream attention projection adds a modest MC2 and margin improvement without losing any keepers; cumulative KL remains 0.000405. Complements earlier MLP edits._

## Arm: `frozen_sae`
**Agent Summary:** Committed three complementary frozen-SAE edits, without gradients: L9 gated suppression of strong-agreement feature 7836 (gain 6), L10 MLP rank-1 suppression of social/historical identity feature 16977 (alpha -0.6), and L11 gated suppression of advice/instruction feature 15707 (gain 6). Practice MC2 rose 0.4078 -> 0.4113 (+0.0035), MC1 0.28 -> 0.29, and mean margin -8.2798 -> -8.2138. Final cumulative WikiText-2 KL 0.00266, keeper retention 1.0, PPL 48.6478 versus baseline 48.1774. Tested both operator types and rejected non-improving candidates, including the final attention edit. Ending at diminishing returns with three edits unused to avoid practice overfitting. Gains are small and held-out generalization is unproven; late gate mechanism is uncertain given reported zero aggregate span fire rates. Downstream SAEs remained frozen.

- **Edit #1 (L9, dormant_neuron_gate):** L9 dormant MLP gate read=[7836] write=-1*[7836] (gain=6.00) — _Conditionally suppress the layer-9 strong-agreement/exclamatory feature on high-activation spans. This yields a modest MC2 and margin gain with full keeper retention and negligible background KL; unlike broad suppression, the gate is selective._
- **Edit #2 (L10, rank1_projection):** L10.mlp SAE proj #16977 (alpha=-0.600) — _Dampen the social/historical identity and societal-norm feature, which is more active on wrong answers, to reduce reliance on stereotyped associations. The sandbox shows complementary MC2 and margin improvement, KL 0.002536 and full keeper retention._
- **Edit #3 (L11, dormant_neuron_gate):** L11 dormant MLP gate read=[15707] write=-1*[15707] (gain=6.00) — _Add a late-layer gated suppression of advice/instruction phrasing, a plausible source of unsupported prescriptive answers. The measured cumulative edit improves MC2 by 0.0022 and MC1 by 0.01 while retaining all keepers and KL 0.00266. Reported zero span-level fire rates mean the precise gating mechanism remains uncertain._

## Arm: `patched_sae`
**Agent Summary:** Committed one L7 dormant-neuron gate suppressing conversational-pattern latent 14413 (gain 4), with automatic downstream SAE patches. Practice MC2 increased from 0.4078 to 0.4113 and mean margin improved from -8.2798 to -8.2099; MC1 decreased from 0.28 to 0.27. WikiText-2 KL remained 0.000092 and keeper retention 0.9286, both within guards. Re-localized downstream layers and tested both projections and gates; further candidates produced only negligible or negative MC2 changes, so left them uncommitted and stopped at diminishing returns. Held-out generalization remains to be evaluated.

- **Edit #1 (L7, dormant_neuron_gate):** L7 dormant MLP gate read=[14413] write=-1*[14413] (gain=4.00) — _Gate suppression of a high-AUC conversational-pattern feature reduces answer-style bias on failing continuations; improves MC2 and margin with negligible KL and keeper retention above 0.9._

**Local Subspace SAE Patching Logs:**
- After Edit #1 (L7) -> Patched L8 SAE (0.1302% latents): MSE `1.46` -> `0.96`
- After Edit #1 (L7) -> Patched L9 SAE (0.1302% latents): MSE `2.05` -> `1.01`
- After Edit #1 (L7) -> Patched L10 SAE (0.1302% latents): MSE `2.93` -> `1.66`
- After Edit #1 (L7) -> Patched L11 SAE (0.1302% latents): MSE `41.07` -> `12.58`
