# BLIMP Benchmark Findings (`gpt-6.1-sol`, seed=44)

| Arm | Base Held-Out | Final Held-Out | Delta | Margin Δ | Keeper | WikiText-2 PPL | WikiText-2 KL | Edits | API Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_sae` | 0.5080 | 0.5140 | +0.0060 | +0.0268 | 98.8% | 48.32 | 0.00229 | 3 | 96,377 |
| `frozen_sae` | 0.5080 | 0.5080 | +0.0000 | -0.0519 | 99.0% | 48.16 | 0.00008 | 1 | 137,890 |
| `patched_sae` | 0.5080 | 0.6210 | +0.1130 | +1.8681 | 98.4% | 48.25 | 0.00086 | 2 | 156,421 |

## Arm: `no_sae`
**Agent Summary:** Committed three no-SAE rank-1 projections in downstream order: L8 MLP alpha=-0.8 (left-branch direction, last pooling), L10 attention alpha=-0.8 (NPI direction), and L11 MLP alpha=-0.6 (tough/raising direction). Tested both projections and dormant-neuron gates; rejected nonprogressing or adverse candidates. Practice accuracy rose from 0.4125 to 0.4188 (one additional correct pair), mean margin improved from -0.4989 to -0.4450. Keeper retention remains 1.0 and WikiText-2 KL is 0.002285, below 0.05; perplexity 48.3201 versus baseline 48.1774. Gains are modest and held-out generalization is not yet established. Ending at turn budget rather than using remaining edit slots.

- **Edit #1 (L8, rank1_projection):** L8.mlp raw diff_dir projection (alpha=-0.800) — _Layer-8 MLP projection improves mean grammatical-pair margin by 0.0227 with unchanged accuracy and keeper retention, at only 0.000065 WikiText-2 KL._
- **Edit #2 (L10, rank1_projection):** L10.attn raw diff_dir projection (alpha=-0.800) — _Complementary layer-10 attention projection increases practice accuracy to 0.4188 and improves mean margin, with full keeper retention and cumulative KL 0.000526._
- **Edit #3 (L11, rank1_projection):** L11.mlp raw diff_dir projection (alpha=-0.600) — _Layer-11 MLP projection adds a 0.0251 mean-margin improvement without losing practice accuracy or keepers; cumulative WikiText-2 KL remains 0.002285, well below 0.05._

## Arm: `frozen_sae`
**Agent Summary:** Committed one selective layer-7 dormant-neuron gate suppressing question-start feature 11434 when features 11434/10767 activate. Practice BLiMP rose from 0.4125 to 0.4188; left-branch accuracy rose 0.25 to 0.275. WikiText-2 KL 0.000077 and keeper retention 1.0. Tested both projection and gated operators across layers 7–11, targeting individual subsets; subsequent candidates provided no clear accuracy gain and were not committed. Downstream SAEs remained frozen. Stopping at diminishing returns rather than stacking unhelpful edits; held-out generalization remains to be evaluated.

- **Edit #1 (L7, dormant_neuron_gate):** L7 dormant MLP gate read=[11434, 10767] write=-1*[11434] (gain=6.00) — _Question-feature gated suppression improves left-branch subset accuracy with unchanged keeper retention and negligible WikiText KL; use selective gate rather than global suppression._

## Arm: `patched_sae`
**Agent Summary:** Committed two dormant-neuron gates, L6 latent 16129 suppression and L11 latent 22295 suppression, both gain 8 and margin 0, targeting left-branch echo questions. Committed practice accuracy increased from 0.4125 to 0.5375; echo-question accuracy from 0.25 to 0.75; mean margin from -0.4989 to 1.4399. Cumulative WikiText-2 KL 0.000856, keeper retention 1.0, PPL 48.2511 versus baseline 48.1774. Tested projection and gate candidates on Principle A and NPI, but did not commit weak or regressive candidates. Last staged Principle A gate is intentionally uncommitted due to tiny net accuracy gain and substantial margin regression. Proceed to strictly held-out evaluation using only the two committed edits.

- **Edit #1 (L6, dormant_neuron_gate):** L6 dormant MLP gate read=[16129] write=-1*[16129] (gain=8.00) — _Conditionally suppress the question-associated feature on left-branch echo-question failures: subset accuracy rises 25% to 30%, mean margin improves 0.0789, KL is 0.000819 and keeper retention remains 1.0._
- **Edit #2 (L11, dormant_neuron_gate):** L11 dormant MLP gate read=[22295] write=-1*[22295] (gain=8.00) — _A highly separating question feature (AUC 0.9949) supports selective gated suppression: echo-question accuracy improves 30% to 75%, total accuracy 42.5% to 53.75%, with all other subsets unchanged, keeper retention 1.0 and cumulative KL 0.000856._

**Local Subspace SAE Patching Logs:**
- After Edit #1 (L6) -> Patched L7 SAE (0.1302% latents): MSE `860268.38` -> `521190.81`
- After Edit #1 (L6) -> Patched L8 SAE (0.1302% latents): MSE `817488.44` -> `281265.47`
- After Edit #1 (L6) -> Patched L9 SAE (0.1302% latents): MSE `722435.31` -> `198012.67`
- After Edit #1 (L6) -> Patched L10 SAE (0.1302% latents): MSE `577278.38` -> `287524.72`
- After Edit #1 (L6) -> Patched L11 SAE (0.1302% latents): MSE `1020126.12` -> `365140.88`
