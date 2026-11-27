# BLIMP Benchmark Findings (`gpt-6.1-sol`, seed=42)

| Arm | Base Held-Out | Final Held-Out | Delta | Margin Δ | Keeper | WikiText-2 PPL | WikiText-2 KL | Edits | API Tokens |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `no_sae` | 0.4910 | 0.5050 | +0.0140 | +0.0843 | 99.4% | 48.28 | 0.00058 | 1 | 80,477 |
| `frozen_sae` | 0.4910 | 0.5170 | +0.0260 | +0.3773 | 99.4% | 48.23 | 0.00067 | 3 | 149,894 |
| `patched_sae` | 0.4910 | 0.5100 | +0.0190 | +0.2878 | 99.8% | 48.19 | 0.00016 | 2 | 122,918 |

## Arm: `no_sae`
**Agent Summary:** Committed one no-SAE rank-1 projection: L10 MLP, NPI contrastive direction, alpha=+0.5. Practice BLiMP accuracy improved 0.5188 -> 0.5250, NPI 0.600 -> 0.625, mean margin 0.2949 -> 0.3671; other subset accuracies unchanged. Keeper retention 1.0, WikiText-2 KL 0.000578, PPL 48.2777 versus baseline 48.1774. Tested projections and dormant neuron gates sequentially across L6/L8/L9/L10; rejected all non-improving candidates. Stopping at diminishing returns rather than stacking unsupported edits. Held-out improvement remains unverified.

- **Edit #1 (L10, rank1_projection):** L10.mlp raw diff_dir projection (alpha=+0.500) — _Layer-10 NPI-targeted MLP projection improves NPI accuracy from 60% to 62.5% and overall mean margin by 0.0722, without other subset accuracy losses; keeper retention is 1.0 and WikiText-2 KL is 0.000578. Reversed projection sign works empirically, unlike suppression._

## Arm: `frozen_sae`
**Agent Summary:** Committed 3 complementary edits, keeping all SAEs frozen: L7 question-feature suppression gate, L9 attention amplification of pronoun/action latent 16914, and L11 probability-feature suppression gate for tough-versus-raising. Practice accuracy improved 83/160 to 87/160 (0.5188 to 0.5437); mean margin 0.2949 to 0.5573. Subset accuracies: left-branch 0.35→0.40, Principle A 0.40→0.425, NPI unchanged 0.60, tough/raising 0.725→0.75. Final keeper retention 1.0, OOD keeper accuracy unchanged 0.875, WikiText-2 KL 0.000669 (<0.05), perplexity 48.2254 versus 48.1774 baseline. Rejected non-improving projection/gate candidates, including NPI edits; stopped at diminishing returns within turn budget. Proceed to strictly held-out evaluation; practice gains are not evidence yet of held-out generalization.

- **Edit #1 (L7, dormant_neuron_gate):** L7 dormant MLP gate read=[15444, 11434] write=-1*[15444] (gain=6.00) — _A question-feature-gated suppression improves left-branch echo-question accuracy from 35% to 40%, with unchanged keeper retention and negligible WikiText-2 KL; the ungated projection did not improve accuracy._
- **Edit #2 (L9, rank1_projection):** L9.attn SAE proj #16914 (alpha=+0.700) — _Amplifying the personal-pronoun/action feature in L9 attention improves Principle A reconstruction from 40% to 42.5%, consistent with a binding-related hypothesis, without reducing other subset accuracies or keeper retention; cumulative KL is 0.000131._
- **Edit #3 (L11, dormant_neuron_gate):** L11 dormant MLP gate read=[24468, 21716] write=-1*[24468] (gain=4.00) — _Gated suppression of likelihood/probability features addresses the raising-versus-tough distinction, improving that subset from 72.5% to 75% and mean margin by 0.0934, with no other subset regression, 100% keeper retention, and cumulative KL 0.000669._

## Arm: `patched_sae`
**Agent Summary:** Committed two low-collateral dormant-neuron gates, at L7 for left-branch island echo questions and L10 for tough/raising action features, with downstream SAE patching after each commit. Tested both rank-1 projections and gates; rejected accuracy-degrading candidates for reconstruction and NPI subsets. Practice accuracy rose from 0.5188 to 0.5312 (83 to 85 of 160), mean margin from 0.2949 to 0.5062. Final WikiText-2 KL 0.000159, keeper retention 1.0, perplexity 48.1871 versus 48.1774 baseline. Stopping at diminishing returns rather than forcing additional edits; held-out gains remain to be evaluated.

- **Edit #1 (L7, dormant_neuron_gate):** L7 dormant MLP gate read=[15444, 11434] write=-1*[15444] (gain=6.00) — _Gated suppression of excessive fronted-question activation improves left-branch island discrimination by 5 percentage points, with negligible background KL and full keeper retention._
- **Edit #2 (L10, dormant_neuron_gate):** L10 dormant MLP gate read=[22214, 1991] write=-1*[22214] (gain=10.00) — _Selective action-feature suppression improves mean grammatical-pair margin by 0.0502 without reducing any subset accuracy; full keeper retention and cumulative KL 0.000159._

**Local Subspace SAE Patching Logs:**
- After Edit #1 (L7) -> Patched L8 SAE (0.1302% latents): MSE `881089.56` -> `381226.88`
- After Edit #1 (L7) -> Patched L9 SAE (0.1302% latents): MSE `783188.06` -> `284896.16`
- After Edit #1 (L7) -> Patched L10 SAE (0.1302% latents): MSE `631446.88` -> `343564.50`
- After Edit #1 (L7) -> Patched L11 SAE (0.1302% latents): MSE `1084996.12` -> `469485.53`
- After Edit #2 (L10) -> Patched L11 SAE (0.1302% latents): MSE `465547.69` -> `58941.32`
