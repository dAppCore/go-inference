# HIP 12B r15 logit-spread receipt

Date: 2026-07-13

Prompt: `why the sky is blue`. Statistics cover the full 262144-token
vocabulary; the device count is 262120 because its projection kernel suppresses
24 control tokens before the receipt. Device values are raw LM-head scores;
host and oracle values are after the one owned `30*tanh(x/30)` softcap.

## Step-0 tables

| model | arm | stage | max | mean | stddev | top tokens (`id:logit`) |
|---|---|---|---:|---:|---:|---|
| 12B before fix | host | softcapped | 5.937215 | -19.543623 | 2.783380 | `107:5.9372, 236761:3.8129, 236753:3.4744, 1390:1.9524, 138:0.9122` |
| 12B after fix | host | softcapped | 17.205996 | -6.110431 | 3.624836 | `107:17.2060, 236761:12.5471, 100:11.6273, 607:11.3429, 532:11.2762` |
| 12B after fix | device top-k | pre-softcap | 19.583164 | -6.318540 | 3.990974 | `107:19.5832, 236761:13.3665, 607:11.9351, 532:11.8573, 236764:11.3557` |
| 12B | chained oracle | softcapped | 17.205996 | -6.110431 | 3.624836 | `107:17.2060, 236761:12.5471, 100:11.6273, 607:11.3429, 532:11.2762` |
| 12B | fused greedy | pre-softcap | not read back | not read back | not read back | fused projection consumes the same final-normalised row and owns softcap internally; its batched prompt kernel does not expose the full vector |
| E2B | host | softcapped | 18.828835 | -4.365543 | 2.654442 | `236761:18.8288, 236764:17.6654, 1547:16.7279, 580:15.5725, 2779:15.0765` |
| E2B | device top-k | pre-softcap | 22.124775 | -4.430504 | 2.730064 | `236761:22.1248, 236764:20.2769, 1547:18.8802, 580:17.2526, 2779:16.5814` |
| E2B | chained oracle | softcapped | 17.491064 | -4.991747 | 2.466730 | `236761:17.4911, 236764:16.2634, 1547:15.5556, 1781:14.2342, 1390:13.8280` |

The 12B device row maps exactly to the host/oracle row under its single owned
softcap. An armed actual-logit oracle also compared every packed-device draw
with `hipGemma4Q4HostSampleResult`; all six recorded tokens matched.

## Step-5 fingerprint

| model/trajectory | arm | max | mean | stddev |
|---|---|---:|---:|---:|
| 12B pre-fix sampled | host softcapped | 22.153374 | -3.205703 | 4.261507 |
| 12B post-fix sampled | host softcapped | 10.731044 | -9.695870 | 3.142602 |
| 12B oracle greedy trajectory | oracle softcapped | 19.975704 | -5.049725 | 3.667173 |
| E2B sampled | host softcapped | 15.170243 | -9.782986 | 3.896669 |
| E2B oracle greedy trajectory | oracle softcapped | 14.589339 | -12.327758 | 3.085456 |

Step 5 is trajectory-dependent, so sampled and greedy-oracle rows are not a
pairwise equality test. Step 0 is the fixed-prompt, same-position fingerprint.

## Root cause and ownership fix

Sampling was incorrectly part of transformer-route selection:
`useBatchedPrefill := ... && !hostSampling` forced both sampled arms through
the token-at-a-time prompt producer, while coherent greedy and the chained
oracle used batched prefill. The same split continued during decode.

Compatible prompts and batch-one decode now use the batched retained-state
producer regardless of sampling. Host or packed-device sampling owns only the
final-normalised row returned by that producer. This is a producer ownership
fix, not a clamp, top-k substitution, or temperature mitigation.

