# HIP 12B r16 explicit-topK "fork" — the third fork does not exist

Date: 2026-07-13

## Verdict

The hypothesised third producer fork on the explicit top-k arm **is not
present**. The r15 fix (`09168bd`/`d6d3e92`) already routes explicit top-k
decode through the SAME batched retained-state producer that greedy and default
sampling use — it dropped `!deviceTopKSampling` from `useBatchedDecode`
(`hip_tiny_model.go:1184`), so nothing selects a different logits/hidden
producer when `TopK` is set. The residual "explicit top-k incoherence" recorded
by r14/r15 was a **reproduction-harness artifact**, not a code defect:

- The coherent baselines were the chat-framed `generate` CLI path
  (`model.Chat`, thinking off). The "incoherent top-k" probes drove the arm
  through the RAW completion path (`model.Generate` on a bare prompt, thinking
  left on).
- On the 12B the raw completion path flattens decode for **every** arm,
  including greedy (this is the same r10/r11 "thought thought thought" collapse
  on the raw prompt). Raw-path incoherence is therefore not a verdict on the
  sampler.
- Once the turn is chat-framed with thinking off, greedy, default and explicit
  top-k are all coherent, and the two sampled arms produce byte-identical
  step-0 logits.

No decode-routing change was made. Patching outputs, clamping, or re-routing
top-k would be mitigating a bug that is not there.

## Reproduce first — where the arms diverge (they do not)

All runs: box gfx1101, worktree HSACO (`make hip-amd AMD_HIP_ARCH=gfx1101`),
model `/tmp/models/gemma-4-12B-it-4bit-clean`, prompt `why is the sky blue`.

### Matched-position producer parity (step 0, deterministic prefill)

Spread instrument armed, both arms chat-framed. Step 0 is the same framed
prompt at the same position, so it is a direct producer comparison. The
`host-full-vocab` softcapped sampler input at step 0:

| arm | count | max | top tokens (`id:logit`) |
|---|---:|---:|---|
| default (host, TopK=0) | 262144 | 19.2617779 | `818:19.2617779, 669:9.90223598, 2021:9.28517914, 82842:8.8560009, 1018:6.34063435, 17596:5.95770502, 109642:5.83426332` |
| explicit top-k=64/top-p=0.95 | 262120 | 19.2617779 | `818:19.2617779, 669:9.90223598, 2021:9.28517818, 82842:8.8560009, 1018:6.34063435, 17596:5.95770502, 109642:5.83426332` |

Identical to ~7 significant figures; the ~1e-6 differences are float readback
noise between the fused-projection and standalone-projection reads. The device
top-k arm's own pre-softcap row (`818:22.85, 669:10.29, 2021:9.60, …`) maps onto
the softcapped row under the single owned `30*tanh(x/30)` softcap. The 24-token
count gap is projection-time suppression of control tokens (device arm) versus
sample-time suppression (host arm) — same tokens, different stage, all outside
the top candidates. **Same producer.**

### Framed vs raw A/B on the SAME top-k arm (48 tokens)

```text
FRAMED   arm=device  "The sky is blue because of a phenomenon called **Rayleigh
                      scattering**. To understand this, you have to look at how
                      sunlight interacts with the Earth's atmosphere. Here is
                      the step-by-step breakdown: ### 1 …"        (coherent)

RAW      arm=device  "\n1YV\nng9\nV11\nng1\nng2\nng1\nng1\nng1 …"  (degenerate)
```

The only variable changed is chat-framing. The sampler, producer, kernels,
draws and HSACO are identical. The raw path also degenerates under greedy, so
the artifact is not top-k specific.

### Robustness (chat-framed, temperature 1, 64 tokens, four runs each)

| arm | coherent runs | note |
|---|---:|---|
| explicit top-k=64/top-p=0.95 | **4 / 4** | every run a clean Rayleigh explanation |
| default full-vocab (TopK=0) | 2 / 4 | runs 2 and 3 collapse mid-answer (`…Earth'd movment_1.0.1.1.1`, `…the rest of the list_0.1.1.1`) |

This inverts the campaign premise. Full-vocab temperature-1 sampling on the
flattened 12B is the UNSTABLE arm — its long tail carries enough mass to draw
garbage. `r15`'s "default coherent at 20.8 tok/s" was an unseeded lucky draw;
at this base the CLI default (`-temp 1`) is also frequently incoherent
(`…Ray1100: a.R.L.E.Y. 1.2.0.0.0…`). top-k=64/top-p=0.95 truncates that chaotic
tail, so it is the MORE robust sampler — closer to greedy, never the problem.

### The packed device sampler still matches the host reference

The r12/r15 armed oracle remains green: every packed-device draw equals
`hipGemma4Q4HostSampleResult` on identical actual logits and draws
(`HIP_DEVICE_SAMPLE_ORACLE draw=… device=N host=N`, all matched). The sampler
math was never the issue.

## Root cause

The diagnostic, not the engine. r14/r15 verified explicit top-k through
`model.Generate` on a bare prompt with the thinking channel left enabled;
`generate`'s coherent runs use `model.Chat` (chat template) with thinking off.
Comparing a framed default against a raw top-k made a harness difference look
like a sampler fork. The engine already gives explicit top-k the shared
producer.

## Fix (the diagnostic)

`hip_gemma4_q4_oracle_test.go` — `TestHIPGemma4Q4LogitSpreadProbe` now drives
the coherent path by default: `model.Chat` with `WithEnableThinking(&false)`,
matching the CLI. `GO_ROCM_SPREAD_RAW=1` restores the old raw path for the A/B
above, but a coherence verdict must come from the framed run. No production
decode path changed.

## Latent note (not fixed — does not manifest, out of scope)

`attentionWorkspaceNeeded` (`hip_gemma4_q4_engine_config.go:94`) turns the
attention workspace on when top-k is requested, because the packed top-k reduce
needs the workspace scratch. With the default `ChunkedAttention: true` both arms
already hold the workspace, so this couples nothing today. If chunked attention
were ever disabled, the top-k arm would take the workspace attention path while
the default arm would not — a genuine top-k-keyed producer divergence risk. It
is latent at the shipped config and a fix touches the attention/kernel layer
(out of this task's scope); recorded here so a future round does not rediscover
it as a live fork.

## Verification

```sh
# corrected probe — coherent every arm, chat-framed (the default now):
GO_ROCM_RUN_HIP_TESTS=1 GO_ROCM_HIP_LOGIT_SPREAD_RECEIPTS=1 \
GO_ROCM_KERNEL_HSACO=<worktree>/build/kernels/rocm_kernels_gfx1101.hsaco \
GO_ROCM_ORACLE_MODEL_PATH=/tmp/models/gemma-4-12B-it-4bit-clean \
GO_ROCM_GEN_MAX_TOKENS=48 GO_ROCM_SPREAD_ARM=device \
go -C go test -count=1 -v ./engine/hip -run '^TestHIPGemma4Q4LogitSpreadProbe$'
# add GO_ROCM_SPREAD_RAW=1 to reproduce the old raw-path degeneracy.

# full suite (armed) green:
GO_ROCM_RUN_HIP_TESTS=1 \
GO_ROCM_KERNEL_HSACO=<worktree>/build/kernels/rocm_kernels_gfx1101.hsaco \
go -C go test -count=1 ./engine/hip/
# ok dappco.re/go/inference/engine/hip 2.641s
```

`go vet ./engine/hip/` reports only the four acknowledged pre-existing
`unsafe.Pointer` findings in `hip_driver_cgo.go` at lines 1255, 1290, 1317, and
1381. The r16 change is a test file and adds none. The known numerical
pre-existing red `TestHIPHardwareTransformerKernelSource` is gated behind the
compile-test env and is unrelated to this change.
