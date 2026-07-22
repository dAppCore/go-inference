<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# dflash

DFlash block-diffusion speculative decoding as a library caller. The real
z-lab convention (z-lab/Qwen3-4B-DFlash-b16 and siblings — arXiv 2602.06036)
pairs a small 5-layer drafter against its target through the SAME
`LoadSpeculativePair` seam [chat/mtp](../chat/mtp) uses for the Gemma 4
assistant / Qwen MTP conventions: one call, family-dispatched on the
drafter's config.json.

The drafter proposes a whole BLOCK of continuation tokens per forward
(conditioned on the target's own hidden states, fused across several
layers) instead of one token per forward; `decode/dflash.AcceptBlock` then
verifies the block against the target with the ordinary greedy
prefix-accept rule, so the emitted sequence stays byte-identical to plain
decode WHATEVER the drafter proposes — losslessness by construction.

darwin/arm64 only — this file imports `engine/metal` directly
(`LoadSpeculativePair`), the same call serve/generate wire in through
`serving.SpeculativeLoader` once the engine's `DFlashEngineProbe` is armed
for live serving.

## Fetch a real target + drafter pair

```sh
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-4B'))"
python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('z-lab/Qwen3-4B-DFlash-b16'))"
```

## Run

```sh
go run ./pkg/dflash \
  -model ~/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/<snap> \
  -draft ~/.cache/huggingface/hub/models--z-lab--Qwen3-4B-DFlash-b16/snapshots/<snap>
```

Flags and behaviour are documented in [main.go](main.go) — the code is the example.

## Current status (2026-07-20)

Determinism and losslessness both hold against the real checkpoint pair
(`engine/metal`'s `TestSpeculativeModel_DFlashZLab_RealPairedGenerate`): the
emitted sequence is byte-identical to the target's own plain greedy decode,
run twice, every time — confirmed both at the token level (the test) and
independently against the target loaded PLAINLY through `inference.LoadModel`
with no drafter at all (same tokens, same repetition). The printed
accept-rate is currently 0% — a pre-existing bug in the engine's
`ForwardCaptureHiddens` intermediate-layer capture (used by this pairing's
aux-hidden tap, `ExtractAuxHiddensAllRaw`) feeds the drafter the wrong
context, so its proposals don't land; see that test's doc comment for the
cross-validated evidence. The pairing is safe to run (never a wrong token,
only a slower one) but does not yet accelerate — `serving.DFlashEngineProbe`
stays `false` until the capture bug is fixed and an accept-rate receipt
lands.

Note separately: `-temp 0` (required for the exact byte-for-byte verification
this example's losslessness rests on) can produce repetitive greedy output on
a checkpoint whose own `generation_config.json` declares sampling defaults
(`temperature: 0.6`) — that is a property of greedy decoding on this
checkpoint, reproduced identically with or without the drafter, not a DFlash
defect.
