# Whisper ASR Design (lem transcribe)

**Status:** DRAFT for review.

**Goal:** Whisper checkpoints (`model_type: whisper`, `WhisperForConditionalGeneration`) transcribe audio through lem — a `lem transcribe` verb and the OpenAI-compatible `/v1/audio/transcriptions` serve route — replacing today's honest refusal ("recognised, forward not yet implemented") with a working ASR forward.

**Why now:** approved for this week (2026-07-19). Speech-to-text is the most lem-useful of the recognition-only arches, and the engine already owns most of the parts.

## What already exists (reused, not rebuilt)

| Piece | Where |
|---|---|
| Log-mel front-end (shared home, golden-tested) | `engine/metal/audio_features.go` (+ `audio_mel_golden_test.go`) |
| Audio encoder attention/matmul primitives | `engine/metal/audio_attention.go`, `audio_encoder.go` (the gemma4 Conformer lane) |
| Cross-attention to injected external K/V | the MTP drafter path (`assistant_dflash.go` — draft layers cross-attend target K/V rows) |
| Non-factory arch precedent (own loader, own session) | mamba2: "route them to their own loader before the registered path" (`engine/metal/load.go`) |
| Config parse + tensor probing | `model/arch/openai/whisper/config.go` (complete; `Arch()` currently refuses) |
| WAV ingestion (16-bit PCM mono 16 kHz) | the `--audio` flag path in generate/serve |
| OpenAI-compat serve plumbing | `serving/` handlers |

## Architecture

Whisper is an encoder–decoder and deliberately does NOT enter the factory (`model.Assemble` is the causal-LM shape). Like mamba2 it registers its own loader and session:

- **`model/arch/openai/whisper`** grows `weights.go` (tensor-name map read from real checkpoints, never guessed) and the load path: encoder stack (conv×2 subsample → sinusoidal positions → pre-LN transformer layers) + decoder stack (learned positions; per layer: causal self-attn → cross-attn over encoder output → FFN) + the task-token machinery (`<|startoftranscript|>`, language tokens, `<|transcribe|>`, `<|notimestamps|>`, `<|endoftext|>`).
- **Engine session (`engine/metal`)**: host-f32 correctness-first forward (the proven #18/#42 method — host reference before device fusion):
  1. **Encode once per request:** mel → encoder → per-decoder-layer cross-K/V precomputed once (the standard Whisper serving trick — cross-attention keys/values depend only on the encoder output).
  2. **Decode loop:** self-attn KV cache grows per token; cross-attn reads the precomputed K/V; greedy argmax v1.
  3. Device fusion of the hot projections rides the existing `MatMulF32NT`/steel seams from day one (they are the host path's matmul); kernel-level fusion is a later slice with its own receipts.
- **Language detection:** one decoder step from `<|startoftranscript|>`, argmax restricted to the language-token ids (the reference method); `--language` overrides.

## Scope (v1, honest bounds)

- **Single ≤30 s window** (`max_source_positions` 1500 = 30 s). Longer audio in v1 returns a truthful "audio exceeds the v1 window" error naming the bound; the sliding-window chunking loop (with the token-context carry) is v2, not silently truncated audio.
- **Plain-text transcription** (`<|notimestamps|>` set). Timestamp emission is v2.
- Greedy decode only; temperature/beam fallback ladder is v2.
- Verification target: **`openai/whisper-tiny`** (39 M — CI-sized, fast to cache) for the live gate; quality spot-check on a larger checkpoint once the shape is proven.

## Surfaces

- `lem transcribe --model <whisper-dir> <audio.wav>` → transcript to stdout (`--json` for `{text, language}`).
- Serve: `POST /v1/audio/transcriptions` (multipart or JSON base64 — match the existing image-input precedent: data-URLs only, no remote fetch), model = the loaded whisper checkpoint. A non-whisper loaded model answers the route with the clean capability refusal (the vision 400 pattern).
- Capability discovery: a separate interface (`Transcriber`), type-assertion on the loaded model — the stability-contract pattern.

## Gates (the ship bar)

- Mel front-end: already golden-tested — reused unchanged.
- Encoder + decoder blocks: FD-free forward parity — golden vectors captured from the reference implementation on whisper-tiny fixtures (small synthetic mel in → per-block out), byte-tolerance banded.
- End-to-end: the live gate transcribes a known WAV fixture on whisper-tiny and asserts the exact reference transcript (greedy is deterministic); language auto-detect asserted on a non-English fixture.
- Serve route: compat-shape test + the capability-refusal test.

## Non-goals (v1)

Chunked long-form audio · timestamps · word-level alignment · translation task (`<|translate|>` — trivial later, out of v1 assertions) · streaming partials · device-fused encoder kernels · training/fine-tuning.

## Open questions for review

1. **Checkpoint set to cache:** whisper-tiny (verification) + which daily-driver — `large-v3-turbo` (809 M, the speed/quality sweet spot) or `small`? My lean: tiny + large-v3-turbo.
2. **Long audio v1 behaviour:** hard error at >30 s (my lean — truthful bound) vs auto-chunk-without-carry (worse quality, silently different from reference)?
3. **Serve route auth for uploads:** the audio route accepts base64 bodies like vision — confirm the same 32 MiB-class request bound is right for audio (≈ 5 min of 16 kHz WAV at that cap; moot while the v1 window is 30 s, but the bound outlives v1).
4. **Verb name:** `lem transcribe` (my lean) vs folding under `lem generate --audio` (conflates conditioning-on-audio with ASR — I'd keep them distinct).
