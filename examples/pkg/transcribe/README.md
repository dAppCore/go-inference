<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# transcribe

Speech-to-text via the host Whisper encoder-decoder. model/arch/openai/whisper is a pure-CPU float32
forward pass — like model/arch/bert (see pkg/embed), it never touches the GPU registry the chat/eval
examples blank-import (no metallib, no engine backend needed), so this example has no
`_ "dappco.re/go/inference/examples/internal/engine"` line.

Fetch a snapshot first:

hf download openai/whisper-tiny

## Run

```sh
go run ./pkg/transcribe -model ~/.cache/huggingface/hub/models--openai--whisper-tiny/snapshots/<rev> -audio clip.wav
```

`clip.wav` must be 16-bit PCM, mono, 16 kHz, and at most 30 seconds (v1's single-window bound — longer
audio is a named refusal, not silent truncation). Pass `-language en` (or any `lang_to_id` code the
checkpoint's generation_config.json carries) to force the source language instead of auto-detecting it.

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
