<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# glmocr

OCR via the host GLM-OCR vision-language decoder. model/arch/zai-org/glmocr is a pure-CPU
float32 forward pass — like model/arch/openai/whisper (see pkg/transcribe), it never touches the
GPU registry the chat/eval examples blank-import (no metallib, no engine backend needed), so this
example has no `_ "dappco.re/go/inference/examples/internal/engine"` line.

Fetch a snapshot first:

hf download zai-org/GLM-OCR

## Run

```sh
go run ./pkg/glmocr -model ~/.cache/huggingface/hub/models--zai-org--GLM-OCR/snapshots/<rev> -image doc.png
```

`doc.png` must already be a smart_resize-stable size — every side an exact multiple of 28px,
within GLM-OCR's min/max pixel bounds (112x112 up to roughly 3500x3500 total). Arbitrary-size
bicubic resampling is not implemented in this lane (a named boundary): a mismatched size is
refused with the exact target dimensions named, not silently resampled.

Pass `-prompt` to select a different task — GLM-OCR documents `"Text Recognition:"` (the
default), `"Formula Recognition:"`, `"Table Recognition:"`, or a JSON-schema information-
extraction instruction. Pass `-max-new-tokens` to raise the generated-token cap for a long
document (this package recomputes the whole sequence every decode step — no growing KV cache
yet — so wall-clock grows with the generation length; the default is conservative).

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
