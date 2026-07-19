<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# ocr

Optical character recognition via the host DeepSeek-OCR dual-tower vision encoder (SAM ViT-B
feeding CLIP-L) + MoE decoder. model/arch/deepseek-ai/deepseekvl2 is a pure-CPU float32 forward
pass — like model/arch/openai/whisper (see pkg/transcribe) and model/arch/bert (see pkg/embed), it
never touches the GPU registry the chat/eval examples blank-import (no metallib, no engine backend
needed), so this example has no `_ "dappco.re/go/inference/examples/internal/engine"` line.

Fetch a snapshot first:

    hf download deepseek-ai/DeepSeek-OCR

## Run

```sh
go run ./pkg/ocr -model ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/<rev> -image page.png
```

`page.png` must be exactly 1024×1024 pixels in this v1 lane — DeepSeek-OCR's "Base" resolution
mode, the one fixed-canvas preset this package implements (see
`deepseekvl2.DecodeAndNormaliseImage`'s doc comment for why: PIL's bicubic resize/letterbox to that
canvas is impractical to port bit-exactly, so a mismatched size is a named refusal, not a silent,
non-matching resize). Pass `-prompt` to override the default "Free OCR" extraction prompt (e.g. the
checkpoint's own layout-aware `"<image>\n<|grounding|>Convert the document to markdown. "` mode) —
it must contain exactly one `<image>` placeholder.

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
