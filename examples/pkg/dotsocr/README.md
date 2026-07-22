<!-- SPDX-Licence-Identifier: EUPL-1.2 -->

# dotsocr

Document OCR via the host DOTS-OCR vision-language forward. model/arch/rednote-hilab/dotsocr is a
pure-CPU float32 forward pass (NaViT-style vision tower + Qwen2 decoder) — like model/arch/openai/
whisper (see pkg/transcribe), it never touches the GPU registry the chat/eval examples blank-import
(no metallib, no engine backend needed), so this example has no
`_ "dappco.re/go/inference/examples/internal/engine"` line.

Fetch a snapshot first:

hf download rednote-hilab/dots.ocr

## Run

```sh
go run ./pkg/dotsocr -model ~/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots/<rev> -image page.png
```

`-image` accepts PNG or JPEG. The default `-prompt` is DOTS-OCR's own README-documented
`prompt_layout_all_en` task (full JSON layout: bbox + category + text per element) — the exact
prompt this package's committed E2E golden was captured against, so running this example
unmodified against `model/arch/rednote-hilab/dotsocr/testdata/fixture.png` reproduces that
golden's output. Pass `-prompt` to run a different instruction (DOTS-OCR is trained specifically
for its documented layout/OCR task prompts — see the checkpoint's own README for the other
`prompt_*` modes — free-form instructions outside that training distribution may not behave like a
general chat assistant).

Image resizing: DOTS-OCR resizes to the nearest `patch_size·merge_size` (28px) grid box within
`[min_pixels,max_pixels]` (`smart_resize`, dimension-exact). When actual pixel resampling is
needed (source dimensions aren't already 28px-aligned), this port uses a standard bilinear filter
— a named approximation of the reference's PIL BICUBIC resampler (see `image.go`'s doc comment),
not bit-exact for that specific path.

Flags and behaviour are documented in [main.go](main.go) — the code is the example.
