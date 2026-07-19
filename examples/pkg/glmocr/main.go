// SPDX-Licence-Identifier: EUPL-1.2

// OCR via the host GLM-OCR vision-language decoder. model/arch/zai-org/glmocr is a pure-CPU
// float32 forward pass — like model/arch/openai/whisper (see pkg/transcribe) and model/arch/bert
// (see pkg/embed), it never touches the GPU registry the chat/eval examples blank-import (no
// metallib, no engine backend needed), so this example has no
// `_ "dappco.re/go/inference/examples/internal/engine"` line. GLM-OCR is a vision-language OCR
// decoder, a different shape from the decoder-only causal-LM the neutral inference.TextModel
// contract assumes, so it is driven directly rather than through inference.LoadModel (mirroring
// the CLI: a `lem ocr` verb wires this package in at a sibling lane, not `lem generate`).
//
//	go run ./pkg/glmocr -model ~/.cache/huggingface/hub/models--zai-org--GLM-OCR/snapshots/<rev> -image doc.png
//
// Fetch a snapshot first:
//
//	hf download zai-org/GLM-OCR
//
// The image must already be a smart_resize-stable size (e.g. any size whose height AND width
// are exact multiples of 28px, within GLM-OCR's min/max pixel bounds — 112x112 up to roughly
// 3500x3500) — arbitrary-size bicubic resampling is not implemented in this lane; a mismatched
// size is refused with the exact target dimensions named, not silently resampled.
package main

import (
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference/model/arch/zai-org/glmocr"
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "GLM-OCR checkpoint directory (config.json, tokenizer.json, *.safetensors)")
	image := flag.String("image", "", "a PNG or JPEG image to run OCR on (must already be a smart_resize-stable size)")
	prompt := flag.String("prompt", "Text Recognition:", "the task prompt — GLM-OCR documents \"Text Recognition:\", \"Formula Recognition:\", \"Table Recognition:\", or a JSON-schema information-extraction instruction")
	maxNewTokens := flag.Int("max-new-tokens", 0, "cap the generated token count (0 uses the package default)")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a GLM-OCR snapshot directory")
		os.Exit(2)
	}
	if *image == "" {
		fmt.Fprintln(os.Stderr, "set -image to a PNG or JPEG file")
		os.Exit(2)
	}

	// glmocr.Load and Model.OCR return plain Go errors, not core.Result — the host vision-
	// language path never crosses the engine registry that LoadModel/Generate report through.
	m, err := glmocr.Load(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}

	imageBytes, err := os.ReadFile(*image)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read image:", err)
		os.Exit(1)
	}

	text, err := m.OCRWithOptions(imageBytes, *prompt, glmocr.GenerateOptions{MaxNewTokens: *maxNewTokens})
	if err != nil {
		fmt.Fprintln(os.Stderr, "ocr:", err)
		os.Exit(1)
	}

	fmt.Println(text)
}
