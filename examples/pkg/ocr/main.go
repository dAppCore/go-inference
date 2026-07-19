// SPDX-Licence-Identifier: EUPL-1.2

// Optical character recognition via the host DeepSeek-OCR dual-tower vision encoder + MoE
// decoder. model/arch/deepseek-ai/deepseekvl2 is a pure-CPU float32 forward pass — like
// model/arch/openai/whisper (see pkg/transcribe) and model/arch/bert (see pkg/embed), it never
// touches the GPU registry the chat/eval examples blank-import (no metallib, no engine backend
// needed), so this example has no `_ "dappco.re/go/inference/examples/internal/engine"` line.
// DeepSeek-OCR's vision-encoder-conditioned MoE decoder is a genuinely different shape from the
// decoder-only causal-LM inference.TextModel assumes, so it is driven directly rather than
// through inference.LoadModel (mirroring the CLI: `lem ocr`, not `lem generate`).
//
//	go run ./pkg/ocr -model ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-OCR/snapshots/<rev> -image page.png
//
// Fetch a snapshot first:
//
//	hf download deepseek-ai/DeepSeek-OCR
package main

import (
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference/model/arch/deepseek-ai/deepseekvl2"
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "DeepSeek-OCR checkpoint directory (config.json, tokenizer.json, *.safetensors)")
	image := flag.String("image", "", "a PNG/JPEG image to run OCR on (v1: must be exactly 1024x1024 — see deepseekvl2.DecodeAndNormaliseImage's doc comment)")
	prompt := flag.String("prompt", "", "override the OCR prompt (must contain exactly one <image> placeholder); empty uses deepseekvl2.DefaultPrompt")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a DeepSeek-OCR snapshot directory")
		os.Exit(2)
	}
	if *image == "" {
		fmt.Fprintln(os.Stderr, "set -image to a PNG/JPEG file")
		os.Exit(2)
	}

	// deepseekvl2.Load and Model.OCR return plain Go errors, not core.Result — the host
	// vision-encoder + MoE-decoder path never crosses the engine registry that LoadModel/Generate
	// report through.
	m, err := deepseekvl2.Load(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}

	imageBytes, err := os.ReadFile(*image)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read image:", err)
		os.Exit(1)
	}

	result, err := m.OCR(imageBytes, deepseekvl2.Options{Prompt: *prompt})
	if err != nil {
		fmt.Fprintln(os.Stderr, "ocr:", err)
		os.Exit(1)
	}

	fmt.Printf("text: %s\n", result.Text)
}
