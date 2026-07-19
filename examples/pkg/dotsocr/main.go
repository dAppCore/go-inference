// SPDX-Licence-Identifier: EUPL-1.2

// Document OCR via the host DOTS-OCR vision-language forward. model/arch/rednote-hilab/dotsocr is a
// pure-CPU float32 forward pass (NaViT-style vision tower + Qwen2 decoder) — like model/arch/openai/
// whisper (see pkg/transcribe), it never touches the GPU registry the chat/eval examples blank-import
// (no metallib, no engine backend needed), so this example has no
// `_ "dappco.re/go/inference/examples/internal/engine"` line. DOTS-OCR's multimodal (image+text ->
// text) shape does not fit the neutral inference.TextModel contract, so it is driven directly via the
// package's own Load/OCR surface, mirroring whisper's Load/Transcribe (and the CLI precedent: `lem
// transcribe`, not `lem generate` — the sibling lane wires an analogous `lem ocr` verb onto this
// package's OCR method at merge time, not this example).
//
//	go run ./pkg/dotsocr -model ~/.cache/huggingface/hub/models--rednote-hilab--dots.ocr/snapshots/<rev> -image page.png
//
// Fetch a snapshot first:
//
//	hf download rednote-hilab/dots.ocr
package main

import (
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference/model/arch/rednote-hilab/dotsocr"
)

// defaultPrompt is DOTS-OCR's own README-documented "prompt_layout_all_en" task — the exact
// prompt this package's E2E golden (testdata/e2e_golden.json) was captured against, so running
// this example unmodified against the committed fixture reproduces that golden's output.
const defaultPrompt = `Please output the layout information from the PDF image, including each layout element's bbox, its category, and the corresponding text content within the bbox.

1. Bbox format: [x1, y1, x2, y2]

2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', 'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].

3. Text Extraction & Formatting Rules:
    - Picture: For the 'Picture' category, the text field should be omitted.
    - Formula: Format its text as LaTeX.
    - Table: Format its text as HTML.
    - All Others (Text, Title, etc.): Format their text as Markdown.

4. Constraints:
    - The output text must be the original text from the image, with no translation.
    - All layout elements must be sorted according to human reading order.

5. Final Output: The entire output must be a single JSON object.
`

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "DOTS-OCR checkpoint directory (config.json, tokenizer.json, *.safetensors)")
	image := flag.String("image", "", "a PNG or JPEG image to OCR")
	prompt := flag.String("prompt", defaultPrompt, "the instruction prompt (defaults to DOTS-OCR's own README prompt_layout_all_en)")
	maxNewTokens := flag.Int("max-new-tokens", 0, "cap generation length (0 uses the package default)")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a DOTS-OCR snapshot directory")
		os.Exit(2)
	}
	if *image == "" {
		fmt.Fprintln(os.Stderr, "set -image to a PNG or JPEG file")
		os.Exit(2)
	}

	// dotsocr.Load and Model.OCR return plain Go errors, not core.Result — the host vision+decoder
	// path never crosses the engine registry that LoadModel/Generate report through.
	m, err := dotsocr.Load(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}
	if *maxNewTokens > 0 {
		m.MaxNewTokens = *maxNewTokens
	}

	imageBytes, err := os.ReadFile(*image)
	if err != nil {
		fmt.Fprintln(os.Stderr, "read image:", err)
		os.Exit(1)
	}

	text, err := m.OCR(imageBytes, *prompt)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ocr:", err)
		os.Exit(1)
	}

	fmt.Println(text)
}
