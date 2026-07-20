// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	"dappco.re/go/inference/model"
)

// init registers dots_ocr and its dots_ocr_1_5 successor
// (https://huggingface.co/rednote-hilab/dots.ocr) as RECOGNISED model_types with a clean
// refusal, not an "unknown model architecture": a user pointing lem at a DOTS-OCR checkpoint
// gets direction (this IS dots_ocr[_1_5], here is what's missing), never a mystery. One
// Config/refusal serves both ids — 1.5 is the same Qwen2-shaped decoder plus ViT tower shape,
// versioned. Both loader entry points refuse: Parse succeeds and Arch refuses (the model.Load
// path), and Composed refuses directly (the model.LoadComposedDir path engine/metal tries
// first) — so whichever entry point a backend reaches, the refusal surfaces with the same
// explanation. This mirrors ../../deepseek-ai/deepseek's registration for the identical
// "recognised, not yet implemented" contract.
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"dots_ocr", "dots_ocr_1_5"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			return ParseConfig(data)
		},
	})
}
