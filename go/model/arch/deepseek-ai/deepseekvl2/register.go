// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	"dappco.re/go/inference/model"
)

// init registers deepseek_vl_v2 — the DeepSeek-VL2 architecture DeepSeek-OCR / OCR-2 reuse
// unchanged (https://huggingface.co/deepseek-ai/DeepSeek-OCR) — as a RECOGNISED model_type with a
// clean, ACCURATE refusal from the decoder-only causal-LM path, not an "unknown model
// architecture" and not a stale "not yet implemented" once OCR itself IS implemented (ocr.go).
// Both loader entry points refuse the SAME way: Parse succeeds and Arch refuses (the model.Load
// path — config.go), and Composed refuses directly (the model.LoadComposedDir path engine/metal
// tries first) — so whichever entry point a backend reaches, the refusal names the real reason
// (a dual-tower vision encoder + MoE decoder is not the decoder-only shape either path assembles)
// and redirects to `lem ocr`. Mirrors whisper's registration shape: Arch/Composed refuse
// permanently (architecturally different, not "not yet implemented"), the real forward lives in
// its own loader (ocr.go's Load/Model.OCR) driven by its own verb.
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"deepseek_vl_v2"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			return ParseConfig(data)
		},
	})
}
