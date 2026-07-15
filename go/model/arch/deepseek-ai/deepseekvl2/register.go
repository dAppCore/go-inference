// SPDX-Licence-Identifier: EUPL-1.2

package deepseekvl2

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// init registers deepseek_vl_v2 — the DeepSeek-VL2 architecture DeepSeek-OCR / OCR-2 reuse
// unchanged (https://huggingface.co/deepseek-ai/DeepSeek-OCR) — as a RECOGNISED model_type with
// a clean refusal, not an "unknown model architecture": a user pointing lem at a DeepSeek-OCR
// checkpoint gets direction (this IS deepseek_vl_v2, here is what's missing), never a mystery.
// Both loader entry points refuse: Parse succeeds and Arch refuses (the model.Load path), and
// Composed refuses directly (the model.LoadComposedDir path engine/metal tries first) — so
// whichever entry point a backend reaches, the refusal surfaces with the same explanation. This
// mirrors ../deepseek's deepseek_v2/deepseek_v3 registration for the identical "recognised, not
// yet implemented" contract.
func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"deepseek_vl_v2"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			return ParseConfig(data)
		},
		Composed: func(_ map[string]safetensors.Tensor, cfgJSON []byte) (model.TokenModel, error) {
			mt, _ := model.ProbeModelTypes(cfgJSON)
			if mt == "" {
				mt = "deepseek_vl_v2"
			}
			return nil, core.NewError(mt + " (DeepSeek-OCR) is a recognised OCR vision-language arch; its vision encoder + OCR decoder forward is not yet implemented in this engine")
		},
	})
}
