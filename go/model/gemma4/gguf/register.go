// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

// init registers gemma-4's dedicated GGUF export lane with model/gguf's
// generic QuantizeModelPack, so a gemma-4 checkpoint gets the canonical
// llama.cpp tensor names, per-tensor type policy, and full metadata +
// tokenizer header this package builds, instead of the byte-generic
// per-tensor quantiser. Mirrors model.RegisterArch's reactive registration
// (model/arch_spec.go) one level down, for the export path — model/gguf
// never imports this package (AX-8, lib never imports consumer).
func init() {
	basegguf.RegisterQuantizeLane(gemma4ModelType, basegguf.QuantizeLane{
		Detect:         isGemma4Config,
		SupportsFormat: isGemma4SupportedQuantizeFormat,
		UnsupportedFormatError: func(format basegguf.QuantizeFormat) error {
			return core.NewError("gguf: gemma4 GGUF conversion does not support " + string(format) + " (supported: q4_k_m, q8_0, q6_k, q5_k_m, q3_k_m)")
		},
		Quantize: quantizeGemma4ModelPack,
	})
}
