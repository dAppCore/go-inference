// SPDX-Licence-Identifier: EUPL-1.2

// Package gguf is gemma-3's dedicated GGUF export lane: it converts a dense
// gemma-3 safetensors checkpoint into a GGUF text model llama.cpp loads and
// generates from — canonical tensor names, the divisibility-aware per-tensor
// quant policy, the gemma "(1 + weight)" RMS-norm fold, the full gemma3.*
// hyperparameter header, and a SentencePiece ("llama") tokenizer block with
// real per-token scores read from the checkpoint's tokenizer.model. It registers
// itself with model/gguf's generic QuantizeModelPack from init(); model/gguf
// never imports this package (AX-8, lib never imports consumer).
package gguf

import (
	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

// init registers gemma-3's GGUF export lane so a gemma-3 checkpoint gets this
// package's canonical export instead of the byte-generic per-tensor quantiser.
// Mirrors model.RegisterArch's reactive registration (model/arch_spec.go) one
// level down, for the export path — the same pattern model/gemma4/gguf uses.
func init() {
	basegguf.RegisterQuantizeLane(gemma3Arch, basegguf.QuantizeLane{
		Detect:         isGemma3Config,
		SupportsFormat: isGemma3SupportedQuantizeFormat,
		UnsupportedFormatError: func(format basegguf.QuantizeFormat) error {
			return core.NewError("gguf: gemma3 GGUF conversion does not support " + string(format) + " (supported: q4_k_m, q8_0, q6_k, q5_k_m, q3_k_m, q2_k)")
		},
		Quantize: quantizeGemma3ModelPack,
	})
}
