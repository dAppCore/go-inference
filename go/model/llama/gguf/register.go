// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

func init() {
	basegguf.RegisterQuantizeLane(llamaArch, basegguf.QuantizeLane{
		Detect:         isLlamaConfig,
		SupportsFormat: isLlamaSupportedQuantizeFormat,
		UnsupportedFormatError: func(format basegguf.QuantizeFormat) error {
			return core.NewError("gguf: llama GGUF conversion does not support " + string(format) + " (supported: q4_k_m, q8_0)")
		},
		Quantize: quantizeLlamaModelPack,
	})
}
