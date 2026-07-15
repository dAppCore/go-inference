// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// adaptFFNWeights exposes JetMoE's packed expert matrices through the generalised
// composed-MoE names. Tensor data remains zero-copy: each expert is a view into
// the checkpoint's [experts, 2*ff, hidden] or [experts, hidden, ff] tensor.
func adaptFFNWeights(tensors map[string]safetensors.Tensor, cfg Config) (map[string]safetensors.Tensor, error) {
	adapted := make(map[string]safetensors.Tensor, len(tensors)+cfg.NumHiddenLayers*(3*cfg.MoENumExperts+1))
	for name, tensor := range tensors {
		adapted[name] = tensor
	}
	for layer := 0; layer < cfg.NumHiddenLayers; layer++ {
		prefix := core.Sprintf("model.layers.%d.mlp.", layer)
		input, inputOK := tensors[prefix+"input_linear.weight"]
		output, outputOK := tensors[prefix+"output_linear.weight"]
		router, routerOK := tensors[prefix+"router.layer.weight"]
		if !inputOK || !outputOK || !routerOK {
			return nil, core.NewError(core.Sprintf("jetmoe.adaptFFNWeights: layer %d packed FFN weights are incomplete", layer))
		}
		if len(input.Shape) != 3 || input.Shape[0] != cfg.MoENumExperts || input.Shape[1] != 2*cfg.FFNHiddenSize || input.Shape[2] != cfg.HiddenSize {
			return nil, core.NewError(core.Sprintf("jetmoe.adaptFFNWeights: layer %d input_linear shape mismatch", layer))
		}
		if len(output.Shape) != 3 || output.Shape[0] != cfg.MoENumExperts || output.Shape[1] != cfg.HiddenSize || output.Shape[2] != cfg.FFNHiddenSize {
			return nil, core.NewError(core.Sprintf("jetmoe.adaptFFNWeights: layer %d output_linear shape mismatch", layer))
		}
		if len(router.Shape) != 2 || router.Shape[0] != cfg.MoENumExperts || router.Shape[1] != cfg.HiddenSize {
			return nil, core.NewError(core.Sprintf("jetmoe.adaptFFNWeights: layer %d router shape mismatch", layer))
		}
		bytesPerValue, err := tensorValueBytes(input.Dtype)
		if err != nil || output.Dtype != input.Dtype {
			return nil, core.NewError(core.Sprintf("jetmoe.adaptFFNWeights: layer %d expert dtype mismatch", layer))
		}
		inputExpertBytes := 2 * cfg.FFNHiddenSize * cfg.HiddenSize * bytesPerValue
		inputHalfBytes := cfg.FFNHiddenSize * cfg.HiddenSize * bytesPerValue
		outputExpertBytes := cfg.HiddenSize * cfg.FFNHiddenSize * bytesPerValue
		if len(input.Data) != cfg.MoENumExperts*inputExpertBytes || len(output.Data) != cfg.MoENumExperts*outputExpertBytes {
			return nil, core.NewError(core.Sprintf("jetmoe.adaptFFNWeights: layer %d packed data size mismatch", layer))
		}
		adapted[prefix+"gate.weight"] = router
		for expert := 0; expert < cfg.MoENumExperts; expert++ {
			expertPrefix := prefix + core.Sprintf("experts.%d.", expert)
			inputStart := expert * inputExpertBytes
			outputStart := expert * outputExpertBytes
			adapted[expertPrefix+"gate_proj.weight"] = safetensors.Tensor{Dtype: input.Dtype, Shape: []int{cfg.FFNHiddenSize, cfg.HiddenSize}, Data: input.Data[inputStart : inputStart+inputHalfBytes]}
			adapted[expertPrefix+"up_proj.weight"] = safetensors.Tensor{Dtype: input.Dtype, Shape: []int{cfg.FFNHiddenSize, cfg.HiddenSize}, Data: input.Data[inputStart+inputHalfBytes : inputStart+inputExpertBytes]}
			adapted[expertPrefix+"down_proj.weight"] = safetensors.Tensor{Dtype: output.Dtype, Shape: []int{cfg.HiddenSize, cfg.FFNHiddenSize}, Data: output.Data[outputStart : outputStart+outputExpertBytes]}
		}
	}
	return adapted, nil
}

func tensorValueBytes(dtype string) (int, error) {
	switch dtype {
	case "BF16", "bfloat16":
		return 2, nil
	case "F32", "float32":
		return 4, nil
	default:
		return 0, core.NewError("jetmoe.tensorValueBytes: unsupported dtype " + dtype)
	}
}
