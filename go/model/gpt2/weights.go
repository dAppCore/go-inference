// SPDX-Licence-Identifier: EUPL-1.2

package gpt2

import (
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// NormalizeWeights applies the HF Conv1D [in,out] convention and exposes fused QKV by role.
func NormalizeWeights(in map[string]safetensors.Tensor) map[string]safetensors.Tensor {
	out := make(map[string]safetensors.Tensor, len(in)+16)
	for name, tensor := range in {
		out[name] = tensor
	}
	for name, tensor := range in {
		if !(hasSuffix(name, ".attn.c_attn.weight") || hasSuffix(name, ".attn.c_proj.weight") || hasSuffix(name, ".mlp.c_fc.weight") || hasSuffix(name, ".mlp.c_proj.weight")) {
			continue
		}
		tr, err := model.TransposeTensor2D(tensor)
		if err != nil {
			continue
		}
		if hasSuffix(name, ".attn.c_attn.weight") {
			base := name[:len(name)-len("c_attn.weight")]
			qRows, remainder := tr.Shape[1], tr.Shape[0]-tr.Shape[1]
			if remainder > 0 && remainder%2 == 0 {
				splitQKVRows(out, base, tr, qRows, remainder/2)
			}
		} else {
			out[name] = tr
		}
	}
	return out
}

func hasSuffix(s, suffix string) bool {
	return len(s) >= len(suffix) && s[len(s)-len(suffix):] == suffix
}

func splitQKVRows(out map[string]safetensors.Tensor, base string, t safetensors.Tensor, qRows, kvRows int) {
	rowBytes := len(t.Data) / t.Shape[0]
	offset := 0
	for i, name := range []string{"q_proj.weight", "k_proj.weight", "v_proj.weight"} {
		rows := kvRows
		if i == 0 {
			rows = qRows
		}
		data := make([]byte, rows*rowBytes)
		copy(data, t.Data[offset*rowBytes:(offset+rows)*rowBytes])
		out[base+name] = safetensors.Tensor{Dtype: t.Dtype, Shape: []int{rows, t.Shape[1]}, Data: data}
		offset += rows
	}
}
