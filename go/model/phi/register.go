// SPDX-Licence-Identifier: EUPL-1.2

package phi

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/phi/gguf"
	"dappco.re/go/inference/model/safetensors"
)

func parse(data []byte) (model.ArchConfig, error) {
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("phi.Parse: config.json parse failed")
	}
	return &cfg, nil
}

func init() {
	phi2 := model.StandardWeightNames()
	phi2.FinalNorm = "model.final_layernorm.weight"
	phi2.O = ".self_attn.dense"
	phi2.MLPNorm = ".input_layernorm.weight"
	phi2.Gate = ".mlp.fc1"
	phi2.Up = ".mlp.fc1"
	phi2.Down = ".mlp.fc2"
	phi2.PostAttnNorm = ""
	phi2.PostFFNorm = ""
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"phi"}, Parse: parse, Weights: phi2})

	phi3 := model.StandardWeightNames()
	phi3.MLPNorm = ".post_attention_layernorm.weight"
	phi3.PostAttnNorm = ""
	phi3.PostFFNorm = ""
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"phi3"}, Parse: parse, Weights: phi3, Normalize: NormalizePhi3Weights})
}

// NormalizePhi3Weights splits Phi-3's fused qkv and gate/up rows into the
// canonical role names consumed by the neutral assembler. The public Phi-3
// checkpoints use equal query and key/value head counts, hence three equal qkv
// row blocks.
func NormalizePhi3Weights(in map[string]safetensors.Tensor) map[string]safetensors.Tensor {
	out := make(map[string]safetensors.Tensor, len(in)*2)
	for name, tensor := range in {
		out[name] = tensor
		if core.HasSuffix(name, ".self_attn.qkv_proj.weight") {
			parts, ok := splitRows(tensor, 3)
			if ok {
				base := core.TrimSuffix(name, "qkv_proj.weight")
				out[base+"q_proj.weight"], out[base+"k_proj.weight"], out[base+"v_proj.weight"] = parts[0], parts[1], parts[2]
			}
		}
		if core.HasSuffix(name, ".mlp.gate_up_proj.weight") {
			parts, ok := splitRows(tensor, 2)
			if ok {
				base := core.TrimSuffix(name, "gate_up_proj.weight")
				out[base+"gate_proj.weight"], out[base+"up_proj.weight"] = parts[0], parts[1]
			}
		}
	}
	return out
}

func splitRows(t safetensors.Tensor, parts int) ([]safetensors.Tensor, bool) {
	if len(t.Shape) < 2 || parts <= 0 || t.Shape[0]%parts != 0 || len(t.Data)%parts != 0 {
		return nil, false
	}
	rows, bytes := t.Shape[0]/parts, len(t.Data)/parts
	out := make([]safetensors.Tensor, parts)
	for i := range parts {
		shape := append([]int(nil), t.Shape...)
		shape[0] = rows
		out[i] = safetensors.Tensor{Dtype: t.Dtype, Shape: shape, Data: t.Data[i*bytes : (i+1)*bytes]}
	}
	return out, true
}
