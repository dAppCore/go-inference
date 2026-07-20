// SPDX-Licence-Identifier: EUPL-1.2

package llama4

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

type wrapperConfig struct {
	ModelType         string  `json:"model_type"`
	TextConfig        *Config `json:"text_config"`
	TieWordEmbeddings *bool   `json:"tie_word_embeddings"`
}

func parseConfig(data []byte) (*Config, error) {
	var wrapper wrapperConfig
	if r := core.JSONUnmarshal(data, &wrapper); !r.OK {
		return nil, core.NewError("llama4.Parse: config.json parse failed")
	}
	if wrapper.TextConfig != nil {
		if wrapper.TextConfig.TieWordEmbeddings == nil {
			wrapper.TextConfig.TieWordEmbeddings = wrapper.TieWordEmbeddings
		}
		return wrapper.TextConfig, nil
	}
	var cfg Config
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil, core.NewError("llama4.Parse: text config parse failed")
	}
	return &cfg, nil
}

func elementBytes(dtype string) int {
	switch dtype {
	case "BF16", "bfloat16", "F16", "float16":
		return 2
	case "F32", "float32":
		return 4
	}
	return 0
}

func transposeMatrix(data []byte, rows, cols, width int) []byte {
	out := make([]byte, len(data))
	for row := range rows {
		for col := range cols {
			src, dst := (row*cols+col)*width, (col*rows+row)*width
			copy(out[dst:dst+width], data[src:src+width])
		}
	}
	return out
}

// NormalizeWeights maps Llama 4's feed_forward names and packed expert matrices
// onto the architecture-neutral composed MoE roles. Expert matrices are
// transposed from the checkpoint's bmm layout to the composed matNT layout.
func NormalizeWeights(in map[string]safetensors.Tensor) (map[string]safetensors.Tensor, error) {
	out := make(map[string]safetensors.Tensor, len(in)*3)
	for name, tensor := range in {
		out[name] = tensor
		alias := core.Replace(name, ".feed_forward.router.weight", ".mlp.gate.weight")
		alias = core.Replace(alias, ".feed_forward.", ".mlp.")
		out[alias] = tensor
		if !core.HasSuffix(name, ".feed_forward.experts.gate_up_proj") && !core.HasSuffix(name, ".feed_forward.experts.down_proj") {
			continue
		}
		if len(tensor.Shape) != 3 {
			return nil, core.NewError("llama4.NormalizeWeights: packed expert tensor must be 3-D")
		}
		width := elementBytes(tensor.Dtype)
		if width == 0 {
			return nil, core.NewError("llama4.NormalizeWeights: unsupported packed expert dtype " + tensor.Dtype)
		}
		experts, rows, cols := tensor.Shape[0], tensor.Shape[1], tensor.Shape[2]
		if len(tensor.Data) != experts*rows*cols*width {
			return nil, core.NewError("llama4.NormalizeWeights: packed expert byte length mismatch")
		}
		gateUp := core.HasSuffix(name, ".feed_forward.experts.gate_up_proj")
		suffix := "experts.down_proj"
		if gateUp {
			suffix = "experts.gate_up_proj"
		}
		base := name[:len(name)-len(suffix)]
		if gateUp && cols%2 != 0 {
			return nil, core.NewError("llama4.NormalizeWeights: gate_up_proj output width must be even")
		}
		for expert := range experts {
			chunk := tensor.Data[expert*rows*cols*width : (expert+1)*rows*cols*width]
			prefix := core.Replace(base, ".feed_forward.", ".mlp.") + core.Sprintf("experts.%d.", expert)
			if gateUp {
				ff := cols / 2
				gateRaw, upRaw := make([]byte, rows*ff*width), make([]byte, rows*ff*width)
				for row := range rows {
					rowData := chunk[row*cols*width : (row+1)*cols*width]
					copy(gateRaw[row*ff*width:(row+1)*ff*width], rowData[:ff*width])
					copy(upRaw[row*ff*width:(row+1)*ff*width], rowData[ff*width:])
				}
				out[prefix+"gate_proj.weight"] = safetensors.Tensor{Dtype: tensor.Dtype, Shape: []int{ff, rows}, Data: transposeMatrix(gateRaw, rows, ff, width)}
				out[prefix+"up_proj.weight"] = safetensors.Tensor{Dtype: tensor.Dtype, Shape: []int{ff, rows}, Data: transposeMatrix(upRaw, rows, ff, width)}
			} else {
				out[prefix+"down_proj.weight"] = safetensors.Tensor{Dtype: tensor.Dtype, Shape: []int{cols, rows}, Data: transposeMatrix(chunk, rows, cols, width)}
			}
		}
	}
	return out, nil
}

func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"llama4", "llama4_text"},
		Parse:      func(data []byte) (model.ArchConfig, error) { return parseConfig(data) },
		// Weights + NormalizeConfig give Llama 4 the factory route (model.Assemble — the #18/#50
		// unification target), dual-registered alongside the Composed hook below exactly as
		// dbrx/qwenmoe/mixtral/granitemoe carry both: Composed stays the A/B reference + the route a
		// caller that deliberately bypasses model.Load still reaches, while model.Load now succeeds
		// instead of rejecting Llama 4 as composed-only. Llama 4 was the LAST arch-zoo entry still
		// Composed-only (#50).
		Weights: FactoryWeightNames(),
		NormalizeConfig: func(tensors map[string]safetensors.Tensor, ac model.ArchConfig) map[string]safetensors.Tensor {
			cfg := ac.(*Config)
			normalised, err := NormalizeWeights(tensors)
			if err != nil {
				return tensors // malformed checkpoint — Assemble's nil-safe load surfaces the gap downstream
			}
			arch, err := cfg.Arch()
			if err != nil {
				return normalised // Parse succeeding doesn't guarantee Arch(); load.go's own Arch() call reports this cleanly
			}
			if packed, err := packExperts(normalised, arch); err == nil {
				return packed
			}
			return normalised // malformed/absent experts on a declared MoE layer — nil ExpGate/ExpUp/ExpDown surfaces downstream
		},
	})
}
