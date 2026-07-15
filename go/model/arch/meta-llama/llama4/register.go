// SPDX-Licence-Identifier: EUPL-1.2

package llama4

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
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
		Composed: func(tensors map[string]safetensors.Tensor, configJSON []byte) (model.TokenModel, error) {
			cfg, err := parseConfig(configJSON)
			if err != nil {
				return nil, err
			}
			arch, err := cfg.Arch()
			if err != nil {
				return nil, core.E("llama4.Load", "resolve architecture", err)
			}
			normalised, err := NormalizeWeights(tensors)
			if err != nil {
				return nil, core.E("llama4.Load", "normalise weights", err)
			}
			// Zero-copy: NormalizeWeights re-exposes the checkpoint tensors (the pass-through map entries keep
			// their mmap-backed bytes), so the packed quant projection weights VIEW the mapped checkpoint
			// rather than being copied. model.LoadComposedDir hands the model the mapping via RetainMmap.
			cm, err := composed.LoadComposedWithArchMmap(normalised, configJSON, arch)
			if err != nil {
				return nil, core.E("llama4.Load", "assemble composed model", err)
			}
			return composed.NewTokenModel(cm), nil
		},
	})
}
