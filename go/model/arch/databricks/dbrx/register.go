// SPDX-Licence-Identifier: EUPL-1.2

package dbrx

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/attn"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
)

// NormalizeWeights aliases the published DBRX checkpoint layout to neutral composed roles.
func NormalizeWeights(in map[string]safetensors.Tensor, cfg Config) map[string]safetensors.Tensor {
	out := make(map[string]safetensors.Tensor, len(in)+cfg.Layers*(8+3*cfg.FFN.Experts))
	for name, tensor := range in {
		out[name] = tensor
	}
	alias := func(dst, src string) {
		if tensor, ok := in[src]; ok {
			out[dst] = tensor
		}
	}
	alias("model.embed_tokens.weight", "transformer.wte.weight")
	alias("model.norm.weight", "transformer.norm_f.weight")
	alias("lm_head.weight", "lm_head.weight")
	headDim := 0
	if cfg.Heads > 0 {
		headDim = cfg.DModel / cfg.Heads
	}
	for layer := range cfg.Layers {
		src := core.Sprintf("transformer.blocks.%d.", layer)
		dst := core.Sprintf("model.layers.%d.", layer)
		alias(dst+"input_layernorm.weight", src+"norm_attn_norm.norm_1.weight")
		alias(dst+"post_attention_layernorm.weight", src+"norm_attn_norm.norm_2.weight")
		alias(dst+"self_attn.o_proj.weight", src+"norm_attn_norm.attn.out_proj.weight")
		out = attn.SplitContiguousQKV(out, src+"norm_attn_norm.attn.Wqkv", dst+"self_attn.q_proj", dst+"self_attn.k_proj", dst+"self_attn.v_proj", cfg.Heads*headDim, cfg.Attention.KVHeads*headDim)
		alias(dst+"mlp.gate.weight", src+"ffn.router.layer.weight")
		for _, part := range []struct{ source, target string }{{"w1", "gate_proj.weight"}, {"v1", "up_proj.weight"}, {"w2", "down_proj.weight"}} {
			tensor, ok := in[src+"ffn.experts.mlp."+part.source]
			width := 4
			if tensor.Dtype == "BF16" || tensor.Dtype == "bfloat16" {
				width = 2
			}
			if !ok || len(tensor.Shape) < 2 || cfg.FFN.HiddenSize <= 0 || cfg.DModel <= 0 || len(tensor.Data) != cfg.FFN.Experts*cfg.FFN.HiddenSize*cfg.DModel*width {
				continue
			}
			bytesPerExpert := len(tensor.Data) / cfg.FFN.Experts
			for expert := range cfg.FFN.Experts {
				shape := []int{cfg.FFN.HiddenSize, cfg.DModel}
				data := tensor.Data[expert*bytesPerExpert : (expert+1)*bytesPerExpert]
				if part.source == "w2" {
					data = transpose(data, tensor.Dtype, cfg.FFN.HiddenSize, cfg.DModel)
					shape = []int{cfg.DModel, cfg.FFN.HiddenSize}
				}
				out[core.Sprintf("%smlp.experts.%d.%s", dst, expert, part.target)] = safetensors.Tensor{Dtype: tensor.Dtype, Shape: shape, Data: data}
			}
		}
	}
	return out
}

func transpose(data []byte, dtype string, rows, columns int) []byte {
	width := 4
	if dtype == "BF16" || dtype == "bfloat16" {
		width = 2
	}
	if len(data) != rows*columns*width {
		return data
	}
	out := make([]byte, len(data))
	for row := range rows {
		for column := range columns {
			source := (row*columns + column) * width
			target := (column*rows + row) * width
			copy(out[target:target+width], data[source:source+width])
		}
	}
	return out
}

func loaderJSON(c Config) []byte {
	return []byte(core.Sprintf(`{"model_type":"dbrx","hidden_size":%d,"intermediate_size":%d,"num_hidden_layers":%d,"num_attention_heads":%d,"num_key_value_heads":%d,"num_experts":%d,"num_experts_per_tok":%d,"vocab_size":%d,"rms_norm_eps":%g,"rope_theta":%g,"use_layer_norm":true,"qkv_clip":%g}`, c.DModel, c.FFN.HiddenSize, c.Layers, c.Heads, c.Attention.KVHeads, c.FFN.Experts, c.FFN.TopK, c.VocabSize, c.LayerNormEps, c.Attention.RopeTheta, c.Attention.ClipQKV))
}

func init() {
	model.RegisterArch(model.ArchSpec{ModelTypes: []string{"dbrx"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("dbrx.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		Composed: func(tensors map[string]safetensors.Tensor, configJSON []byte) (model.TokenModel, error) {
			var cfg Config
			if r := core.JSONUnmarshal(configJSON, &cfg); !r.OK {
				return nil, core.NewError("dbrx.Load: config.json parse failed")
			}
			arch, err := cfg.Arch()
			if err != nil {
				return nil, core.E("dbrx.Load", "resolve architecture", err)
			}
			// Zero-copy: NormalizeWeights re-exposes the checkpoint tensors (the pass-through map entries keep
			// their mmap-backed bytes), so the packed quant projection weights VIEW the mapped checkpoint
			// rather than being copied. model.LoadComposedDir hands the model the mapping via RetainMmap.
			cm, err := composed.LoadComposedWithArchMmap(NormalizeWeights(tensors, cfg), loaderJSON(cfg), arch)
			if err != nil {
				return nil, core.E("dbrx.Load", "assemble composed model", err)
			}
			return composed.NewTokenModel(cm), nil
		},
	})
}
