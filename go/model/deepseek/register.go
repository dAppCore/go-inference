// SPDX-Licence-Identifier: EUPL-1.2

package deepseek

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// Names documents the DeepSeek MLA and routed-expert tensor layout.
type Names struct {
	Q, QA, QANorm, QB, KVA, KVANorm, KVB, O  string
	Router, ExpertGate, ExpertUp, ExpertDown string
	SharedGate, SharedUp, SharedDown         string
}

// WeightNames returns the exact DeepSeek-V2/V3 attention and expert tensor suffixes.
func WeightNames() Names {
	return Names{
		Q: ".self_attn.q_proj.weight", QA: ".self_attn.q_a_proj.weight", QANorm: ".self_attn.q_a_layernorm.weight", QB: ".self_attn.q_b_proj.weight",
		KVA: ".self_attn.kv_a_proj_with_mqa.weight", KVANorm: ".self_attn.kv_a_layernorm.weight", KVB: ".self_attn.kv_b_proj.weight", O: ".self_attn.o_proj.weight",
		Router: ".mlp.gate.weight", ExpertGate: ".mlp.experts.%d.gate_proj.weight",
		ExpertUp: ".mlp.experts.%d.up_proj.weight", ExpertDown: ".mlp.experts.%d.down_proj.weight",
		SharedGate: ".mlp.shared_experts.gate_proj.weight", SharedUp: ".mlp.shared_experts.up_proj.weight", SharedDown: ".mlp.shared_experts.down_proj.weight",
	}
}

func init() {
	model.RegisterArch(model.ArchSpec{
		ModelTypes: []string{"deepseek_v2", "deepseek_v3"},
		Parse: func(data []byte) (model.ArchConfig, error) {
			var cfg Config
			if r := core.JSONUnmarshal(data, &cfg); !r.OK {
				return nil, core.NewError("deepseek.Parse: config.json parse failed")
			}
			return &cfg, nil
		},
		Composed: func(map[string]safetensors.Tensor, []byte) (model.TokenModel, error) {
			return nil, core.NewError("deepseek.Load: MLA attention core is not implemented; refusing standard-attention fallback")
		},
	})
}
