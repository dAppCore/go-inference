// SPDX-Licence-Identifier: EUPL-1.2

package granitemoe

import (
	"math"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
)

type seededValues struct{ state uint32 }

func (s *seededValues) next(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		s.state = 1664525*s.state + 1013904223
		out[i] = float32(int32(s.state>>24)-128) / 256
	}
	return out
}

func tinyWeights() map[string]safetensors.Tensor {
	const hidden, vocab, ff, heads, kvHeads, headDim, experts = 8, 32, 4, 2, 1, 4, 4
	seed := seededValues{state: 0x6772616e}
	norm := func() safetensors.Tensor {
		values := seed.next(hidden)
		for i := range values {
			values[i] += 1
		}
		return tensor(values, hidden)
	}
	return map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                            tensor(seed.next(vocab*hidden), vocab, hidden),
		"model.norm.weight":                                    norm(),
		"model.layers.0.input_layernorm.weight":                norm(),
		"model.layers.0.post_attention_layernorm.weight":       norm(),
		"model.layers.0.self_attn.q_proj.weight":               tensor(seed.next(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":               tensor(seed.next(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":               tensor(seed.next(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":               tensor(seed.next(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.block_sparse_moe.input_linear.weight":  tensor(seed.next(experts*2*ff*hidden), experts, 2*ff, hidden),
		"model.layers.0.block_sparse_moe.output_linear.weight": tensor(seed.next(experts*hidden*ff), experts, hidden, ff),
		"model.layers.0.block_sparse_moe.router.layer.weight":  tensor(seed.next(experts*hidden), experts, hidden),
	}
}

func TestTinyGraniteMoEForward_Good(t *testing.T) {
	config := []byte(`{"model_type":"granitemoe","hidden_size":8,"intermediate_size":4,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"num_local_experts":4,"num_experts_per_tok":2,"vocab_size":32,"rms_norm_eps":0.00001,"rope_theta":10000,"tie_word_embeddings":true,"hidden_act":"silu","logits_scaling":6,"residual_multiplier":0.22,"embedding_multiplier":12,"attention_multiplier":0.125}`)
	spec, ok := model.LookupArch("granitemoe")
	if !ok {
		t.Fatal("GraniteMoE architecture not registered")
	}
	tensors := tinyWeights()
	tm, err := spec.Composed(tensors, config)
	if err != nil {
		t.Fatalf("load tiny GraniteMoE: %v", err)
	}
	tokens := []int32{1, 5, 9, 17, 23}
	inputs := make([][]byte, len(tokens))
	for i, tokenID := range tokens {
		inputs[i], err = tm.Embed(tokenID)
		if err != nil {
			t.Fatalf("embed token %d: %v", tokenID, err)
		}
	}
	hidden, err := tm.DecodeForward(inputs)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(hidden) != len(tokens) || len(hidden[0]) != 16 {
		t.Fatalf("forward shape = [%d,%d]", len(hidden), len(hidden[0]))
	}

	cfgResult := ParseConfig(config)
	cfg := cfgResult.Value.(*Config)
	arch, _ := cfg.Arch()
	normalized, err := NormalizeWeights(tensors, cfg)
	if err != nil {
		t.Fatalf("normalise: %v", err)
	}
	cm, err := composed.LoadComposedWithArch(normalized, config, arch)
	if err != nil {
		t.Fatalf("inspect composed model: %v", err)
	}
	moe := cm.Layers[0].MLP.(*composed.MoEMLP)
	if cm.EmbedScale != 12 || cm.LogitsScaling != 6 || cm.ResidualScale != .22 {
		t.Fatalf("Granite multipliers = embed %g logits %g residual %g", cm.EmbedScale, cm.LogitsScaling, cm.ResidualScale)
	}
	if moe.TopK != 2 || !moe.NormTopKProb || moe.Shared != nil {
		t.Fatalf("router policy = top-k %d normalise %v shared %v", moe.TopK, moe.NormTopKProb, moe.Shared != nil)
	}

	// Router-distribution receipt over deliberately varied fills. This independently
	// computes Granite's declared top-2 score ordering, while the forward above drives
	// the production router with the same rows.
	distributions := map[[2]int]bool{}
	for fill := float32(-2); fill <= 2; fill++ {
		x := make([]float32, 8)
		for i := range x {
			x[i] = fill + float32(i%3-1)*.25
		}
		best, second := -1, -1
		bestScore, secondScore := float32(-math.MaxFloat32), float32(-math.MaxFloat32)
		for expert := range 4 {
			var score float32
			for d := range 8 {
				score += x[d] * moe.Router[expert*8+d]
			}
			if score > bestScore {
				second, secondScore, best, bestScore = best, bestScore, expert, score
			} else if score > secondScore {
				second, secondScore = expert, score
			}
		}
		distributions[[2]int{best, second}] = true
	}
	if len(distributions) < 2 {
		t.Fatalf("varied fills produced %d router distributions: %v", len(distributions), distributions)
	}
	t.Logf("GraniteMoE router distribution receipt: %v", distributions)
	for row, bytes := range hidden {
		for i := 0; i < len(bytes); i += 2 {
			value := math.Float32frombits(uint32(uint16(bytes[i])|uint16(bytes[i+1])<<8) << 16)
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("hidden row %d contains non-finite value", row)
			}
		}
	}
}
