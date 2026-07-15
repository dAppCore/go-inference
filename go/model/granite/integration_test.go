// SPDX-Licence-Identifier: EUPL-1.2

package granite_test

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	_ "dappco.re/go/inference/model/granite"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

type seededWeights struct{ state uint32 }

func (s *seededWeights) values(n int) []float32 {
	out := make([]float32, n)
	for i := range out {
		s.state = 1664525*s.state + 1013904223
		out[i] = (float32(s.state>>8)/float32(1<<24) - 0.5) * 0.1
	}
	return out
}

func f32Tensor(values []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		bits := math.Float32bits(value)
		data[4*i], data[4*i+1] = byte(bits), byte(bits>>8)
		data[4*i+2], data[4*i+3] = byte(bits>>16), byte(bits>>24)
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: data}
}

func tinyGraniteWeights() map[string]safetensors.Tensor {
	const hidden, vocab, ff, heads, kvHeads, headDim = 8, 32, 16, 2, 1, 4
	seed := seededWeights{state: 0x6a72616e}
	norm := func() safetensors.Tensor {
		values := seed.values(hidden)
		for i := range values {
			values[i] += 1
		}
		return f32Tensor(values, hidden)
	}
	return map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                      f32Tensor(seed.values(vocab*hidden), vocab, hidden),
		"model.norm.weight":                              norm(),
		"model.layers.0.input_layernorm.weight":          norm(),
		"model.layers.0.post_attention_layernorm.weight": norm(),
		"model.layers.0.self_attn.q_proj.weight":         f32Tensor(seed.values(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":         f32Tensor(seed.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":         f32Tensor(seed.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":         f32Tensor(seed.values(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.mlp.gate_proj.weight":            f32Tensor(seed.values(ff*hidden), ff, hidden),
		"model.layers.0.mlp.up_proj.weight":              f32Tensor(seed.values(ff*hidden), ff, hidden),
		"model.layers.0.mlp.down_proj.weight":            f32Tensor(seed.values(hidden*ff), hidden, ff),
	}
}

// TestTinyGraniteForward_Good is synthetic-only: no public dense Granite checkpoint
// is below 500 MB (the cited 2B fixture's index declares 5,067,051,008 bytes).
func TestTinyGraniteForward_Good(t *testing.T) {
	config := []byte(`{"model_type":"granite","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":32,"rms_norm_eps":0.00001,"rope_theta":10000,"tie_word_embeddings":true,"hidden_act":"silu","logits_scaling":8,"residual_multiplier":0.22,"embedding_multiplier":12,"attention_multiplier":0.5}`)
	tensors := tinyGraniteWeights()
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), string(config)); err != nil {
		t.Fatalf("write config: %v", err)
	}
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		t.Fatalf("encode weights: %v", err)
	}
	if err := coreio.Local.Write(core.PathJoin(dir, "model.safetensors"), string(blob)); err != nil {
		t.Fatalf("write weights: %v", err)
	}
	loaded, mapping, err := model.Load(dir)
	if err != nil {
		t.Fatalf("reactive model.Load: %v", err)
	}
	defer func() { _ = mapping.Close() }()
	if loaded.Arch.LogitsScaling != 8 || loaded.Arch.ResidualMultiplier != 0.22 {
		t.Fatalf("loaded Granite scalars = %+v", loaded.Arch)
	}
	composedModel, err := composed.LoadComposed(tensors, config)
	if err != nil {
		t.Fatalf("load composed Granite: %v", err)
	}
	hidden, err := composed.NewSession(composedModel).Forward([]int32{1, 5, 9})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(hidden) != 24 {
		t.Fatalf("forward shape = %d, want 24", len(hidden))
	}
	for i, value := range hidden {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			t.Fatalf("hidden[%d] is not finite: %g", i, value)
		}
	}
}
