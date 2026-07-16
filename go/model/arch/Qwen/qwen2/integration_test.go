// SPDX-Licence-Identifier: EUPL-1.2

package qwen2_test

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/arch/Qwen/qwen2"
	"dappco.re/go/inference/model/composed"
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

func tinyQwen2Weights() map[string]safetensors.Tensor {
	const hidden, vocab, ff, heads, kvHeads, headDim = 8, 32, 16, 2, 1, 4
	seed := seededWeights{state: 0x25eed123}
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
		"model.layers.0.self_attn.q_proj.bias":           f32Tensor(seed.values(heads*headDim), heads*headDim),
		"model.layers.0.self_attn.k_proj.weight":         f32Tensor(seed.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.bias":           f32Tensor(seed.values(kvHeads*headDim), kvHeads*headDim),
		"model.layers.0.self_attn.v_proj.weight":         f32Tensor(seed.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.bias":           f32Tensor(seed.values(kvHeads*headDim), kvHeads*headDim),
		"model.layers.0.self_attn.o_proj.weight":         f32Tensor(seed.values(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.mlp.gate_proj.weight":            f32Tensor(seed.values(ff*hidden), ff, hidden),
		"model.layers.0.mlp.up_proj.weight":              f32Tensor(seed.values(ff*hidden), ff, hidden),
		"model.layers.0.mlp.down_proj.weight":            f32Tensor(seed.values(hidden*ff), hidden, ff),
	}
}

// TestTinyQwen2Forward_Good is synthetic-only parity: seeded varied weights exercise
// Qwen2's QKV biases through the shared forward path. It is not a checkpoint golden.
func TestTinyQwen2Forward_Good(t *testing.T) {
	config := []byte(`{"model_type":"qwen2","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,` +
		`"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"vocab_size":32,` +
		`"rms_norm_eps":1e-5,"rope_theta":1000000,"tie_word_embeddings":true}`)
	tensors := tinyQwen2Weights()
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
	if loaded.Layers[0].Q.Bias == nil || loaded.Layers[0].K.Bias == nil || loaded.Layers[0].V.Bias == nil {
		t.Fatal("Qwen2 QKV biases were not mapped")
	}

	composedModel, err := composed.LoadComposed(tensors, config)
	if err != nil {
		t.Fatalf("load composed Qwen2: %v", err)
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
