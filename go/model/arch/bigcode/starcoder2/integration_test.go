// SPDX-Licence-Identifier: EUPL-1.2

package starcoder2

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
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

func tinyStarCoder2Weights() map[string]safetensors.Tensor {
	const hidden, vocab, ff, heads, kvHeads, headDim = 8, 32, 16, 2, 1, 4
	seed := seededWeights{state: 0x5ca2c0de}
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
		"model.layers.0.mlp.c_fc.weight":                 f32Tensor(seed.values(ff*hidden), ff, hidden),
		"model.layers.0.mlp.c_proj.weight":               f32Tensor(seed.values(hidden*ff), hidden, ff),
	}
}

// TestTinyStarCoder2Forward_Good is synthetic-only: seeded varied weights exercise
// the reactive StarCoder2 load and existing GQA/rotary KV cache. It is not a checkpoint golden.
func TestTinyStarCoder2Forward_Good(t *testing.T) {
	config := []byte(`{"model_type":"starcoder2","hidden_size":8,"intermediate_size":16,` +
		`"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,` +
		`"max_position_embeddings":16,"sliding_window":4,"vocab_size":32,` +
		`"norm_epsilon":1e-5,"rope_theta":1000000,"hidden_act":"gelu_pytorch_tanh"}`)
	tensors := tinyStarCoder2Weights()
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
	if loaded.Arch.SlidingWindow != 4 || loaded.Arch.Layer[0].Attention != model.SlidingAttention || loaded.Layers[0].K == nil || loaded.Layers[0].V == nil {
		t.Fatalf("loaded StarCoder2 cache declaration = arch %+v layer %+v", loaded.Arch, loaded.Layers[0])
	}

	seed := seededWeights{state: 0x51deca5e}
	mixer := composed.NewAttnMixer(&composed.AttnWeights{
		QProj: seed.values(2 * 4 * 8), KProj: seed.values(1 * 4 * 8),
		VProj: seed.values(1 * 4 * 8), OProj: seed.values(8 * 2 * 4),
	}, composed.AttnConfig{Heads: 2, KVHeads: 1, HeadDim: 4, RotaryDim: 4, RopeTheta: 1_000_000, NormEps: 1e-5})
	hidden := seed.values(6 * 8)
	prefill, _, err := mixer.Forward(hidden, 6, 8, nil)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	var cache any
	for token := range 6 {
		decoded, next, err := mixer.Forward(hidden[token*8:(token+1)*8], 1, 8, cache)
		if err != nil {
			t.Fatalf("decode token %d: %v", token, err)
		}
		cache = next
		for i := range 8 {
			if math.Float32bits(decoded[i]) != math.Float32bits(prefill[token*8+i]) {
				t.Fatalf("token %d dim %d cache=%g prefill=%g", token, i, decoded[i], prefill[token*8+i])
			}
		}
	}
}
