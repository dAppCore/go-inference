// SPDX-Licence-Identifier: EUPL-1.2

package cohere_test

import (
	"math"
	"testing"

	core "dappco.re/go"
	_ "dappco.re/go/inference/model/cohere"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/safetensors"
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
		data[4*i], data[4*i+1], data[4*i+2], data[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: data}
}

func tinyWeights(layers int, qkNorm, untied bool) map[string]safetensors.Tensor {
	const hidden, vocab, ff, heads, kvHeads, headDim = 8, 32, 16, 2, 1, 4
	s := seededWeights{state: 0xc0e2e123}
	norm := func(n int) safetensors.Tensor {
		values := s.values(n)
		for i := range values {
			values[i] += 1
		}
		return f32Tensor(values, n)
	}
	tensors := map[string]safetensors.Tensor{"model.embed_tokens.weight": f32Tensor(s.values(vocab*hidden), vocab, hidden), "model.norm.weight": norm(hidden)}
	if untied {
		tensors["lm_head.weight"] = f32Tensor(s.values(vocab*hidden), vocab, hidden)
	}
	for i := 0; i < layers; i++ {
		p := core.Sprintf("model.layers.%d", i)
		tensors[p+".input_layernorm.weight"] = norm(hidden)
		tensors[p+".self_attn.q_proj.weight"] = f32Tensor(s.values(heads*headDim*hidden), heads*headDim, hidden)
		tensors[p+".self_attn.k_proj.weight"] = f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden)
		tensors[p+".self_attn.v_proj.weight"] = f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden)
		tensors[p+".self_attn.o_proj.weight"] = f32Tensor(s.values(hidden*heads*headDim), hidden, heads*headDim)
		tensors[p+".mlp.gate_proj.weight"] = f32Tensor(s.values(ff*hidden), ff, hidden)
		tensors[p+".mlp.up_proj.weight"] = f32Tensor(s.values(ff*hidden), ff, hidden)
		tensors[p+".mlp.down_proj.weight"] = f32Tensor(s.values(hidden*ff), hidden, ff)
		if qkNorm {
			tensors[p+".self_attn.q_norm.weight"] = norm(headDim)
			tensors[p+".self_attn.k_norm.weight"] = norm(headDim)
		}
	}
	return tensors
}

// TestTinyCohereForward_Good is synthetic-only: seeded varied weights exercise
// per-head LayerNorm, parallel residuals and logit scaling. No sub-500 MB public
// Cohere checkpoint exists, so this is deliberately not a checkpoint golden.
func TestTinyCohereForward_Good(t *testing.T) {
	config := []byte(`{"model_type":"cohere","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":32,"layer_norm_eps":1e-5,"rope_theta":10000,"use_qk_norm":true,"logit_scale":0.25,"tie_word_embeddings":true}`)
	m, err := composed.LoadComposed(tinyWeights(1, true, false), config)
	if err != nil {
		t.Fatalf("load tiny Cohere: %v", err)
	}
	hidden, err := composed.NewSession(m).Forward([]int32{1, 5, 9})
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
	logits := composed.HeadLogitsHost(m, hidden[len(hidden)-8:])
	m.LogitScale = 1
	unscaled := composed.HeadLogitsHost(m, hidden[len(hidden)-8:])
	for i := range logits {
		if math.Abs(float64(logits[i]-0.25*unscaled[i])) > 1e-6 {
			t.Fatalf("logit %d = %g, want 0.25 * %g", i, logits[i], unscaled[i])
		}
	}
}

func TestTinyCohereDefaults_Ugly(t *testing.T) {
	config := []byte(`{"model_type":"cohere","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":32,"rope_theta":10000,"tie_word_embeddings":true}`)
	m, err := composed.LoadComposed(tinyWeights(1, false, false), config)
	if err != nil {
		t.Fatalf("load tiny Cohere defaults: %v", err)
	}
	if m.Eps != 1e-5 || m.LogitScale != 0.0625 {
		t.Fatalf("defaults = eps %g logit scale %g", m.Eps, m.LogitScale)
	}
}

// TestTinyCohere2SlidingInterleave_Good is synthetic-only parity for the existing
// attention mixer's local-window path: layers 0..2 are sliding and layer 3 is full.
func TestTinyCohere2SlidingInterleave_Good(t *testing.T) {
	config := []byte(`{"model_type":"cohere2","hidden_size":8,"intermediate_size":16,"num_hidden_layers":4,"num_attention_heads":2,"num_key_value_heads":1,"vocab_size":32,"layer_norm_eps":1e-5,"rope_theta":10000,"sliding_window":2,"sliding_window_pattern":4,"logit_scale":0.25,"tie_word_embeddings":false}`)
	m, err := composed.LoadComposed(tinyWeights(4, false, true), config)
	if err != nil {
		t.Fatalf("load tiny Cohere2: %v", err)
	}
	for i, want := range []string{"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"} {
		if got := m.Layers[i].Mixer.Kind(); got != want {
			t.Fatalf("layer %d kind = %q, want %q", i, got, want)
		}
	}
	hidden, err := composed.NewSession(m).Forward([]int32{1, 5, 9, 3})
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(hidden) != 32 {
		t.Fatalf("forward shape = %d, want 32", len(hidden))
	}
}
