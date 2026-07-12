// SPDX-Licence-Identifier: EUPL-1.2

package llama_test

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

func tinyLlamaWeights(tied bool) map[string]safetensors.Tensor {
	const hidden, vocab, ff, heads, kvHeads, headDim = 8, 32, 16, 2, 1, 4
	s := seededWeights{state: 0x5eed1234}
	norm := func() safetensors.Tensor {
		values := s.values(hidden)
		for i := range values {
			values[i] += 1
		}
		return f32Tensor(values, hidden)
	}
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                      f32Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.norm.weight":                              norm(),
		"model.layers.0.input_layernorm.weight":          norm(),
		"model.layers.0.post_attention_layernorm.weight": norm(),
		"model.layers.0.self_attn.q_proj.weight":         f32Tensor(s.values(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":         f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":         f32Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":         f32Tensor(s.values(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.mlp.gate_proj.weight":            f32Tensor(s.values(ff*hidden), ff, hidden),
		"model.layers.0.mlp.up_proj.weight":              f32Tensor(s.values(ff*hidden), ff, hidden),
		"model.layers.0.mlp.down_proj.weight":            f32Tensor(s.values(hidden*ff), hidden, ff),
	}
	if !tied {
		tensors["lm_head.weight"] = f32Tensor(s.values(vocab*hidden), vocab, hidden)
	}
	return tensors
}

func TestTinyLlamaForwardAndGenerate_Good(t *testing.T) {
	const configPrefix = `{"model_type":"llama","hidden_size":8,"intermediate_size":16,"num_hidden_layers":1,` +
		`"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"vocab_size":32,` +
		`"rms_norm_eps":1e-5,"rope_theta":500000,"rope_scaling":{"rope_type":"llama3","factor":8,` +
		`"low_freq_factor":1,"high_freq_factor":4,"original_max_position_embeddings":8192},` +
		`"layer_types":["full_attention"],"tie_word_embeddings":`
	for _, tied := range []bool{true, false} {
		name := "untied"
		if tied {
			name = "tied"
		}
		t.Run(name, func(t *testing.T) {
			config := configPrefix + core.Sprintf("%t}", tied)
			tensors := tinyLlamaWeights(tied)
			dir := t.TempDir()
			if err := coreio.Local.Write(core.PathJoin(dir, "config.json"), config); err != nil {
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
			if loaded.Tied() != tied || len(loaded.Arch.RopeFreqs) != 2 {
				t.Fatalf("reactive load = tied %v rope freqs %d", loaded.Tied(), len(loaded.Arch.RopeFreqs))
			}

			m, err := composed.LoadComposed(tensors, []byte(config))
			if err != nil {
				t.Fatalf("load tiny Llama: %v", err)
			}
			if (m.Output == nil) != tied {
				t.Fatalf("Output nil = %v, want tied = %v", m.Output == nil, tied)
			}
			hidden, err := composed.NewSession(m).Forward([]int32{1, 5, 9})
			if err != nil {
				t.Fatalf("forward: %v", err)
			}
			if len(hidden) != 3*8 {
				t.Fatalf("forward shape = %d, want [3,8]", len(hidden))
			}
			for i, value := range hidden {
				if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
					t.Fatalf("hidden[%d] is not finite: %g", i, value)
				}
			}
			generated, err := composed.NewSession(m).Generate([]int32{1, 5, 9}, 4, -1)
			if err != nil {
				t.Fatalf("generate: %v", err)
			}
			if len(generated) != 4 {
				t.Fatalf("generated %d tokens, want 4", len(generated))
			}
		})
	}
}

func TestTinyLlamaHeadDeclarationMismatch_Bad(t *testing.T) {
	for _, tc := range []struct {
		declaredTied, checkpointTied bool
	}{
		{declaredTied: true, checkpointTied: false},
		{declaredTied: false, checkpointTied: true},
	} {
		arch := model.Arch{
			Hidden: 8, Heads: 2, KVHeads: 1, HeadDim: 4, GlobalHeadDim: 4, GlobalKVHeads: 1,
			FF: 16, Vocab: 32, TieWordEmbeddings: &tc.declaredTied,
		}
		if _, err := model.Assemble(tinyLlamaWeights(tc.checkpointTied), arch, model.StandardWeightNames()); err == nil {
			t.Fatalf("declared tied %v accepted checkpoint tied %v", tc.declaredTied, tc.checkpointTied)
		}
	}
}
