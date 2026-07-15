// SPDX-Licence-Identifier: EUPL-1.2

package mixtral_test

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/mixtral"
	"dappco.re/go/inference/model/safetensors"
)

type seededWeights struct{ state uint32 }

func (s *seededWeights) values(n int) []uint16 {
	out := make([]uint16, n)
	for i := range out {
		s.state = 1664525*s.state + 1013904223
		out[i] = 0x3c00 + uint16((s.state>>28)&7)
	}
	return out
}

func bf16Tensor(values []uint16, shape ...int) safetensors.Tensor {
	data := make([]byte, len(values)*2)
	for i, value := range values {
		data[2*i], data[2*i+1] = byte(value), byte(value>>8)
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
}

func tinyMixtralWeights() map[string]safetensors.Tensor {
	const hidden, vocab, expertFF, heads, kvHeads, headDim, experts = 8, 32, 12, 2, 1, 4, 2
	s := seededWeights{state: 0x5eed1234}
	norm := func() safetensors.Tensor {
		values := make([]uint16, hidden)
		for i := range values {
			values[i] = 0x3f80
		}
		return bf16Tensor(values, hidden)
	}
	tensors := map[string]safetensors.Tensor{
		"model.embed_tokens.weight":                      bf16Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.norm.weight":                              norm(),
		"lm_head.weight":                                 bf16Tensor(s.values(vocab*hidden), vocab, hidden),
		"model.layers.0.input_layernorm.weight":          norm(),
		"model.layers.0.post_attention_layernorm.weight": norm(),
		"model.layers.0.self_attn.q_proj.weight":         bf16Tensor(s.values(heads*headDim*hidden), heads*headDim, hidden),
		"model.layers.0.self_attn.k_proj.weight":         bf16Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.v_proj.weight":         bf16Tensor(s.values(kvHeads*headDim*hidden), kvHeads*headDim, hidden),
		"model.layers.0.self_attn.o_proj.weight":         bf16Tensor(s.values(hidden*heads*headDim), hidden, heads*headDim),
		"model.layers.0.block_sparse_moe.gate.weight":    bf16Tensor(s.values(experts*hidden), experts, hidden),
	}
	for expert := range experts {
		prefix := core.Sprintf("model.layers.0.block_sparse_moe.experts.%d", expert)
		tensors[prefix+".w1.weight"] = bf16Tensor(s.values(expertFF*hidden), expertFF, hidden)
		tensors[prefix+".w2.weight"] = bf16Tensor(s.values(hidden*expertFF), hidden, expertFF)
		tensors[prefix+".w3.weight"] = bf16Tensor(s.values(expertFF*hidden), expertFF, hidden)
	}
	return tensors
}

func TestTinyMixtralForwardAndGenerate_Good(t *testing.T) {
	config := []byte(`{"model_type":"mixtral","hidden_size":8,"intermediate_size":12,"num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,"num_local_experts":2,"num_experts_per_tok":1,"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":10000,"tie_word_embeddings":false}`)
	spec, ok := model.LookupArch("mixtral")
	if !ok {
		t.Fatal("Mixtral architecture not registered")
	}
	tm, err := spec.Composed(tinyMixtralWeights(), config)
	if err != nil {
		t.Fatalf("load tiny Mixtral: %v", err)
	}
	inputs := make([][]byte, 3)
	for i, token := range []int32{1, 5, 9} {
		inputs[i], err = tm.Embed(token)
		if err != nil {
			t.Fatalf("embed token %d: %v", token, err)
		}
	}
	hidden, err := tm.DecodeForward(inputs)
	if err != nil {
		t.Fatalf("forward: %v", err)
	}
	if len(hidden) != 3 || len(hidden[0]) != 16 {
		t.Fatalf("forward shape = [%d,%d], want [3,16 bf16 bytes]", len(hidden), len(hidden[0]))
	}
	generated, err := model.Generate(tm, []int32{1, 5, 9}, 4, -1)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(generated) != 4 {
		t.Fatalf("generated %d tokens, want 4", len(generated))
	}
}
