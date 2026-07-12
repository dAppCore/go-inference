// SPDX-Licence-Identifier: EUPL-1.2

package llama4_test

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	_ "dappco.re/go/inference/model/llama4"
	"dappco.re/go/inference/model/safetensors"
)

func tensor(n int, shape ...int) safetensors.Tensor {
	data := make([]byte, n*2)
	state := uint32(n + 17)
	for i := range n {
		state = 1664525*state + 1013904223
		value := uint16(0x3c00 + (state>>29)&3)
		data[2*i], data[2*i+1] = byte(value), byte(value>>8)
	}
	return safetensors.Tensor{Dtype: "BF16", Shape: shape, Data: data}
}

func tinyWeights() map[string]safetensors.Tensor {
	const d, vocab, expertFF, denseFF, heads, kvHeads, headDim, experts = 8, 32, 4, 12, 2, 1, 4, 2
	norm := func() safetensors.Tensor { return tensor(d, d) }
	w := map[string]safetensors.Tensor{
		"language_model.model.embed_tokens.weight": tensor(vocab*d, vocab, d),
		"language_model.model.norm.weight":         norm(), "language_model.lm_head.weight": tensor(vocab*d, vocab, d),
	}
	for layer := range 2 {
		p := core.Sprintf("language_model.model.layers.%d.", layer)
		w[p+"input_layernorm.weight"], w[p+"post_attention_layernorm.weight"] = norm(), norm()
		w[p+"self_attn.q_proj.weight"] = tensor(heads*headDim*d, heads*headDim, d)
		w[p+"self_attn.k_proj.weight"] = tensor(kvHeads*headDim*d, kvHeads*headDim, d)
		w[p+"self_attn.v_proj.weight"] = tensor(kvHeads*headDim*d, kvHeads*headDim, d)
		w[p+"self_attn.o_proj.weight"] = tensor(d*heads*headDim, d, heads*headDim)
		if layer == 0 {
			w[p+"feed_forward.gate_proj.weight"] = tensor(denseFF*d, denseFF, d)
			w[p+"feed_forward.up_proj.weight"] = tensor(denseFF*d, denseFF, d)
			w[p+"feed_forward.down_proj.weight"] = tensor(d*denseFF, d, denseFF)
		} else {
			w[p+"feed_forward.router.weight"] = tensor(experts*d, experts, d)
			w[p+"feed_forward.experts.gate_up_proj"] = tensor(experts*d*2*expertFF, experts, d, 2*expertFF)
			w[p+"feed_forward.experts.down_proj"] = tensor(experts*expertFF*d, experts, expertFF, d)
			w[p+"feed_forward.shared_expert.gate_proj.weight"] = tensor(expertFF*d, expertFF, d)
			w[p+"feed_forward.shared_expert.up_proj.weight"] = tensor(expertFF*d, expertFF, d)
			w[p+"feed_forward.shared_expert.down_proj.weight"] = tensor(d*expertFF, d, expertFF)
		}
	}
	return w
}

func TestTinyLlama4TextForwardAndGenerate_Good(t *testing.T) {
	config := []byte(`{"model_type":"llama4","text_config":{"model_type":"llama4_text","hidden_size":8,"intermediate_size":4,"intermediate_size_mlp":12,"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,"num_local_experts":2,"num_experts_per_tok":1,"moe_layers":[1],"no_rope_layers":[1,0],"vocab_size":32,"rms_norm_eps":1e-5,"rope_theta":500000,"use_qk_norm":true},"tie_word_embeddings":false,"vision_config":{"model_type":"llama4_vision_model"}}`)
	spec, ok := model.LookupArch("llama4")
	if !ok {
		t.Fatal("Llama 4 architecture not registered")
	}
	tm, err := spec.Composed(tinyWeights(), config)
	if err != nil {
		t.Fatalf("load tiny Llama 4 text model: %v", err)
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
		t.Fatalf("forward shape = [%d,%d], want [3,16]", len(hidden), len(hidden[0]))
	}
	generated, err := model.Generate(tm, []int32{1, 5, 9}, 3, -1)
	if err != nil || len(generated) != 3 {
		t.Fatalf("generate = %v, %v", generated, err)
	}
	// vision_config is intentionally present: this receipt covers sparse text only.
}
