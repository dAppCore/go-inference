// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe

import (
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

func TestRegister_LookupArch_Good(t *testing.T) {
	spec, ok := model.LookupArch("jetmoe")
	if !ok {
		t.Fatal("jetmoe architecture not registered")
	}
	parsed, err := spec.Parse([]byte(`{"model_type":"jetmoe","hidden_size":8}`))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := parsed.(*Config); !ok {
		t.Fatalf("parsed config = %T, want *jetmoe.Config", parsed)
	}
}

func TestRegister_LookupArch_Bad(t *testing.T) {
	spec, _ := model.LookupArch("jetmoe")
	if _, err := spec.Parse([]byte(`{"model_type":`)); err == nil {
		t.Fatal("malformed config accepted")
	}
}

func TestRegister_JetMoE_Ugly(t *testing.T) {
	spec, _ := model.LookupArch("jetmoe")
	config := []byte(`{"model_type":"jetmoe","hidden_size":2,"ffn_hidden_size":1,"num_hidden_layers":1,"num_attention_heads":1,"moe_num_experts":1,"moe_top_k":1,"vocab_size":4}`)
	tensors := map[string]safetensors.Tensor{
		"model.layers.0.mlp.input_linear.weight":                    {Dtype: "BF16", Shape: []int{1, 2, 2}, Data: make([]byte, 8)},
		"model.layers.0.mlp.output_linear.weight":                   {Dtype: "BF16", Shape: []int{1, 2, 1}, Data: make([]byte, 4)},
		"model.layers.0.mlp.router.layer.weight":                    {Dtype: "BF16", Shape: []int{1, 2}, Data: make([]byte, 4)},
		"model.layers.0.self_attention.experts.input_linear.weight": {Dtype: "BF16"},
	}
	if _, err := spec.Composed(tensors, config); err == nil || err.Error() != "jetmoe.Load: MoA requires routed query/output attention projections with shared KV; composed attention does not yet expose that primitive" {
		t.Fatalf("MoA gap error = %v", err)
	}
}
