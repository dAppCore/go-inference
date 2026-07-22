// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

func TestAdaptFFNWeights_WeightMap_Good(t *testing.T) {
	data, err := coreio.Local.Read(core.PathJoin("testdata", "jetmoe-jetmoe-8b-model.safetensors.index.json"))
	if err != nil {
		t.Fatal(err)
	}
	var index struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if r := core.JSONUnmarshal([]byte(data), &index); !r.OK {
		t.Fatal(r.Error())
	}
	for _, name := range []string{
		"model.layers.0.mlp.input_linear.weight",
		"model.layers.0.mlp.output_linear.weight",
		"model.layers.0.mlp.router.layer.weight",
		"model.layers.0.self_attention.experts.input_linear.weight",
		"model.layers.0.self_attention.experts.output_linear.weight",
		"model.layers.0.self_attention.experts.router.layer.weight",
		"model.layers.0.self_attention.kv_proj.weight",
	} {
		if index.WeightMap[name] == "" {
			t.Fatalf("published INDEX fixture lacks %s", name)
		}
	}
}

func TestAdaptFFNWeights_Packed_Good(t *testing.T) {
	cfg := Config{HiddenSize: 2, FFNHiddenSize: 3, NumHiddenLayers: 1, MoENumExperts: 2}
	input := make([]byte, 2*2*3*2*2)
	output := make([]byte, 2*2*3*2)
	for i := range input {
		input[i] = byte(i)
	}
	for i := range output {
		output[i] = byte(100 + i)
	}
	tensors := map[string]safetensors.Tensor{
		"model.layers.0.mlp.input_linear.weight":  {Dtype: "BF16", Shape: []int{2, 6, 2}, Data: input},
		"model.layers.0.mlp.output_linear.weight": {Dtype: "BF16", Shape: []int{2, 2, 3}, Data: output},
		"model.layers.0.mlp.router.layer.weight":  {Dtype: "BF16", Shape: []int{2, 2}, Data: make([]byte, 8)},
	}
	adapted, err := adaptFFNWeights(tensors, cfg)
	if err != nil {
		t.Fatal(err)
	}
	gate := adapted["model.layers.0.mlp.experts.1.gate_proj.weight"]
	up := adapted["model.layers.0.mlp.experts.1.up_proj.weight"]
	down := adapted["model.layers.0.mlp.experts.1.down_proj.weight"]
	if len(gate.Data) != 12 || gate.Data[0] != 24 || up.Data[0] != 36 || down.Data[0] != 112 {
		t.Fatalf("expert views = gate %d/%d up %d down %d", len(gate.Data), gate.Data[0], up.Data[0], down.Data[0])
	}
}

func TestAdaptFFNWeights_Packed_Bad(t *testing.T) {
	_, err := adaptFFNWeights(map[string]safetensors.Tensor{}, Config{NumHiddenLayers: 1})
	if err == nil {
		t.Fatal("missing packed weights accepted")
	}
}

func TestAdaptFFNWeights_Packed_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 2, FFNHiddenSize: 3, NumHiddenLayers: 1, MoENumExperts: 2}
	tensors := map[string]safetensors.Tensor{
		"model.layers.0.mlp.input_linear.weight":  {Dtype: "BF16", Shape: []int{2, 5, 2}},
		"model.layers.0.mlp.output_linear.weight": {Dtype: "BF16", Shape: []int{2, 2, 3}},
		"model.layers.0.mlp.router.layer.weight":  {Dtype: "BF16", Shape: []int{2, 2}},
	}
	_, err := adaptFFNWeights(tensors, cfg)
	if err == nil {
		t.Fatal("malformed packed shape accepted")
	}
}
