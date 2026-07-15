// SPDX-Licence-Identifier: EUPL-1.2

package granitemoe

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

func tensor(values []float32, shape ...int) safetensors.Tensor {
	data := make([]byte, len(values)*4)
	for i, value := range values {
		bits := math.Float32bits(value)
		data[4*i], data[4*i+1], data[4*i+2], data[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
	}
	return safetensors.Tensor{Dtype: "F32", Shape: shape, Data: data}
}

func TestNormalizeWeights_Good(t *testing.T) {
	cfg := &Config{HiddenSize: 2, IntermediateSize: 3, NumHiddenLayers: 1, NumLocalExperts: 2}
	in := map[string]safetensors.Tensor{
		"model.layers.0.block_sparse_moe.input_linear.weight":  tensor(make([]float32, 24), 2, 6, 2),
		"model.layers.0.block_sparse_moe.output_linear.weight": tensor(make([]float32, 12), 2, 2, 3),
		"model.layers.0.block_sparse_moe.router.layer.weight":  tensor(make([]float32, 4), 2, 2),
	}
	out, err := NormalizeWeights(in, cfg)
	if err != nil {
		t.Fatalf("NormalizeWeights: %v", err)
	}
	if got := out["model.layers.0.mlp.experts.1.up_proj.weight"]; len(got.Data) != 24 || got.Shape[0] != 3 {
		t.Fatalf("expert up alias = shape %v bytes %d", got.Shape, len(got.Data))
	}
}

func TestNormalizeWeights_Bad(t *testing.T) {
	if _, err := NormalizeWeights(map[string]safetensors.Tensor{}, &Config{NumHiddenLayers: 1, NumLocalExperts: 2}); err == nil {
		t.Fatal("NormalizeWeights accepted missing packed weights")
	}
}

func TestNormalizeWeights_Ugly(t *testing.T) {
	cfg := &Config{HiddenSize: 2, IntermediateSize: 3, NumHiddenLayers: 1, NumLocalExperts: 2}
	in := map[string]safetensors.Tensor{
		"model.layers.0.block_sparse_moe.input_linear.weight":  tensor(nil, 2, 5, 2),
		"model.layers.0.block_sparse_moe.output_linear.weight": tensor(nil, 2, 2, 3),
		"model.layers.0.block_sparse_moe.router.layer.weight":  tensor(nil, 2, 2),
	}
	if _, err := NormalizeWeights(in, cfg); err == nil {
		t.Fatal("NormalizeWeights accepted malformed packed shape")
	}
}

// Index source: https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base/blob/main/model.safetensors.index.json
func TestWeightMapFixture_Good(t *testing.T) {
	data, err := coreio.Local.Read("testdata/ibm-granite-granite-3.1-1b-a400m-base-model.safetensors.index.json")
	if err != nil {
		t.Fatalf("read index fixture: %v", err)
	}
	var index struct {
		Metadata struct {
			TotalSize int64 `json:"total_size"`
		} `json:"metadata"`
		WeightMap map[string]string `json:"weight_map"`
	}
	if r := core.JSONUnmarshal([]byte(data), &index); !r.OK {
		t.Fatalf("parse index fixture: %s", r.Error())
	}
	for _, name := range []string{"input_linear.weight", "output_linear.weight", "router.layer.weight"} {
		key := "model.layers.0.block_sparse_moe." + name
		if index.WeightMap[key] == "" {
			t.Fatalf("index missing %s", key)
		}
	}
	if index.Metadata.TotalSize != 2669250560 {
		t.Fatalf("total_size = %d", index.Metadata.TotalSize)
	}
}

func TestRegister_Good(t *testing.T) {
	spec, ok := model.LookupArch("granitemoe")
	if !ok || spec.Parse == nil || spec.Composed == nil {
		t.Fatalf("granitemoe registration = ok %v parse %v composed %v", ok, spec.Parse != nil, spec.Composed != nil)
	}
}
