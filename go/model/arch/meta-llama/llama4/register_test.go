// SPDX-Licence-Identifier: EUPL-1.2

package llama4

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// Fixture excerpt source: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/model.safetensors.index.json
func TestWeightMap_Good(t *testing.T) {
	data, err := coreio.Local.Read(core.PathJoin("testdata", "meta-llama-llama-4-scout-17b-16e-instruct-model-index-excerpt.json"))
	if err != nil {
		t.Fatalf("read index excerpt: %v", err)
	}
	var index struct {
		Metadata struct {
			TotalSize         int64 `json:"total_size"`
			SourceWeightCount int   `json:"source_weight_count"`
		} `json:"metadata"`
		WeightMap map[string]string `json:"weight_map"`
	}
	if r := core.JSONUnmarshal([]byte(data), &index); !r.OK {
		t.Fatalf("parse index excerpt: %s", r.Error())
	}
	if index.Metadata.TotalSize != 217_283_587_072 || index.Metadata.SourceWeightCount != 1133 || len(index.WeightMap) != 6 {
		t.Fatalf("index receipt = size %d source weights %d excerpt weights %d", index.Metadata.TotalSize, index.Metadata.SourceWeightCount, len(index.WeightMap))
	}
	for name, shard := range index.WeightMap {
		if !core.HasPrefix(name, "language_model.model.layers.0.feed_forward.") || shard != "model-00002-of-00050.safetensors" {
			t.Fatalf("unexpected index mapping %q -> %q", name, shard)
		}
	}
}

func TestLlama4Registered_Good(t *testing.T) {
	for _, modelType := range []string{"llama4", "llama4_text"} {
		spec, ok := model.LookupArch(modelType)
		if !ok || spec.Parse == nil {
			t.Fatalf("Llama 4 registration %q = found %v spec %+v", modelType, ok, spec)
		}
	}
}

func TestParse_Bad(t *testing.T) {
	spec, _ := model.LookupArch("llama4")
	if _, err := spec.Parse([]byte("{")); err == nil {
		t.Fatal("malformed config accepted")
	}
}

func TestParse_Ugly(t *testing.T) {
	spec, _ := model.LookupArch("llama4")
	parsed, err := spec.Parse([]byte(`{"model_type":"llama4","text_config":{"hidden_size":8}}`))
	if err != nil {
		t.Fatalf("valid wrapper rejected: %v", err)
	}
	if _, err := parsed.Arch(); err == nil {
		t.Fatal("incomplete text_config accepted by Arch")
	}
}

func TestRegister_NormalizeWeights_Good(t *testing.T) {
	values := func(n int) []byte {
		out := make([]byte, n*4)
		for i := range n {
			bits := math.Float32bits(float32(i + 1))
			out[4*i], out[4*i+1], out[4*i+2], out[4*i+3] = byte(bits), byte(bits>>8), byte(bits>>16), byte(bits>>24)
		}
		return out
	}
	in := map[string]safetensors.Tensor{
		"language_model.model.layers.0.feed_forward.router.weight":                {Dtype: "F32", Shape: []int{2, 2}, Data: values(4)},
		"language_model.model.layers.0.feed_forward.experts.gate_up_proj":         {Dtype: "F32", Shape: []int{2, 2, 4}, Data: values(16)},
		"language_model.model.layers.0.feed_forward.experts.down_proj":            {Dtype: "F32", Shape: []int{2, 2, 2}, Data: values(8)},
		"language_model.model.layers.0.feed_forward.shared_expert.up_proj.weight": {Dtype: "F32", Shape: []int{4, 2}, Data: values(8)},
	}
	out, err := NormalizeWeights(in)
	if err != nil {
		t.Fatalf("NormalizeWeights: %v", err)
	}
	for _, name := range []string{
		"language_model.model.layers.0.mlp.gate.weight",
		"language_model.model.layers.0.mlp.experts.0.gate_proj.weight",
		"language_model.model.layers.0.mlp.experts.0.up_proj.weight",
		"language_model.model.layers.0.mlp.experts.0.down_proj.weight",
		"language_model.model.layers.0.mlp.experts.1.gate_proj.weight",
		"language_model.model.layers.0.mlp.shared_expert.up_proj.weight",
	} {
		if _, ok := out[name]; !ok {
			t.Errorf("missing normalised weight %q", name)
		}
	}
	gate := out["language_model.model.layers.0.mlp.experts.0.gate_proj.weight"]
	if len(gate.Shape) != 2 || gate.Shape[0] != 2 || gate.Shape[1] != 2 {
		t.Fatalf("expert gate shape = %v, want [2 2]", gate.Shape)
	}
}

func TestRegister_NormalizeWeights_Bad(t *testing.T) {
	_, err := NormalizeWeights(map[string]safetensors.Tensor{"model.layers.0.feed_forward.experts.gate_up_proj": {Dtype: "F32", Shape: []int{2, 3}}})
	if err == nil {
		t.Fatal("malformed packed experts accepted")
	}
}

func TestRegister_NormalizeWeights_Ugly(t *testing.T) {
	_, err := NormalizeWeights(map[string]safetensors.Tensor{"model.layers.0.feed_forward.experts.gate_up_proj": {Dtype: "I8", Shape: []int{1, 2, 4}, Data: make([]byte, 8)}})
	if err == nil {
		t.Fatal("unsupported expert dtype accepted")
	}
}
