// SPDX-Licence-Identifier: EUPL-1.2

package qwenmoe

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	coreio "dappco.re/go/io"
)

func TestRegister_Good(t *testing.T) {
	for _, modelType := range []string{"qwen2_moe", "qwen3_moe"} {
		spec, ok := model.LookupArch(modelType)
		if !ok || spec.Parse == nil || spec.Composed == nil {
			t.Fatalf("%s registration = found %v spec %+v", modelType, ok, spec)
		}
	}
}

func TestRegister_Bad(t *testing.T) {
	spec, _ := model.LookupArch("qwen2_moe")
	if _, err := spec.Parse([]byte("{")); err == nil {
		t.Fatal("malformed config accepted")
	}
}

func TestRegister_Ugly(t *testing.T) {
	spec, _ := model.LookupArch("qwen3_moe")
	parsed, err := spec.Parse([]byte(`{"model_type":"qwen3_moe","hidden_size":8}`))
	if err != nil {
		t.Fatalf("valid JSON rejected: %v", err)
	}
	if _, err := parsed.Arch(); err == nil {
		t.Fatal("incomplete config accepted by Arch")
	}
}

// Fixture sources:
// https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B/blob/main/model.safetensors.index.json
// https://huggingface.co/Qwen/Qwen3-30B-A3B/blob/main/model.safetensors.index.json
func TestWeightMap_Good(t *testing.T) {
	tests := []struct {
		name       string
		total      int64
		count      int
		sharedGate bool
	}{
		{"Qwen-Qwen1.5-MoE-A2.7B-model.safetensors.index.json", 28_631_568_384, 4659, true},
		{"Qwen-Qwen3-30B-A3B-model.safetensors.index.json", 61_064_245_248, 18867, false},
	}
	for _, test := range tests {
		data, err := coreio.Local.Read(core.PathJoin("testdata", test.name))
		if err != nil {
			t.Fatalf("read %s: %v", test.name, err)
		}
		var index struct {
			Metadata struct {
				TotalSize int64 `json:"total_size"`
			} `json:"metadata"`
			WeightMap map[string]string `json:"weight_map"`
		}
		if r := core.JSONUnmarshal([]byte(data), &index); !r.OK {
			t.Fatalf("parse %s: %s", test.name, r.Error())
		}
		if index.Metadata.TotalSize != test.total || len(index.WeightMap) != test.count {
			t.Fatalf("%s receipt = size %d weights %d", test.name, index.Metadata.TotalSize, len(index.WeightMap))
		}
		for _, suffix := range []string{"gate.weight", "experts.0.gate_proj.weight", "experts.0.up_proj.weight", "experts.0.down_proj.weight"} {
			if index.WeightMap["model.layers.0.mlp."+suffix] == "" {
				t.Fatalf("%s lacks layer-0 %s", test.name, suffix)
			}
		}
		_, hasSharedGate := index.WeightMap["model.layers.0.mlp.shared_expert_gate.weight"]
		if hasSharedGate != test.sharedGate {
			t.Fatalf("%s shared expert gate = %v, want %v", test.name, hasSharedGate, test.sharedGate)
		}
	}
}
