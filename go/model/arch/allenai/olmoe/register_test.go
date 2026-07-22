// SPDX-Licence-Identifier: EUPL-1.2

package olmoe

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	coreio "dappco.re/go/io"
)

// Fixture source: https://huggingface.co/allenai/OLMoE-1B-7B-0924/blob/main/model.safetensors.index.json
func TestWeightMap_Good(t *testing.T) {
	data, err := coreio.Local.Read(core.PathJoin("testdata", "allenai-olmoe-1b-7b-0924-model.safetensors.index.json"))
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
	if index.Metadata.TotalSize != 13_838_323_712 || len(index.WeightMap) != 3219 {
		t.Fatalf("index receipt = total size %d weights %d", index.Metadata.TotalSize, len(index.WeightMap))
	}
	want := map[string]string{
		"model.layers.0.mlp.gate.weight":                "model-00001-of-00003.safetensors",
		"model.layers.0.mlp.experts.0.gate_proj.weight": "model-00001-of-00003.safetensors",
		"model.layers.0.mlp.experts.0.down_proj.weight": "model-00001-of-00003.safetensors",
		"model.layers.0.mlp.experts.0.up_proj.weight":   "model-00001-of-00003.safetensors",
	}
	for name, shard := range want {
		if index.WeightMap[name] != shard {
			t.Fatalf("weight_map[%q] = %q, want %q", name, index.WeightMap[name], shard)
		}
	}
}

func TestOLMoERegistered_Good(t *testing.T) {
	spec, ok := model.LookupArch("olmoe")
	if !ok || spec.Parse == nil {
		t.Fatalf("OLMoE registration = found %v spec %+v", ok, spec)
	}
}

func TestParse_Bad(t *testing.T) {
	spec, _ := model.LookupArch("olmoe")
	if _, err := spec.Parse([]byte("{")); err == nil {
		t.Fatal("malformed config accepted")
	}
}

func TestParse_Ugly(t *testing.T) {
	spec, _ := model.LookupArch("olmoe")
	parsed, err := spec.Parse([]byte(`{"model_type":"olmoe","hidden_size":8}`))
	if err != nil {
		t.Fatalf("valid JSON rejected: %v", err)
	}
	if _, err := parsed.Arch(); err == nil {
		t.Fatal("incomplete config accepted by Arch")
	}
}
