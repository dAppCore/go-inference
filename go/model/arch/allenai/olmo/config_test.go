// SPDX-Licence-Identifier: EUPL-1.2

package olmo

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

func fixture(t *testing.T, name string) []byte {
	t.Helper()
	r := core.ReadFile(core.PathJoin("testdata", name))
	if !r.OK {
		t.Fatalf("read %s: %v", name, r.Error())
	}
	return r.Value.([]byte)
}

// TestParseConfig_OLMo_Good parses the unmodified allenai/OLMo-1B-hf config.
// Source: https://huggingface.co/allenai/OLMo-1B-hf/blob/main/config.json
func TestParseConfig_OLMo_Good(t *testing.T) {
	cfg, err := ParseConfig(fixture(t, "allenai-olmo-1b-hf-config.json"))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if cfg.ModelType != "olmo" || a.Hidden != 2048 || a.FF != 8192 || a.HeadDim != 128 || a.Vocab != 50304 {
		t.Fatalf("OLMo arch = %+v", a)
	}
	if a.NormPlacement != model.NormPlacementPre || !a.NonParametricLayerNorm || a.Activation != "silu" {
		t.Fatalf("OLMo strategy = placement %q non-parametric %v activation %q", a.NormPlacement, a.NonParametricLayerNorm, a.Activation)
	}
}

// TestParseConfig_OLMo2_Good parses the unmodified allenai/OLMo-2-0425-1B config.
// Source: https://huggingface.co/allenai/OLMo-2-0425-1B/blob/main/config.json
func TestParseConfig_OLMo2_Good(t *testing.T) {
	cfg, err := ParseConfig(fixture(t, "allenai-olmo-2-0425-1b-config.json"))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if cfg.ModelType != "olmo2" || a.RopeBase != 500000 || a.Eps != 1e-6 || a.TieWordEmbeddings == nil || *a.TieWordEmbeddings {
		t.Fatalf("OLMo2 arch = %+v", a)
	}
	if a.NormPlacement != model.NormPlacementPost || a.NonParametricLayerNorm {
		t.Fatalf("OLMo2 strategy = placement %q non-parametric %v", a.NormPlacement, a.NonParametricLayerNorm)
	}
}

func TestParseConfig_Bad(t *testing.T) {
	if _, err := ParseConfig([]byte(`{"model_type":`)); err == nil {
		t.Fatal("ParseConfig accepted malformed JSON")
	}
}

func TestParseConfig_Ugly(t *testing.T) {
	cfg, err := ParseConfig([]byte(`{"model_type":"olmo2"}`))
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("Arch accepted empty geometry")
	}
}

func TestConfigArch_Bad(t *testing.T) {
	valid := Config{ModelType: "olmo2", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 32}
	cases := []Config{
		{ModelType: "unknown", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32},
		{ModelType: "olmo2", HiddenSize: 7, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32},
		{ModelType: "olmo2", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 4, NumKeyValueHeads: 3, VocabSize: 32},
	}
	negativeEps := valid
	negativeEps.RMSNormEps = -1
	cases = append(cases, negativeEps)
	negativeRope := valid
	negativeRope.RopeTheta = -1
	cases = append(cases, negativeRope)
	for i := range cases {
		if _, err := cases[i].Arch(); err == nil {
			t.Fatalf("invalid config %d accepted", i)
		}
	}
}

func TestConfigArch_Ugly(t *testing.T) {
	cfg := Config{ModelType: "olmo", HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32}
	a, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch defaults: %v", err)
	}
	if a.KVHeads != 2 || a.Eps != defaultLayerNormEps || a.RopeBase != defaultRopeTheta || a.Activation != "silu" {
		t.Fatalf("Arch defaults = %+v", a)
	}
}
