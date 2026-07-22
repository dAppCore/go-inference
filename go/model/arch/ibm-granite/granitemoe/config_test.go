// SPDX-Licence-Identifier: EUPL-1.2

package granitemoe

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// Fixture source: https://huggingface.co/ibm-granite/granite-3.1-1b-a400m-base/blob/main/config.json
func TestConfig_ParseConfig_Good(t *testing.T) {
	data, err := coreio.Local.Read("testdata/ibm-granite-granite-3.1-1b-a400m-base-config.json")
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	r := ParseConfig([]byte(data))
	if !r.OK {
		t.Fatalf("ParseConfig: %s", r.Error())
	}
	cfg := r.Value.(*Config)
	if cfg.NumLocalExperts != 32 || cfg.NumExpertsPerTok != 8 || cfg.LogitsScaling != 6 {
		t.Fatalf("parsed policy = experts %d top-k %d logits %g", cfg.NumLocalExperts, cfg.NumExpertsPerTok, cfg.LogitsScaling)
	}
}

func TestConfig_ParseConfig_Bad(t *testing.T) {
	if r := ParseConfig([]byte(`{"model_type":`)); r.OK {
		t.Fatal("ParseConfig malformed JSON succeeded")
	}
}

func TestConfig_ParseConfig_Ugly(t *testing.T) {
	r := ParseConfig([]byte(`{}`))
	if !r.OK {
		t.Fatalf("ParseConfig empty object parse: %s", r.Error())
	}
	if _, err := r.Value.(*Config).Arch(); err == nil {
		t.Fatal("empty config Arch succeeded")
	}
}

func TestConfig_Arch_Good(t *testing.T) {
	cfg := Config{ModelType: "granitemoe", HiddenSize: 8, IntermediateSize: 4, NumHiddenLayers: 2, NumAttentionHeads: 2, NumKeyValueHeads: 1, NumLocalExperts: 4, NumExpertsPerTok: 2, VocabSize: 32, RMSNormEps: 1e-5, RopeTheta: 10000, LogitsScaling: 6, ResidualMultiplier: .22, EmbeddingMultiplier: 12, AttentionMultiplier: .125}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Experts != 4 || arch.TopK != 2 || !arch.NormaliseMoETopK || arch.SharedExperts != 0 || arch.LogitsScaling != 6 || arch.ResidualMultiplier != .22 || !arch.HasMoE() {
		t.Fatalf("GraniteMoE architecture = %+v", arch)
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	cfg := Config{ModelType: "granite"}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("Arch accepted dense model_type")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{ModelType: "granitemoe", HiddenSize: 8, IntermediateSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, NumLocalExperts: 2, NumExpertsPerTok: 3, VocabSize: 32, RMSNormEps: 1e-5, RopeTheta: 10000, LogitsScaling: 6, ResidualMultiplier: .22, EmbeddingMultiplier: 12, AttentionMultiplier: .125}
	if _, err := cfg.Arch(); err == nil || !core.Contains(err.Error(), "exceeds") {
		t.Fatalf("Arch over-wide top-k error = %v", err)
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != 8 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}

func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{ModelType: "granite"}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("dense model_type became valid after InferFromWeights")
	}
}

// TestConfig_InferFromWeights_Ugly proves the no-op does not paper over the
// top-k-exceeds-experts guard — distinct from _Bad's wrong-model_type
// rejection.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{ModelType: "granitemoe", HiddenSize: 8, IntermediateSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, NumLocalExperts: 2, NumExpertsPerTok: 3, VocabSize: 32, RMSNormEps: 1e-5, RopeTheta: 10000, LogitsScaling: 6, ResidualMultiplier: .22, EmbeddingMultiplier: 12, AttentionMultiplier: .125}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("top-k exceeding experts became valid after InferFromWeights")
	}
}
