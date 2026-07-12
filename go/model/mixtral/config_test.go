// SPDX-Licence-Identifier: EUPL-1.2

package mixtral

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// Fixture source: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1/blob/main/config.json
func TestConfig_Arch_Good(t *testing.T) {
	data, err := coreio.Local.Read(core.PathJoin("testdata", "mistralai-mixtral-8x7b-v0.1-config.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cfg Config
	if r := core.JSONUnmarshal([]byte(data), &cfg); !r.OK {
		t.Fatalf("parse fixture: %s", r.Error())
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 4096 || arch.Heads != 32 || arch.KVHeads != 8 || arch.HeadDim != 128 {
		t.Fatalf("attention geometry = %+v", arch)
	}
	if arch.Experts != 8 || arch.TopK != 2 || arch.ExpertFF != 14336 || !arch.HasMoE() {
		t.Fatalf("MoE geometry = experts %d top-k %d expert FF %d", arch.Experts, arch.TopK, arch.ExpertFF)
	}
	if len(arch.Layer) != 32 || arch.RopeBase != 1_000_000 || arch.Eps != 1e-5 {
		t.Fatalf("architecture receipt = layers %d rope %g eps %g", len(arch.Layer), arch.RopeBase, arch.Eps)
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	if _, err := (Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, VocabSize: 32, NumLocalExperts: 2, NumExpertsPerTok: 3}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("top-k greater than expert count accepted")
	}
}

func TestConfig_ArchDefaultsAndGeometryErrors_Ugly(t *testing.T) {
	base := Config{HiddenSize: 8, IntermediateSize: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32, NumLocalExperts: 2, NumExpertsPerTok: 1}
	arch, err := base.Arch()
	if err != nil {
		t.Fatalf("default geometry: %v", err)
	}
	if arch.KVHeads != 2 || arch.Eps != defaultRMSNormEps || arch.RopeBase != defaultRopeTheta {
		t.Fatalf("defaults = kv %d eps %g rope %g", arch.KVHeads, arch.Eps, arch.RopeBase)
	}
	badHidden := base
	badHidden.HiddenSize = 9
	if _, err := badHidden.Arch(); err == nil {
		t.Fatal("indivisible hidden size accepted")
	}
	badKV := base
	badKV.NumKeyValueHeads = 3
	if _, err := badKV.Arch(); err == nil {
		t.Fatal("non-divisible KV heads accepted")
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"ignored": {Shape: []int{1}}})
	if cfg.HiddenSize != 8 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}
