// SPDX-Licence-Identifier: EUPL-1.2

package llama

import (
	"math"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

func loadConfigFixture(t *testing.T, name string) Config {
	t.Helper()
	data, err := coreio.Local.Read(core.PathJoin("testdata", name))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cfg Config
	if r := core.JSONUnmarshal([]byte(data), &cfg); !r.OK {
		t.Fatalf("parse fixture: %s", r.Error())
	}
	return cfg
}

func TestConfigArchLlama31_8B_Good(t *testing.T) {
	cfg := loadConfigFixture(t, "meta-llama-llama-3.1-8b-config.json")
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 4096 || arch.FF != 14336 || len(arch.Layer) != 32 {
		t.Fatalf("geometry = hidden %d ff %d layers %d", arch.Hidden, arch.FF, len(arch.Layer))
	}
	if arch.Heads != 32 || arch.KVHeads != 8 || arch.HeadDim != 128 {
		t.Fatalf("GQA = heads %d kv %d headDim %d", arch.Heads, arch.KVHeads, arch.HeadDim)
	}
	if arch.RopeBase != 500000 || len(arch.RopeFreqs) != 64 {
		t.Fatalf("llama3 rope = base %g freqs %d", arch.RopeBase, len(arch.RopeFreqs))
	}
	if cfg.TieWordEmbeddings == nil || *cfg.TieWordEmbeddings {
		t.Fatal("Meta-Llama-3.1-8B fixture must declare an untied output head")
	}
}

func TestConfigArchLlama32_1B_Good(t *testing.T) {
	cfg := loadConfigFixture(t, "meta-llama-llama-3.2-1b-config.json")
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 2048 || arch.FF != 8192 || len(arch.Layer) != 16 {
		t.Fatalf("geometry = hidden %d ff %d layers %d", arch.Hidden, arch.FF, len(arch.Layer))
	}
	if arch.Heads != 32 || arch.KVHeads != 8 || arch.HeadDim != 64 {
		t.Fatalf("GQA = heads %d kv %d headDim %d", arch.Heads, arch.KVHeads, arch.HeadDim)
	}
	if cfg.TieWordEmbeddings == nil || !*cfg.TieWordEmbeddings {
		t.Fatal("Llama-3.2-1B fixture must declare tied word embeddings")
	}
}

func TestLlama3InvFreqs_Bad(t *testing.T) {
	if got := Llama3InvFreqs(500000, RopeScaling{RopeType: "llama3", Factor: 1, OriginalMaxPositionEmbeddings: 8192}, 128); got != nil {
		t.Fatalf("invalid extension returned %d frequencies", len(got))
	}
}

func TestLlama3InvFreqs_Ugly(t *testing.T) {
	rp := RopeScaling{RopeType: "llama3", Factor: 8, LowFreqFactor: 1, HighFreqFactor: 4, OriginalMaxPositionEmbeddings: 8192}
	got := Llama3InvFreqs(500000, rp, 128)
	if len(got) != 64 {
		t.Fatalf("frequencies = %d, want 64", len(got))
	}
	plainHigh := float32(1)
	if got[0] != plainHigh {
		t.Fatalf("high frequency = %g, want unchanged %g", got[0], plainHigh)
	}
	plainLow := float32(math.Pow(500000, -126.0/128.0))
	if relDiff(got[63], plainLow/8) > 1e-5 {
		t.Fatalf("low frequency = %g, want scaled %g", got[63], plainLow/8)
	}
	foundSmooth := false
	for i, freq := range got {
		plain := float32(math.Pow(500000, -float64(2*i)/128.0))
		if freq > plain/8 && freq < plain {
			foundSmooth = true
			break
		}
	}
	if !foundSmooth {
		t.Fatal("llama3 rope has no smoothly interpolated middle frequency")
	}
}

func TestConfigArchLinearRope_Good(t *testing.T) {
	cfg := Config{
		HiddenSize: 64, IntermediateSize: 128, NumHiddenLayers: 1,
		NumAttentionHeads: 8, VocabSize: 32, RopeTheta: 10000,
		RopeScaling: &RopeScaling{Type: "linear", Factor: 4},
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.RopeScale != 0.25 || arch.RopeFreqs != nil {
		t.Fatalf("linear rope = scale %g freqs %v", arch.RopeScale, arch.RopeFreqs)
	}
}

func TestConfigArchDefaultRope_Good(t *testing.T) {
	cfg := Config{HiddenSize: 64, IntermediateSize: 128, NumHiddenLayers: 1, NumAttentionHeads: 8, VocabSize: 32}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.RopeBase != 10000 || arch.RopeScale != 1 || arch.RopeFreqs != nil {
		t.Fatalf("default rope = base %g scale %g freqs %v", arch.RopeBase, arch.RopeScale, arch.RopeFreqs)
	}
}

func TestConfigArch_Bad(t *testing.T) {
	cfg := Config{HiddenSize: 65, IntermediateSize: 128, NumHiddenLayers: 2, NumAttentionHeads: 8, VocabSize: 32}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("indivisible hidden size accepted without head_dim")
	}
}

func TestConfigArch_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 64, IntermediateSize: 128, NumHiddenLayers: 1, NumAttentionHeads: 8, NumKeyValueHeads: 3, VocabSize: 32}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("non-divisible GQA geometry accepted")
	}
}

func TestConfigInferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 64}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"ignored": {Shape: []int{1}}})
	if cfg.HiddenSize != 64 {
		t.Fatalf("InferFromWeights changed declared geometry: %+v", cfg)
	}
}

func TestConfigArchRopeErrors_Bad(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  Config
	}{
		{name: "missing geometry", cfg: Config{}},
		{name: "negative head dim", cfg: Config{HiddenSize: 64, IntermediateSize: 128, NumHiddenLayers: 1, NumAttentionHeads: 8, HeadDim: -1, VocabSize: 32}},
		{name: "negative norm epsilon", cfg: Config{HiddenSize: 64, IntermediateSize: 128, NumHiddenLayers: 1, NumAttentionHeads: 8, VocabSize: 32, RMSNormEps: -1}},
		{name: "negative rope theta", cfg: Config{HiddenSize: 64, IntermediateSize: 128, NumHiddenLayers: 1, NumAttentionHeads: 8, VocabSize: 32, RopeTheta: -1}},
		{name: "invalid linear factor", cfg: Config{HiddenSize: 64, IntermediateSize: 128, NumHiddenLayers: 1, NumAttentionHeads: 8, VocabSize: 32, RopeScaling: &RopeScaling{RopeType: "linear"}}},
		{name: "invalid llama3 scaling", cfg: Config{HiddenSize: 64, IntermediateSize: 128, NumHiddenLayers: 1, NumAttentionHeads: 8, VocabSize: 32, RopeScaling: &RopeScaling{RopeType: "llama3", Factor: 8}}},
		{name: "unsupported variant", cfg: Config{HiddenSize: 64, IntermediateSize: 128, NumHiddenLayers: 1, NumAttentionHeads: 8, VocabSize: 32, RopeScaling: &RopeScaling{RopeType: "dynamic", Factor: 2}}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.cfg.Arch(); err == nil {
				t.Fatal("invalid config accepted")
			}
		})
	}
}

func relDiff(a, b float32) float32 {
	d := float32(math.Abs(float64(a - b)))
	den := max(float32(math.Abs(float64(a))), float32(math.Abs(float64(b))))
	if den == 0 {
		return d
	}
	return d / den
}
