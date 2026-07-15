// SPDX-Licence-Identifier: EUPL-1.2

package opt

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	"testing"
)

// Fixtures are verbatim architecture fields from the cited public config.json files.
// Sources: https://huggingface.co/facebook/opt-125m/blob/main/config.json
// https://huggingface.co/facebook/opt-350m/blob/main/config.json
// https://huggingface.co/facebook/opt-1.3b/blob/main/config.json
func TestConfig_Arch_Good(t *testing.T) {
	fixtures := []struct {
		name, file                       string
		hidden, embed, heads, layers, ff int
		before                           bool
	}{
		{"OPT-125M", "facebook-opt-125m-config.json", 768, 768, 12, 12, 3072, true},
		{"OPT-350M", "facebook-opt-350m-config.json", 1024, 512, 16, 24, 4096, false},
		{"OPT-1.3B", "facebook-opt-1.3b-config.json", 2048, 2048, 32, 24, 8192, true},
	}
	for _, fixture := range fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			read := core.ReadFile(core.PathJoin("testdata", fixture.file))
			if !read.OK {
				t.Fatal("read fixture")
			}
			var cfg Config
			if r := core.JSONUnmarshal(read.Value.([]byte), &cfg); !r.OK {
				t.Fatal("parse")
			}
			arch, err := cfg.Arch()
			if err != nil {
				t.Fatal(err)
			}
			if arch.Hidden != fixture.hidden || arch.EmbeddingDim != fixture.embed || arch.Heads != fixture.heads || len(arch.Layer) != fixture.layers || arch.FF != fixture.ff {
				t.Fatalf("arch=%+v", arch)
			}
			if arch.LayerNormBefore != fixture.before || !arch.LearnedAbsolutePositions || arch.PositionOffset != 2 {
				t.Fatalf("OPT declarations=%+v", arch)
			}
			if arch.NoFinalNorm == fixture.before {
				t.Fatalf("NoFinalNorm=%v before=%v", arch.NoFinalNorm, fixture.before)
			}
		})
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	if _, err := (&Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	if _, err := (&Config{Hidden: 7, Heads: 2, Layers: 1, FF: 8, Positions: 4, Vocab: 8, EmbedDim: 7}).Arch(); err == nil {
		t.Fatal("indivisible heads accepted")
	}
}

func TestConfig_Config_Good(t *testing.T) {
	config := Config{Hidden: 8, Heads: 2, Layers: 1, FF: 16, Positions: 8, Vocab: 12}
	if config.Hidden != 8 || config.Heads != 2 {
		t.Fatalf("config=%+v", config)
	}
}

func TestConfig_Config_Bad(t *testing.T) {
	config := Config{}
	if _, err := config.Arch(); err == nil {
		t.Fatal("zero Config accepted")
	}
}

func TestConfig_Config_Ugly(t *testing.T) {
	config := Config{Hidden: -1}
	if _, err := config.Arch(); err == nil {
		t.Fatal("negative Config accepted")
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	config := Config{Hidden: 8}
	config.InferFromWeights(nil)
	if config.Hidden != 8 {
		t.Fatalf("InferFromWeights changed config: %+v", config)
	}
}

func TestConfig_InferFromWeights_Bad(t *testing.T) {
	config := Config{}
	config.InferFromWeights(nil)
	if config.Hidden != 0 {
		t.Fatalf("InferFromWeights invented geometry: %+v", config)
	}
}

func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	config := Config{EmbedDim: -1}
	config.InferFromWeights(map[string]safetensors.Tensor{})
	if config.EmbedDim != -1 {
		t.Fatalf("InferFromWeights changed boundary value: %+v", config)
	}
}
