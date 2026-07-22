// SPDX-Licence-Identifier: EUPL-1.2

package llama4

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// Fixture source: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct/blob/main/config.json
func TestConfig_Arch_Good(t *testing.T) {
	data, err := coreio.Local.Read(core.PathJoin("testdata", "meta-llama-llama-4-scout-17b-16e-instruct-config.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var wrapper struct {
		TextConfig Config `json:"text_config"`
	}
	if r := core.JSONUnmarshal([]byte(data), &wrapper); !r.OK {
		t.Fatalf("parse fixture: %s", r.Error())
	}
	arch, err := wrapper.TextConfig.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.Hidden != 5120 || arch.Heads != 40 || arch.KVHeads != 8 || arch.HeadDim != 128 {
		t.Fatalf("attention geometry = %+v", arch)
	}
	if arch.Experts != 16 || arch.TopK != 1 || arch.ExpertFF != 8192 || arch.SharedExperts != 1 || arch.NormaliseMoETopK {
		t.Fatalf("MoE declaration = experts %d top-k %d expert FF %d shared %d normalise %v", arch.Experts, arch.TopK, arch.ExpertFF, arch.SharedExperts, arch.NormaliseMoETopK)
	}
	// SharedExpertFF (#61): Llama 4's shared expert reuses intermediate_size_mlp (16384) — genuinely
	// distinct from ExpertFF's intermediate_size (8192) above, a live 2x mismatch that reaches
	// engine/metal/arch_qwen_moe.go's encQwenMoEHalf shared-expert dispatch on a real checkpoint.
	if arch.SharedExpertFF != 16384 {
		t.Fatalf("SharedExpertFF = %d, want 16384 (intermediate_size_mlp — the shared expert's OWN width, distinct from ExpertFF's routed 8192)", arch.SharedExpertFF)
	}
	if arch.MoEGating != model.MoEGatingSigmoid || arch.FF != 16384 || arch.QKNormalization != model.QKL2Norm {
		t.Fatalf("text declaration = gating %q dense FF %d qk norm %q", arch.MoEGating, arch.FF, arch.QKNormalization)
	}
	if len(arch.Layer) != 48 || !arch.Layer[0].MoE || arch.Layer[0].DisableRotary || !arch.Layer[3].DisableRotary {
		t.Fatalf("layer pattern not preserved: %#v", arch.Layer[:4])
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	if _, err := (Config{}).Arch(); err == nil {
		t.Fatal("empty config accepted")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, IntermediateSize: 12, IntermediateSizeMLP: 16, NumHiddenLayers: 2, NumAttentionHeads: 2, HeadDim: 4, NumLocalExperts: 2, NumExpertsPerTok: 1, VocabSize: 32, NoRopeLayers: []int{1}}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("no_rope_layers length mismatch accepted")
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"ignored": {Shape: []int{1}}})
	if cfg.HiddenSize != 8 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}

func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("empty config became valid after InferFromWeights")
	}
}

func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, IntermediateSize: 12, IntermediateSizeMLP: 16, NumHiddenLayers: 1, NumAttentionHeads: 2, HeadDim: 4, NumLocalExperts: 2, NumExpertsPerTok: 3, VocabSize: 32, NoRopeLayers: []int{0}}
	cfg.InferFromWeights(map[string]safetensors.Tensor{})
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("top-k greater than expert count accepted")
	}
}
