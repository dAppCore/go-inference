// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// realConfig returns a fully-populated, structurally valid Config matching the real
// openai/gpt-oss-20b geometry (hand-built rather than parsed from the testdata fixture, so buildArch's
// tests are independent of ParseConfig/testdata — config_test.go covers the parsed-fixture path).
func realConfig() Config {
	types := make([]string, 24)
	for i := range types {
		if i%2 == 0 {
			types[i] = "sliding_attention"
		} else {
			types[i] = "full_attention"
		}
	}
	return Config{
		HiddenSize: 2880, NumHiddenLayers: 24, NumAttentionHeads: 64, NumKeyValueHeads: 8, HeadDim: 64,
		VocabSize: 201088, NumLocalExperts: 32, NumExpertsPerTok: 4, IntermediateSize: 2880,
		RMSNormEps: 1e-5, RopeTheta: 150000, SlidingWindow: 128, LayerTypes: types,
		RopeScaling: RopeScaling{RopeType: "yarn", Factor: 32, BetaFast: 32, BetaSlow: 1, OriginalMaxPositionEmbeddings: 4096, Truncate: false},
	}
}

// TestConfig_buildArch_Good proves the full geometry resolves correctly against realistic gpt_oss
// dimensions: attention/MoE dims, the alternating sliding/full schedule (every layer MoE, none gated-
// delta), and a YaRN RopeFreqs table of the right length.
func TestConfig_buildArch_Good(t *testing.T) {
	cfg := realConfig()
	a, err := cfg.buildArch()
	if err != nil {
		t.Fatalf("buildArch: %v", err)
	}
	if a.Hidden != 2880 || a.Heads != 64 || a.KVHeads != 8 || a.HeadDim != 64 || a.Vocab != 201088 {
		t.Fatalf("buildArch attention dims = %+v", a)
	}
	if a.Experts != 32 || a.TopK != 4 || a.ExpertFF != 2880 {
		t.Fatalf("buildArch MoE dims = experts %d topK %d expertFF %d", a.Experts, a.TopK, a.ExpertFF)
	}
	if a.MoEGating != model.MoEGatingSoftmax || !a.NormaliseMoETopK {
		t.Fatalf("buildArch MoE gating = %q normalise=%v, want softmax+normalise", a.MoEGating, a.NormaliseMoETopK)
	}
	if a.Activation == "silu" || a.Activation == "swish" {
		t.Fatalf("buildArch Activation = %q must NOT be a plain-SiLU-matching string (see config.go doc)", a.Activation)
	}
	if a.SlidingWindow != 128 {
		t.Fatalf("buildArch SlidingWindow = %d, want 128", a.SlidingWindow)
	}
	if len(a.Layer) != 24 {
		t.Fatalf("buildArch layer count = %d, want 24", len(a.Layer))
	}
	for i, l := range a.Layer {
		if l.Mixer != model.MixerAttention {
			t.Fatalf("layer %d Mixer = %v, want MixerAttention (gpt_oss has no linear_attention layers)", i, l.Mixer)
		}
		if !l.MoE {
			t.Fatalf("layer %d MoE = false, want true (gpt_oss has no dense layer)", i)
		}
		if l.HeadDim != 64 || l.KVHeads != 8 {
			t.Fatalf("layer %d geometry = headDim %d kvHeads %d, want 64/8", i, l.HeadDim, l.KVHeads)
		}
	}
	if len(a.RopeFreqs) != 32 {
		t.Fatalf("buildArch RopeFreqs length = %d, want 32 (rotaryDim/2)", len(a.RopeFreqs))
	}
}

// TestConfig_buildArch_Bad proves the kv-heads divisibility guard fires — a DIFFERENT structural check
// than config_test.go's TestConfig_Arch_Bad (which covers the more basic hidden_size==0 case).
func TestConfig_buildArch_Bad(t *testing.T) {
	cfg := realConfig()
	cfg.NumKeyValueHeads = 7 // 64 % 7 != 0
	_, err := cfg.buildArch()
	if err == nil || !core.Contains(err.Error(), "multiple of num_key_value_heads") {
		t.Fatalf("buildArch %v must reject a kv-head count that doesn't divide num_attention_heads", err)
	}
}

// TestConfig_buildArch_Ugly proves a router top-k that exceeds the local-expert count is rejected —
// syntactically plausible (both are positive integers) but semantically impossible, distinct from
// _Bad's divisibility failure and config_test.go's layer_types-length failure.
func TestConfig_buildArch_Ugly(t *testing.T) {
	cfg := realConfig()
	cfg.NumExpertsPerTok = 33 // > NumLocalExperts (32)
	_, err := cfg.buildArch()
	if err == nil || !core.Contains(err.Error(), "num_experts_per_tok/experts_per_token must be in (0, num_local_experts]") {
		t.Fatalf("buildArch %v must reject top_k > num_local_experts", err)
	}
}

func TestConfig_resolvedExpertsPerTok_Good(t *testing.T) {
	cfg := Config{NumExpertsPerTok: 4, ExpertsPerToken: 4}
	if got := cfg.resolvedExpertsPerTok(); got != 4 {
		t.Fatalf("resolvedExpertsPerTok = %d, want 4", got)
	}
}

// TestConfig_resolvedExpertsPerTok_Bad proves the experts_per_token synonym is used as a fallback when
// num_experts_per_tok is absent — a checkpoint conversion that only sets the synonym still resolves.
func TestConfig_resolvedExpertsPerTok_Bad(t *testing.T) {
	cfg := Config{ExpertsPerToken: 6}
	if got := cfg.resolvedExpertsPerTok(); got != 6 {
		t.Fatalf("resolvedExpertsPerTok = %d, want 6 (fallback to experts_per_token)", got)
	}
}

// TestConfig_resolvedExpertsPerTok_Ugly proves the zero-value case returns 0 without error — the
// function never validates, it only resolves the synonym; buildArch is what turns 0 into a rejection.
func TestConfig_resolvedExpertsPerTok_Ugly(t *testing.T) {
	if got := (&Config{}).resolvedExpertsPerTok(); got != 0 {
		t.Fatalf("resolvedExpertsPerTok on an empty config = %d, want 0", got)
	}
}
