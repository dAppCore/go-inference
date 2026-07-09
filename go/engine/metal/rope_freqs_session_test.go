// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/mistral"
)

// TestMistralYaRNExecutor_Good gates the freqs-rope threaded through the decode
// executor: a Ministral session whose arch carries the PLAIN-rope spectrum must
// decode identically to one with no RopeFreqs (the base-derived rope) — proving
// the periods buffer flows through stepToken → encAttnHalfKV/Shared correctly — and
// a session carrying a YaRN spectrum decodes valid tokens (the long-context rope
// runs end to end).
func TestMistralYaRNExecutor_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	const numLayers, maxLen, n = 2, 16, 4
	cfg := mistral.Config{
		HiddenSize: dModel, NumHiddenLayers: numLayers, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim, VocabSize: vocab, RMSNormEps: 1e-6,
		RopeParameters: &mistral.RopeParams{RopeTheta: 1_000_000}, // default rope → arch.RopeFreqs nil
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if arch.RopeFreqs != nil {
		t.Fatal("base arch should have no RopeFreqs")
	}
	ts := mistralBF16Tensors(t, dModel, nHeads, nKV, headDim, dFF, vocab, numLayers)
	// mistral's weight layout (same as pkg/model/mistral/register.go): standard names with the
	// pre-MLP norm at post_attention_layernorm and no gemma-style post-attention norm.
	w := model.StandardWeightNames()
	w.MLPNorm = ".post_attention_layernorm.weight"
	w.PostAttnNorm = ""
	lm, err := model.Assemble(ts, arch, w)
	if err != nil {
		t.Fatalf("model.Assemble: %v", err)
	}
	g := loadedToBF16(lm)
	prompt := []int32{1, 5, 3}

	// base: no RopeFreqs → the base-derived rope.
	sessBase, err := NewArchSession(g, arch, maxLen)
	if err != nil {
		t.Fatalf("base session: %v", err)
	}
	genBase, err := sessBase.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("base generate: %v", err)
	}

	// plain spectrum through the freqs path → must equal the base rope exactly.
	archPlain := arch
	archPlain.RopeFreqs = plainRopeInvFreqs(float64(arch.RopeBase), arch.RotaryDim)
	sessPlain, err := NewArchSession(g, archPlain, maxLen)
	if err != nil {
		t.Fatalf("plain session: %v", err)
	}
	genPlain, err := sessPlain.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("plain generate: %v", err)
	}
	if !idsEqual(genPlain, genBase) {
		t.Fatalf("plain-freqs executor %v != base-rope %v", genPlain, genBase)
	}

	// YaRN spectrum → decodes valid tokens (the long-context rope runs end to end).
	archYarn := arch
	archYarn.RopeFreqs = mistral.YaRNInvFreqs(float64(arch.RopeBase), 16, 32, 1, 16384, arch.RotaryDim)
	sessYarn, err := NewArchSession(g, archYarn, maxLen)
	if err != nil {
		t.Fatalf("yarn session: %v", err)
	}
	genYarn, err := sessYarn.Generate(prompt, n, -1)
	if err != nil {
		t.Fatalf("yarn generate: %v", err)
	}
	for i, id := range genYarn {
		if id < 0 || int(id) >= vocab {
			t.Fatalf("yarn token %d = %d out of range", i, id)
		}
	}
	t.Logf("YaRN through the executor: plain-freqs ≡ base %v; yarn-freqs decodes %v", genBase, genYarn)
}
