// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"os"
	"testing"

	"dappco.re/go/inference/model"
	g4 "dappco.re/go/inference/model/gemma4"
)

// TestGenerateBF16 gates the token loop end to end: on a small (but SDPA-real,
// headDim 64) bf16 gemma4 it generates maxNew in-range tokens, is deterministic (greedy),
// stops at EOS, and its first token equals the manual embed → DecodeForward → LM head →
// greedy chain (the loop wires the components correctly).
func TestGenerateBF16(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 32
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	mk := func(n, salt int) []float32 {
		s := make([]float32, n)
		for i := range s {
			s[i] = float32((i*salt+13)%97-48) * 0.02
		}
		return s
	}
	layers := make([]DecodeLayerWeights, len(arch.Layer))
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
	}
	g := &BF16Model{
		Layers:    layers,
		Embed:     toBF16Bytes(mk(vocab*dModel, 11)),
		FinalNorm: toBF16Bytes(mk(dModel, 7)),
	}
	g.LMHead, g.Tied = g.Embed, true // tied head

	prompt := []int32{1, 5, 3, 9}
	const maxNew, maxLen = 5, 16

	out, err := GenerateBF16(g, arch, prompt, maxNew, maxLen, -1)
	if err != nil {
		t.Fatalf("GenerateBF16: %v", err)
	}
	if len(out) != maxNew {
		t.Fatalf("generated %d tokens, want %d", len(out), maxNew)
	}
	for i, id := range out {
		if id < 0 || int(id) >= vocab {
			t.Fatalf("generated token %d = %d out of [0,%d)", i, id, vocab)
		}
	}

	// deterministic: greedy re-run is identical.
	out2, err := GenerateBF16(g, arch, prompt, maxNew, maxLen, -1)
	if err != nil {
		t.Fatalf("GenerateBF16 re-run: %v", err)
	}
	for i := range out {
		if out[i] != out2[i] {
			t.Fatalf("non-deterministic at %d: %d vs %d", i, out[i], out2[i])
		}
	}

	// the first generated token equals the manual component chain.
	backend, err := NewBF16Backend(arch, layers, maxLen)
	if err != nil {
		t.Fatalf("NewBF16Backend: %v", err)
	}
	embs, err := EmbedTokensBF16(g.Embed, prompt, vocab, dModel, float32(math.Sqrt(float64(dModel))))
	if err != nil {
		t.Fatalf("EmbedTokensBF16: %v", err)
	}
	hidden, err := backend.DecodeForward(embs)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	logits, err := LMHeadBF16(hidden[len(hidden)-1], g.FinalNorm, g.LMHead, dModel, vocab, arch.Eps, arch.SoftCap)
	if err != nil {
		t.Fatalf("LMHeadBF16: %v", err)
	}
	first, err := model.Greedy(logits, vocab)
	if err != nil {
		t.Fatalf("Greedy: %v", err)
	}
	if out[0] != first {
		t.Fatalf("loop first token %d != manual embed→decode→lm_head→greedy %d", out[0], first)
	}

	// EOS stops the loop: setting eosID to the first generated token yields exactly it.
	outEos, err := GenerateBF16(g, arch, prompt, maxNew, maxLen, int(out[0]))
	if err != nil {
		t.Fatalf("GenerateBF16 eos: %v", err)
	}
	if len(outEos) != 1 || outEos[0] != out[0] {
		t.Fatalf("EOS stop: got %v, want exactly [%d]", outEos, out[0])
	}

	t.Logf("token loop: %d-token prompt → %d greedy tokens %v (deterministic, in-range, first ≡ manual chain, EOS stops) — embed→decode→lm_head→sample end to end on a real-SDPA gemma4", len(prompt), len(out), out)
}

func TestGenerateBF16OneTokenAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	g, arch := gemma4BF16Fixture(t, 64, 1, 1, 64, 128, 32, 1)
	prompt := []int32{1, 5}
	if _, err := GenerateBF16(g, arch, prompt, 1, 4, -1); err != nil {
		t.Fatalf("GenerateBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(3, func() {
		if _, err := GenerateBF16(g, arch, prompt, 1, 4, -1); err != nil {
			t.Fatalf("GenerateBF16: %v", err)
		}
	})
	if allocs > 125 {
		t.Fatalf("GenerateBF16 allocations = %.0f, want <= 125", allocs)
	}
}
