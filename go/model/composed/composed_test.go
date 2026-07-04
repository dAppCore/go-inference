// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model/qwen3"
)

func syn(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*seed+7)%101-50) * 0.02
	}
	return out
}

func mkGatedDeltaMixer(cfg qwen3.GatedDeltaConfig, D, seed int) Mixer {
	qd, vd, cd := cfg.KeyHeads*cfg.HeadDim, cfg.ValueHeads*cfg.HeadDim, 2*cfg.KeyHeads*cfg.HeadDim+cfg.ValueHeads*cfg.HeadDim
	_ = qd
	w := &qwen3.GatedDeltaWeights{
		InProjQKV:  syn(cd*D, seed+1),
		ConvWeight: syn(cd*cfg.ConvKernel, seed+2),
		ConvBias:   syn(cd, seed+3),
		InProjA:    syn(cfg.ValueHeads*D, seed+4),
		ALog:       syn(cfg.ValueHeads, seed+5),
		DtBias:     syn(cfg.ValueHeads, seed+6),
		InProjB:    syn(cfg.ValueHeads*D, seed+7),
		InProjZ:    syn(vd*D, seed+8),
		Norm:       syn(cfg.HeadDim, seed+9),
		OutProj:    syn(D*vd, seed+10),
	}
	return NewGatedDeltaMixer(w, cfg)
}

func mkComposedModel(nLayers, D, vocab, FF int) *ComposedModel {
	cfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 8, ConvKernel: 4, Eps: 1e-5}
	layers := make([]Layer, nLayers)
	for li := range layers {
		layers[li] = Layer{
			InputNorm:    syn(D, li*13+1),
			Mixer:        mkGatedDeltaMixer(cfg, D, li*13+20),
			PostAttnNorm: syn(D, li*13+2),
			MLP:          &MLP{Gate: syn(FF*D, li*13+3), Up: syn(FF*D, li*13+4), Down: syn(D*FF, li*13+5), FF: FF},
		}
	}
	return &ComposedModel{
		Embed: syn(vocab*D, 100), Layers: layers, NormF: syn(D, 101), Output: nil,
		D: D, Vocab: vocab, Eps: 1e-5,
	}
}

// TestComposedDecodeEqualsPrefill is the orchestration correctness: stepping a sequence one token at a
// time through a fresh session (each layer threading its gated-delta state) produces hidden states
// BIT-EXACT to a single prefill pass — the layer loop (norm → mixer → residual → norm → SwiGLU → residual)
// plus the recurrent state threading reproduce prefill, the requirement for streaming hybrid decode.
func TestComposedDecodeEqualsPrefill(t *testing.T) {
	const D, vocab, nLayers, FF = 8, 32, 3, 16
	m := mkComposedModel(nLayers, D, vocab, FF)
	tokens := []int32{1, 5, 9, 2, 7, 3}

	prefill, err := NewSession(m).Forward(tokens)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	dec := NewSession(m)
	for t0, tok := range tokens {
		h, err := dec.Forward([]int32{tok})
		if err != nil {
			t.Fatalf("decode step %d: %v", t0, err)
		}
		for i := 0; i < D; i++ {
			if h[i] != prefill[t0*D+i] {
				t.Fatalf("token %d hidden[%d] = %v != prefill %v (composed decode diverged)", t0, i, h[i], prefill[t0*D+i])
			}
		}
	}
	t.Logf("composed decode == prefill bit-exact over %d tokens, %d gated-delta layers + SwiGLU", len(tokens), nLayers)
}

// TestComposedGenerate checks the greedy generate loop runs and is deterministic.
func TestComposedGenerate(t *testing.T) {
	m := mkComposedModel(2, 8, 32, 16)
	prompt := []int32{1, 2, 3}
	g1, err := NewSession(m).Generate(prompt, 5, -1)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	if len(g1) != 5 {
		t.Fatalf("generated %d, want 5", len(g1))
	}
	g2, _ := NewSession(m).Generate(prompt, 5, -1)
	for i := range g1 {
		if g1[i] != g2[i] {
			t.Fatalf("non-deterministic at %d: %d != %d", i, g1[i], g2[i])
		}
	}
	t.Logf("composed Generate: prefill→recurrent decode→head produced %v (deterministic)", g1)
}
