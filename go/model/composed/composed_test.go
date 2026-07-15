// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/qwen3"
)

func syn(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*seed+7)%101-50) * 0.02
	}
	return out
}

// countingMixer wraps a Mixer that also implements projMixer, counting calls to each method — used to
// prove forwardEmb's dispatch actually takes the isProj path (forwardNoProj) rather than the standard
// path (Forward) for a given composed config. Unlike engine/metal's composed_backend_test.go pattern
// (which counts a bound DEVICE hook), there is no device hook at this level to intercept: projMixer is
// composed's own arch-neutral dispatch contract, so the count is taken directly on the mixer.
type countingMixer struct {
	inner interface {
		Mixer
		projMixer
	}
	forwardCalls, noProjCalls int
}

func (c *countingMixer) Kind() string { return c.inner.Kind() }

func (c *countingMixer) Forward(h []float32, L, D int, prior any) ([]float32, any, error) {
	c.forwardCalls++
	return c.inner.Forward(h, L, D, prior)
}

func (c *countingMixer) forwardNoProj(h []float32, L, D int, prior any) (mixerHidden, projW []float32, mixCols int, next any, err error) {
	c.noProjCalls++
	return c.inner.forwardNoProj(h, L, D, prior)
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
		for i := range D {
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

// TestMatNTIntoDeviceHook pins the ProjMatMulInto seam: the hook fires only at
// or above the deviceMinWork floor, its result is returned verbatim, and a
// device error falls back to the host path (deterministic either way).
func TestMatNTIntoDeviceHook(t *testing.T) {
	defer func() { ProjMatMulInto = nil }()

	var calls int
	sentinel := []float32{42}
	ProjMatMulInto = func(out, x, w []float32, M, K, N int) ([]float32, error) {
		calls++
		return sentinel, nil
	}
	// Below the floor: 4·4·4 MACs — hook must NOT fire.
	small := matNTInto(nil, make([]float32, 16), make([]float32, 16), 4, 4, 4)
	if calls != 0 {
		t.Fatalf("device hook fired below deviceMinWork (calls=%d)", calls)
	}
	if len(small) != 16 {
		t.Fatalf("host path returned %d elems, want 16", len(small))
	}
	// At the floor: 1<<20 MACs (M=1, K=1024, N=1024) — hook result verbatim.
	in := make([]float32, 1024)
	w := make([]float32, 1024*1024)
	got := matNTInto(nil, in, w, 1, 1024, 1024)
	if calls != 1 || len(got) != 1 || got[0] != 42 {
		t.Fatalf("device hook not used verbatim at floor (calls=%d, got=%v)", calls, got[:min(len(got), 4)])
	}
	// Device error: host fallback produces the real result.
	ProjMatMulInto = func(out, x, w []float32, M, K, N int) ([]float32, error) {
		return nil, core.NewError("synthetic device failure")
	}
	fallback := matNTInto(nil, in, w, 1, 1024, 1024)
	if len(fallback) != 1024 {
		t.Fatalf("error fallback returned %d elems, want 1024", len(fallback))
	}
	for i, v := range fallback {
		if v != 0 {
			t.Fatalf("fallback[%d]=%v, want 0 (zero inputs)", i, v)
		}
	}
}
