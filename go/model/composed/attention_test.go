// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/qwen3"
)

func mkAttnMixer(cfg AttnConfig, D, seed int) Mixer {
	return NewAttnMixer(&AttnWeights{
		QProj: syn(cfg.Heads*cfg.HeadDim*D, seed+1),
		KProj: syn(cfg.KVHeads*cfg.HeadDim*D, seed+2),
		VProj: syn(cfg.KVHeads*cfg.HeadDim*D, seed+3),
		OProj: syn(D*cfg.Heads*cfg.HeadDim, seed+4),
		QNorm: syn(cfg.HeadDim, seed+5),
		KNorm: syn(cfg.HeadDim, seed+6),
	}, cfg)
}

// TestRmsNormHead pins the per-head RMSNorm against hand-computed values — the primitive the
// decode==prefill tests structurally can't isolate (a shared bug cancels on both sides).
// x=[3,4]: rms = sqrt((9+16)/2 + eps) = sqrt(12.5); out_i = x_i/rms · w_i.
func TestRmsNormHead(t *testing.T) {
	x := []float32{3, 4}
	w := []float32{2, 0.5}
	rmsNormHead(x, w, 0)
	rms := math.Sqrt(12.5)
	want0, want1 := float32(3/rms*2), float32(4/rms*0.5)
	if math.Abs(float64(x[0]-want0)) > 1e-6 || math.Abs(float64(x[1]-want1)) > 1e-6 {
		t.Fatalf("rmsNormHead([3,4],[2,0.5]) = %v, want [%v %v]", x, want0, want1)
	}
	// eps sits inside the root: rms = sqrt(ss/n + eps), so a zero vector stays finite.
	z := []float32{0, 0}
	rmsNormHead(z, []float32{1, 1}, 1e-6)
	if z[0] != 0 || z[1] != 0 {
		t.Fatalf("rmsNormHead(zero vector) = %v, want [0 0] (eps keeps the divide finite)", z)
	}
}

// TestApplyRotaryHalf pins the rotate_half rotation against hand-computed values: position 0
// is the identity, position 1 rotates pair (i, i+half) through angle pos·theta^(−2i/rotaryDim),
// and dims at or beyond rotaryDim are untouched (partial rotary).
func TestApplyRotaryHalf(t *testing.T) {
	// pos 0 → identity everywhere.
	x := []float32{1, 2, 3, 4}
	applyRotaryHalf(x, 0, 4, 10000)
	for i, v := range []float32{1, 2, 3, 4} {
		if x[i] != v {
			t.Fatalf("pos 0 must be the identity, got %v", x)
		}
	}
	// pos 1, rotaryDim 2 of headDim 4: half=1 pairs (0,1) at freq 1; dims 2,3 untouched.
	x = []float32{1, 2, 3, 4}
	applyRotaryHalf(x, 1, 2, 10000)
	c, s := math.Cos(1), math.Sin(1)
	if want0, want1 := float32(1*c-2*s), float32(2*c+1*s); math.Abs(float64(x[0]-want0)) > 1e-6 || math.Abs(float64(x[1]-want1)) > 1e-6 {
		t.Fatalf("pos 1 rotation = [%v %v], want [%v %v]", x[0], x[1], want0, want1)
	}
	if x[2] != 3 || x[3] != 4 {
		t.Fatalf("dims beyond rotaryDim must be untouched, got %v", x)
	}
	// full rotary (rotaryDim 4): half=2 pairs (0,2) and (1,3); pair 1's frequency is
	// theta^(−2/4) — the per-pair frequency progression.
	x = []float32{1, 2, 3, 4}
	applyRotaryHalf(x, 1, 4, 10000)
	f1 := 1.0 / math.Pow(10000, 2.0/4.0)
	c0, s0 := math.Cos(1), math.Sin(1)
	c1, s1 := math.Cos(f1), math.Sin(f1)
	wants := []float32{float32(1*c0 - 3*s0), float32(2*c1 - 4*s1), float32(3*c0 + 1*s0), float32(4*c1 + 2*s1)}
	for i, want := range wants {
		if math.Abs(float64(x[i]-want)) > 1e-6 {
			t.Fatalf("full-rotary pos 1: x[%d] = %v, want %v (pair freq progression)", i, x[i], want)
		}
	}
}

// TestAttnMixerSingleTokenClosedForm checks Forward against an INDEPENDENT closed form the
// decode==prefill tests can't provide (they compare Forward to itself, so a bug shared by both
// paths passes). At L=1, pos=0: softmax over the single key is exactly 1 whatever Q/K compute,
// and rotary at position 0 is the identity — so out MUST equal OProj · (V per head,
// GQA-expanded), assembled here from the raw weights without calling Forward's attention code.
// Catches a V-path, GQA head-mapping, or output-projection defect.
func TestAttnMixerSingleTokenClosedForm(t *testing.T) {
	cfg := AttnConfig{Heads: 4, KVHeads: 2, HeadDim: 8, RotaryDim: 4, RopeTheta: 1e6, NormEps: 1e-6}
	const D = 8
	w := &AttnWeights{
		QProj: syn(cfg.Heads*cfg.HeadDim*D, 1),
		KProj: syn(cfg.KVHeads*cfg.HeadDim*D, 2),
		VProj: syn(cfg.KVHeads*cfg.HeadDim*D, 3),
		OProj: syn(D*cfg.Heads*cfg.HeadDim, 4),
		QNorm: syn(cfg.HeadDim, 5),
		KNorm: syn(cfg.HeadDim, 6),
	}
	m := NewAttnMixer(w, cfg)
	x := syn(D, 7)
	got, _, err := m.Forward(x, 1, D, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	// independent reference: v = VProj·x, expanded per query head by the GQA repeat, then OProj.
	HD, rep := cfg.HeadDim, cfg.Heads/cfg.KVHeads
	v := matNT(x, w.VProj, 1, D, cfg.KVHeads*HD)
	expanded := make([]float32, cfg.Heads*HD)
	for hd := 0; hd < cfg.Heads; hd++ {
		copy(expanded[hd*HD:hd*HD+HD], v[(hd/rep)*HD:(hd/rep)*HD+HD])
	}
	want := matNT(expanded, w.OProj, 1, cfg.Heads*HD, D)
	for i := range D {
		if got[i] != want[i] {
			t.Fatalf("out[%d] = %v, want %v (single-token closed form: softmax(1 key)=1 → out = OProj·GQA(V))", i, got[i], want[i])
		}
	}
	t.Logf("single-token attention matches the closed form OProj·GQA(VProj·x) — V path, GQA map, o_proj verified independently")
}

// TestAttnMixerDecodeEqualsPrefill is the KV-cache correctness: stepping tokens one at a time through the
// attention mixer (growing the cache) produces outputs BIT-EXACT to a single prefill pass — causal
// attention over the cache reproduces full-sequence attention.
func TestAttnMixerDecodeEqualsPrefill(t *testing.T) {
	cfg := AttnConfig{Heads: 4, KVHeads: 2, HeadDim: 8, RotaryDim: 4, RopeTheta: 1e6, NormEps: 1e-6}
	const D, L = 8, 6
	m := mkAttnMixer(cfg, D, 0)
	h := syn(L*D, 1)

	full, _, err := m.Forward(h, L, D, nil)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	var st any
	for t0 := range L {
		o, next, err := m.Forward(h[t0*D:(t0+1)*D], 1, D, st)
		if err != nil {
			t.Fatalf("decode %d: %v", t0, err)
		}
		st = next
		for i := range D {
			if o[i] != full[t0*D+i] {
				t.Fatalf("token %d out[%d] = %v != prefill %v (KV cache diverged)", t0, i, o[i], full[t0*D+i])
			}
		}
	}
	t.Logf("attention mixer decode == prefill bit-exact over %d tokens (KV cache + partial rotary + GQA)", L)
}

// TestHybridDecodeEqualsPrefill is the orchestration's reason to exist: a ComposedModel that INTERLEAVES
// gated-delta and full-attention layers (the Qwen 3.6 schedule shape) decodes token-by-token BIT-EXACT to
// prefill — the session threads each layer's own state type (recurrent for gated-delta, KV for attention)
// through the same loop.
func TestHybridDecodeEqualsPrefill(t *testing.T) {
	const D, vocab, FF = 8, 32, 16
	gdCfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 8, ConvKernel: 4, Eps: 1e-5}
	atCfg := AttnConfig{Heads: 4, KVHeads: 2, HeadDim: 8, RotaryDim: 4, RopeTheta: 1e6, NormEps: 1e-6}
	mk := func(li int, mx Mixer) Layer {
		return Layer{
			InputNorm:    syn(D, li*13+1),
			Mixer:        mx,
			PostAttnNorm: syn(D, li*13+2),
			MLP:          &MLP{Gate: syn(FF*D, li*13+3), Up: syn(FF*D, li*13+4), Down: syn(D*FF, li*13+5), FF: FF},
		}
	}
	m := &ComposedModel{
		Embed: syn(vocab*D, 100), NormF: syn(D, 101), D: D, Vocab: vocab, Eps: 1e-5,
		Layers: []Layer{
			mk(0, mkGatedDeltaMixer(gdCfg, D, 20)), // linear_attention
			mk(1, mkAttnMixer(atCfg, D, 40)),       // full_attention
			mk(2, mkGatedDeltaMixer(gdCfg, D, 60)), // linear_attention
			mk(3, mkAttnMixer(atCfg, D, 80)),       // full_attention
		},
	}
	tokens := []int32{1, 5, 9, 2, 7, 3}

	prefill, err := NewSession(m).Forward(tokens)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	dec := NewSession(m)
	for t0, tok := range tokens {
		h, err := dec.Forward([]int32{tok})
		if err != nil {
			t.Fatalf("decode %d: %v", t0, err)
		}
		for i := range D {
			if h[i] != prefill[t0*D+i] {
				t.Fatalf("token %d hidden[%d] = %v != prefill %v (hybrid decode diverged)", t0, i, h[i], prefill[t0*D+i])
			}
		}
	}
	t.Logf("hybrid (gated-delta + full-attention interleaved) decode == prefill bit-exact over %d tokens", len(tokens))
}
