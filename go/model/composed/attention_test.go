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

// TestAttnMixer_Forward_Golden pins the exact f32 bit-pattern of the attention mixer's output AND the
// grown K/V cache over a fixed 3-token input, gating the cache-buffer fusion refactor on bit-identical
// behaviour (the closed-form and decode==prefill tests above are tolerance / self-referential).
func TestAttnMixer_Forward_Golden(t *testing.T) {
	cfg := AttnConfig{Heads: 4, KVHeads: 2, HeadDim: 8, RotaryDim: 4, RopeTheta: 1e6, NormEps: 1e-6}
	const D, L = 8, 3
	w := &AttnWeights{
		QProj: syn(cfg.Heads*cfg.HeadDim*D, 1),
		KProj: syn(cfg.KVHeads*cfg.HeadDim*D, 2),
		VProj: syn(cfg.KVHeads*cfg.HeadDim*D, 3),
		OProj: syn(D*cfg.Heads*cfg.HeadDim, 4),
		QNorm: syn(cfg.HeadDim, 5),
		KNorm: syn(cfg.HeadDim, 6),
	}
	out, st, err := NewAttnMixer(w, cfg).Forward(syn(L*D, 7), L, D, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	stt := st.(attnState)
	wantOut := []uint32{0xc08bbc1a, 0xbef063d6, 0x401b677e, 0xc0ba7631, 0xbfef3f94, 0x3fcba40d, 0x3fd0f16f, 0x40afcefa, 0xbd60fcd5, 0x3e41dde7, 0x3effaf96, 0xbfa8c070, 0x3e1d28c3, 0x3fabcede, 0xbe44b596, 0x3f19bb3e, 0xbfb7ba85, 0xbe0a72f3, 0x3f87aeb6, 0xc0126fef, 0xbf97ccb3, 0x3f98c4f0, 0x3f41e0d8, 0x3f8a15c0}
	wantK := []uint32{0xbf8df4bc, 0xbf128303, 0xbe234d40, 0x3e04967b, 0x3e96cf92, 0x3ebd7c07, 0xbe4fff68, 0xbc992422, 0xbf0d6be7, 0x3de326c5, 0x3f1588b7, 0x3f5dffb6, 0x3d97cb8f, 0xbedfee5c, 0xbdff2962, 0xbb000bf8, 0x3eb9d031, 0x3efdf818, 0x3f5e1b36, 0xbd919815, 0xbe55f728, 0xbf03abdd, 0x3e2c9b49, 0x3c82a1ac, 0x3f3d0598, 0xbcacc85f, 0x3e849ed1, 0xbf42fb51, 0x3ea6b75b, 0x3ee1fb6f, 0x3e084baa, 0x3b859a6c, 0x3f250273, 0xbf168eba, 0xbf5c31df, 0x3ce6aa50, 0x3e3cdab7, 0x3edefe14, 0xbe4234dd, 0xbc980b70, 0xbce4fbb1, 0xbda899c9, 0xbf359de0, 0x3f126596, 0x3f190746, 0xbecf94f2, 0xbe05ae31, 0xbbc847ab}
	wantV := []uint32{0x4011b717, 0x3f5b22d0, 0xbf1096bb, 0xbffe2823, 0x4024a8c1, 0x3f9374bc, 0xbe89a027, 0xbfd844cf, 0x3f90d845, 0x3fb9580f, 0x3cded285, 0xbfb2617c, 0xbf99652b, 0x3fdf3b64, 0x3ea57a78, 0xbf8c7e28, 0xbfd2fec5, 0xbf352545, 0x3e6ecbfd, 0x3f9645a2, 0xbfec154c, 0xbf675253, 0x3d185f0a, 0x3f7a5e34, 0xbfc1f213, 0xbf8cbfb1, 0xbe229c76, 0x3f483128, 0x3e401a31, 0xbfa5d638, 0xbeb5a858, 0x3f160418, 0x3f90ff97, 0x3f0e8a71, 0xbc9d4954, 0xbf185f06, 0x3fa05bc0, 0x3f2d42c3, 0x3dce703a, 0xbef34d69, 0x3da3d70d, 0x3f4bfb15, 0x3e621964, 0xbeb5dcc6, 0xbfc4c2f7, 0x3f6ab368, 0x3eae7d56, 0xbe70d844}
	chk := func(name string, got []float32, want []uint32) {
		if len(got) != len(want) {
			t.Fatalf("%s len %d, want %d", name, len(got), len(want))
		}
		for i, v := range got {
			if b := math.Float32bits(v); b != want[i] {
				t.Fatalf("%s[%d] bits 0x%08x, want 0x%08x", name, i, b, want[i])
			}
		}
	}
	chk("out", out, wantOut)
	chk("cache K", stt.k, wantK)
	chk("cache V", stt.v, wantV)
}
