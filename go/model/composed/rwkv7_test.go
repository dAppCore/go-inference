// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model/rwkv7"
)

// mkRWKV7Weights builds synthetic RWKV-7 block weights shaped for cfg/D. hk/hv mirror rwkv7.BlockConfig's
// own (unexported) hk()/hv() — re-derived here rather than duplicated via an exported accessor that has
// no OTHER caller (see model/rwkv7/block.go).
func mkRWKV7Weights(cfg rwkv7.BlockConfig, D, seed int) *rwkv7.BlockWeights {
	hk := cfg.NumHeads * cfg.KeyDim
	hv := cfg.NumHeads * cfg.ValueDim
	return &rwkv7.BlockWeights{
		RProj: syn(hk*D, seed+1), WProj: syn(hk*D, seed+2), KProj: syn(hk*D, seed+3),
		VProj: syn(hv*D, seed+4), AProj: syn(hk*D, seed+5), BProj: syn(hk*D, seed+6),
		OutProj: syn(D*hv, seed+7),
	}
}

func mkRWKV7Mixer(cfg rwkv7.BlockConfig, D, seed int) *rwkv7Mixer {
	return &rwkv7Mixer{w: mkRWKV7Weights(cfg, D, seed), cfg: cfg}
}

// TestRWKV7MixerForwardNoProjParity proves rwkv7Mixer.forwardNoProj (the composed-level projMixer
// capability) plus a host projection reproduces rwkv7Mixer.Forward's own output bit-for-bit — the same
// invariant model/rwkv7's own TestBlockForwardScratchNoProjF32_Parity pins at the block level, re-verified
// through the composed Mixer adapter (the prior/next state boxing + interface dispatch this wrapper adds).
func TestRWKV7MixerForwardNoProjParity(t *testing.T) {
	cfg := rwkv7.BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	mm := mkRWKV7Mixer(cfg, D, 11)
	h := syn(L*D, 61)

	wantOut, _, err := mm.Forward(h, L, D, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	mixerHidden, projW, mixCols, _, err := mm.forwardNoProj(h, L, D, nil)
	if err != nil {
		t.Fatalf("forwardNoProj: %v", err)
	}
	if want := cfg.NumHeads * cfg.ValueDim; mixCols != want {
		t.Fatalf("mixCols = %d, want %d", mixCols, want)
	}
	gotOut := matNT(mixerHidden, projW, L, mixCols, D)

	if len(gotOut) != len(wantOut) {
		t.Fatalf("out len %d, want %d", len(gotOut), len(wantOut))
	}
	for i := range wantOut {
		if gotOut[i] != wantOut[i] {
			t.Fatalf("out[%d] = %v, want %v (composed rwkv7Mixer forwardNoProj+host projection diverged from Forward)", i, gotOut[i], wantOut[i])
		}
	}
	t.Logf("composed rwkv7Mixer: forwardNoProj+host projection byte-identical to Forward over [%d,%d], mixCols=%d", L, D, mixCols)
}

// TestComposedForwardEmbTakesRWKV7ProjPath proves forwardEmb dispatches an rwkv7-mixer layer through the
// isProj path (pm.forwardNoProj) rather than the standard path (Mixer.Forward) — the dispatch the
// proj-fused tail depends on to ever engage for an RWKV-7 layer once a backend wires the device hooks.
func TestComposedForwardEmbTakesRWKV7ProjPath(t *testing.T) {
	const D, FF, vocab = 8, 32, 64
	cfg := rwkv7.BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 4}
	counting := &countingMixer{inner: mkRWKV7Mixer(cfg, D, 21)}
	m := &ComposedModel{
		Embed: syn(vocab*D, 100), NormF: syn(D, 101), D: D, Vocab: vocab, Eps: 1e-5,
		Layers: []Layer{{
			InputNorm:    syn(D, 3),
			Mixer:        counting,
			PostAttnNorm: syn(D, 4),
			MLP:          &MLP{Gate: syn(FF*D, 5), Up: syn(FF*D, 6), Down: syn(D*FF, 7), FF: FF},
		}},
	}
	tokens := []int32{1, 2, 3}

	dev, err := NewSession(m).Forward(tokens)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if counting.noProjCalls == 0 {
		t.Fatal("forwardEmb never called forwardNoProj — isProj path not taken for an rwkv7 mixer layer")
	}
	if counting.forwardCalls != 0 {
		t.Fatalf("forwardEmb called Forward %d time(s) — the standard path ran instead of isProj", counting.forwardCalls)
	}

	// Cross-check: an UNWRAPPED session over identical (deterministically re-synthesised) weights produces
	// the same hidden — the counting wrapper is transparent, so this pins the isProj path's OUTPUT, not
	// just that it fired.
	plain := &ComposedModel{
		Embed: m.Embed, NormF: m.NormF, D: D, Vocab: vocab, Eps: 1e-5,
		Layers: []Layer{{
			InputNorm:    m.Layers[0].InputNorm,
			Mixer:        mkRWKV7Mixer(cfg, D, 21),
			PostAttnNorm: m.Layers[0].PostAttnNorm,
			MLP:          m.Layers[0].MLP,
		}},
	}
	host, err := NewSession(plain).Forward(tokens)
	if err != nil {
		t.Fatalf("plain Forward: %v", err)
	}
	if len(dev) != len(host) {
		t.Fatalf("length: %d != %d", len(dev), len(host))
	}
	for i := range dev {
		if dev[i] != host[i] {
			t.Fatalf("hidden[%d] = %v, want %v (counting wrapper altered the result)", i, dev[i], host[i])
		}
	}
	t.Logf("composed forwardEmb: %d forwardNoProj call(s), 0 Forward calls — isProj path confirmed for an rwkv7 layer", counting.noProjCalls)
}
