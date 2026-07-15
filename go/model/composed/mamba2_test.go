// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	"dappco.re/go/inference/model/mamba2"
)

// mkMamba2Weights builds synthetic Mamba-2 block weights shaped for cfg/D. The projDim/convDim/dInner
// formulas mirror mamba2.BlockConfig's own (unexported, so re-derived here rather than duplicated via an
// exported accessor that has no OTHER caller — see model/mamba2/block.go's dInner/convDim/projDim).
func mkMamba2Weights(cfg mamba2.BlockConfig, D, seed int) *mamba2.BlockWeights {
	dInner := cfg.NumHeads * cfg.HeadDim
	convDim := dInner + 2*cfg.NumGroups*cfg.StateDim
	projDim := 2*dInner + 2*cfg.NumGroups*cfg.StateDim + cfg.NumHeads
	return &mamba2.BlockWeights{
		InProj:     syn(projDim*D, seed+1),
		ConvWeight: syn(convDim*cfg.ConvKernel, seed+2),
		ConvBias:   syn(convDim, seed+3),
		ALog:       syn(cfg.NumHeads, seed+4),
		D:          syn(cfg.NumHeads, seed+5),
		DtBias:     syn(cfg.NumHeads, seed+6),
		Norm:       syn(dInner, seed+7),
		OutProj:    syn(D*dInner, seed+8),
	}
}

func mkMamba2Mixer(cfg mamba2.BlockConfig, D, seed int) *mamba2Mixer {
	return &mamba2Mixer{w: mkMamba2Weights(cfg, D, seed), cfg: cfg}
}

// TestMamba2MixerForwardNoProjParity proves mamba2Mixer.forwardNoProj (the composed-level projMixer
// capability) plus a host projection reproduces mamba2Mixer.Forward's own output bit-for-bit — the same
// invariant model/mamba2's own TestBlockForwardScratchNoProjF32_Parity pins at the block level, re-verified
// through the composed Mixer adapter (the prior/next state boxing + interface dispatch this wrapper adds).
func TestMamba2MixerForwardNoProjParity(t *testing.T) {
	cfg := mamba2.BlockConfig{NumHeads: 4, HeadDim: 4, StateDim: 4, NumGroups: 2, ConvKernel: 3, Eps: 1e-5}
	const L, D = 5, 16
	mm := mkMamba2Mixer(cfg, D, 1)
	h := syn(L*D, 51)

	wantOut, _, err := mm.Forward(h, L, D, nil)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}

	mixerHidden, projW, mixCols, _, err := mm.forwardNoProj(h, L, D, nil)
	if err != nil {
		t.Fatalf("forwardNoProj: %v", err)
	}
	if want := cfg.NumHeads * cfg.HeadDim; mixCols != want {
		t.Fatalf("mixCols = %d, want %d", mixCols, want)
	}
	gotOut := matNT(mixerHidden, projW, L, mixCols, D)

	if len(gotOut) != len(wantOut) {
		t.Fatalf("out len %d, want %d", len(gotOut), len(wantOut))
	}
	for i := range wantOut {
		if gotOut[i] != wantOut[i] {
			t.Fatalf("out[%d] = %v, want %v (composed mamba2Mixer forwardNoProj+host projection diverged from Forward)", i, gotOut[i], wantOut[i])
		}
	}
	t.Logf("composed mamba2Mixer: forwardNoProj+host projection byte-identical to Forward over [%d,%d], mixCols=%d", L, D, mixCols)
}

// TestComposedForwardEmbTakesMamba2ProjPath proves forwardEmb dispatches a mamba2-mixer layer through the
// isProj path (pm.forwardNoProj) rather than the standard path (Mixer.Forward) — the dispatch the
// proj-fused tail depends on to ever engage for a Mamba-2 layer once a backend wires the device hooks.
func TestComposedForwardEmbTakesMamba2ProjPath(t *testing.T) {
	const D, FF, vocab = 16, 32, 64
	cfg := mamba2.BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	counting := &countingMixer{inner: mkMamba2Mixer(cfg, D, 2)}
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
		t.Fatal("forwardEmb never called forwardNoProj — isProj path not taken for a mamba2 mixer layer")
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
			Mixer:        mkMamba2Mixer(cfg, D, 2),
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
	t.Logf("composed forwardEmb: %d forwardNoProj call(s), 0 Forward calls — isProj path confirmed for a mamba2 layer", counting.noProjCalls)
}
