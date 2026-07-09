// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import "testing"

func gdSyn(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*seed+7)%101-50) * 0.02
	}
	return out
}

func mkGatedDeltaWeights(cfg GatedDeltaConfig, D int) *GatedDeltaWeights {
	return &GatedDeltaWeights{
		InProjQKV:  gdSyn(cfg.convDim()*D, 11),
		ConvWeight: gdSyn(cfg.convDim()*cfg.ConvKernel, 12),
		ConvBias:   gdSyn(cfg.convDim(), 13),
		InProjA:    gdSyn(cfg.ValueHeads*D, 14),
		ALog:       gdSyn(cfg.ValueHeads, 15),
		DtBias:     gdSyn(cfg.ValueHeads, 16),
		InProjB:    gdSyn(cfg.ValueHeads*D, 17),
		InProjZ:    gdSyn(cfg.vDim()*D, 18),
		Norm:       gdSyn(cfg.HeadDim, 19),
		OutProj:    gdSyn(D*cfg.vDim(), 20),
	}
}

// TestGatedDeltaForwardShape checks the block produces [L,D] and advances both state slots (conv ring +
// delta state).
func TestGatedDeltaForwardShape(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 8, ConvKernel: 4, Eps: 1e-5}
	const L, D = 5, 8
	out, nc, nd, err := GatedDeltaForwardF32(gdSyn(L*D, 1), mkGatedDeltaWeights(cfg, D), cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("GatedDeltaForwardF32: %v", err)
	}
	if len(out) != L*D {
		t.Fatalf("out len %d, want %d", len(out), L*D)
	}
	if len(nc) != (cfg.ConvKernel-1)*cfg.convDim() || len(nd) != cfg.ValueHeads*cfg.HeadDim*cfg.HeadDim {
		t.Fatalf("state shapes wrong: conv %d delta %d", len(nc), len(nd))
	}
	t.Logf("qwen3 gated-delta block: [%d,%d] in → out, conv-state %d + delta-state %d advanced", L, D, len(nc), len(nd))
}

// TestGatedDeltaForwardCarry is the full-block decode invariant: one pass over a sequence is BIT-EXACT to
// two chunks carrying BOTH the conv-state ring AND the delta state across the boundary — Qwen 3.6 streaming
// decode reproduces prefill.
func TestGatedDeltaForwardCarry(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 8, ConvKernel: 4, Eps: 1e-5}
	const L, split, D = 7, 4, 8
	w := mkGatedDeltaWeights(cfg, D)
	x := gdSyn(L*D, 1)

	outFull, _, _, err := GatedDeltaForwardF32(x, w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("full: %v", err)
	}
	o1, nc1, nd1, err := GatedDeltaForwardF32(x[:split*D], w, cfg, nil, nil, split, D)
	if err != nil {
		t.Fatalf("chunk1: %v", err)
	}
	rem := L - split
	o2, _, _, err := GatedDeltaForwardF32(x[split*D:], w, cfg, nc1, nd1, rem, D)
	if err != nil {
		t.Fatalf("chunk2: %v", err)
	}
	for i := range o1 {
		if o1[i] != outFull[i] {
			t.Fatalf("chunk1 out[%d] = %v != full %v", i, o1[i], outFull[i])
		}
	}
	for i := range o2 {
		if o2[i] != outFull[split*D+i] {
			t.Fatalf("chunk2 out[%d] = %v != full %v", i, o2[i], outFull[split*D+i])
		}
	}
	t.Logf("qwen3 gated-delta decode invariant: split %d|%d, conv + delta state carry → output bit-exact to one-pass", split, rem)
}
