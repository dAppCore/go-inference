// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"math"
	"testing"
)

func mkBlockWeights(cfg BlockConfig, D int) *BlockWeights {
	return &BlockWeights{
		InProj:     syn(cfg.projDim()*D, 11),
		ConvWeight: syn(cfg.convDim()*cfg.ConvKernel, 12),
		ConvBias:   syn(cfg.convDim(), 13),
		ALog:       syn(cfg.NumHeads, 14),
		D:          syn(cfg.NumHeads, 15),
		DtBias:     syn(cfg.NumHeads, 16),
		Norm:       syn(cfg.dInner(), 17),
		OutProj:    syn(D*cfg.dInner(), 18),
	}
}

// TestBlockForwardShape checks the block produces [L,D] and advances both state slots.
func TestBlockForwardShape(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	out, nc, ns, err := BlockForwardF32(syn(L*D, 1), w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("BlockForwardF32: %v", err)
	}
	if len(out) != L*D {
		t.Fatalf("out len %d, want %d", len(out), L*D)
	}
	if len(nc) != (cfg.ConvKernel-1)*cfg.convDim() || len(ns) != cfg.NumHeads*cfg.HeadDim*cfg.StateDim {
		t.Fatalf("state shapes wrong: conv %d ssm %d", len(nc), len(ns))
	}
	t.Logf("mamba2 block: [%d,%d] in → out, conv-state %d + ssm-state %d advanced", L, D, len(nc), len(ns))
}

// TestBlockForwardGatedNormReference pins the block's glue stages — the projection split, the
// B/C group→head expansion, dt = softplus(dt + dt_bias), A = −exp(A_log), and above all the
// gated RMSNorm ORDERING — against an independent in-test pipeline. The two core ops
// (CausalConv1dF32, SSDScanF32) are reused because each has its own closed-form test; every
// glue stage between them is re-derived here from the documented layout, so a regression in
// any of them diverges from this reference. The load-bearing pin is the gate order: the
// reference computes g = y·SiLU(z) FIRST and normalises g (HF MambaRMSNormGated) — the
// gate-AFTER form (RMSNorm(y)·SiLU(z), which metal's shared flakernel used) is a CONFIRMED
// past real bug (~5× activation inflation on a real mamba2 checkpoint, per block.go's own
// doc), and the shape/carry tests cannot catch it (both sides of a carry comparison share the
// wrong order). H=4, G=2 so the group expansion is non-trivial (2 heads per group).
func TestBlockForwardGatedNormReference(t *testing.T) {
	cfg := BlockConfig{NumHeads: 4, HeadDim: 4, StateDim: 4, NumGroups: 2, ConvKernel: 3, Eps: 1e-5}
	const L, D = 3, 8
	H, P, N, G, K := cfg.NumHeads, cfg.HeadDim, cfg.StateDim, cfg.NumGroups, cfg.ConvKernel
	dInner, convDim, projDim := cfg.dInner(), cfg.convDim(), cfg.projDim()
	w := mkBlockWeights(cfg, D)
	x := syn(L*D, 21)

	got, _, _, err := BlockForwardF32(x, w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("BlockForwardF32: %v", err)
	}

	// ---- independent reference (documented layout: in-proj → z | xBC | dt → conv → SiLU →
	// x | B | C with group expansion → dt/A transforms → scan → gate-before-norm → out-proj).
	proj := matNT(x, w.InProj, L, D, projDim)
	z := make([]float32, L*dInner)
	xBC := make([]float32, L*convDim)
	dtRaw := make([]float32, L*H)
	for tt := range L {
		row := proj[tt*projDim:]
		copy(z[tt*dInner:(tt+1)*dInner], row[0:dInner])
		copy(xBC[tt*convDim:(tt+1)*convDim], row[dInner:dInner+convDim])
		copy(dtRaw[tt*H:(tt+1)*H], row[dInner+convDim:dInner+convDim+H])
	}
	convOut, _, err := CausalConv1dF32(xBC, w.ConvWeight, w.ConvBias, nil, L, convDim, K)
	if err != nil {
		t.Fatalf("reference conv: %v", err)
	}
	for i := range convOut {
		convOut[i] = float32(silu(float64(convOut[i])))
	}
	xHeads := make([]float32, L*H*P)
	bHeads := make([]float32, L*H*N)
	cHeads := make([]float32, L*H*N)
	headsPerGroup := H / G
	for tt := range L {
		crow := convOut[tt*convDim:]
		copy(xHeads[tt*dInner:(tt+1)*dInner], crow[0:dInner])
		for h := range H {
			g := h / headsPerGroup
			copy(bHeads[(tt*H+h)*N:(tt*H+h+1)*N], crow[dInner+g*N:dInner+g*N+N])
			copy(cHeads[(tt*H+h)*N:(tt*H+h+1)*N], crow[dInner+G*N+g*N:dInner+G*N+g*N+N])
		}
	}
	dt := make([]float32, L*H)
	for i := 0; i < L*H; i++ {
		dt[i] = float32(softplus(float64(dtRaw[i]) + float64(w.DtBias[i%H])))
	}
	a := make([]float32, H)
	for h := range H {
		a[h] = float32(-math.Exp(float64(w.ALog[h])))
	}
	y, _, err := SSDScanF32(xHeads, dt, a, bHeads, cHeads, w.D, nil, L, H, P, N)
	if err != nil {
		t.Fatalf("reference scan: %v", err)
	}
	// gate BEFORE norm — the documented correct order.
	gated := make([]float32, L*dInner)
	for tt := range L {
		g := make([]float64, dInner)
		var ss float64
		for i := range dInner {
			g[i] = float64(y[tt*dInner+i]) * silu(float64(z[tt*dInner+i]))
			ss += g[i] * g[i]
		}
		rms := math.Sqrt(ss/float64(dInner) + float64(cfg.Eps))
		for i := range dInner {
			gated[tt*dInner+i] = float32(g[i] / rms * float64(w.Norm[i]))
		}
	}
	want := matNT(gated, w.OutProj, L, dInner, D)

	for i := range want {
		diff := math.Abs(float64(got[i]) - float64(want[i]))
		if diff > 1e-5*(1+math.Abs(float64(want[i]))) {
			t.Fatalf("out[%d] = %v, want %v — block diverged from the documented pipeline (gate-before-norm / dt / A / group expansion)", i, got[i], want[i])
		}
	}
	t.Logf("block matches the independent reference: split, %d-group expansion, softplus(dt+bias), −exp(A_log), gate-BEFORE-norm all pinned", G)
}

// TestBlockForwardCarry is the full-block decode invariant: running the block over a sequence in one
// pass is BIT-EXACT to running it as two chunks that carry BOTH the conv-state ring AND the SSM state
// across the boundary — so streaming Mamba-2 decode (state resident across calls) reproduces the
// one-pass prefill exactly.
func TestBlockForwardCarry(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const L, split, D = 7, 4, 8
	w := mkBlockWeights(cfg, D)
	x := syn(L*D, 1)

	outFull, _, _, err := BlockForwardF32(x, w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("full: %v", err)
	}
	o1, nc1, ns1, err := BlockForwardF32(x[:split*D], w, cfg, nil, nil, split, D)
	if err != nil {
		t.Fatalf("chunk1: %v", err)
	}
	rem := L - split
	o2, _, _, err := BlockForwardF32(x[split*D:], w, cfg, nc1, ns1, rem, D)
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
	t.Logf("mamba2 block decode invariant: split %d|%d, conv-state + SSM-state carry → output bit-exact to one-pass", split, rem)
}
