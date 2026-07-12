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

// TestBlockForwardF32_Golden pins the exact f32 bit-pattern of the block's three outputs (out plus the
// advanced conv-state ring and SSM state) for a fixed input, gating alloc-reduction refactors of the
// projection/head-split scratch on bit-identical behaviour (the reference test above is 1e-5 tolerant).
func TestBlockForwardF32_Golden(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	const L, D = 3, 8
	w := mkBlockWeights(cfg, D)
	out, nc, ns, err := BlockForwardF32(syn(L*D, 1), w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("BlockForwardF32: %v", err)
	}
	wantOut := []uint32{0x40439ca3, 0xbca1f3aa, 0xbef48f32, 0xbfc28edb, 0xbfed7b0e, 0xbeec103e, 0x3fc893f8, 0x4052d757, 0xbe963563, 0x3f5eca97, 0x3f66adae, 0x3e94778e, 0xbf52ca1c, 0xbf27c410, 0x3f189267, 0xbed389da, 0x403eff6c, 0x3f938b91, 0xbe6527a9, 0xc00c4740, 0xc0064dd3, 0xbf78b1ca, 0x3fd679bd, 0x403f5eed}
	wantNC := []uint32{0x3fc18936, 0xbe8ebee0, 0xbe958107, 0xbe8793db, 0xbe49eeca, 0xbdb6ae78, 0x3d727bad, 0x3e7df3b6, 0x3e1db22d, 0x3dcd9e81, 0x3db295e7, 0x3dea4a96, 0x3e3a5e36, 0xbf9a36e3, 0xbf820c4a, 0x3f20902d, 0x3f087fcc, 0x3ef58e21, 0x3eeecbfa, 0x3efcb926, 0xbf79580f, 0xbf535a86, 0xbf230553, 0x3f80d1b7, 0x3f69930c, 0x3f5bda50, 0xbf4538ef, 0xbf33eab2, 0xbf1844cf, 0xbee48e8c, 0xbe83e426, 0x3fb15b57, 0x3f9f212d, 0xbe6f9db6, 0xbe837b4a, 0xbe74f0d6, 0xbe398c81, 0xbda99306, 0x3d65604f, 0x3e70d846, 0x3e06c226, 0x3d8c1543, 0x3d3ac711, 0x3d816f06, 0x3df837ae, 0xbf73b645, 0xbf45d638, 0x3f09a027, 0x3ede353f, 0x3ebdd97d, 0x3eb22d0e, 0x3ebb2fee, 0xbf49d494, 0xbf264c2f, 0xbef0d844, 0x3f570a3d, 0x3f3c84b6, 0x3f2c56d4, 0xbf247454, 0xbf159b3d, 0xbef8d4fc, 0xbeb1c432, 0xbe2c0830, 0x3f923a29, 0x3f797246, 0xbe41bda3, 0xbe62eb1c, 0xbe5ab9f4, 0xbe292a30, 0xbd9c779b, 0x3d5844cc, 0x3e63bcd3, 0x3ddfa443, 0x3d15182f, 0x3b831271, 0x3c449bb9, 0x3d7765fa, 0xbf32fec6, 0xbf0793dd, 0x3ee56040, 0x3eab6ae9, 0x3e8624de, 0x3e6b1c43, 0x3e734d6b, 0xbf1a5119, 0xbef27bb4, 0xbe9ba5e3, 0x3f2c710c, 0x3f0f7660, 0x3ef9a6b5, 0xbf03afb8, 0xbeee978d, 0xbec1205c, 0xbe7df3b8, 0xbda0902e, 0x3f6631f8}
	wantNS := []uint32{0x3dc35ab4, 0x3db8bf5c, 0xbdd5c37b, 0x3cfba8b3, 0x3ddbb3a1, 0xbeaebf7c, 0xbd3765bc, 0x3ddc0fb1, 0x3dca4d23, 0x3dca9f00, 0xbdee4b73, 0x3d197e1e, 0x3defb968, 0xbec6ec2f, 0xbd3c414a, 0x3de8072a, 0x3cbc4dae, 0x3c72c6d3, 0xbc650e25, 0x3960bb06, 0x3c93a9ce, 0xbd1ce134, 0xbc46eeaf, 0x3cbea34e, 0x3d5db833, 0x3d662471, 0xbd89fdd5, 0x3cb9653b, 0x3d8851a7, 0xbe66cfd1, 0xbcc772a9, 0x3d80c670, 0xbda3177b, 0xbdb63cec, 0x3dde1897, 0xbd22deed, 0xbdd666c7, 0x3ebe5c39, 0x3d117147, 0xbdc228cc, 0xbda6b2f8, 0xbda30e3a, 0x3dbd06b0, 0xbcec4489, 0xbdc0ac38, 0x3e9db950, 0x3d1eca51, 0xbdbd9a83, 0xbe215a58, 0xbe1c306a, 0x3e34ea04, 0xbd5e2752, 0xbe38e660, 0x3f160f4e, 0x3d991975, 0xbe36f763, 0xbea53d21, 0xbea5a101, 0x3ec3107d, 0xbdfb60ac, 0xbec40bd8, 0x3fa2ae6a, 0x3e19428e, 0xbebd977c, 0x3d3f65bf, 0x3d43969c, 0xbd67cceb, 0x3c995054, 0x3d673b5a, 0xbe425f7d, 0xbcb07009, 0x3d5d00c4, 0x3d0bde3c, 0x3d10d153, 0xbd2b8f73, 0x3c6785a2, 0x3d2abad5, 0xbe110c57, 0xbc8208c2, 0x3d221f93, 0x3cc8d0e2, 0x3cc51f2d, 0xbce92475, 0x3c10b82b, 0x3ceac34d, 0xbdbf0f07, 0xbc360671, 0x3ce522a0, 0xb7eb2255, 0x3a2fbee0, 0xba813692, 0x3a383900, 0x3a39f75b, 0xbb8de4de, 0x3816f0f7, 0x396786fd, 0xbc16aa4e, 0xbc48cfde, 0x3c6845d7, 0xbbd43b8f, 0xbc5fc788, 0x3d63ac2d, 0x3bae7d27, 0xbc3cf0c4, 0xbd35775f, 0xbd62e6e7, 0x3d8bcf0e, 0xbce5e8f8, 0xbd83852f, 0x3e7a3107, 0x3ca5e151, 0xbd607b8c, 0xbdc18560, 0xbd7764e5, 0x3d4c6717, 0x39d5d2ab, 0xbd90916f, 0x3e1e304a, 0x3d69a786, 0xbdc208f6, 0x3d47596c, 0x3d4f2686, 0xbd7607db, 0x3ca69c2d, 0x3d745015, 0xbe4fea13, 0xbcb86723, 0x3d676284}
	checkGoldenBits(t, "out", out, wantOut)
	checkGoldenBits(t, "newConv", nc, wantNC)
	checkGoldenBits(t, "newSSM", ns, wantNS)
}

// TestBlockForwardScratchNoProjF32_Parity proves BlockForwardScratchNoProjF32 (everything up to but NOT
// including out_proj) plus a host out_proj GEMM reproduces BlockForwardScratchF32's own output, advanced
// conv-state and advanced SSM-state bit-for-bit. This is the split the composed session's projMixer path
// relies on: mixerHidden (the returned `gated`) @ projW (OutProj) must equal what the full forward computes
// internally, so folding the projection into the FFN-tail command buffer changes WHERE the GEMM runs, never
// its result. H=4, G=2 exercises the non-trivial group expansion the same as TestBlockForwardGatedNormReference.
func TestBlockForwardScratchNoProjF32_Parity(t *testing.T) {
	cfg := BlockConfig{NumHeads: 4, HeadDim: 4, StateDim: 4, NumGroups: 2, ConvKernel: 3, Eps: 1e-5}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	x := syn(L*D, 41)

	wantOut, wantConv, wantSSM, err := BlockForwardF32(x, w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("BlockForwardF32: %v", err)
	}

	gated, dInner, gotConv, gotSSM, err := BlockForwardScratchNoProjF32(x, w, cfg, nil, nil, L, D, nil)
	if err != nil {
		t.Fatalf("BlockForwardScratchNoProjF32: %v", err)
	}
	if dInner != cfg.dInner() {
		t.Fatalf("dInner = %d, want %d", dInner, cfg.dInner())
	}
	gotOut := matNT(gated, w.OutProj, L, dInner, D)

	if len(gotOut) != len(wantOut) {
		t.Fatalf("out len %d, want %d", len(gotOut), len(wantOut))
	}
	for i := range wantOut {
		if gotOut[i] != wantOut[i] {
			t.Fatalf("out[%d] = %v, want %v (forwardNoProj+host out_proj diverged from full forward)", i, gotOut[i], wantOut[i])
		}
	}
	if len(gotConv) != len(wantConv) {
		t.Fatalf("newConv len %d, want %d", len(gotConv), len(wantConv))
	}
	for i := range wantConv {
		if gotConv[i] != wantConv[i] {
			t.Fatalf("newConv[%d] = %v, want %v", i, gotConv[i], wantConv[i])
		}
	}
	if len(gotSSM) != len(wantSSM) {
		t.Fatalf("newSSM len %d, want %d", len(gotSSM), len(wantSSM))
	}
	for i := range wantSSM {
		if gotSSM[i] != wantSSM[i] {
			t.Fatalf("newSSM[%d] = %v, want %v", i, gotSSM[i], wantSSM[i])
		}
	}
	t.Logf("mamba2 forwardNoProj+host out_proj byte-identical to full BlockForwardF32 over [%d,%d], dInner=%d", L, D, dInner)
}

func checkGoldenBits(t *testing.T, name string, got []float32, want []uint32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len %d, want %d", name, len(got), len(want))
	}
	for i := range got {
		if b := math.Float32bits(got[i]); b != want[i] {
			t.Fatalf("%s[%d] bits 0x%08x, want 0x%08x", name, i, b, want[i])
		}
	}
}
