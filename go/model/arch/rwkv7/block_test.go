// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	"math"
	"testing"
)

func mkBlockWeights(cfg BlockConfig, D int) *BlockWeights {
	hk, hv := cfg.hk(), cfg.hv()
	return &BlockWeights{
		RProj: syn(hk*D, 11), WProj: syn(hk*D, 12), KProj: syn(hk*D, 13),
		VProj: syn(hv*D, 14), AProj: syn(hk*D, 15), BProj: syn(hk*D, 16),
		OutProj: syn(D*hv, 17),
	}
}

// TestBlockForwardShape checks the block produces [L,D] and advances the [H,K,V] state.
func TestBlockForwardShape(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	out, st, err := BlockForwardF32(syn(L*D, 1), mkBlockWeights(cfg, D), cfg, nil, L, D)
	if err != nil {
		t.Fatalf("BlockForwardF32: %v", err)
	}
	if len(out) != L*D {
		t.Fatalf("out len %d, want %d", len(out), L*D)
	}
	if len(st) != cfg.NumHeads*cfg.KeyDim*cfg.ValueDim {
		t.Fatalf("state len %d, want %d", len(st), cfg.NumHeads*cfg.KeyDim*cfg.ValueDim)
	}
	t.Logf("rwkv7 block: [%d,%d] in → out, [H,K,V] state %d advanced", L, D, len(st))
}

// TestBlock_BlockForwardScratchFromInputF32_Good proves calling FromInputF32 with manually-computed
// input projections (via the same matNT the package uses internally) reproduces EXACTLY what
// BlockForwardScratchNoProjF32 gets by computing those same projections itself.
func TestBlock_BlockForwardScratchFromInputF32_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	w, x := mkBlockWeights(cfg, D), syn(L*D, 21)
	want, _, wantState, err := BlockForwardScratchNoProjF32(x, w, cfg, nil, L, D, nil)
	if err != nil {
		t.Fatal(err)
	}
	hk, hv := cfg.hk(), cfg.hv()
	r := matNT(x, w.RProj, L, D, hk)
	wd := matNT(x, w.WProj, L, D, hk)
	k := matNT(x, w.KProj, L, D, hk)
	v := matNT(x, w.VProj, L, D, hv)
	a := matNT(x, w.AProj, L, D, hk)
	b := matNT(x, w.BProj, L, D, hk)
	got, _, gotState, err := BlockForwardScratchFromInputF32(r, wd, k, v, a, b, w, cfg, nil, L, D, nil)
	if err != nil {
		t.Fatal(err)
	}
	for name, pair := range map[string][2][]float32{"hidden": {got, want}, "state": {gotState, wantState}} {
		for i := range pair[0] {
			if pair[0][i] != pair[1][i] {
				t.Fatalf("%s[%d] = %v, want %v", name, i, pair[0][i], pair[1][i])
			}
		}
	}
}

func TestBlock_BlockForwardScratchFromInputF32_Bad(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	if _, _, _, err := BlockForwardScratchFromInputF32(nil, nil, nil, nil, nil, nil, nil, cfg, nil, 5, 8, nil); err == nil {
		t.Fatal("nil weights accepted")
	}
}

// TestBlock_BlockForwardScratchFromInputF32_Ugly rejects a malformed projected-input length (r one
// element short of L*hk) — distinct from _Bad's nil-weights case.
func TestBlock_BlockForwardScratchFromInputF32_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	hk := cfg.hk()
	r := make([]float32, L*hk-1)
	if _, _, _, err := BlockForwardScratchFromInputF32(r, nil, nil, nil, nil, nil, w, cfg, nil, L, D, nil); err == nil {
		t.Fatal("malformed projected-input length accepted")
	}
}

func TestBlock_BlockForwardF32_Bad(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	if _, _, err := BlockForwardF32(syn(5*8, 1), nil, cfg, nil, 5, 8); err == nil {
		t.Fatal("nil weights accepted")
	}
}

// TestBlock_BlockForwardF32_Ugly is the full-block decode invariant: one pass over a sequence is
// BIT-EXACT to two chunks carrying the [H,K,V] state across the boundary — streaming RWKV-7 decode
// reproduces prefill. A genuine distinct edge from the single-pass golden-bits _Good case.
func TestBlock_BlockForwardF32_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, split, D = 7, 4, 8
	w := mkBlockWeights(cfg, D)
	x := syn(L*D, 1)

	outFull, _, err := BlockForwardF32(x, w, cfg, nil, L, D)
	if err != nil {
		t.Fatalf("full: %v", err)
	}
	o1, s1, err := BlockForwardF32(x[:split*D], w, cfg, nil, split, D)
	if err != nil {
		t.Fatalf("chunk1: %v", err)
	}
	rem := L - split
	o2, _, err := BlockForwardF32(x[split*D:], w, cfg, s1, rem, D)
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
	t.Logf("rwkv7 block decode invariant: split %d|%d, [H,K,V] state carry → output bit-exact to one-pass", split, rem)
}

// TestBlockForwardScratchNoProjF32_Parity proves BlockForwardScratchNoProjF32 (everything up to but NOT
// including out_proj — RWKV-7's time-mix has no gate/norm between the WKV7 recurrence and out_proj, so
// this is just the recurrence read-out) plus a host out_proj GEMM reproduces BlockForwardScratchF32's own
// output and advanced [H,K,V] state bit-for-bit. This is the split the composed session's projMixer path
// relies on: o (the recurrence read-out) @ projW (OutProj) must equal what the full forward computes
// internally, so folding the projection into the FFN-tail command buffer changes WHERE the GEMM runs,
// never its result.
func TestBlock_BlockForwardScratchNoProjF32_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	x := syn(L*D, 21)

	wantOut, wantState, err := BlockForwardF32(x, w, cfg, nil, L, D)
	if err != nil {
		t.Fatalf("BlockForwardF32: %v", err)
	}

	o, hv, gotState, err := BlockForwardScratchNoProjF32(x, w, cfg, nil, L, D, nil)
	if err != nil {
		t.Fatalf("BlockForwardScratchNoProjF32: %v", err)
	}
	if want := cfg.NumHeads * cfg.ValueDim; hv != want {
		t.Fatalf("hv = %d, want %d", hv, want)
	}
	gotOut := matNT(o, w.OutProj, L, hv, D)

	if len(gotOut) != len(wantOut) {
		t.Fatalf("out len %d, want %d", len(gotOut), len(wantOut))
	}
	for i := range wantOut {
		if gotOut[i] != wantOut[i] {
			t.Fatalf("out[%d] = %v, want %v (forwardNoProj+host out_proj diverged from full forward)", i, gotOut[i], wantOut[i])
		}
	}
	if len(gotState) != len(wantState) {
		t.Fatalf("state len %d, want %d", len(gotState), len(wantState))
	}
	for i := range wantState {
		if gotState[i] != wantState[i] {
			t.Fatalf("state[%d] = %v, want %v", i, gotState[i], wantState[i])
		}
	}
	t.Logf("rwkv7 forwardNoProj+host out_proj byte-identical to full BlockForwardF32 over [%d,%d], hv=%d", L, D, hv)
}

func TestBlock_BlockForwardScratchNoProjF32_Bad(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	if _, _, _, err := BlockForwardScratchNoProjF32(syn(5*8, 1), nil, cfg, nil, 5, 8, nil); err == nil {
		t.Fatal("nil weights accepted")
	}
}

// TestBlock_BlockForwardScratchNoProjF32_Ugly rejects a malformed VProj shape (one element short) —
// distinct from _Bad's nil-weights case.
func TestBlock_BlockForwardScratchNoProjF32_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	w.VProj = w.VProj[:len(w.VProj)-1]
	if _, _, _, err := BlockForwardScratchNoProjF32(syn(L*D, 1), w, cfg, nil, L, D, nil); err == nil {
		t.Fatal("malformed VProj shape accepted")
	}
}

// TestBlock_BlockForwardF32_Good pins the exact f32 bit-pattern of the block's output and advanced
// [H,K,V] state for a fixed input, gating the in-place log-decay refactor on bit-identical behaviour —
// a golden bit-pattern pin over real outputs IS the AX-7 "documented happy path with real assertions".
func TestBlock_BlockForwardF32_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 3, 8
	out, st, err := BlockForwardF32(syn(L*D, 1), mkBlockWeights(cfg, D), cfg, nil, L, D)
	if err != nil {
		t.Fatalf("BlockForwardF32: %v", err)
	}
	wantOut := []uint32{0xbf9a719d, 0xbf91f4e0, 0xbf897823, 0xbf80fb66, 0xbf6b6297, 0xbfa4705e, 0xbf9bf3a1, 0xbf9376e4, 0xbf923f4b, 0xbf8a77d1, 0xbf82b057, 0xbf75d1b9, 0xbf5d731b, 0xbf9808f1, 0xbf904177, 0xbf8879fd, 0xbf401f9f, 0xbf365d39, 0xbf2c9ad3, 0xbf22d86c, 0xbf10f7fd, 0xbf419908, 0xbf37d6a2, 0xbf2e143b}
	wantState := []uint32{0x3f8040c6, 0x3f8d0a30, 0xbec1d5b7, 0xbe6cb64f, 0xbd145d49, 0x3e533182, 0x3e93b25b, 0x3e95eac9, 0xbdd26b8e, 0xbd9d38e4, 0xbceed520, 0x3d243e90, 0x3da48b82, 0x3d923263, 0xbcd51bd5, 0xbcd1fee8, 0xbc82609c, 0x3b4df8a7, 0xbe30664d, 0xbe492c08, 0x3d88fa70, 0x3d16c3cb, 0x3a5e2076, 0xbd27ecf4, 0xbdfee1b5, 0x3e7d57fa, 0x3e23d110, 0xbea4bc76, 0x3d4d3fa7, 0x3ce2ad5c, 0xbe48deb2, 0x3ec86ed9, 0x3e7c1c9e, 0xbf03da39, 0x3d94a704, 0x3d352de5, 0x3e6c001b, 0xbeeb905e, 0xbe93ccd4, 0x3f1b1cf3, 0xbdad5a19, 0xbd551e14, 0x3d9dbb8e, 0xbe1d526c, 0xbdc68062, 0x3e4eb86d, 0xbcebbe2c, 0xbc8e0ca4}
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
	chk("state", st, wantState)
}

// TestBlock_BlockForwardScratchF32_Good proves a caller-supplied *BlockScratch produces bit-identical
// output to the nil-scratch (BlockForwardF32) path, and that the scratch's out buffer is populated for
// reuse.
func TestBlock_BlockForwardScratchF32_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	x := syn(L*D, 1)
	wantOut, wantState, err := BlockForwardF32(x, w, cfg, nil, L, D)
	if err != nil {
		t.Fatalf("reference BlockForwardF32: %v", err)
	}
	sc := &BlockScratch{}
	out, state, err := BlockForwardScratchF32(x, w, cfg, nil, L, D, sc)
	if err != nil {
		t.Fatalf("BlockForwardScratchF32: %v", err)
	}
	for i := range out {
		if out[i] != wantOut[i] {
			t.Fatalf("scratch out[%d] = %v, want bit-identical %v", i, out[i], wantOut[i])
		}
	}
	if len(state) != len(wantState) {
		t.Fatalf("state len %d, want %d", len(state), len(wantState))
	}
	if len(sc.out) == 0 {
		t.Fatal("scratch.out was not populated for reuse")
	}
}

func TestBlock_BlockForwardScratchF32_Bad(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	if _, _, err := BlockForwardScratchF32(syn(5*8, 1), nil, cfg, nil, 5, 8, nil); err == nil {
		t.Fatal("nil weights accepted")
	}
}

// TestBlock_BlockForwardScratchF32_Ugly proves scratch-buffer REUSE across successive calls doesn't
// corrupt correctness: a second call sharing the same *BlockScratch (fed the first call's advanced
// state) stays bit-identical to the no-scratch reference.
func TestBlock_BlockForwardScratchF32_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 6}
	const L, D = 5, 8
	w := mkBlockWeights(cfg, D)
	x1, x2 := syn(L*D, 1), syn(L*D, 2)
	sc := &BlockScratch{}
	_, state1, err := BlockForwardScratchF32(x1, w, cfg, nil, L, D, sc)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	out2, _, err := BlockForwardScratchF32(x2, w, cfg, state1, L, D, sc)
	if err != nil {
		t.Fatalf("second call (reused scratch): %v", err)
	}
	wantOut2, _, err := BlockForwardF32(x2, w, cfg, state1, L, D)
	if err != nil {
		t.Fatalf("reference second call: %v", err)
	}
	for i := range out2 {
		if out2[i] != wantOut2[i] {
			t.Fatalf("reused-scratch out[%d] = %v, want %v", i, out2[i], wantOut2[i])
		}
	}
}
