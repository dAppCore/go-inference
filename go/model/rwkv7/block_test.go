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

// TestBlockForwardCarry is the full-block decode invariant: one pass over a sequence is BIT-EXACT to two
// chunks carrying the [H,K,V] state across the boundary — streaming RWKV-7 decode reproduces prefill.
func TestBlockForwardCarry(t *testing.T) {
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

// TestBlockForwardF32_Golden pins the exact f32 bit-pattern of the block's output and advanced [H,K,V]
// state for a fixed input, gating the in-place log-decay refactor on bit-identical behaviour.
func TestBlockForwardF32_Golden(t *testing.T) {
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
