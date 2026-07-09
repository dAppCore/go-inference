// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import "testing"

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
