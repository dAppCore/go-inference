// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import "testing"

// mkTimeMixWeights builds a synthetic time-mix layer's weights for cfg/D. VLora is populated only when
// layerIdx>0 (the real checkpoint's own layer-0 exception — see loader.go).
func mkTimeMixWeights(cfg BlockConfig, D, decayLow, gateLow, aLow, vLow, seed, layerIdx int) *timeMixWeights {
	H, K, V := cfg.NumHeads, cfg.KeyDim, cfg.ValueDim
	Dv := H * V
	w := &timeMixWeights{
		XR: syn(D, seed+1), XW: syn(D, seed+2), XK: syn(D, seed+3),
		XV: syn(D, seed+4), XA: syn(D, seed+5), XG: syn(D, seed+6),
		RProj: syn(D*D, seed+7), KProj: syn(D*D, seed+8),
		VProj: syn(Dv*D, seed+9), OProj: syn(D*Dv, seed+10),
		WLora: mkLora(D, decayLow, D, seed+20, true),
		ALora: mkLora(D, aLow, D, seed+30, true),
		GLora: mkLora(D, gateLow, Dv, seed+40, false),
		KK:    syn(D, seed+11), KA: syn(D, seed+12),
		RK:         syn(H*K, seed+13),
		GroupNormW: syn(Dv, seed+14), GroupNormB: syn(Dv, seed+15),
	}
	if layerIdx > 0 {
		vl := mkLora(D, vLow, Dv, seed+50, true)
		w.VLora = &vl
	}
	return w
}

// TestTimemix_timeMixForward_Good proves the layer-0 shape contract: out is [L,D], vFirstOut is [L,H*V]
// (layer 0 always defines it), and the state carries an [H,K,V] WKV matrix plus a [D] shift register.
func TestTimemix_timeMixForward_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	const L, D = 5, 8
	w := mkTimeMixWeights(cfg, D, 2, 2, 2, 2, 1, 0)
	x := syn(L*D, 100)

	out, vFirst, st, err := timeMixForward(x, w, cfg, 0, nil, timeMixState{}, L, D, 1e-5)
	if err != nil {
		t.Fatalf("timeMixForward: %v", err)
	}
	if len(out) != L*D {
		t.Fatalf("out len %d, want %d", len(out), L*D)
	}
	if len(vFirst) != L*cfg.NumHeads*cfg.ValueDim {
		t.Fatalf("vFirst len %d, want %d", len(vFirst), L*cfg.NumHeads*cfg.ValueDim)
	}
	if len(st.WKV) != cfg.NumHeads*cfg.KeyDim*cfg.ValueDim {
		t.Fatalf("state WKV len %d, want H*K*V=%d", len(st.WKV), cfg.NumHeads*cfg.KeyDim*cfg.ValueDim)
	}
	if len(st.Shift) != D {
		t.Fatalf("state Shift len %d, want D=%d", len(st.Shift), D)
	}
}

func TestTimemix_timeMixForward_Bad(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	if _, _, _, err := timeMixForward(syn(5*8, 1), nil, cfg, 0, nil, timeMixState{}, 5, 8, 1e-5); err == nil {
		t.Fatal("nil weights accepted")
	}
}

// TestTimemix_timeMixForward_Ugly is the decode-boundary invariant: one pass over a sequence is
// bit-exact to two chunks carrying (WKV state, shift register) across the boundary — the same carry
// discipline recurrence.go/block.go pin for the simplified block, now proven for the real chain
// (token-shift + LoRA gates + kk/k-update + GroupNorm + bonus, all state-dependent).
func TestTimemix_timeMixForward_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	const L, split, D = 7, 3, 8
	w := mkTimeMixWeights(cfg, D, 2, 2, 2, 2, 1, 0)
	x := syn(L*D, 100)

	outFull, _, _, err := timeMixForward(x, w, cfg, 0, nil, timeMixState{}, L, D, 1e-5)
	if err != nil {
		t.Fatalf("full: %v", err)
	}
	o1, _, st1, err := timeMixForward(x[:split*D], w, cfg, 0, nil, timeMixState{}, split, D, 1e-5)
	if err != nil {
		t.Fatalf("chunk1: %v", err)
	}
	o2, _, _, err := timeMixForward(x[split*D:], w, cfg, 0, nil, st1, L-split, D, 1e-5)
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
	t.Logf("timeMixForward decode invariant: split %d|%d bit-exact to one-pass", split, L-split)
}

// TestTimemix_timeMixForward_layerGT0 proves the layer>0 branch (value-residual lerp against a supplied
// vFirst) actually uses vFirst: swapping it for a different vector must change the output. A distinct
// scenario from the three canonical Good/Bad/Ugly cases above (all layer 0), covering the loader's other
// real branch.
func TestTimemix_timeMixForward_layerGT0(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	const L, D = 4, 8
	w := mkTimeMixWeights(cfg, D, 2, 2, 2, 2, 1, 1) // layerIdx=1 ⇒ VLora populated
	x := syn(L*D, 200)
	vFirstA := syn(L*cfg.NumHeads*cfg.ValueDim, 300)
	vFirstB := syn(L*cfg.NumHeads*cfg.ValueDim, 301)

	outA, _, _, err := timeMixForward(x, w, cfg, 1, vFirstA, timeMixState{}, L, D, 1e-5)
	if err != nil {
		t.Fatalf("vFirstA: %v", err)
	}
	outB, _, _, err := timeMixForward(x, w, cfg, 1, vFirstB, timeMixState{}, L, D, 1e-5)
	if err != nil {
		t.Fatalf("vFirstB: %v", err)
	}
	same := true
	for i := range outA {
		if outA[i] != outB[i] {
			same = false
			break
		}
	}
	if same {
		t.Fatal("changing vFirst made no difference to a layer>0 forward — value-residual lerp not wired")
	}
}
