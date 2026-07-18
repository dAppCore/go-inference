// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// moe_fuse_test.go gates the composed MoE gate+up fusion (#17): the fused [gate‖up] expert projection
// (model.ConcatQuantRows, synthesised by fuseExpertGateUp at load when Arch.FuseExpertGateUp opts in) and
// the one-matvec swigluExpertQuantInto path it feeds. The load-bearing receipt is byte-identity — the
// fused halves must equal the two separate gate/up matvecs exactly, so the fusion is a pure launch-count
// saving, not a rounding change (TestSwigluExpertQuantInto_FusedMatchesUnfused). The 35B-A3B decode bench
// that justifies the heap-copy RSS cost is the device slice (#18).

// TestSwigluExpertQuantInto_FusedMatchesUnfused is the byte-identity receipt: a packed expert run through
// the fused path (one 2·FF matvec, halves sliced) produces output BYTE-IDENTICAL to the same expert run
// unfused (separate gate + up matvecs) — because matNTQuant dequantises each output row independently, so
// concatenating gate's rows ahead of up's changes nothing but the launch count. Not a tolerance: exact.
func TestSwigluExpertQuantInto_FusedMatchesUnfused(t *testing.T) {
	for _, tc := range []struct{ D, FF, bits, gs int }{
		{64, 24, 4, 8},
		{128, 40, 8, 8},
		{96, 16, 2, 16},
	} {
		unfused, _ := mkMoEExpertQuant(t, tc.D, tc.FF, tc.bits, tc.gs, 1)
		fused := unfused // struct copy shares the QuantWeight pointers; fuse mutates only the copy's fields
		fuseExpertGateUp(&fused)
		if fused.GateUpQ == nil || fused.GateQ != nil || fused.UpQ != nil {
			t.Fatalf("D=%d FF=%d: fuseExpertGateUp left GateUpQ=%v GateQ=%v UpQ=%v",
				tc.D, tc.FF, fused.GateUpQ != nil, fused.GateQ != nil, fused.UpQ != nil)
		}

		xt := make([]float32, tc.D)
		for i := range xt {
			xt[i] = float32((i%7)-3) * 0.1
		}
		gotUnfused := make([]float32, tc.D)
		gotFused := make([]float32, tc.D)
		swigluExpertQuantInto(xt, unfused, tc.D, gotUnfused)
		swigluExpertQuantInto(xt, fused, tc.D, gotFused)

		for i := range gotUnfused {
			if gotFused[i] != gotUnfused[i] {
				t.Fatalf("D=%d FF=%d bits=%d gs=%d: fused[%d]=%v != unfused %v — fusion is NOT byte-identical",
					tc.D, tc.FF, tc.bits, tc.gs, i, gotFused[i], gotUnfused[i])
			}
		}
	}
}

// TestFuseExpertGateUp_Good pins the load-time synthesis: a packed expert's separate GateQ/UpQ become one
// GateUpQ of 2·FF rows and the originals are dropped (their mmap views are no longer read), DownQ untouched.
func TestFuseExpertGateUp_Good(t *testing.T) {
	const D, FF, bits, gs = 64, 24, 4, 8
	e, _ := mkMoEExpertQuant(t, D, FF, bits, gs, 5)
	downBefore := e.DownQ
	fuseExpertGateUp(&e)
	if e.GateUpQ == nil {
		t.Fatal("GateUpQ nil after fuse")
	}
	if e.GateQ != nil || e.UpQ != nil {
		t.Error("separate GateQ/UpQ not dropped after fuse")
	}
	if e.GateUpQ.OutDim != 2*FF {
		t.Errorf("GateUpQ.OutDim = %d, want %d (gate‖up)", e.GateUpQ.OutDim, 2*FF)
	}
	if e.GateUpQ.InDim != D {
		t.Errorf("GateUpQ.InDim = %d, want %d", e.GateUpQ.InDim, D)
	}
	if e.DownQ != downBefore {
		t.Error("DownQ must be untouched by the gate+up fusion")
	}
}

// TestFuseExpertGateUp_Ugly pins the no-op boundaries: a dense (f32) expert is left untouched, and a
// second fuse on an already-fused expert is idempotent (does not re-materialise a fresh concat).
func TestFuseExpertGateUp_Ugly(t *testing.T) {
	dense := MoEExpert{Gate: syn(24*64, 1), Up: syn(24*64, 2), Down: syn(64*24, 3)}
	fuseExpertGateUp(&dense)
	if dense.GateUpQ != nil || dense.Gate == nil {
		t.Error("dense expert must be untouched (no GateQ/UpQ to fuse)")
	}

	e, _ := mkMoEExpertQuant(t, 64, 24, 4, 8, 6)
	fuseExpertGateUp(&e)
	first := e.GateUpQ
	fuseExpertGateUp(&e)
	if e.GateUpQ != first {
		t.Error("re-fusing an already-fused expert re-materialised the concat — not idempotent")
	}
}

// TestMoeExpertFF_Fused pins moeExpertFF's fused branch: GateUpQ.OutDim/2 (gate and up share the concat's
// 2·FF rows), NOT GateUpQ.OutDim.
func TestMoeExpertFF_Fused(t *testing.T) {
	const D, FF = 8, 12
	e := MoEExpert{GateUpQ: &model.QuantWeight{OutDim: 2 * FF, InDim: D}}
	if got := moeExpertFF(&e, D); got != FF {
		t.Fatalf("moeExpertFF(fused) = %d, want %d (GateUpQ.OutDim/2)", got, FF)
	}
}

// TestMoEExpert_Packed pins the forward-dispatch marker: a packed expert reports packed for EITHER the
// separate (GateQ) or fused (GateUpQ) form; a dense expert does not (it runs swigluExpertInto instead).
func TestMoEExpert_Packed(t *testing.T) {
	if !(&MoEExpert{GateQ: &model.QuantWeight{}}).packed() {
		t.Error("expert with GateQ must be packed")
	}
	if !(&MoEExpert{GateUpQ: &model.QuantWeight{}}).packed() {
		t.Error("expert with GateUpQ must be packed")
	}
	if (&MoEExpert{Gate: []float32{1}}).packed() {
		t.Error("dense expert must not be packed")
	}
}

// TestBuildMoE_FuseExpertGateUp pins the loader gating: buildMoE fuses every packed expert's gate+up
// (routed AND shared) when the arch opts in (Arch.FuseExpertGateUp), and leaves them separate when it does
// not — the off-by-default that keeps the composed lane's mmap zero-copy until the device bench (#18)
// justifies the fused-path heap copy.
func TestBuildMoE_FuseExpertGateUp(t *testing.T) {
	const D, FF, nE, bits, gs = 64, 24, 3, 4, 8
	qw := map[string]*model.QuantWeight{}
	put := func(p string, seed int) {
		g, _ := quantiseSynthetic(t, FF, D, bits, gs, seed+1)
		u, _ := quantiseSynthetic(t, FF, D, bits, gs, seed+2)
		d, _ := quantiseSynthetic(t, D, FF, bits, gs, seed+3)
		qw[p+"gate_proj.weight"], qw[p+"up_proj.weight"], qw[p+"down_proj.weight"] = g, u, d
	}
	for e := range nE {
		put("experts."+itoa(e)+".", e*10)
	}
	put("shared_expert.", 900)

	get := func(name string) (safetensors.Tensor, bool) { _, ok := qw[name]; return safetensors.Tensor{}, ok }
	proj := func(name string) ([]float32, *model.QuantWeight, *model.BF16Weight, error) {
		if q, ok := qw[name]; ok {
			return nil, q, nil, nil
		}
		return nil, nil, nil, core.NewError("buildMoE test: unexpected proj " + name)
	}
	f32 := func(string) ([]float32, error) { return syn(nE*D, 500), nil } // router

	build := func(fuse bool) *MoEMLP {
		arch := &model.Arch{Experts: nE, TopK: 2, SharedExperts: 1, FuseExpertGateUp: fuse}
		ffn, err := buildMoE(get, proj, f32, "", &loaderConfig{NumExpertsPerTok: 2}, arch, D)
		if err != nil {
			t.Fatalf("buildMoE(fuse=%v): %v", fuse, err)
		}
		return ffn.(*MoEMLP)
	}

	off := build(false)
	for e := range off.Experts {
		if off.Experts[e].GateUpQ != nil || off.Experts[e].GateQ == nil {
			t.Fatalf("fuse OFF: routed expert %d fused anyway", e)
		}
	}
	if off.Shared.GateUpQ != nil || off.Shared.GateQ == nil {
		t.Fatal("fuse OFF: shared expert fused anyway")
	}

	on := build(true)
	for e := range on.Experts {
		if on.Experts[e].GateUpQ == nil || on.Experts[e].GateQ != nil {
			t.Fatalf("fuse ON: routed expert %d not fused", e)
		}
	}
	if on.Shared.GateUpQ == nil || on.Shared.GateQ != nil {
		t.Fatal("fuse ON: shared expert not fused")
	}
}

// BenchmarkSwigluExpertQuant_Unfused and _Fused measure the fusion on the HOST path (matNTQuantHost): the
// fused path makes ONE quant matvec at 2·FF where the unfused makes two at FF, but on the host that is the
// SAME per-row dequant work with no per-launch overhead to save, so the two come out performance-NEUTRAL
// (measured within noise, ~identical allocs) — the bench's receipt is that fusion adds no host regression.
// The launch-count saving is a DEVICE property (one GPU kernel dispatch per routed expert instead of two);
// the ~34% MoE figure the metal lane measures is the 35B-A3B device bench (#18), not visible on the host.
func BenchmarkSwigluExpertQuant_Unfused(b *testing.B) {
	e, _ := mkMoEExpertQuant(b, 128, 384, 4, 32, 1)
	xt := syn(128, 7)
	out := make([]float32, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		swigluExpertQuantInto(xt, e, 128, out)
	}
}

func BenchmarkSwigluExpertQuant_Fused(b *testing.B) {
	e, _ := mkMoEExpertQuant(b, 128, 384, 4, 32, 1)
	fuseExpertGateUp(&e)
	xt := syn(128, 7)
	out := make([]float32, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for range b.N {
		swigluExpertQuantInto(xt, e, 128, out)
	}
}
