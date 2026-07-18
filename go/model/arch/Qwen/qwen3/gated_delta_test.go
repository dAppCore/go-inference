// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import (
	"math"
	"testing"
)

func TestGatedDelta_GatedDeltaConfig_QDim_Good(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, HeadDim: 4}
	if got := cfg.QDim(); got != 8 {
		t.Fatalf("QDim = %d, want 8 (2*4)", got)
	}
}

func TestGatedDelta_GatedDeltaConfig_QDim_Bad(t *testing.T) {
	if got := (GatedDeltaConfig{}).QDim(); got != 0 {
		t.Fatalf("QDim = %d, want 0 for an unconfigured layer", got)
	}
}

// TestGatedDelta_GatedDeltaConfig_QDim_Ugly pins the extreme-GQA edge:
// KeyHeads=1 (every value head reads the same single key head).
func TestGatedDelta_GatedDeltaConfig_QDim_Ugly(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 1, HeadDim: 128}
	if got := cfg.QDim(); got != 128 {
		t.Fatalf("QDim = %d, want 128 (1*128)", got)
	}
}

func TestGatedDelta_GatedDeltaConfig_VDim_Good(t *testing.T) {
	cfg := GatedDeltaConfig{ValueHeads: 4, HeadDim: 4}
	if got := cfg.VDim(); got != 16 {
		t.Fatalf("VDim = %d, want 16 (4*4)", got)
	}
}

func TestGatedDelta_GatedDeltaConfig_VDim_Bad(t *testing.T) {
	if got := (GatedDeltaConfig{}).VDim(); got != 0 {
		t.Fatalf("VDim = %d, want 0 for an unconfigured layer", got)
	}
}

// TestGatedDelta_GatedDeltaConfig_VDim_Ugly proves VDim is independent of
// KeyHeads — unlike QDim, which scales with KeyHeads, not ValueHeads.
func TestGatedDelta_GatedDeltaConfig_VDim_Ugly(t *testing.T) {
	cfg := GatedDeltaConfig{ValueHeads: 8, HeadDim: 4, KeyHeads: 2}
	if got := cfg.VDim(); got != 32 {
		t.Fatalf("VDim = %d, want 32 (8*4, independent of KeyHeads=2)", got)
	}
}

// TestGatedDelta_GatedDeltaConfig_ConvDim_Good pins the q|k|v packed width
// formula (2*qDim + vDim) for a plain MHA-shaped config (KeyHeads == ValueHeads,
// no GQA repeat).
func TestGatedDelta_GatedDeltaConfig_ConvDim_Good(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 4, ValueHeads: 4, HeadDim: 4}
	if got, want := cfg.ConvDim(), 2*cfg.QDim()+cfg.VDim(); got != want {
		t.Fatalf("ConvDim = %d, want %d (2*qDim + vDim)", got, want)
	}
}

func TestGatedDelta_GatedDeltaConfig_ConvDim_Bad(t *testing.T) {
	if got := (GatedDeltaConfig{}).ConvDim(); got != 0 {
		t.Fatalf("ConvDim = %d, want 0 for an unconfigured layer", got)
	}
}

// TestGatedDelta_GatedDeltaConfig_ConvDim_Ugly proves ConvDim packs q|k at the
// COMPRESSED KeyHeads width, not value-head-expanded by the GQA repeat factor
// — the conv projects the pre-repeat q/k, matching qDim()'s own KeyHeads-based
// formula, never the fully-expanded ValueHeads width.
func TestGatedDelta_GatedDeltaConfig_ConvDim_Ugly(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 8, HeadDim: 4} // GQA: 4x repeat
	qDim, vDim := cfg.QDim(), cfg.VDim()
	want := 2*qDim + vDim
	if got := cfg.ConvDim(); got != want || got == 2*vDim+vDim {
		t.Fatalf("ConvDim = %d, want %d (packs the compressed q|k width, never the GQA-expanded one)", got, want)
	}
}

// TestGatedDeltaForwardF32_Golden pins the exact f32 bit-pattern of the block's three outputs (out plus
// both advanced states) for a fixed input, gating alloc-reduction refactors on bit-identical behaviour:
// any change that shifts an output bit fails here. Renamed to TestGatedDelta_GatedDeltaForwardF32_Good:
// a golden bit-pattern pin over real outputs IS the AX-7 "documented happy path with real assertions".
func TestGatedDelta_GatedDeltaForwardF32_Good(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	const L, D = 3, 6
	w := mkGatedDeltaWeights(cfg, D)
	out, nc, nd, err := GatedDeltaForwardF32(gdSyn(L*D, 1), w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("GatedDeltaForwardF32: %v", err)
	}
	wantOut := []uint32{0x3def3267, 0x3e430a0c, 0xbd8376f2, 0xbadc41db, 0xbe45c8e2, 0xbe07c07e, 0xbe2f93fc, 0x3d57897c, 0x3e8386e5, 0x3e476a90, 0x3d996529, 0xbe194d8e, 0xbd7272d5, 0x3dbe9d19, 0x3dc2df96, 0xbbf2891e, 0x3e4acd71, 0xbe38ce94}
	wantNC := []uint32{0x3fae2196, 0xbdd6a161, 0xbf9a6b50, 0x3fd88659, 0x3e67d565, 0xbf600d1b, 0x3f072b00, 0xbf50e560, 0xbf0b4395, 0x3f5bf487, 0xbef837b6, 0xbe59e83d, 0x3f985f06, 0xbe1d4951, 0x3df27bb9, 0x3d89a028, 0xbf93eab3, 0x3ee631f8, 0x3ecbfb15, 0xbf530be0, 0x3f47e282, 0x3f3ac710, 0xbfe49ba5, 0x3f8e5604, 0xbeb4d6a1, 0xbfba36e2, 0x3fb8bac7, 0xbcb4395e, 0xbf8fd21f, 0x3fe31f8a, 0x3e9e4f78, 0xbf4adab9, 0x3f918fc4, 0xbdd6a161, 0xbf7bb2fd, 0x3fb49518, 0x3e2cd9e7, 0xbf35a858, 0x3edd2f18, 0xbf305532, 0xbedf3b63, 0x3f34a233, 0xbed49519, 0xbe264c2e, 0x3f7aacd9, 0xbe10ff97, 0x3de3bcd9, 0x3d271de7, 0xbf7573ea, 0x3ec50481, 0x3ea0f909, 0xbf2f6943, 0x3f288ce7, 0x3f16872a, 0xbfbb22d0, 0x3f6e978c, 0xbe9eb852, 0xbf981d7d, 0x3f9a5119, 0xbd15182d, 0xbf6a3054, 0x3fbd566c, 0x3e72e490, 0xbf2425ae}
	wantND := []uint32{0xbc3016ee, 0x3d54dae4, 0x3bb38a96, 0xbc811334, 0x3df2afb7, 0x3e0dbc25, 0xbbc11f4e, 0x3da93b41, 0x3de4006b, 0x3e1b92de, 0xbbe9597e, 0x3d816eb6, 0xbd0bbb08, 0xbd350116, 0x3bd8f0ae, 0xbb931a57, 0x3d14b750, 0xbe2c70e7, 0x3c23112a, 0x3d3c8c75, 0x3d99d118, 0xbef4f262, 0xbf856cac, 0x3e00f8ba, 0x3d9463e1, 0xbf04ab23, 0xbf98637c, 0x3e0d198f, 0xbb8325fb, 0x3e168c73, 0x3f199606, 0xbd242cda, 0xbced8f11, 0x3e3ed3f1, 0x3bf99f18, 0x3ba65574, 0xbd4cc003, 0x3d4352fe, 0xbce163f3, 0x3bb7cc56, 0xbe10c84e, 0x3dff31f1, 0xbdabc948, 0x3c7b0f91, 0x3dcfdbe2, 0xbd467998, 0x3d8a6acb, 0xbc2a8a9d, 0x3df4b6ca, 0x3e21589f, 0x3d7dfae8, 0xbd031950, 0x3ec63c85, 0x3e0b989c, 0x3e099f57, 0x3cb87eef, 0x3f903018, 0x3ebae0e7, 0x3ec62ec0, 0x3d9c973a, 0xbf57bc6e, 0xbe730931, 0xbe913aa0, 0xbd8df5a9}
	checkGoldenBits(t, "out", out, wantOut)
	checkGoldenBits(t, "newConv", nc, wantNC)
	checkGoldenBits(t, "newDelta", nd, wantND)
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

func gdSyn(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*seed+7)%101-50) * 0.02
	}
	return out
}

func mkGatedDeltaWeights(cfg GatedDeltaConfig, D int) *GatedDeltaWeights {
	return &GatedDeltaWeights{
		InProjQKV:  gdSyn(cfg.ConvDim()*D, 11),
		ConvWeight: gdSyn(cfg.ConvDim()*cfg.ConvKernel, 12),
		ConvBias:   gdSyn(cfg.ConvDim(), 13),
		InProjA:    gdSyn(cfg.ValueHeads*D, 14),
		ALog:       gdSyn(cfg.ValueHeads, 15),
		DtBias:     gdSyn(cfg.ValueHeads, 16),
		InProjB:    gdSyn(cfg.ValueHeads*D, 17),
		InProjZ:    gdSyn(cfg.VDim()*D, 18),
		Norm:       gdSyn(cfg.HeadDim, 19),
		OutProj:    gdSyn(D*cfg.VDim(), 20),
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
	if len(nc) != (cfg.ConvKernel-1)*cfg.ConvDim() || len(nd) != cfg.ValueHeads*cfg.HeadDim*cfg.HeadDim {
		t.Fatalf("state shapes wrong: conv %d delta %d", len(nc), len(nd))
	}
	t.Logf("qwen3 gated-delta block: [%d,%d] in → out, conv-state %d + delta-state %d advanced", L, D, len(nc), len(nd))
}

func TestGatedDelta_GatedDeltaForwardF32_Bad(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	if _, _, _, err := GatedDeltaForwardF32(gdSyn(3*6, 1), nil, cfg, nil, nil, 3, 6); err == nil {
		t.Fatal("nil weights accepted")
	}
}

// TestGatedDelta_GatedDeltaForwardF32_Ugly is the full-block decode invariant: one pass over a sequence is
// BIT-EXACT to two chunks carrying BOTH the conv-state ring AND the delta state across the boundary — Qwen
// 3.6 streaming decode reproduces prefill. A genuine distinct edge from the single-pass _Good case.
func TestGatedDelta_GatedDeltaForwardF32_Ugly(t *testing.T) {
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

// TestGatedDelta_GatedDeltaForwardScratchF32_Good proves a caller-supplied *GatedDeltaScratch
// produces bit-identical output to the nil-scratch (GatedDeltaForwardF32) path, and that the
// scratch's out buffer is populated for reuse.
func TestGatedDelta_GatedDeltaForwardScratchF32_Good(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	const L, D = 3, 6
	w := mkGatedDeltaWeights(cfg, D)
	x := gdSyn(L*D, 1)
	wantOut, wantNC, wantND, err := GatedDeltaForwardF32(x, w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("reference GatedDeltaForwardF32: %v", err)
	}
	sc := &GatedDeltaScratch{}
	out, nc, nd, err := GatedDeltaForwardScratchF32(x, w, cfg, nil, nil, L, D, sc)
	if err != nil {
		t.Fatalf("GatedDeltaForwardScratchF32: %v", err)
	}
	for i := range out {
		if out[i] != wantOut[i] {
			t.Fatalf("scratch out[%d] = %v, want bit-identical %v", i, out[i], wantOut[i])
		}
	}
	if len(nc) != len(wantNC) || len(nd) != len(wantND) {
		t.Fatalf("state shapes: conv %d/%d delta %d/%d", len(nc), len(wantNC), len(nd), len(wantND))
	}
	if len(sc.out) == 0 {
		t.Fatal("scratch.out was not populated for reuse")
	}
}

func TestGatedDelta_GatedDeltaForwardScratchF32_Bad(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	if _, _, _, err := GatedDeltaForwardScratchF32(gdSyn(3*6, 1), nil, cfg, nil, nil, 3, 6, nil); err == nil {
		t.Fatal("nil weights accepted")
	}
}

// TestGatedDelta_GatedDeltaForwardScratchF32_Ugly proves scratch-buffer REUSE across successive
// calls doesn't corrupt correctness: a second call sharing the same *GatedDeltaScratch (fed the
// first call's advanced state) stays bit-identical to the no-scratch reference — the buffers are
// overwritten, not stale-read.
func TestGatedDelta_GatedDeltaForwardScratchF32_Ugly(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	const L, D = 3, 6
	w := mkGatedDeltaWeights(cfg, D)
	x1, x2 := gdSyn(L*D, 1), gdSyn(L*D, 2)
	sc := &GatedDeltaScratch{}
	_, nc1, nd1, err := GatedDeltaForwardScratchF32(x1, w, cfg, nil, nil, L, D, sc)
	if err != nil {
		t.Fatalf("first call: %v", err)
	}
	out2, _, _, err := GatedDeltaForwardScratchF32(x2, w, cfg, nc1, nd1, L, D, sc)
	if err != nil {
		t.Fatalf("second call (reused scratch): %v", err)
	}
	wantOut2, _, _, err := GatedDeltaForwardF32(x2, w, cfg, nc1, nd1, L, D)
	if err != nil {
		t.Fatalf("reference second call: %v", err)
	}
	for i := range out2 {
		if out2[i] != wantOut2[i] {
			t.Fatalf("reused-scratch out[%d] = %v, want %v", i, out2[i], wantOut2[i])
		}
	}
}

// TestGatedDelta_GatedDeltaForwardScratchNoProjF32_Good proves the NoProj variant's pre-projection
// gated output, when projected through out_proj manually, round-trips to EXACTLY the wrapper's
// (GatedDeltaForwardF32's) output bits — the wrapper is nothing more than NoProjF32 + one GEMM.
func TestGatedDelta_GatedDeltaForwardScratchNoProjF32_Good(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	const L, D = 3, 6
	w := mkGatedDeltaWeights(cfg, D)
	x := gdSyn(L*D, 1)
	gated, vDim, nc, nd, err := GatedDeltaForwardScratchNoProjF32(x, w, cfg, nil, nil, L, D, nil)
	if err != nil {
		t.Fatalf("GatedDeltaForwardScratchNoProjF32: %v", err)
	}
	if vDim != cfg.VDim() {
		t.Fatalf("vDim = %d, want %d", vDim, cfg.VDim())
	}
	wantOut, wantNC, wantND, err := GatedDeltaForwardF32(x, w, cfg, nil, nil, L, D)
	if err != nil {
		t.Fatalf("reference: %v", err)
	}
	gotOut := matNT(gated, w.OutProj, L, vDim, D)
	for i := range gotOut {
		if gotOut[i] != wantOut[i] {
			t.Fatalf("projected gated[%d] = %v, want %v (out_proj applied manually must match the wrapper)", i, gotOut[i], wantOut[i])
		}
	}
	if len(nc) != len(wantNC) || len(nd) != len(wantND) {
		t.Fatalf("state shapes: conv %d/%d delta %d/%d", len(nc), len(wantNC), len(nd), len(wantND))
	}
}

func TestGatedDelta_GatedDeltaForwardScratchNoProjF32_Bad(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	if _, _, _, _, err := GatedDeltaForwardScratchNoProjF32(gdSyn(3*6, 1), nil, cfg, nil, nil, 3, 6, nil); err == nil {
		t.Fatal("nil weights accepted")
	}
}

// TestGatedDelta_GatedDeltaForwardScratchNoProjF32_Ugly rejects a non-GQA-divisible geometry
// (ValueHeads not a multiple of KeyHeads) — distinct from _Bad's nil-weights rejection.
func TestGatedDelta_GatedDeltaForwardScratchNoProjF32_Ugly(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 3, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5} // 4 % 3 != 0
	w := mkGatedDeltaWeights(cfg, 6)
	if _, _, _, _, err := GatedDeltaForwardScratchNoProjF32(gdSyn(3*6, 1), w, cfg, nil, nil, 3, 6, nil); err == nil {
		t.Fatal("non-GQA-divisible KeyHeads/ValueHeads accepted")
	}
}

// TestGatedDelta_GatedDeltaForwardScratchFromInputF32_Good proves calling FromInputF32 directly
// with manually-computed input projections (qkv/z/alpha/beta, via the same matNT the package uses
// internally) reproduces EXACTLY the gated output GatedDeltaForwardScratchNoProjF32 gets by
// computing those same projections itself — the two entry points differ only in HOW the inputs
// were produced.
func TestGatedDelta_GatedDeltaForwardScratchFromInputF32_Good(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	const L, D = 3, 6
	w := mkGatedDeltaWeights(cfg, D)
	x := gdSyn(L*D, 1)
	vDim, convDim := cfg.VDim(), cfg.ConvDim()
	qkv := matNT(x, w.InProjQKV, L, D, convDim)
	alpha := matNT(x, w.InProjA, L, D, cfg.ValueHeads)
	beta := matNT(x, w.InProjB, L, D, cfg.ValueHeads)
	zProj := matNT(x, w.InProjZ, L, D, vDim)

	gated, gotVDim, nc, nd, err := GatedDeltaForwardScratchFromInputF32(qkv, zProj, alpha, beta, w, cfg, nil, nil, L, D, nil)
	if err != nil {
		t.Fatalf("GatedDeltaForwardScratchFromInputF32: %v", err)
	}
	if gotVDim != vDim {
		t.Fatalf("vDim = %d, want %d", gotVDim, vDim)
	}
	wantGated, wantVDim, wantNC, wantND, err := GatedDeltaForwardScratchNoProjF32(x, w, cfg, nil, nil, L, D, nil)
	if err != nil {
		t.Fatalf("reference NoProjF32: %v", err)
	}
	if wantVDim != vDim {
		t.Fatalf("reference vDim = %d, want %d", wantVDim, vDim)
	}
	for i := range gated {
		if gated[i] != wantGated[i] {
			t.Fatalf("gated[%d] = %v, want %v (must match NoProjF32 computing the same inputs internally)", i, gated[i], wantGated[i])
		}
	}
	if len(nc) != len(wantNC) || len(nd) != len(wantND) {
		t.Fatalf("state shapes: conv %d/%d delta %d/%d", len(nc), len(wantNC), len(nd), len(wantND))
	}
}

func TestGatedDelta_GatedDeltaForwardScratchFromInputF32_Bad(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5}
	if _, _, _, _, err := GatedDeltaForwardScratchFromInputF32(nil, nil, nil, nil, nil, cfg, nil, nil, 3, 6, nil); err == nil {
		t.Fatal("nil weights accepted")
	}
}

// TestGatedDelta_GatedDeltaForwardScratchFromInputF32_Ugly rejects a non-GQA-divisible geometry
// before ever touching the (here nil) input projections — distinct from _Bad's nil-weights case.
func TestGatedDelta_GatedDeltaForwardScratchFromInputF32_Ugly(t *testing.T) {
	cfg := GatedDeltaConfig{KeyHeads: 3, ValueHeads: 4, HeadDim: 4, ConvKernel: 3, Eps: 1e-5} // 4 % 3 != 0
	w := mkGatedDeltaWeights(cfg, 6)
	if _, _, _, _, err := GatedDeltaForwardScratchFromInputF32(nil, nil, nil, nil, w, cfg, nil, nil, 3, 6, nil); err == nil {
		t.Fatal("non-GQA-divisible KeyHeads/ValueHeads accepted")
	}
}
