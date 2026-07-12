// SPDX-Licence-Identifier: EUPL-1.2

package qwen3

import (
	"math"
	"testing"
)

// TestGatedDeltaForwardF32_Golden pins the exact f32 bit-pattern of the block's three outputs (out plus
// both advanced states) for a fixed input, gating alloc-reduction refactors on bit-identical behaviour:
// any change that shifts an output bit fails here.
func TestGatedDeltaForwardF32_Golden(t *testing.T) {
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
