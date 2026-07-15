// SPDX-Licence-Identifier: EUPL-1.2

package mlxaffine_test

import (
	"testing"

	"dappco.re/go/inference/model/quant/mlxaffine"
)

// b1HostDequant is the reference the repack is gated against: the b1 pack read straight
// through DequantizeTensor at bits=1. Writing it first (before the repack) is the whole
// point — the repack must reproduce THIS, byte-for-byte, through the b2 kernels.
func b1HostDequant(t *testing.T, packed, scales, biases []byte, outDim, inDim, groupSize int) []float32 {
	t.Helper()
	got, err := mlxaffine.DequantizeTensor(packed, scales, biases, outDim, inDim, 1, groupSize)
	if err != nil {
		t.Fatalf("b1 host dequant: %v", err)
	}
	return got
}

// TestRepackB1ToB2_ExactDequant is the core exactness gate: a synthetic b1 pack, widened to
// b2, must dequantise byte-identically (bit-for-bit float32) to the b1 host dequant — the
// repack changes only the code WIDTH, never the value (w = scale·q + bias, q ∈ {0,1}).
func TestRepackB1ToB2_ExactDequant(t *testing.T) {
	for _, tc := range []struct {
		outDim, inDim, groupSize int
	}{
		{4, 128, 128}, // Bonsai's gs
		{8, 256, 128},
		{3, 64, 64},
		{5, 96, 32},
	} {
		w := synthWeight(tc.outDim, tc.inDim)
		packed, scales, biases, err := mlxaffine.QuantizeTensor(w, tc.outDim, tc.inDim, 1, tc.groupSize)
		if err != nil {
			t.Fatalf("gs=%d: quantise b1: %v", tc.groupSize, err)
		}
		ref := b1HostDequant(t, packed, scales, biases, tc.outDim, tc.inDim, tc.groupSize)

		p2, s2, b2, err := mlxaffine.RepackB1ToB2(packed, scales, biases, tc.outDim, tc.inDim, tc.groupSize)
		if err != nil {
			t.Fatalf("gs=%d: repack: %v", tc.groupSize, err)
		}
		if want := tc.outDim * mlxaffine.PackedWords(tc.inDim, 2) * 4; len(p2) != want {
			t.Fatalf("gs=%d: repacked length %d, want %d (b2 = inDim/16 words/row)", tc.groupSize, len(p2), want)
		}
		assertBytes(t, "scales carried through", s2, scales)
		assertBytes(t, "biases carried through", b2, biases)

		got, err := mlxaffine.DequantizeTensor(p2, s2, b2, tc.outDim, tc.inDim, 2, tc.groupSize)
		if err != nil {
			t.Fatalf("gs=%d: dequant b2: %v", tc.groupSize, err)
		}
		if len(got) != len(ref) {
			t.Fatalf("gs=%d: length %d != %d", tc.groupSize, len(got), len(ref))
		}
		for i := range got {
			if got[i] != ref[i] { // exact float32 equality — same scale·q + bias
				t.Fatalf("gs=%d i=%d: b2 dequant %g != b1 dequant %g (repack must be exact)", tc.groupSize, i, got[i], ref[i])
			}
		}
	}
}

// TestRepackB1ToB2_Errors exercises the rejection paths: a non-32-multiple inDim (b1 packs
// 32 codes per word), a wrong packed length, and a scales/biases length that disagrees.
func TestRepackB1ToB2_Errors(t *testing.T) {
	const outDim, inDim, groupSize = 2, 128, 128
	packed, scales, biases, err := mlxaffine.QuantizeTensor(synthWeight(outDim, inDim), outDim, inDim, 1, groupSize)
	if err != nil {
		t.Fatalf("setup quantise: %v", err)
	}
	for _, tc := range []struct {
		name                     string
		packed, scales, biases   []byte
		outDim, inDim, groupSize int
	}{
		{"inDim not 32-multiple", packed, scales, biases, outDim, 100, groupSize},
		{"packed length", packed[:len(packed)-4], scales, biases, outDim, inDim, groupSize},
		{"scales length", packed, scales[:len(scales)-2], biases, outDim, inDim, groupSize},
		{"zero group", packed, scales, biases, outDim, inDim, 0},
	} {
		if _, _, _, err := mlxaffine.RepackB1ToB2(tc.packed, tc.scales, tc.biases, tc.outDim, tc.inDim, tc.groupSize); err == nil {
			t.Errorf("%s: expected an error, got nil", tc.name)
		}
	}
}
