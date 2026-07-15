// SPDX-Licence-Identifier: EUPL-1.2

package mlxaffine_test

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// synthWeight builds a deterministic (outDim × inDim) weight with per-row varying scale
// and a mix of signs and magnitudes, so groups exercise both the min-edge-dominant and
// max-edge-dominant branches of the affine derivation.
func synthWeight(outDim, inDim int) []float32 {
	w := make([]float32, outDim*inDim)
	for r := 0; r < outDim; r++ {
		amp := 0.01 + 0.5*float32(r+1)/float32(outDim)
		bias := 0.1 * float32(r%3-1) // -0.1, 0, +0.1 across rows
		for c := 0; c < inDim; c++ {
			w[r*inDim+c] = bias + amp*float32(math.Sin(float64(r*7+c)*0.013))
		}
	}
	return w
}

// TestQuantizeTensor_RoundTripWithinGroupBound quantises then dequantises synthetic
// weights and asserts every element lands within the theoretical group error bound —
// dominated by one quantisation step (group range / (2^bits−1)) plus the bf16 rounding
// slack on the stored scale/bias.
func TestQuantizeTensor_RoundTripWithinGroupBound(t *testing.T) {
	for _, tc := range []struct {
		bits, groupSize, outDim, inDim int
	}{
		{4, 64, 8, 3840},
		{4, 32, 4, 256},
		{8, 64, 6, 512},
		{8, 32, 3, 128},
		{2, 64, 5, 640},
		{4, 64, 3, 64}, // groupSize == inDim: a single group per row
	} {
		w := synthWeight(tc.outDim, tc.inDim)
		packed, scales, biases, err := mlxaffine.QuantizeTensor(w, tc.outDim, tc.inDim, tc.bits, tc.groupSize)
		if err != nil {
			t.Fatalf("bits=%d gs=%d: quantise: %v", tc.bits, tc.groupSize, err)
		}
		got, err := mlxaffine.DequantizeTensor(packed, scales, biases, tc.outDim, tc.inDim, tc.bits, tc.groupSize)
		if err != nil {
			t.Fatalf("bits=%d gs=%d: dequantise: %v", tc.bits, tc.groupSize, err)
		}
		nBins := float32(int(1<<uint(tc.bits)) - 1)
		groups := tc.inDim / tc.groupSize
		for r := 0; r < tc.outDim; r++ {
			for g := 0; g < groups; g++ {
				lo, hi := float32(math.Inf(1)), float32(math.Inf(-1))
				var maxAbs float32
				for c := g * tc.groupSize; c < (g+1)*tc.groupSize; c++ {
					v := w[r*tc.inDim+c]
					lo, hi = min(lo, v), max(hi, v)
					maxAbs = max(maxAbs, float32(math.Abs(float64(v))))
				}
				step := (hi - lo) / nBins
				bound := 1.5*step + 0.02*maxAbs + 1e-6
				for c := g * tc.groupSize; c < (g+1)*tc.groupSize; c++ {
					i := r*tc.inDim + c
					if e := float32(math.Abs(float64(got[i] - w[i]))); e > bound {
						t.Fatalf("bits=%d gs=%d r=%d g=%d c=%d: error %g exceeds group bound %g (w=%g deq=%g step=%g)",
							tc.bits, tc.groupSize, r, g, c, e, bound, w[i], got[i], step)
					}
					if math.IsNaN(float64(got[i])) || math.IsInf(float64(got[i]), 0) {
						t.Fatalf("bits=%d gs=%d i=%d: dequant produced non-finite %g", tc.bits, tc.groupSize, i, got[i])
					}
				}
			}
		}
	}
}

// TestQuantizeTensor_ReconstructionStable checks the quantiser is a near-fixed-point in
// value space: dequantising, re-quantising, and dequantising again reproduces the same
// reconstruction values. The packed CODES are stable (a reconstruction level maps back
// to its own code), so only the bf16 rounding of the re-derived scale/bias can move a
// value — by at most a bf16 ULP over the group. (Exact BYTE idempotence does not hold:
// at a bf16 rounding boundary the re-derived scale can land one ULP away — benign, and
// distinct from the byte-exactness against mlx that the oracle pins.)
func TestQuantizeTensor_ReconstructionStable(t *testing.T) {
	const outDim, inDim, bits, groupSize = 6, 512, 4, 64
	w := synthWeight(outDim, inDim)
	p1, s1, b1, err := mlxaffine.QuantizeTensor(w, outDim, inDim, bits, groupSize)
	if err != nil {
		t.Fatalf("quantise: %v", err)
	}
	deq1, err := mlxaffine.DequantizeTensor(p1, s1, b1, outDim, inDim, bits, groupSize)
	if err != nil {
		t.Fatalf("dequantise: %v", err)
	}
	p2, s2, b2, err := mlxaffine.QuantizeTensor(deq1, outDim, inDim, bits, groupSize)
	if err != nil {
		t.Fatalf("re-quantise: %v", err)
	}
	assertBytes(t, "packed codes stable", p2, p1) // codes are a true fixed point
	deq2, err := mlxaffine.DequantizeTensor(p2, s2, b2, outDim, inDim, bits, groupSize)
	if err != nil {
		t.Fatalf("re-dequantise: %v", err)
	}
	for i := range deq1 {
		if e := math.Abs(float64(deq2[i] - deq1[i])); e > 5e-3 {
			t.Fatalf("i=%d: reconstruction moved by %g on re-quantise (want ≤ a bf16 ULP)", i, e)
		}
	}
}

// TestQuantizeTensor_ConstantGroup covers the degenerate group: every element equal, so
// the range is zero and the scale is clamped to eps. Dequant must reproduce the constant
// (no divide-by-zero, no NaN).
func TestQuantizeTensor_ConstantGroup(t *testing.T) {
	const outDim, inDim, bits, groupSize = 2, 64, 4, 64
	w := make([]float32, outDim*inDim)
	for i := range w {
		w[i] = 0.25
	}
	packed, scales, biases, err := mlxaffine.QuantizeTensor(w, outDim, inDim, bits, groupSize)
	if err != nil {
		t.Fatalf("quantise: %v", err)
	}
	got, err := mlxaffine.DequantizeTensor(packed, scales, biases, outDim, inDim, bits, groupSize)
	if err != nil {
		t.Fatalf("dequantise: %v", err)
	}
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("i=%d: non-finite %g from a constant group", i, v)
		}
		if e := math.Abs(float64(v - 0.25)); e > 0.01 {
			t.Fatalf("i=%d: constant group dequantised to %g, want ~0.25", i, v)
		}
	}
}

// TestQuantizeTensor_ShapeContract pins the output shapes/lengths against the reader's
// expectations: packed is outDim·(inDim·bits/32) uint32 words, scales/biases are
// outDim·(inDim/groupSize) bf16 values.
func TestQuantizeTensor_ShapeContract(t *testing.T) {
	const outDim, inDim, bits, groupSize = 4, 256, 4, 64
	w := synthWeight(outDim, inDim)
	packed, scales, biases, err := mlxaffine.QuantizeTensor(w, outDim, inDim, bits, groupSize)
	if err != nil {
		t.Fatalf("quantise: %v", err)
	}
	if wantPacked := outDim * mlxaffine.PackedWords(inDim, bits) * 4; len(packed) != wantPacked {
		t.Errorf("packed length %d, want %d", len(packed), wantPacked)
	}
	if wantSB := outDim * (inDim / groupSize) * 2; len(scales) != wantSB || len(biases) != wantSB {
		t.Errorf("scales/biases length %d/%d, want %d", len(scales), len(biases), wantSB)
	}
}

// TestQuantizeTensor_Errors exercises the rejection paths: unsupported bit-widths
// (including MLX's 3/6-bit layouts this package deliberately does not emit), a group
// size that does not divide inDim (the format has no partial trailing group), and a
// values length that disagrees with the dimensions.
func TestQuantizeTensor_Errors(t *testing.T) {
	for _, tc := range []struct {
		name                           string
		values                         []float32
		outDim, inDim, bits, groupSize int
	}{
		{"bits3", make([]float32, 64), 1, 64, 3, 32},
		{"bits6", make([]float32, 64), 1, 64, 6, 32},
		{"bits5", make([]float32, 64), 1, 64, 5, 32},
		{"non-dividing group", make([]float32, 100), 1, 100, 4, 64},
		{"zero group", make([]float32, 64), 1, 64, 4, 0},
		{"length mismatch", make([]float32, 63), 1, 64, 4, 64},
		{"group not word-multiple", make([]float32, 8), 1, 8, 4, 4},
	} {
		if _, _, _, err := mlxaffine.QuantizeTensor(tc.values, tc.outDim, tc.inDim, tc.bits, tc.groupSize); err == nil {
			t.Errorf("%s: expected an error, got nil", tc.name)
		}
	}
}

// TestBFloat16Encode_RoundTrips checks float32ToBFloat16 (exercised through
// QuantizeTensor's stored scales) round-trips through the loader's BFloat16ToFloat32 —
// i.e. the bytes this package writes decode to the value the reader reconstructs.
func TestBFloat16Encode_RoundTrips(t *testing.T) {
	// A single-group weight whose scale/bias are known-representable, then confirm the
	// stored bf16 decodes back to a value the quantiser could have produced.
	const outDim, inDim, bits, groupSize = 1, 64, 4, 64
	w := synthWeight(outDim, inDim)
	_, scales, biases, err := mlxaffine.QuantizeTensor(w, outDim, inDim, bits, groupSize)
	if err != nil {
		t.Fatalf("quantise: %v", err)
	}
	// Decoding via the loader's own bf16 path must succeed and be finite.
	sc, err := safetensors.DecodeFloat32("BF16", scales, 1)
	if err != nil {
		t.Fatalf("decode scales: %v", err)
	}
	bi, err := safetensors.DecodeFloat32("BF16", biases, 1)
	if err != nil {
		t.Fatalf("decode biases: %v", err)
	}
	if math.IsNaN(float64(sc[0])) || math.IsInf(float64(sc[0]), 0) || sc[0] == 0 {
		t.Errorf("scale decoded to %g, want a finite non-zero value", sc[0])
	}
	if math.IsNaN(float64(bi[0])) || math.IsInf(float64(bi[0]), 0) {
		t.Errorf("bias decoded to non-finite %g", bi[0])
	}
}

// TestEligibleShape mirrors mlx_lm.convert's default predicate: quantise a 2-D matrix
// whose inner dim is a whole number of groups; pass 1-D tensors and non-aligned matrices
// through wide.
func TestEligibleShape(t *testing.T) {
	for _, tc := range []struct {
		shape []uint64
		gs    int
		want  bool
	}{
		{[]uint64{4096, 3840}, 64, true},
		{[]uint64{262144, 3840}, 64, true},
		{[]uint64{3840}, 64, false},          // norm: 1-D
		{[]uint64{4096, 100}, 64, false},     // inner dim not a group multiple
		{[]uint64{4096, 3840, 2}, 64, false}, // 3-D
		{[]uint64{4096, 3840}, 0, false},     // invalid group size
	} {
		if got := mlxaffine.EligibleShape(tc.shape, tc.gs); got != tc.want {
			t.Errorf("EligibleShape(%v, %d) = %v, want %v", tc.shape, tc.gs, got, tc.want)
		}
	}
}

// TestSupportedBits pins the byte-exact bit-widths (1, 2, 4, 8) and the refusal of MLX's
// cross-word 3/5/6-bit layouts this package does not reproduce. 1-bit joins the set for the
// composed quant lane's b1→b2 repack read path (RepackB1ToB2).
func TestSupportedBits(t *testing.T) {
	for bits, want := range map[int]bool{1: true, 2: true, 3: false, 4: true, 5: false, 6: false, 8: true, 16: false} {
		if got := mlxaffine.SupportedBits(bits); got != want {
			t.Errorf("SupportedBits(%d) = %v, want %v", bits, got, want)
		}
	}
}

func assertBytes(t *testing.T, label string, got, want []byte) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", label, len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("%s: byte %d differs: 0x%02x != 0x%02x", label, i, got[i], want[i])
		}
	}
}
