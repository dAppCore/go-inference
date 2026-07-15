// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"fmt"
	"math"
	"testing"
)

// f32ToBf16 encodes a float32 to a bf16 uint16 using the same round-to-nearest-
// even truncation the package uses on the dequant side, so integers and other
// bf16-representable values round-trip exactly.
func f32ToBf16(f float32) uint16 {
	b := math.Float32bits(f)
	b += 0x7FFF + ((b >> 16) & 1)
	return uint16(b >> 16)
}

// f32GroupToBf16 encodes a slice of float32 into little-endian bf16 bytes — the
// layout kvQuantRows / kvQ8QuantRows read (low byte first = top f32 bits 16-23).
func f32GroupToBf16(vals []float32) []byte {
	out := make([]byte, len(vals)*2)
	for i, v := range vals {
		h := f32ToBf16(v)
		out[i*2] = byte(h)
		out[i*2+1] = byte(h >> 8)
	}
	return out
}

// bf16At decodes element i of a bf16 byte buffer back to float32.
func bf16At(b []byte, i int) float32 {
	raw := uint32(b[i*2]) | uint32(b[i*2+1])<<8
	return math.Float32frombits(raw << 16)
}

// relL2 is the relative L2 (Euclidean) reconstruction error of xhat against x.
func relL2(x, xhat []float32) float64 {
	var num, den float64
	for i := range x {
		d := float64(xhat[i]) - float64(x[i])
		num += d * d
		den += float64(x[i]) * float64(x[i])
	}
	if den == 0 {
		return 0
	}
	return math.Sqrt(num / den)
}

// roundTripRelL2 quantises one bf16 group at the given bit-depth, dequantises,
// and returns the rel-L2 error between the original and reconstructed values.
func roundTripRelL2(t *testing.T, src []byte, bits int) float64 {
	t.Helper()
	elems := len(src) / 2
	packed := make([]byte, kvQuantPackedLen(elems, bits))
	scales := make([]float32, elems/kvQ8GroupSize)
	kvQuantRows(packed, scales, src, bits)

	dst := make([]byte, elems*2)
	kvDequantRows(dst, packed, scales, bits)

	orig := make([]float32, elems)
	recon := make([]float32, elems)
	for i := range orig {
		orig[i] = bf16At(src, i)
		recon[i] = bf16At(dst, i)
	}
	return relL2(orig, recon)
}

// realisticGroup builds one 64-element bf16 group with varied magnitudes: a
// deterministic pseudo-random spread in [-3,3), a forced near-zero, and a
// forced outlier that pins the group scale.
func realisticGroup() []byte {
	vals := make([]float32, kvQ8GroupSize)
	seed := uint32(0x1234_5678)
	for i := range vals {
		seed = seed*1664525 + 1013904223
		u := float32(seed>>8) / float32(1<<24) // [0,1)
		vals[i] = (u*2 - 1) * 3.0              // [-3,3)
	}
	vals[7] = 1e-3  // near-zero
	vals[40] = 12.0 // outlier — sets maxabs, hence the group scale
	return f32GroupToBf16(vals)
}

// TestNBitQuantRoundTripMonotonic asserts more bits → strictly less error:
// rel-L2(q4) > rel-L2(q6) > rel-L2(q8) on a realistic group.
func TestNBitQuantRoundTripMonotonic(t *testing.T) {
	src := realisticGroup()
	e4 := roundTripRelL2(t, src, 4)
	e6 := roundTripRelL2(t, src, 6)
	e8 := roundTripRelL2(t, src, 8)
	t.Logf("rel-L2: q4=%.6g q6=%.6g q8=%.6g", e4, e6, e8)
	if !(e4 > e6) {
		t.Errorf("expected rel-L2 q4 > q6, got q4=%.6g q6=%.6g", e4, e6)
	}
	if !(e6 > e8) {
		t.Errorf("expected rel-L2 q6 > q8, got q6=%.6g q8=%.6g", e6, e8)
	}
}

// TestKVQuantByteIdenticalToQ8 asserts the bits==8 path is byte-for-byte the
// existing q8: same packed bytes, same scales, same dequantised bf16 bytes.
func TestKVQuantByteIdenticalToQ8(t *testing.T) {
	src := realisticGroup()
	elems := len(src) / 2
	groups := elems / kvQ8GroupSize

	// existing q8
	q8 := make([]int8, elems)
	scalesQ8 := make([]float32, groups)
	kvQ8QuantRows(q8, scalesQ8, src)

	// new N-bit path at bits==8
	packed := make([]byte, kvQuantPackedLen(elems, 8))
	scalesN := make([]float32, groups)
	kvQuantRows(packed, scalesN, src, 8)

	for i := range q8 {
		if byte(q8[i]) != packed[i] {
			t.Fatalf("packed byte %d differs: q8=%d nbit=%d", i, byte(q8[i]), packed[i])
		}
	}
	for g := range scalesQ8 {
		if scalesQ8[g] != scalesN[g] {
			t.Fatalf("scale %d differs: q8=%v nbit=%v", g, scalesQ8[g], scalesN[g])
		}
	}

	// dequant must also match byte-for-byte
	dstQ8 := make([]byte, elems*2)
	kvQ8DequantRows(dstQ8, q8, scalesQ8)
	dstN := make([]byte, elems*2)
	kvDequantRows(dstN, packed, scalesN, 8)
	if !bytes.Equal(dstQ8, dstN) {
		t.Fatalf("dequant bytes differ between kvQ8DequantRows and kvDequantRows(bits=8)")
	}
}

// TestNBitQuantGridExact asserts values already on the quant grid reconstruct
// exactly. For each bit-depth we build a group whose max magnitude equals qmax
// and whose other values are integers within range, so scale == 1 and every
// value is bf16-representable — quant→dequant is a no-op.
func TestNBitQuantGridExact(t *testing.T) {
	for _, bits := range []int{4, 6, 8} {
		qmax := (1 << (bits - 1)) - 1
		vals := make([]float32, kvQ8GroupSize)
		vals[0] = float32(qmax) // pins scale = qmax/qmax = 1
		// a deterministic spread of integers within [-qmax, qmax]
		for i := 1; i < kvQ8GroupSize; i++ {
			step := (2*qmax + 1)
			vals[i] = float32((i*7)%step - qmax)
		}
		src := f32GroupToBf16(vals)

		packed := make([]byte, kvQuantPackedLen(kvQ8GroupSize, bits))
		scales := make([]float32, 1)
		kvQuantRows(packed, scales, src, bits)
		if scales[0] != 1 {
			t.Fatalf("bits=%d: expected scale 1, got %v", bits, scales[0])
		}
		dst := make([]byte, kvQ8GroupSize*2)
		kvDequantRows(dst, packed, scales, bits)
		if !bytes.Equal(src, dst) {
			for i := 0; i < kvQ8GroupSize; i++ {
				if src[i*2] != dst[i*2] || src[i*2+1] != dst[i*2+1] {
					t.Fatalf("bits=%d: element %d not exact: orig=%v recon=%v",
						bits, i, bf16At(src, i), bf16At(dst, i))
				}
			}
		}
	}
}

// Example_kvQuantRows shows a quant→dequant round trip on a small (one-group)
// vector at q8, with an outlier that pins the group scale to 1 so the low-
// magnitude entries reconstruct exactly.
func Example_kvQuantRows() {
	vals := make([]float32, kvQ8GroupSize)
	vals[0] = 127 // outlier → scale = 127/127 = 1
	vals[1] = 10
	vals[2] = -5
	vals[3] = 3
	src := f32GroupToBf16(vals)

	packed := make([]byte, kvQuantPackedLen(kvQ8GroupSize, 8))
	scales := make([]float32, 1)
	kvQuantRows(packed, scales, src, 8)

	dst := make([]byte, kvQ8GroupSize*2)
	kvDequantRows(dst, packed, scales, 8)

	fmt.Println(bf16At(dst, 0), bf16At(dst, 1), bf16At(dst, 2), bf16At(dst, 3))
	// Output: 127 10 -5 3
}
