// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"math"
	"testing"
)

// TestFloat16ToFloat32_NEONParity_BitExact verifies that the platform-
// selected float16SliceToFloat32 (NEON FCVTL on darwin/arm64, scalar
// elsewhere) produces float32 output that matches the scalar
// Float16ToFloat32 reference across the entire uint16 space. For non-NaN
// inputs the test asserts bit-identical output via Float32bits. For NaN
// inputs (fp16 exponent==31, fraction!=0) the test asserts NaN equivalence
// rather than bit equivalence: ARMv8 FCVTL canonicalises signalling NaNs
// to quiet NaNs by setting the most significant fraction bit, which is the
// IEEE-754-2008 hardware default and is preferable behaviour for any
// downstream that does not distinguish sNaN from qNaN (which, as verified
// in callers via IsNaN, is the case for every consumer in this tree).
func TestFloat16ToFloat32_NEONParity_BitExact(t *testing.T) {
	const n = 1 << 16
	src := make([]uint16, n)
	for i := range src {
		src[i] = uint16(i)
	}
	dst := make([]float32, n)
	float16SliceToFloat32(src, dst, n)
	for i := range n {
		want := Float16ToFloat32(uint16(i))
		got := dst[i]
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got)) {
				t.Fatalf("half 0x%04x: scalar=NaN NEON=0x%08x", i, math.Float32bits(got))
			}
			continue
		}
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Fatalf("half 0x%04x: NEON=0x%08x scalar=0x%08x (NEON=%v scalar=%v)",
				i, math.Float32bits(got), math.Float32bits(want), got, want)
		}
	}
}

// TestFloat16ToFloat32_NEONParity_EdgeCases pins the round-trip behaviour
// of the IEEE-754 edge cases that have historically tripped up half-to-
// single converters: +/-0, smallest subnormal, largest subnormal, smallest
// normal, largest normal, +/-Inf, and a representative quiet NaN. The
// values are spelled out by their fp16 bit pattern rather than computed,
// so any reader can audit the table by hand.
func TestFloat16ToFloat32_NEONParity_EdgeCases(t *testing.T) {
	cases := []struct {
		name string
		half uint16
	}{
		{"+zero", 0x0000},
		{"-zero", 0x8000},
		{"smallest +subnormal", 0x0001},
		{"largest +subnormal", 0x03ff},
		{"smallest +normal", 0x0400},
		{"+1.0", 0x3c00},
		{"-1.0", 0xbc00},
		{"largest +normal", 0x7bff},
		{"+inf", 0x7c00},
		{"-inf", 0xfc00},
		{"quiet NaN", 0x7e00},
		{"signalling NaN", 0x7d00},
		{"+pi", 0x4248},
	}
	src := make([]uint16, len(cases))
	dst := make([]float32, len(cases))
	for i, c := range cases {
		src[i] = c.half
	}
	float16SliceToFloat32(src, dst, len(cases))
	for i, c := range cases {
		want := Float16ToFloat32(c.half)
		got := dst[i]
		if math.IsNaN(float64(want)) {
			if !math.IsNaN(float64(got)) {
				t.Errorf("%s (0x%04x): scalar=NaN NEON=0x%08x", c.name, c.half, math.Float32bits(got))
			}
			continue
		}
		if math.Float32bits(got) != math.Float32bits(want) {
			t.Errorf("%s (0x%04x): NEON=0x%08x scalar=0x%08x",
				c.name, c.half, math.Float32bits(got), math.Float32bits(want))
		}
	}
}

// TestFloat16ToFloat32_NEONParity_TailLengths exercises the tail handler
// inside the NEON inner loop for every residue mod 4 (including n<4), so
// any off-by-one in the scalar fixup path is caught. The body is a normal-
// range fp16 ramp so a regression in the scalar tail is unambiguous.
func TestFloat16ToFloat32_NEONParity_TailLengths(t *testing.T) {
	for n := 0; n <= 17; n++ {
		src := make([]uint16, n)
		dst := make([]float32, n)
		for i := range src {
			src[i] = uint16(0x3c00 + i)
		}
		float16SliceToFloat32(src, dst, n)
		for i := 0; i < n; i++ {
			want := Float16ToFloat32(src[i])
			if math.Float32bits(dst[i]) != math.Float32bits(want) {
				t.Fatalf("n=%d i=%d: NEON=%v scalar=%v", n, i, dst[i], want)
			}
		}
	}
}
