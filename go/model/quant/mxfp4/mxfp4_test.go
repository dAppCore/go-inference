// SPDX-Licence-Identifier: EUPL-1.2
package mxfp4

import (
	"math"
	"testing"
)

func TestQuantize_Good(t *testing.T) {
	values := make([]float32, 65)
	for i := range values {
		values[i] = float32(i-32) / 32
	}
	q, err := Quantize(values, []int{5, 13})
	if err != nil {
		t.Fatal(err)
	}
	got, err := Dequantize(q)
	if err != nil {
		t.Fatal(err)
	}
	for i := range values {
		// Half the largest gap in the E2M1 table (4 to 6) is 1, in units of
		// the block's own scale — a safe, format-derived error bound.
		tol := q.Scale[i/BlockSize]
		if math.Abs(float64(got[i]-values[i])) > float64(tol) {
			t.Fatalf("value %d error %g (scale %g)", i, got[i]-values[i], tol)
		}
	}
}
func TestQuantize_Bad(t *testing.T) {
	if _, err := Quantize(nil, nil); err == nil {
		t.Fatal("Quantize(nil) error = nil")
	}
}
func TestQuantize_Ugly(t *testing.T) {
	if _, err := Quantize([]float32{1}, []int{2}); err == nil {
		t.Fatal("Quantize(shape mismatch) error = nil")
	}
}
func TestDequantize_Good(t *testing.T) {
	// High nibble 0xC = sign|4 -> -magnitudes[4] = -2; low nibble 0x4 -> +2.
	got, err := Dequantize(Tensor{Data: []byte{0xC4}, Scale: []float32{1}, Shape: []int{2}})
	if err != nil || got[0] != -2 || got[1] != 2 {
		t.Fatalf("Dequantize = %v, %v", got, err)
	}
}
func TestDequantize_Bad(t *testing.T) {
	if _, err := Dequantize(Tensor{}); err == nil {
		t.Fatal("Dequantize(empty) error = nil")
	}
}
func TestDequantize_Ugly(t *testing.T) {
	if _, err := Dequantize(Tensor{Data: []byte{0}, Shape: []int{3}}); err == nil {
		t.Fatal("Dequantize(malformed) error = nil")
	}
}

// TestEncodeE2M1_Good — every table magnitude, signed and unsigned, encodes
// to its own index with the sign bit in bit 3.
func TestEncodeE2M1_Good(t *testing.T) {
	for i, m := range magnitudes {
		if got := encodeE2M1(m); got != byte(i) {
			t.Errorf("encodeE2M1(%g): got code %#x, want %#x", m, got, i)
		}
		if m == 0 {
			continue // no distinct negative zero code
		}
		if got := encodeE2M1(-m); got != byte(i)|0x8 {
			t.Errorf("encodeE2M1(%g): got code %#x, want %#x", -m, got, byte(i)|0x8)
		}
	}
}

// TestEncodeE2M1_Ugly — a magnitude far outside the table clamps to the
// largest representable code (7 -> magnitude 6) instead of overflowing.
func TestEncodeE2M1_Ugly(t *testing.T) {
	if got := encodeE2M1(100); got != 7 {
		t.Errorf("encodeE2M1(100): got %#x, want 0x7 (clamped to magnitude 6)", got)
	}
	if got := encodeE2M1(-100); got != 7|0x8 {
		t.Errorf("encodeE2M1(-100): got %#x, want %#x (clamped, signed)", got, byte(7)|0x8)
	}
}

// TestDecodeE2M1_Good — all 16 codes decode without panicking, matching the
// magnitude table plus sign bit.
func TestDecodeE2M1_Good(t *testing.T) {
	for code := 0; code < 16; code++ {
		want := magnitudes[code&0x7]
		if code&0x8 != 0 {
			want = -want
		}
		if got := decodeE2M1(byte(code)); got != want {
			t.Errorf("decodeE2M1(%#x): got %g, want %g", code, got, want)
		}
	}
}

// TestBlockScale_Good — an exact-boundary peak (6) needs no scaling; an
// all-zero block (peak 0) avoids dividing by zero.
func TestBlockScale_Good(t *testing.T) {
	if got := blockScale(6); got != 1 {
		t.Errorf("blockScale(6): got %g, want 1", got)
	}
	if got := blockScale(0); got != 1 {
		t.Errorf("blockScale(0): got %g, want 1 (avoid divide by zero)", got)
	}
}

// TestBlockScale_Ugly — a vanishingly small (denormal-range) peak needs an
// exponent below E8M0's storable floor; the result clamps to 2^-127 rather
// than underflowing to zero.
func TestBlockScale_Ugly(t *testing.T) {
	got := blockScale(1e-40)
	want := float32(math.Ldexp(1, -127))
	if got != want {
		t.Errorf("blockScale(1e-40): got %g, want %g (clamped to E8M0 min exponent)", got, want)
	}
	if got == 0 {
		t.Fatalf("blockScale(1e-40): got 0, want a nonzero clamped scale")
	}
}

// TestClampExponent_Good — E8M0 is an unsigned 8-bit field biased by 127
// (0xFF reserved for NaN per the OCP MX spec), so both directions clamp at
// +/-127.
func TestClampExponent_Good(t *testing.T) {
	cases := map[int]int{-500: -127, -127: -127, 0: 0, 127: 127, 500: 127}
	for exp, want := range cases {
		if got := clampExponent(exp); got != want {
			t.Errorf("clampExponent(%d): got %d, want %d", exp, got, want)
		}
	}
}
