// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

func le16(v uint16) []byte {
	buf := make([]byte, 2)
	binary.LittleEndian.PutUint16(buf, v)
	return buf
}

func le32(v uint32) []byte {
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, v)
	return buf
}

func le64(v uint64) []byte {
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, v)
	return buf
}

func TestValues_DecodeFloat32_Good(t *core.T) {
	// F32 identity reinterpret: 1.0, 2.0, -1.0.
	raw := append(append(le32(0x3F800000), le32(0x40000000)...), le32(0xBF800000)...)
	values, err := DecodeFloat32("F32", raw, 3)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{1, 2, -1}, values)
}

func TestValues_DecodeFloat32_Bad(t *core.T) {
	values, err := DecodeFloat32("F32", []byte{0, 0, 0}, 1)
	core.AssertErrorIs(t, err, errDecodeF32PayloadMismatch)
	core.AssertNil(t, values)
}

func TestValues_DecodeFloat32_Ugly(t *core.T) {
	values, err := DecodeFloat32("I64", []byte{1, 2, 3, 4}, 1)
	core.AssertError(t, err, "unsupported safetensors dtype")
	core.AssertNil(t, values)
}

func TestValues_DecodeFloat32_F16(t *core.T) {
	// 1.0 = 0x3C00, 2.0 = 0x4000, -1.0 = 0xBC00 in IEEE 754 binary16.
	raw := append(append(le16(0x3C00), le16(0x4000)...), le16(0xBC00)...)
	values, err := DecodeFloat32("F16", raw, 3)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{1, 2, -1}, values)
}

func TestValues_DecodeFloat32_F16_Subnormal(t *core.T) {
	// Smallest positive F16 subnormal (0x0001) is 2^-24.
	values, err := DecodeFloat32("F16", le16(0x0001), 1)
	core.RequireNoError(t, err)
	core.AssertInDelta(t, 5.960464477539063e-08, float64(values[0]), 1e-15)
}

func TestValues_DecodeFloat32_F16_LengthMismatch(t *core.T) {
	values, err := DecodeFloat32("F16", []byte{0}, 1)
	core.AssertErrorIs(t, err, errDecodeF16PayloadMismatch)
	core.AssertNil(t, values)
}

func TestValues_DecodeFloat32_BF16(t *core.T) {
	// bf16 is the high 16 bits of float32: 1.0 = 0x3F80, 2.0 = 0x4000, -1.0 = 0xBF80.
	raw := append(append(le16(0x3F80), le16(0x4000)...), le16(0xBF80)...)
	values, err := DecodeFloat32("bf16", raw, 3)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{1, 2, -1}, values)
}

func TestValues_DecodeFloat32_BF16_LengthMismatch(t *core.T) {
	values, err := DecodeFloat32("BF16", []byte{0}, 1)
	core.AssertErrorIs(t, err, errDecodeBF16PayloadMismatch)
	core.AssertNil(t, values)
}

func TestValues_DecodeFloat32_F64(t *core.T) {
	raw := append(append(le64(0x3FF0000000000000), le64(0x4000000000000000)...), le64(0xBFF0000000000000)...)
	values, err := DecodeFloat32("F64", raw, 3)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{1, 2, -1}, values)
}

func TestValues_DecodeFloat32_F64_LengthMismatch(t *core.T) {
	values, err := DecodeFloat32("F64", []byte{0}, 1)
	core.AssertErrorIs(t, err, errDecodeF64PayloadMismatch)
	core.AssertNil(t, values)
}

func TestValues_EncodeFloat32_Good(t *core.T) {
	got := EncodeFloat32([]float32{1, 2, -1})
	want := append(append(le32(0x3F800000), le32(0x40000000)...), le32(0xBF800000)...)
	core.AssertEqual(t, want, got)
}

func TestValues_EncodeFloat32_Bad(t *core.T) {
	got := EncodeFloat32(nil)
	core.AssertEmpty(t, got)
	core.AssertEmpty(t, EncodeFloat32([]float32{}))
}

func TestValues_EncodeFloat32_Ugly(t *core.T) {
	// Round-trips through DecodeFloat32 for a value with no exact
	// power-of-two bit pattern, proving the pair is inverse over F32.
	values := []float32{3.14159, -0.5, 1e10}
	raw := EncodeFloat32(values)
	decoded, err := DecodeFloat32("F32", raw, len(values))
	core.RequireNoError(t, err)
	core.AssertEqual(t, values, decoded)
}

// --- Float16ToFloat32 ---

// Good: ordinary positive and negative normal values decode exactly —
// known-good IEEE 754 binary16 bit patterns, also exercised indirectly via
// DecodeFloat32("F16", ...) above.
func TestValues_Float16ToFloat32_Good(t *core.T) {
	core.AssertEqual(t, float32(1), Float16ToFloat32(0x3C00))
	core.AssertEqual(t, float32(-1), Float16ToFloat32(0xBC00))
	core.AssertEqual(t, float32(2), Float16ToFloat32(0x4000))
}

// Bad: zero and negative zero both decode to a numerically-zero float32,
// with the sign bit preserved — 0x8000 must not collapse to +0.0.
func TestValues_Float16ToFloat32_Bad(t *core.T) {
	pos := Float16ToFloat32(0x0000)
	neg := Float16ToFloat32(0x8000)
	core.AssertEqual(t, float32(0), pos)
	core.AssertEqual(t, float32(0), neg)
	if math.Signbit(float64(pos)) {
		t.Fatalf("Float16ToFloat32(0x0000) = %v, want +0.0", pos)
	}
	if !math.Signbit(float64(neg)) {
		t.Fatalf("Float16ToFloat32(0x8000) = %v, want -0.0", neg)
	}
}

// Ugly: subnormals, infinities, and NaN — the bit-pattern families that
// drive float16ToFloat32's non-default switch branches.
func TestValues_Float16ToFloat32_Ugly(t *core.T) {
	// Smallest positive/negative subnormal: +/-2^-24.
	core.AssertInDelta(t, 5.960464477539063e-08, float64(Float16ToFloat32(0x0001)), 1e-15)
	core.AssertInDelta(t, -5.960464477539063e-08, float64(Float16ToFloat32(0x8001)), 1e-15)

	posInf := Float16ToFloat32(0x7C00)
	negInf := Float16ToFloat32(0xFC00)
	if !math.IsInf(float64(posInf), 1) {
		t.Fatalf("Float16ToFloat32(0x7C00) = %v, want +Inf", posInf)
	}
	if !math.IsInf(float64(negInf), -1) {
		t.Fatalf("Float16ToFloat32(0xFC00) = %v, want -Inf", negInf)
	}

	// NaN — exponent field all-ones (0x1F) with a non-zero fraction.
	if got := Float16ToFloat32(0x7E00); !math.IsNaN(float64(got)) {
		t.Fatalf("Float16ToFloat32(0x7E00) = %v, want NaN", got)
	}
}

// --- BFloat16ToFloat32 ---

// Good: ordinary positive and negative normal values decode exactly — bf16
// is the top 16 bits of float32, so 1.0/-1.0 carry the same well-known
// short bit patterns as their F32 encodings, truncated.
func TestValues_BFloat16ToFloat32_Good(t *core.T) {
	core.AssertEqual(t, float32(1), BFloat16ToFloat32(0x3F80))
	core.AssertEqual(t, float32(-1), BFloat16ToFloat32(0xBF80))
	core.AssertEqual(t, float32(2), BFloat16ToFloat32(0x4000))
}

// Bad: zero and negative zero both decode to a numerically-zero float32,
// with the sign bit preserved through the widening shift.
func TestValues_BFloat16ToFloat32_Bad(t *core.T) {
	pos := BFloat16ToFloat32(0x0000)
	neg := BFloat16ToFloat32(0x8000)
	core.AssertEqual(t, float32(0), pos)
	core.AssertEqual(t, float32(0), neg)
	if math.Signbit(float64(pos)) {
		t.Fatalf("BFloat16ToFloat32(0x0000) = %v, want +0.0", pos)
	}
	if !math.Signbit(float64(neg)) {
		t.Fatalf("BFloat16ToFloat32(0x8000) = %v, want -0.0", neg)
	}
}

// Ugly: subnormals, infinities, and NaN survive the widening shift exactly
// — bf16 shares float32's 8-bit exponent field, so these bit patterns are
// simultaneously subnormal/Inf/NaN in both formats.
func TestValues_BFloat16ToFloat32_Ugly(t *core.T) {
	// Smallest positive/negative bf16 "subnormal" (exponent field zero,
	// non-zero mantissa) widens into a float32 that is subnormal too.
	if got := BFloat16ToFloat32(0x0001); got <= 0 {
		t.Fatalf("BFloat16ToFloat32(0x0001) = %v, want a tiny positive value", got)
	}
	if got := BFloat16ToFloat32(0x8001); got >= 0 {
		t.Fatalf("BFloat16ToFloat32(0x8001) = %v, want a tiny negative value", got)
	}

	// Infinities: bf16 shares float32's exponent field, so its truncated
	// Inf pattern (0x7F80/0xFF80) widens to an exact float32 Inf.
	if got := BFloat16ToFloat32(0x7F80); !math.IsInf(float64(got), 1) {
		t.Fatalf("BFloat16ToFloat32(0x7F80) = %v, want +Inf", got)
	}
	if got := BFloat16ToFloat32(0xFF80); !math.IsInf(float64(got), -1) {
		t.Fatalf("BFloat16ToFloat32(0xFF80) = %v, want -Inf", got)
	}

	// NaN — exponent field all-ones with a non-zero mantissa.
	if got := BFloat16ToFloat32(0x7FC0); !math.IsNaN(float64(got)) {
		t.Fatalf("BFloat16ToFloat32(0x7FC0) = %v, want NaN", got)
	}
}
