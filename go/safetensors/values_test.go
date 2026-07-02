// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"encoding/binary"

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
	_, err := DecodeFloat32("F32", []byte{0, 0, 0}, 1)
	core.AssertErrorIs(t, err, errDecodeF32PayloadMismatch)
}

func TestValues_DecodeFloat32_Ugly(t *core.T) {
	_, err := DecodeFloat32("I64", []byte{1, 2, 3, 4}, 1)
	core.AssertError(t, err, "unsupported safetensors dtype")
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
	_, err := DecodeFloat32("F16", []byte{0}, 1)
	core.AssertErrorIs(t, err, errDecodeF16PayloadMismatch)
}

func TestValues_DecodeFloat32_BF16(t *core.T) {
	// bf16 is the high 16 bits of float32: 1.0 = 0x3F80, 2.0 = 0x4000, -1.0 = 0xBF80.
	raw := append(append(le16(0x3F80), le16(0x4000)...), le16(0xBF80)...)
	values, err := DecodeFloat32("bf16", raw, 3)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{1, 2, -1}, values)
}

func TestValues_DecodeFloat32_BF16_LengthMismatch(t *core.T) {
	_, err := DecodeFloat32("BF16", []byte{0}, 1)
	core.AssertErrorIs(t, err, errDecodeBF16PayloadMismatch)
}

func TestValues_DecodeFloat32_F64(t *core.T) {
	raw := append(append(le64(0x3FF0000000000000), le64(0x4000000000000000)...), le64(0xBFF0000000000000)...)
	values, err := DecodeFloat32("F64", raw, 3)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{1, 2, -1}, values)
}

func TestValues_DecodeFloat32_F64_LengthMismatch(t *core.T) {
	_, err := DecodeFloat32("F64", []byte{0}, 1)
	core.AssertErrorIs(t, err, errDecodeF64PayloadMismatch)
}

func TestValues_EncodeFloat32_Good(t *core.T) {
	got := EncodeFloat32([]float32{1, 2, -1})
	want := append(append(le32(0x3F800000), le32(0x40000000)...), le32(0xBF800000)...)
	core.AssertEqual(t, want, got)
}

func TestValues_EncodeFloat32_Bad(t *core.T) {
	got := EncodeFloat32(nil)
	core.AssertEmpty(t, got)
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
