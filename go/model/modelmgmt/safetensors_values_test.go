// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import (
	"encoding/binary"

	core "dappco.re/go"
)

// The value-codec deep tests (subnormals, per-dtype length mismatches,
// sentinel identity) moved with the implementation to the safetensors
// package. These tests cover the delegating wrappers modelmgmt keeps for
// compatibility.

func le32(v uint32) []byte {
	buf := make([]byte, 4)
	binary.LittleEndian.PutUint32(buf, v)
	return buf
}

func TestSafetensorsValues_DecodeFloat32_Good(t *core.T) {
	// F32 identity reinterpret: 1.0, 2.0, -1.0.
	raw := append(append(le32(0x3F800000), le32(0x40000000)...), le32(0xBF800000)...)
	values, err := DecodeFloat32("F32", raw, 3)
	core.RequireNoError(t, err)
	core.AssertEqual(t, []float32{1, 2, -1}, values)
}

func TestSafetensorsValues_DecodeFloat32_Bad(t *core.T) {
	values, err := DecodeFloat32("F32", []byte{0, 0, 0}, 1)
	core.AssertError(t, err, "payload length does not match element count")
	core.AssertNil(t, values)
}

func TestSafetensorsValues_DecodeFloat32_Ugly(t *core.T) {
	values, err := DecodeFloat32("I64", []byte{1, 2, 3, 4}, 1)
	core.AssertError(t, err, "unsupported safetensors dtype")
	core.AssertNil(t, values)
}

func TestSafetensorsValues_EncodeFloat32_Good(t *core.T) {
	got := EncodeFloat32([]float32{1, 2, -1})
	want := append(append(le32(0x3F800000), le32(0x40000000)...), le32(0xBF800000)...)
	core.AssertEqual(t, want, got)
}

func TestSafetensorsValues_EncodeFloat32_Bad(t *core.T) {
	got := EncodeFloat32(nil)
	core.AssertEmpty(t, got)
	core.AssertEmpty(t, EncodeFloat32([]float32{}))
}

func TestSafetensorsValues_EncodeFloat32_Ugly(t *core.T) {
	// Round-trips through DecodeFloat32, proving the wrapper pair stays
	// inverse over F32 exactly like the underlying codec.
	values := []float32{3.14159, -0.5, 1e10}
	raw := EncodeFloat32(values)
	decoded, err := DecodeFloat32("F32", raw, len(values))
	core.RequireNoError(t, err)
	core.AssertEqual(t, values, decoded)
}
