// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
)

// Sentinel errors hoisted to package vars rather than allocated fresh per
// call on the (rare) malformed-payload path — mirrors the convention used
// throughout this codebase's format readers.
var (
	errDecodeF32PayloadMismatch  = core.NewError("safetensors: F32 payload length does not match element count")
	errDecodeF16PayloadMismatch  = core.NewError("safetensors: F16 payload length does not match element count")
	errDecodeBF16PayloadMismatch = core.NewError("safetensors: BF16 payload length does not match element count")
	errDecodeF64PayloadMismatch  = core.NewError("safetensors: F64 payload length does not match element count")
)

// DecodeFloat32 decodes a safetensors tensor's raw byte payload to float32
// values according to its dtype. Supports the dense floating-point dtypes
// safetensors stores model weights in: F32 (identity reinterpret), F16 and
// BF16 (half-precision upcast), and F64 (double-precision downcast). dtype
// is matched case-insensitively against safetensors' canonical spellings
// (as found in SafetensorsTensorInfo.Dtype).
//
//	info := tensors["model.embed_tokens.weight"]
//	raw := safetensors.GetTensorData(info, data)
//	values, err := safetensors.DecodeFloat32(info.Dtype, raw, len(info.Shape))
//	if err != nil { return err }
func DecodeFloat32(dtype string, raw []byte, elements int) ([]float32, error) {
	values := make([]float32, elements)
	switch core.Upper(dtype) {
	case "F32":
		if len(raw) != elements*4 {
			return nil, errDecodeF32PayloadMismatch
		}
		for i := range values {
			values[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	case "F16":
		if len(raw) != elements*2 {
			return nil, errDecodeF16PayloadMismatch
		}
		for i := range values {
			values[i] = float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "BF16":
		if len(raw) != elements*2 {
			return nil, errDecodeBF16PayloadMismatch
		}
		for i := range values {
			values[i] = bfloat16ToFloat32(binary.LittleEndian.Uint16(raw[i*2:]))
		}
	case "F64":
		if len(raw) != elements*8 {
			return nil, errDecodeF64PayloadMismatch
		}
		for i := range values {
			values[i] = float32(math.Float64frombits(binary.LittleEndian.Uint64(raw[i*8:])))
		}
	default:
		return nil, core.NewError("safetensors: unsupported safetensors dtype for float decode: " + dtype)
	}
	return values, nil
}

// EncodeFloat32 encodes values as little-endian F32 safetensors bytes — the
// on-disk data-section layout WriteSafetensors expects for an "F32" tensor.
//
//	tensorData["merged.weight"] = safetensors.EncodeFloat32(merged)
func EncodeFloat32(values []float32) []byte {
	raw := make([]byte, len(values)*4)
	for i, v := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
	}
	return raw
}

// float16ToFloat32 converts one IEEE 754 binary16 bit pattern to float32,
// handling subnormals, infinities, and NaN.
func float16ToFloat32(bits uint16) float32 {
	sign := uint32(bits>>15) & 0x1
	exp := int((bits >> 10) & 0x1f)
	frac := uint32(bits & 0x03ff)
	switch {
	case exp == 0 && frac == 0:
		return math.Float32frombits(sign << 31)
	case exp == 0:
		// Subnormal: normalise by shifting the fraction left until its
		// implicit leading bit lands, decrementing exp to match.
		for frac&0x0400 == 0 {
			frac <<= 1
			exp--
		}
		exp++
		frac &= 0x03ff
	case exp == 31:
		// Inf/NaN: widen the exponent field, preserve the payload.
		return math.Float32frombits((sign << 31) | 0x7f800000 | (frac << 13))
	}
	exp += 127 - 15
	return math.Float32frombits((sign << 31) | (uint32(exp) << 23) | (frac << 13))
}

// bfloat16ToFloat32 converts one bfloat16 bit pattern to float32. bf16 is
// defined as the high 16 bits of a float32, so the conversion is a widening
// left-shift; math.Float32frombits keeps NaN payloads and Inf exact.
func bfloat16ToFloat32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}
