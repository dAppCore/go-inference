// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"encoding/binary"
	"math"
	"unsafe"

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
	// Decode via reinterpret-casts of the little-endian on-disk bytes (arm64 + amd64 are both
	// little-endian, the only Go targets this tree builds), matching index.go's DecodeFloatData:
	// F16 then rides the NEON FCVTL batch path on darwin/arm64. Bit-identical to the previous
	// per-element binary.LittleEndian decode — the parity is pinned for F16 by
	// TestFloat16ToFloat32_NEONParity_BitExact — but skips the per-iter raw[i*n:] re-slice and
	// byte-combine. Only reached for elements > 0, so raw is non-empty and unsafe.SliceData is valid.
	switch core.Upper(dtype) {
	case "F32":
		if len(raw) != elements*4 {
			return nil, errDecodeF32PayloadMismatch
		}
		dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), elements*4)
		copy(dst, raw)
	case "F16":
		if len(raw) != elements*2 {
			return nil, errDecodeF16PayloadMismatch
		}
		src16 := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(raw))), elements)
		float16SliceToFloat32(src16, values, elements)
	case "BF16":
		if len(raw) != elements*2 {
			return nil, errDecodeBF16PayloadMismatch
		}
		src16 := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(raw))), elements)
		for i, v := range src16 {
			values[i] = math.Float32frombits(uint32(v) << 16)
		}
	case "F64":
		if len(raw) != elements*8 {
			return nil, errDecodeF64PayloadMismatch
		}
		src64 := unsafe.Slice((*float64)(unsafe.Pointer(unsafe.SliceData(raw))), elements)
		for i, v := range src64 {
			values[i] = float32(v)
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
// handling subnormals, infinities, and NaN. The bit-twiddling engine behind
// the exported Float16ToFloat32 — see that function for the documented,
// tested public contract.
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

// Float16ToFloat32 converts one IEEE 754 binary16 (half precision) bit
// pattern to float32, handling zero, subnormals, infinities, and NaN.
// safetensors stores "F16" tensors in this format; DecodeFloat32 upcasts
// every element of an "F16" tensor through this exact conversion, so
// calling it directly reproduces DecodeFloat32's per-element math for a
// single value (e.g. to inspect one weight without decoding a whole
// tensor).
//
//	f := safetensors.Float16ToFloat32(0x3C00) // 1.0
func Float16ToFloat32(bits uint16) float32 {
	return float16ToFloat32(bits)
}

// bfloat16ToFloat32 converts one bfloat16 bit pattern to float32. bf16 is
// defined as the high 16 bits of a float32, so the conversion is a widening
// left-shift; math.Float32frombits keeps NaN payloads and Inf exact. The
// engine behind the exported BFloat16ToFloat32 — see that function for the
// documented, tested public contract.
func bfloat16ToFloat32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}

// BFloat16ToFloat32 converts one bfloat16 (brain float16) bit pattern to
// float32. bf16 is defined as the high 16 bits of a float32, so the
// conversion is an exact widening left-shift — no precision is gained or
// lost, and NaN payloads / infinities survive unchanged. safetensors
// stores "BF16" tensors in this format; DecodeFloat32 upcasts every
// element of a "BF16" tensor through this exact conversion.
//
//	f := safetensors.BFloat16ToFloat32(0x3F80) // 1.0
func BFloat16ToFloat32(bits uint16) float32 {
	return bfloat16ToFloat32(bits)
}
