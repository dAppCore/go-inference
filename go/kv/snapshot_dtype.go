// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"encoding/binary"
	"math"
	"unsafe"

	sharedsafetensors "dappco.re/go/inference/safetensors"
)

func normalizeKVSnapshotTensorDType(dtype string) (string, int) {
	switch dtype {
	case "float32", "F32":
		return "float32", 4
	case "float16", "F16":
		return "float16", 2
	case "bfloat16", "BF16":
		return "bfloat16", 2
	default:
		return "", 0
	}
}

// kvSnapshotQ8Validate scans values for NaN/Inf and tracks the running
// max-abs in one walk. Returns (maxAbs, ok). Bit-tricks:
//   - NaN/Inf detect: the f32 bit pattern with exponent == 0xff has
//     (bits & 0x7f800000) == 0x7f800000. Mask + compare is one ANDS +
//     CCMP on ARM64 vs. math.IsNaN's float64 conversion + double bit
//     decompose.
//   - abs: bit-clear the sign bit (W10-H gguf maxAbsFloat32 pattern).
//     Lowers to ARM64 FABS vs. math.Abs's float64 round-trip.
//
// 4-way unroll exposes ILP across M3's wide back-end so the per-
// iteration FCMPS chain doesn't bottleneck on the loop-carried max.
func kvSnapshotQ8Validate(values []float32) (float32, bool) {
	const absMask = 0x7fffffff
	const expMask = 0x7f800000
	var m0, m1, m2, m3 float32
	i := 0
	n := len(values)
	for ; i+4 <= n; i += 4 {
		b0 := math.Float32bits(values[i])
		b1 := math.Float32bits(values[i+1])
		b2 := math.Float32bits(values[i+2])
		b3 := math.Float32bits(values[i+3])
		if (b0&expMask) == expMask || (b1&expMask) == expMask || (b2&expMask) == expMask || (b3&expMask) == expMask {
			return 0, false
		}
		a0 := math.Float32frombits(b0 & absMask)
		a1 := math.Float32frombits(b1 & absMask)
		a2 := math.Float32frombits(b2 & absMask)
		a3 := math.Float32frombits(b3 & absMask)
		if a0 > m0 {
			m0 = a0
		}
		if a1 > m1 {
			m1 = a1
		}
		if a2 > m2 {
			m2 = a2
		}
		if a3 > m3 {
			m3 = a3
		}
	}
	maxAbs := m0
	if m1 > maxAbs {
		maxAbs = m1
	}
	if m2 > maxAbs {
		maxAbs = m2
	}
	if m3 > maxAbs {
		maxAbs = m3
	}
	for ; i < n; i++ {
		b := math.Float32bits(values[i])
		if (b & expMask) == expMask {
			return 0, false
		}
		abs := math.Float32frombits(b & absMask)
		if abs > maxAbs {
			maxAbs = abs
		}
	}
	return maxAbs, true
}

func kvSnapshotCanQuantizeQ8(values []float32) bool {
	_, ok := kvSnapshotQ8Validate(values)
	return ok
}

func quantizeKVSnapshotQ8(values []float32) (float32, []byte) {
	maxAbs, _ := kvSnapshotQ8Validate(values)
	return quantizeKVSnapshotQ8WithMaxAbs(values, maxAbs)
}

// quantizeKVSnapshotQ8WithMaxAbs is the inner quantise that skips the
// validation walk when the caller already computed maxAbs. Used by the
// fused validate+quantise path on the encode side; avoids a second walk
// over the f32 values when both calls fire back-to-back.
func quantizeKVSnapshotQ8WithMaxAbs(values []float32, maxAbs float32) (float32, []byte) {
	scale := float32(1)
	if maxAbs > 0 {
		scale = maxAbs / 127
	}
	quantized := make([]byte, len(values))
	for i, value := range values {
		q := min(int(math.Round(float64(value/scale))), 127)
		if q < -127 {
			q = -127
		}
		quantized[i] = byte(int8(q))
	}
	return scale, quantized
}

func validateKVSnapshotNativeTensor(dtype string, raw []byte, elements int) (string, error) {
	dtype, bytesPerValue := normalizeKVSnapshotTensorDType(dtype)
	if dtype == "" || bytesPerValue <= 0 {
		return "", errUnsupportedNativeDtype
	}
	if elements < 0 || len(raw) != elements*bytesPerValue {
		return "", errNativeByteLenMismatch
	}
	return dtype, nil
}

func decodeKVSnapshotNativeTensor(dtype string, raw []byte, elements int) ([]float32, error) {
	dtype, err := validateKVSnapshotNativeTensor(dtype, raw, elements)
	if err != nil {
		return nil, err
	}
	values := make([]float32, elements)
	switch dtype {
	case "float32":
		// Reinterpret-cast bytes → float32 via memcpy; same pattern
		// as f32s() reader. Single copy vs N×Uint32+Float32frombits.
		dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), elements*4)
		copy(dst, raw)
	case "float16":
		for i := range values {
			values[i] = sharedsafetensors.Float16ToFloat32(binary.LittleEndian.Uint16(raw[i*2 : i*2+2]))
		}
	case "bfloat16":
		for i := range values {
			values[i] = math.Float32frombits(uint32(binary.LittleEndian.Uint16(raw[i*2:i*2+2])) << 16)
		}
	default:
		return nil, errUnsupportedNativeDtype
	}
	return values, nil
}
