// SPDX-Licence-Identifier: EUPL-1.2

package turboquant

import "math"

// GroupSize is the per-group element count the plain group-quant baselines
// use — mirrors engine/metal's paged KV q8 group size (kvQ8GroupSize, #357:
// int8 rows + f32 group scales, lthn_kv_q8_store_bf16 in
// kernels/lthn_sdpa_paged.metal). Reimplemented host-side here rather than
// imported: this package is engine-neutral and must build without the
// darwin/arm64 Metal engine.
const GroupSize = 64

// groupScale returns the symmetric quantisation scale for one group: the
// group's absolute maximum divided by the signed integer range's positive
// limit (maxCode) — identical in form to the metal kernel's `scale = m /
// 127.0f`. A group that is entirely zero scales to 0 (dequantises back to
// exactly zero, matching the kernel's `inv = scale > 0 ? 1/scale : 0`
// guard).
func groupScale(group []float64, maxCode float64) float64 {
	var m float64
	for _, v := range group {
		if a := math.Abs(v); a > m {
			m = a
		}
	}
	return m / maxCode
}

// quantiseGroupSymmetric quantises x in fixed-size groups of GroupSize
// elements (the final, possibly-short group is scaled on its own elements
// only), each against its own absmax scale, codes clamped to
// [-maxCode, maxCode] and rounded to nearest. Returns the per-element codes
// (as float64, exact small integers) and one scale per group.
func quantiseGroupSymmetric(x []float64, maxCode float64) (codes []float64, scales []float64) {
	d := len(x)
	numGroups := (d + GroupSize - 1) / GroupSize
	codes = make([]float64, d)
	scales = make([]float64, numGroups)
	for g := 0; g < numGroups; g++ {
		lo := g * GroupSize
		hi := min(lo+GroupSize, d)
		scale := groupScale(x[lo:hi], maxCode)
		scales[g] = scale
		inv := 0.0
		if scale > 0 {
			inv = 1 / scale
		}
		for i := lo; i < hi; i++ {
			c := math.Round(x[i] * inv)
			codes[i] = math.Max(-maxCode, math.Min(maxCode, c))
		}
	}
	return codes, scales
}

func dequantiseGroupSymmetric(codes []float64, scales []float64) []float64 {
	d := len(codes)
	out := make([]float64, d)
	for g, scale := range scales {
		lo := g * GroupSize
		hi := min(lo+GroupSize, d)
		for i := lo; i < hi; i++ {
			out[i] = codes[i] * scale
		}
	}
	return out
}

// --- int8 group quant (maxCode = 127, one byte per element) ---

// GroupQuantInt8Encoded is one row's plain symmetric int8 group-quant
// payload — the baseline this package's TurboQuant codecs are measured
// against.
type GroupQuantInt8Encoded struct {
	Codes  []int8
	Scales []float32
	D      int
}

// EncodeGroupQuantInt8 quantises row x in groups of GroupSize elements,
// symmetric int8 (scale = group absmax / 127, matching engine/metal's paged
// q8 KV cache scheme).
//
//	e := EncodeGroupQuantInt8([]float32{3, 4})
//	x := DecodeGroupQuantInt8(e)
func EncodeGroupQuantInt8(x []float32) GroupQuantInt8Encoded {
	codesF, scalesF := quantiseGroupSymmetric(toFloat64(x), 127)
	codes := make([]int8, len(codesF))
	for i, c := range codesF {
		codes[i] = int8(c)
	}
	scales := make([]float32, len(scalesF))
	for i, s := range scalesF {
		scales[i] = float32(s)
	}
	return GroupQuantInt8Encoded{Codes: codes, Scales: scales, D: len(x)}
}

// DecodeGroupQuantInt8 reverses EncodeGroupQuantInt8.
//
//	e := EncodeGroupQuantInt8([]float32{3, 4})
//	x := DecodeGroupQuantInt8(e)
func DecodeGroupQuantInt8(e GroupQuantInt8Encoded) []float32 {
	codesF := make([]float64, len(e.Codes))
	for i, c := range e.Codes {
		codesF[i] = float64(c)
	}
	scalesF := make([]float64, len(e.Scales))
	for i, s := range e.Scales {
		scalesF[i] = float64(s)
	}
	return toFloat32(dequantiseGroupSymmetric(codesF, scalesF))
}

// MarshalGroupQuantInt8 serialises e as: one byte per code, in order,
// followed by one f32-LE scale per group.
func MarshalGroupQuantInt8(e GroupQuantInt8Encoded) []byte {
	out := make([]byte, len(e.Codes)+4*len(e.Scales))
	for i, c := range e.Codes {
		out[i] = byte(c)
	}
	off := len(e.Codes)
	for _, s := range e.Scales {
		putFloat32LE(out[off:], s)
		off += 4
	}
	return out
}

// UnmarshalGroupQuantInt8 reverses MarshalGroupQuantInt8, given the row
// dimension d the encoder used.
func UnmarshalGroupQuantInt8(data []byte, d int) GroupQuantInt8Encoded {
	numGroups := (d + GroupSize - 1) / GroupSize
	if len(data) < d+4*numGroups {
		return GroupQuantInt8Encoded{D: d}
	}
	codes := make([]int8, d)
	for i := 0; i < d; i++ {
		codes[i] = int8(data[i])
	}
	scales := make([]float32, numGroups)
	off := d
	for g := 0; g < numGroups; g++ {
		scales[g] = getFloat32LE(data[off:])
		off += 4
	}
	return GroupQuantInt8Encoded{Codes: codes, Scales: scales, D: d}
}

// --- int4 group quant (maxCode = 7, symmetric nibbles biased by +8) ---

// GroupQuantInt4Encoded is one row's plain symmetric int4 group-quant
// payload. Codes range over [-7,7] (kept symmetric like the int8 scheme's
// 127-not-128 choice, rather than the full asymmetric [-8,7] two's-complement
// nibble range) so a zero-centred group never carries a systematic bias.
type GroupQuantInt4Encoded struct {
	Codes  []int8 // logical value, range [-7,7]
	Scales []float32
	D      int
}

// EncodeGroupQuantInt4 quantises row x in groups of GroupSize elements,
// symmetric int4 (scale = group absmax / 7).
//
//	e := EncodeGroupQuantInt4([]float32{3, 4})
//	x := DecodeGroupQuantInt4(e)
func EncodeGroupQuantInt4(x []float32) GroupQuantInt4Encoded {
	codesF, scalesF := quantiseGroupSymmetric(toFloat64(x), 7)
	codes := make([]int8, len(codesF))
	for i, c := range codesF {
		codes[i] = int8(c)
	}
	scales := make([]float32, len(scalesF))
	for i, s := range scalesF {
		scales[i] = float32(s)
	}
	return GroupQuantInt4Encoded{Codes: codes, Scales: scales, D: len(x)}
}

// DecodeGroupQuantInt4 reverses EncodeGroupQuantInt4.
//
//	e := EncodeGroupQuantInt4([]float32{3, 4})
//	x := DecodeGroupQuantInt4(e)
func DecodeGroupQuantInt4(e GroupQuantInt4Encoded) []float32 {
	codesF := make([]float64, len(e.Codes))
	for i, c := range e.Codes {
		codesF[i] = float64(c)
	}
	scalesF := make([]float64, len(e.Scales))
	for i, s := range e.Scales {
		scalesF[i] = float64(s)
	}
	return toFloat32(dequantiseGroupSymmetric(codesF, scalesF))
}

// MarshalGroupQuantInt4 serialises e as: packed 4-bit nibbles (each code
// biased by +8 into [1,15] before packing, LSB-first per packBits), followed
// by one f32-LE scale per group.
func MarshalGroupQuantInt4(e GroupQuantInt4Encoded) []byte {
	biased := make([]int, len(e.Codes))
	for i, c := range e.Codes {
		biased[i] = int(c) + 8
	}
	packed := packBits(biased, 4)
	out := make([]byte, len(packed)+4*len(e.Scales))
	off := copy(out, packed)
	for _, s := range e.Scales {
		putFloat32LE(out[off:], s)
		off += 4
	}
	return out
}

// UnmarshalGroupQuantInt4 reverses MarshalGroupQuantInt4, given the row
// dimension d the encoder used.
func UnmarshalGroupQuantInt4(data []byte, d int) GroupQuantInt4Encoded {
	numGroups := (d + GroupSize - 1) / GroupSize
	packedLen := packedByteLen(d, 4)
	if len(data) < packedLen+4*numGroups {
		return GroupQuantInt4Encoded{D: d}
	}
	biased := unpackBits(data[:packedLen], d, 4)
	codes := make([]int8, d)
	for i, b := range biased {
		codes[i] = int8(b - 8)
	}
	scales := make([]float32, numGroups)
	off := packedLen
	for g := 0; g < numGroups; g++ {
		scales[g] = getFloat32LE(data[off:])
		off += 4
	}
	return GroupQuantInt4Encoded{Codes: codes, Scales: scales, D: d}
}
