// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"math"
	"testing"

	core "dappco.re/go"
)

func TestHIPProjectionReferenceFP16_Good(t *testing.T) {
	got, err := hipReferenceFP16Projection(
		[]float32{1, 2},
		[]uint16{0x3c00, 0x3800, 0xbc00, 0x4000},
		2,
		2,
		[]float32{0.25, -0.5},
	)

	core.AssertNoError(t, err)
	assertFloat32Near(t, 2.25, got[0])
	assertFloat32Near(t, 2.5, got[1])
}

func TestHIPProjectionReferenceBF16_Good(t *testing.T) {
	got, err := hipReferenceBF16Projection(
		[]float32{1.5, -2},
		[]uint16{0x3f80, 0xc000, 0x4000, 0x3f00},
		2,
		2,
		[]float32{0.25, -0.5},
	)

	core.AssertNoError(t, err)
	assertFloat32Near(t, 5.75, got[0])
	assertFloat32Near(t, 1.5, got[1])
}

func TestHIPProjectionReferenceF32_Good(t *testing.T) {
	got, err := hipReferenceF32Projection(
		[]float32{1, 2},
		[]float32{1, 0.5, -1, 2},
		2,
		2,
		[]float32{0.25, -0.5},
	)

	core.AssertNoError(t, err)
	assertFloat32Near(t, 2.25, got[0])
	assertFloat32Near(t, 2.5, got[1])
}

func TestHIPProjectionReferenceQ8_Good(t *testing.T) {
	got, err := hipReferenceQ8Projection(
		[]float32{2, -1},
		[]int8{4, -2, 1, 3},
		0.25,
		2,
		2,
		nil,
	)

	core.AssertNoError(t, err)
	assertFloat32Near(t, 2.5, got[0])
	assertFloat32Near(t, -0.25, got[1])
}

func TestHIPProjectionReferenceMLXQ4_Good(t *testing.T) {
	got, err := hipReferenceMLXQ4Projection(
		[]float32{1, 1, 1, 1, 1, 1, 1, 1},
		[]uint32{0x76543210, 0xfedcba98},
		[]uint16{0x3f80, 0x3f00},
		[]uint16{0x0000, 0xbf80},
		2,
		8,
		8,
	)

	core.AssertNoError(t, err)
	assertFloat32Near(t, 28, got[0])
	assertFloat32Near(t, 38, got[1])
}

func TestHIPProjectionReferenceMLXAffineQ6Q8_Good(t *testing.T) {
	input := []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	q6Weights := hipPackMLXAffineValuesForTest([]uint32{
		0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
	}, 16, 6)
	q6, err := hipReferenceMLXAffineProjection(input, q6Weights, []uint16{0x3f80, 0x3f80}, []uint16{0, 0}, 2, 16, 16, 6)
	core.AssertNoError(t, err)
	assertFloat32Near(t, 120, q6[0])
	assertFloat32Near(t, 136, q6[1])

	q8Weights := hipPackMLXAffineValuesForTest([]uint32{
		1, 2, 3, 4,
		5, 6, 7, 8,
	}, 4, 8)
	q8, err := hipReferenceMLXAffineProjection([]float32{1, 1, 1, 1}, q8Weights, []uint16{0x3f80, 0x3f80}, []uint16{0, 0}, 2, 4, 4, 8)
	core.AssertNoError(t, err)
	assertFloat32Near(t, 10, q8[0])
	assertFloat32Near(t, 26, q8[1])
}

func hipPackMLXAffineValuesForTest(values []uint32, cols, bits int) []uint32 {
	packedPerRow, err := hipMLXAffinePackedCols(cols, bits)
	if err != nil {
		panic(err)
	}
	rows := (len(values) + cols - 1) / cols
	out := make([]uint32, rows*packedPerRow)
	mask := uint32(1<<uint(bits)) - 1
	for index, value := range values {
		row := index / cols
		col := index % cols
		bitOffset := uint64(col) * uint64(bits)
		wordIndex := row*packedPerRow + int(bitOffset/32)
		shift := uint(bitOffset % 32)
		value &= mask
		out[wordIndex] |= value << shift
		if shift+uint(bits) > 32 {
			out[wordIndex+1] |= value >> (32 - shift)
		}
	}
	return out
}

func TestHIPProjectionReferenceBadShape_Bad(t *testing.T) {
	_, err := hipReferenceFP16Projection([]float32{1}, []uint16{0x3c00, 0x3c00}, 1, 2, nil)

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "input length")

	_, err = hipReferenceF32Projection([]float32{1}, []float32{1}, 0, 1, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rows and cols")

	_, err = hipReferenceF32Projection([]float32{1}, []float32{1}, 1, 0, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "rows and cols")

	_, err = hipReferenceF32Projection([]float32{1, 2}, []float32{1}, 1, 2, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "weight length")

	_, err = hipReferenceMLXQ4Projection([]float32{1}, []uint32{0}, []uint16{0x3f80}, []uint16{0}, 1, 7, 7)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "cols*bits")

	_, err = hipReferenceMLXQ4Projection([]float32{1, 1, 1, 1, 1, 1, 1, 1}, []uint32{0}, []uint16{0x3f80}, []uint16{0}, 1, 8, 3)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "group size")
}

func TestHIPProjectionReferenceUglyBiasAndScale_Ugly(t *testing.T) {
	_, err := hipReferenceFP16Projection([]float32{1}, []uint16{0x3c00}, 1, 1, []float32{0, 1})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "bias length")

	_, err = hipReferenceQ8Projection([]float32{1}, []int8{1}, 0, 1, 1, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "scale must be positive and finite")

	_, err = hipReferenceQ8Projection([]float32{1}, []int8{1}, float32(math.Inf(1)), 1, 1, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "scale must be positive and finite")

	_, err = hipReferenceQ8Projection([]float32{1}, []int8{1}, -1, 1, 1, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "scale must be positive and finite")

	_, err = hipReferenceQ8Projection([]float32{1}, []int8{1}, float32(math.NaN()), 1, 1, nil)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "scale must be positive and finite")
}

func TestHIPFloat16ToFloat32_Good(t *testing.T) {
	assertFloat32Near(t, 1, hipFloat16ToFloat32(0x3c00))
	assertFloat32Near(t, -2, hipFloat16ToFloat32(0xc000))
	if !math.IsInf(float64(hipFloat16ToFloat32(0x7c00)), 1) {
		t.Fatalf("float16 inf conversion failed")
	}
}

func TestHIPFloat16ToFloat32UglySpecialValues_Ugly(t *testing.T) {
	assertFloat32Near(t, 0, hipFloat16ToFloat32(0x0000))
	if !math.Signbit(float64(hipFloat16ToFloat32(0x8000))) {
		t.Fatalf("float16 negative zero conversion lost sign")
	}
	if got := hipFloat16ToFloat32(0x0001); got <= 0 || got >= 0.0001 {
		t.Fatalf("float16 subnormal conversion = %f, want positive subnormal", got)
	}
	if !math.IsNaN(float64(hipFloat16ToFloat32(0x7e00))) {
		t.Fatalf("float16 NaN conversion failed")
	}
}

func TestHIPBFloat16ToFloat32_Good(t *testing.T) {
	assertFloat32Near(t, 1, hipBFloat16ToFloat32(0x3f80))
	assertFloat32Near(t, -2, hipBFloat16ToFloat32(0xc000))
	if !math.IsInf(float64(hipBFloat16ToFloat32(0x7f80)), 1) {
		t.Fatalf("bfloat16 inf conversion failed")
	}
	if !math.IsNaN(float64(hipBFloat16ToFloat32(0x7fc0))) {
		t.Fatalf("bfloat16 NaN conversion failed")
	}
}

func assertFloat32Near(t *testing.T, want, got float32) {
	t.Helper()
	if math.Abs(float64(want-got)) > 0.0001 {
		t.Fatalf("value = %f, want %f", got, want)
	}
}
