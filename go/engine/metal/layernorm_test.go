// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"
)

func layerNormBF16Fixture(rows, axisSize int) ([]byte, []byte, []byte) {
	x := toBF16Bytes(syntheticFloat32(rows*axisSize, 3))
	w := toBF16Bytes(syntheticFloat32(axisSize, 5))
	b := toBF16Bytes(syntheticFloat32(axisSize, 7))
	return x, w, b
}

func TestLayerNormBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 4, 512
	const eps = float32(1e-5)
	x, w, b := layerNormBF16Fixture(rows, axisSize)
	if _, err := LayerNormBF16(x, w, b, rows, axisSize, eps); err != nil {
		t.Fatalf("LayerNormBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := LayerNormBF16(x, w, b, rows, axisSize, eps); err != nil {
			t.Fatalf("LayerNormBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("LayerNormBF16 allocations = %.0f, want <= 10", allocs)
	}
}

func TestLayerNormF32AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const rows, axisSize = 4, 512
	const eps = float32(1e-5)
	x := syntheticFloat32(rows*axisSize, 3)
	w := syntheticFloat32(axisSize, 5)
	b := syntheticFloat32(axisSize, 7)
	if _, err := LayerNormF32(x, w, b, rows, axisSize, eps); err != nil {
		t.Fatalf("LayerNormF32 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := LayerNormF32(x, w, b, rows, axisSize, eps); err != nil {
			t.Fatalf("LayerNormF32: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("LayerNormF32 allocations = %.0f, want <= 10", allocs)
	}
}

// TestLayerNormBF16 (BYTE-IDENTICAL to pkg/metal.LayerNorm) lives in layernorm_metal_test.go — it
// needs the real cgo metal package as its oracle, so it's gated behind metal_runtime.

func TestLayerNormBF16IntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const rows, ax = 4, 64
	const eps = float32(1e-5)
	x, w, b := layerNormBF16Fixture(rows, ax)
	want, err := LayerNormBF16(x, w, b, rows, ax, eps)
	if err != nil {
		t.Fatalf("LayerNormBF16 reference: %v", err)
	}

	out := make([]byte, rows*ax*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVBF16Scratch(rows*ax, rows*ax)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := LayerNormBF16Into(out, x, w, b, rows, ax, eps)
	if err != nil {
		t.Fatalf("LayerNormBF16Into: %v", err)
	}
	if len(got) != len(out) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("LayerNormBF16Into did not reuse caller-owned output backing")
	}
	eqBytes(t, "LayerNormBF16Into", got, want)

	scratch, err = getQMVBF16Scratch(rows*ax, rows*ax)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("LayerNormBF16Into wrote through pooled scratch output instead of caller output")
	}
}

// TestLayerNormF32 (BYTE-IDENTICAL to pkg/metal.LayerNorm) lives in layernorm_metal_test.go — same
// reason as TestLayerNormBF16 above.

func TestLayerNormF32IntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)

	const rows, ax = 4, 64
	const eps = float32(1e-5)
	x := syntheticFloat32(rows*ax, 3)
	w := syntheticFloat32(ax, 5)
	b := syntheticFloat32(ax, 7)
	want, err := LayerNormF32(x, w, b, rows, ax, eps)
	if err != nil {
		t.Fatalf("LayerNormF32 reference: %v", err)
	}

	out := make([]float32, rows*ax)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVFloatScratch(rows*ax, rows*ax)
	if err != nil {
		t.Fatalf("getQMVFloatScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0xa5}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVFloatScratch(scratch)

	got, err := LayerNormF32Into(out, x, w, b, rows, ax, eps)
	if err != nil {
		t.Fatalf("LayerNormF32Into: %v", err)
	}
	if len(got) != len(out) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("LayerNormF32Into did not reuse caller-owned output backing")
	}
	if !bytes.Equal(float32Bytes(got), float32Bytes(want)) {
		t.Fatal("LayerNormF32Into output differs from allocating wrapper")
	}

	scratch, err = getQMVFloatScratch(rows*ax, rows*ax)
	if err != nil {
		t.Fatalf("getQMVFloatScratch after call: %v", err)
	}
	defer putQMVFloatScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("LayerNormF32Into wrote through pooled scratch output instead of caller output")
	}
}

func TestLayerNormF32LoopedAxis(t *testing.T) {
	requireNativeRuntime(t)
	const rows, ax = 2, 7000
	eps := float32(1e-5)
	x := syntheticFloat32(rows*ax, 23)
	w := syntheticFloat32(ax, 29)
	b := syntheticFloat32(ax, 31)

	got, err := LayerNormF32(x, w, b, rows, ax, eps)
	if err != nil {
		t.Fatalf("LayerNormF32 looped axis: %v", err)
	}
	want := hostLayerNormF32(x, w, b, rows, ax, eps)
	assertFloat32Near(t, "LayerNormF32 looped axis", got, want, 2e-4)
}

func TestLayerNormBF16LoopedAxis(t *testing.T) {
	requireNativeRuntime(t)
	const rows, ax = 1, 7000
	eps := float32(1e-5)
	x := toBF16Bytes(syntheticFloat32(rows*ax, 37))
	w := toBF16Bytes(syntheticFloat32(ax, 41))
	b := toBF16Bytes(syntheticFloat32(ax, 43))

	got, err := LayerNormBF16(x, w, b, rows, ax, eps)
	if err != nil {
		t.Fatalf("LayerNormBF16 looped axis: %v", err)
	}
	want := bf16Floats(toBF16Bytes(hostLayerNormF32(bf16Floats(x), bf16Floats(w), bf16Floats(b), rows, ax, eps)))
	assertFloat32Near(t, "LayerNormBF16 looped axis", bf16Floats(got), want, 0.035)
}

func hostLayerNormF32(x, weight, bias []float32, rows, axisSize int, eps float32) []float32 {
	out := make([]float32, len(x))
	for r := 0; r < rows; r++ {
		row := x[r*axisSize : (r+1)*axisSize]
		var mean float64
		for _, v := range row {
			mean += float64(v)
		}
		mean /= float64(axisSize)
		var variance float64
		for _, v := range row {
			d := float64(v) - mean
			variance += d * d
		}
		variance /= float64(axisSize)
		invStd := float32(1 / math.Sqrt(variance+float64(eps)))
		dst := out[r*axisSize : (r+1)*axisSize]
		for i, v := range row {
			dst[i] = (v-float32(mean))*invStd*weight[i] + bias[i]
		}
	}
	return out
}
