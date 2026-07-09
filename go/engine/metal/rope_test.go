// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"
)

// TestRope_RoPE_Good proves the base invariant every rotation must satisfy: offset zero rotates
// by angle zero, so the output is the exact input, unchanged.
func TestRope_RoPE_Good(t *testing.T) {
	requireNativeRuntime(t)

	x := []float32{1, 2, 3, 4, -1, -2, -3, -4}
	got, err := RoPE(x, 1, 2, 4, 10000, 1, 0, false)
	if err != nil {
		t.Fatalf("RoPE: %v", err)
	}
	assertFloat32Near(t, "RoPE offset zero", got, x, 0)
}

// TestRope_RoPE_Bad proves the shape guard: len(x) must equal b*nHeads*headDim.
func TestRope_RoPE_Bad(t *testing.T) {
	requireNativeRuntime(t)

	if _, err := RoPE([]float32{1, 2, 3}, 1, 2, 4, 10000, 1, 0, false); err == nil {
		t.Fatal("expected RoPE to reject input length mismatch")
	}
}

// TestRope_RoPE_Ugly exercises the traditional=true pairing (the less-used adjacent-pair
// rotation, vs the default half-split convention) at a nonzero offset: RoPE is an orthogonal
// per-pair rotation, so it must preserve each head's L2 norm exactly (within float rounding)
// while still changing the values — a convention-agnostic proof that the rotation actually ran.
func TestRope_RoPE_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const b, nHeads, headDim = 1, 2, 8
	x := syntheticFloat32(b*nHeads*headDim, 5)
	got, err := RoPE(x, b, nHeads, headDim, 10000, 1, 17, true)
	if err != nil {
		t.Fatalf("RoPE traditional: %v", err)
	}
	if float32SliceEqual(got, x) {
		t.Fatal("RoPE at a nonzero offset returned the input unchanged — rotation did not run")
	}
	for h := range nHeads {
		var normIn, normOut float64
		for d := range headDim {
			i := h*headDim + d
			normIn += float64(x[i]) * float64(x[i])
			normOut += float64(got[i]) * float64(got[i])
		}
		if diff := math.Abs(math.Sqrt(normIn) - math.Sqrt(normOut)); diff > 1e-3 {
			t.Fatalf("head %d: ||RoPE(x)||=%.6f vs ||x||=%.6f (diff %.2e) — rotation is not norm-preserving", h, math.Sqrt(normOut), math.Sqrt(normIn), diff)
		}
	}
}

// float32SliceEqual reports whether got and want are elementwise identical — used only to
// sanity-check that a rotation actually perturbed its input (not a correctness assertion by
// itself; see the norm-preservation check alongside its call site).
func float32SliceEqual(got, want []float32) bool {
	if len(got) != len(want) {
		return false
	}
	for i := range got {
		if got[i] != want[i] {
			return false
		}
	}
	return true
}

// TestRope_RoPEInto_Good proves RoPEInto reuses caller-owned output backing, bypasses the pooled
// scratch output, and matches the allocating wrapper byte-for-byte.
func TestRope_RoPEInto_Good(t *testing.T) {
	requireNativeRuntime(t)

	const batch, nHeads, headDim = 1, 8, 64
	x := syntheticFloat32(batch*nHeads*headDim, 3)
	want, err := RoPE(x, batch, nHeads, headDim, 10000, 1, 17, false)
	if err != nil {
		t.Fatalf("RoPE reference: %v", err)
	}
	out := syntheticFloat32(len(x), 11)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVFloatScratch(len(x), len(x))
	if err != nil {
		t.Fatalf("getQMVFloatScratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x8e}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVFloatScratch(scratch)

	got, err := RoPEInto(out, x, batch, nHeads, headDim, 10000, 1, 17, false)
	if err != nil {
		t.Fatalf("RoPEInto: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("RoPEInto did not reuse caller-owned output backing")
	}
	if !bytes.Equal(float32Bytes(got), float32Bytes(want)) {
		t.Fatal("RoPEInto output differs from allocating wrapper")
	}

	scratch, err = getQMVFloatScratch(len(x), len(x))
	if err != nil {
		t.Fatalf("getQMVFloatScratch after call: %v", err)
	}
	defer putQMVFloatScratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("RoPEInto wrote through pooled scratch output instead of caller output")
	}
}

// TestRope_RoPEInto_Bad mirrors RoPE's shape guard through the caller-output entry point.
func TestRope_RoPEInto_Bad(t *testing.T) {
	requireNativeRuntime(t)

	out := make([]float32, 8)
	if _, err := RoPEInto(out, []float32{1, 2, 3}, 1, 2, 4, 10000, 1, 0, false); err == nil {
		t.Fatal("expected RoPEInto to reject input length mismatch")
	}
}

// TestRope_RoPEInto_Ugly proves the too-small-capacity path: when cap(out) is smaller than
// len(x), RoPEInto must allocate fresh storage rather than write out of bounds, and still return
// the correct rotation.
func TestRope_RoPEInto_Ugly(t *testing.T) {
	requireNativeRuntime(t)

	const batch, nHeads, headDim = 1, 8, 64
	x := syntheticFloat32(batch*nHeads*headDim, 3)
	want, err := RoPE(x, batch, nHeads, headDim, 10000, 1, 17, false)
	if err != nil {
		t.Fatalf("RoPE reference: %v", err)
	}

	tooSmall := make([]float32, 1)
	got, err := RoPEInto(tooSmall, x, batch, nHeads, headDim, 10000, 1, 17, false)
	if err != nil {
		t.Fatalf("RoPEInto (undersized out): %v", err)
	}
	assertFloat32Near(t, "RoPEInto (undersized out)", got, want, 0)
}

func TestRoPEAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	x := syntheticFloat32(8*64, 3)
	if _, err := RoPE(x, 1, 8, 64, 10000, 1, 17, false); err != nil {
		t.Fatalf("RoPE warmup: %v", err)
	}

	var ropeErr error
	allocs := testing.AllocsPerRun(5, func() {
		_, ropeErr = RoPE(x, 1, 8, 64, 10000, 1, 17, false)
	})
	if ropeErr != nil {
		t.Fatalf("RoPE: %v", ropeErr)
	}
	if allocs > 10 {
		t.Fatalf("RoPE allocations = %.0f, want <= 10", allocs)
	}
}
