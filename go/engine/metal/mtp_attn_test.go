// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"math"
	"testing"
	"unsafe"
)

// TestSDPACausalSelfAttention (BYTE-IDENTICAL to the real metal.ScaledDotProductAttention) lives in
// mtp_attn_metal_test.go — it needs the real cgo metal package as its oracle, so it's gated behind
// metal_runtime. The tests below are hermetic: scratch-pool residency, allocation budgets, and
// output-buffer reuse for SDPACausalBF16/SDPACausalBF16Into, needing no external oracle.

func sdpaScale(D int) float32 { return float32(1.0 / math.Sqrt(float64(D))) }

func TestSDPACausalBF16ScratchPoolKeepsShapesResident(t *testing.T) {
	small := getSDPACausalBF16Scratch(2, 1, 4, 4, 64)
	putSDPACausalBF16Scratch(small)
	large := getSDPACausalBF16Scratch(4, 2, 8, 8, 64)
	putSDPACausalBF16Scratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall := getSDPACausalBF16Scratch(2, 1, 4, 4, 64)
	defer putSDPACausalBF16Scratch(gotSmall)
	if gotSmall != small {
		t.Fatal("SDPA causal BF16 scratch pool evicted the small shape after using a larger shape")
	}

	gotLarge := getSDPACausalBF16Scratch(4, 2, 8, 8, 64)
	defer putSDPACausalBF16Scratch(gotLarge)
	if gotLarge != large {
		t.Fatal("SDPA causal BF16 scratch pool evicted the large shape after reusing the small shape")
	}
}

func TestSDPACausalBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)

	const H, Hkv, qL, kL, D = 2, 1, 4, 4, 64
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))
	if _, err := SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale); err != nil {
		t.Fatalf("SDPACausalBF16 warmup: %v", err)
	}

	var attnErr error
	allocs := testing.AllocsPerRun(3, func() {
		_, attnErr = SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale)
	})
	if attnErr != nil {
		t.Fatalf("SDPACausalBF16: %v", attnErr)
	}
	if allocs > 390 {
		t.Fatalf("SDPACausalBF16 allocations = %.0f, want <= 390", allocs)
	}
}

func TestSDPACausalBF16IntoReusesOutputBackingAndMatchesSDPACausalBF16(t *testing.T) {
	requireNativeRuntime(t)

	const H, Hkv, qL, kL, D = 2, 1, 4, 4, 64
	scale := sdpaScale(D)
	q := toBF16Bytes(syntheticFloat32(H*qL*D, 3))
	k := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 5))
	v := toBF16Bytes(syntheticFloat32(Hkv*kL*D, 7))
	want, err := SDPACausalBF16(q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		t.Fatalf("SDPACausalBF16 reference: %v", err)
	}
	out := bytes.Repeat([]byte{0xa5}, len(want))
	outPtr := unsafe.Pointer(&out[0])

	got, err := SDPACausalBF16Into(out, q, k, v, H, Hkv, qL, kL, D, scale)
	if err != nil {
		t.Fatalf("SDPACausalBF16Into: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("SDPACausalBF16Into did not reuse caller-owned output backing")
	}
	eqBytes(t, "SDPACausalBF16Into", got, want)
}

// TestSDPACausalSelfAttention lives in mtp_attn_metal_test.go (see the file-header comment above).
