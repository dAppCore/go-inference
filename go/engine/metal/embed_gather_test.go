// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"testing"
	"unsafe"
)

func embedGatherQuantFixture(vocab, dModel, groupSize, bits int) ([]byte, []byte, []byte) {
	packed := make([]byte, vocab*dModel*bits/8)
	for i := range packed {
		packed[i] = byte((i*131 + 17) % 256)
	}
	nSB := vocab * (dModel / groupSize)
	scales := toBF16Bytes(syntheticFloat32(nSB, 11))
	biases := toBF16Bytes(syntheticFloat32(nSB, 13))
	return packed, scales, biases
}

func TestEmbedGatherQuantBF16AllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}

	const vocab, dModel, groupSize, bits = 256, 512, 64, 4
	const scale = float32(0.5)
	packed, scales, biases := embedGatherQuantFixture(vocab, dModel, groupSize, bits)
	if _, err := EmbedGatherQuantBF16(42, packed, scales, biases, dModel, groupSize, bits, scale); err != nil {
		t.Fatalf("EmbedGatherQuantBF16 warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := EmbedGatherQuantBF16(42, packed, scales, biases, dModel, groupSize, bits, scale); err != nil {
			t.Fatalf("EmbedGatherQuantBF16: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("EmbedGatherQuantBF16 allocations = %.0f, want <= 10", allocs)
	}
}

// TestEmbedGather_EmbedGatherQuantBF16_Good gates the GPU embed-gather:
// EmbedGatherQuantBF16 must reproduce the host embedTokenQuant for a token's
// embedding row (same f32 affine arithmetic, same bf16 round) at EVERY width
// MLX's affine quantiser emits — the kernel's generic LSB-first unpack must
// track the host extractAffineCode oracle BYTE-for-byte (the chained decode's
// input seam depends on exact bytes; 3/5/6-bit span byte boundaries — the
// widths the old nibble-only kernel could not decode).
func TestEmbedGather_EmbedGatherQuantBF16_Good(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}
	const vocab, dModel, gs = 256, 1536, 64
	const scale = float32(0.5)
	for _, bits := range []int{2, 3, 4, 5, 6, 8} {
		packed, scales, biases := embedGatherQuantFixture(vocab, dModel, gs, bits)
		for _, tok := range []int32{0, 5, 42, 255} {
			ref, err := embedTokenQuant(packed, scales, biases, tok, vocab, dModel, gs, bits, scale)
			if err != nil {
				t.Fatalf("b%d tok %d: embedTokenQuant: %v", bits, tok, err)
			}
			got, err := EmbedGatherQuantBF16(tok, packed, scales, biases, dModel, gs, bits, scale)
			if err != nil {
				t.Fatalf("b%d tok %d: EmbedGatherQuantBF16: %v", bits, tok, err)
			}
			if !bytes.Equal(got, ref) {
				t.Fatalf("b%d tok %d: GPU embed-gather differs from host embedTokenQuant (cosine=%.7f)", bits, tok, cosineBF16(got, ref))
			}
		}
	}
	t.Logf("GPU embed-gather matches host embedTokenQuant byte-for-byte at every affine width")
}

// TestEmbedGather_EmbedGatherQuantBF16_Bad pins the affine-width gate: a width
// the kernel's unpack does not decode (7, and the degenerate 0/1) is rejected
// with an error, never dispatched as garbage.
func TestEmbedGather_EmbedGatherQuantBF16_Bad(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}

	const vocab, dModel, groupSize = 256, 512, 64
	packed, scales, biases := embedGatherQuantFixture(vocab, dModel, groupSize, 4)
	for _, bits := range []int{0, 1, 7} {
		if _, err := EmbedGatherQuantBF16(42, packed, scales, biases, dModel, groupSize, bits, 0.5); err == nil {
			t.Fatalf("expected EmbedGatherQuantBF16 to reject affine width %d", bits)
		}
	}
}

// TestEmbedGather_EmbedGatherQuantBF16_Ugly pins the zero-dModel corner: a
// valid width with nothing to gather returns an empty row without dispatching.
func TestEmbedGather_EmbedGatherQuantBF16_Ugly(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}

	empty, err := EmbedGatherQuantBF16(0, nil, nil, nil, 0, 64, 4, 0.5)
	if err != nil {
		t.Fatalf("EmbedGatherQuantBF16 dModel 0: %v", err)
	}
	if len(empty) != 0 {
		t.Fatalf("EmbedGatherQuantBF16 dModel 0 returned %d bytes, want none", len(empty))
	}
}

// TestEmbedGather_EmbedGatherQuantBF16Into_Good pins the Into contract: the
// caller-owned output backing is used directly and the bytes are identical to
// the allocating wrapper.
func TestEmbedGather_EmbedGatherQuantBF16Into_Good(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}

	const vocab, dModel, groupSize, bits = 256, 512, 64, 4
	const scale = float32(0.5)
	packed, scales, biases := embedGatherQuantFixture(vocab, dModel, groupSize, bits)
	out := make([]byte, dModel*bf16Size)
	for i := range out {
		out[i] = 0xA5
	}

	got, err := EmbedGatherQuantBF16Into(out, 42, packed, scales, biases, dModel, groupSize, bits, scale)
	if err != nil {
		t.Fatalf("EmbedGatherQuantBF16Into: %v", err)
	}
	if len(got) != len(out) {
		t.Fatalf("EmbedGatherQuantBF16Into len = %d, want %d", len(got), len(out))
	}
	if unsafe.Pointer(&got[0]) != unsafe.Pointer(&out[0]) {
		t.Fatal("EmbedGatherQuantBF16Into did not return caller-owned output backing")
	}
	want, err := EmbedGatherQuantBF16(42, packed, scales, biases, dModel, groupSize, bits, scale)
	if err != nil {
		t.Fatalf("EmbedGatherQuantBF16 reference: %v", err)
	}
	if !bytes.Equal(got, want) {
		t.Fatal("EmbedGatherQuantBF16Into output differs from allocating wrapper")
	}
}

// TestEmbedGather_EmbedGatherQuantBF16Into_Bad pins the affine-width gate
// through the Into surface: the same rejection fires regardless of the output
// the caller offers.
func TestEmbedGather_EmbedGatherQuantBF16Into_Bad(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}

	const vocab, dModel, groupSize = 256, 512, 64
	packed, scales, biases := embedGatherQuantFixture(vocab, dModel, groupSize, 4)
	out := make([]byte, dModel*bf16Size)
	if _, err := EmbedGatherQuantBF16Into(out, 42, packed, scales, biases, dModel, groupSize, 7, 0.5); err == nil {
		t.Fatal("expected EmbedGatherQuantBF16Into to reject affine width 7")
	}
}

// TestEmbedGather_EmbedGatherQuantBF16Into_Ugly pins the fallback allocation
// contract: an under-capacity out cannot be reused, so the call allocates a
// fresh row — still byte-identical to the allocating wrapper.
func TestEmbedGather_EmbedGatherQuantBF16Into_Ugly(t *testing.T) {
	requireNativeRuntime(t)
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}

	const vocab, dModel, groupSize, bits = 256, 512, 64, 4
	const scale = float32(0.5)
	packed, scales, biases := embedGatherQuantFixture(vocab, dModel, groupSize, bits)
	want, err := EmbedGatherQuantBF16(42, packed, scales, biases, dModel, groupSize, bits, scale)
	if err != nil {
		t.Fatalf("EmbedGatherQuantBF16 reference: %v", err)
	}

	short := make([]byte, 0, dModel*bf16Size-1)
	fromShort, err := EmbedGatherQuantBF16Into(short, 42, packed, scales, biases, dModel, groupSize, bits, scale)
	if err != nil {
		t.Fatalf("EmbedGatherQuantBF16Into(short): %v", err)
	}
	if len(fromShort) != dModel*bf16Size {
		t.Fatalf("EmbedGatherQuantBF16Into(short) returned %d bytes, want %d", len(fromShort), dModel*bf16Size)
	}
	if !bytes.Equal(fromShort, want) {
		t.Fatal("EmbedGatherQuantBF16Into(short) output differs from allocating wrapper")
	}
}

func TestEmbedGatherScratchPoolKeepsDimensionsResident(t *testing.T) {
	requireNativeRuntime(t)

	small, err := getEmbedGatherScratch(256)
	if err != nil {
		t.Fatalf("get small embed-gather scratch: %v", err)
	}
	putEmbedGatherScratch(small)

	large, err := getEmbedGatherScratch(512)
	if err != nil {
		t.Fatalf("get large embed-gather scratch: %v", err)
	}
	putEmbedGatherScratch(large)
	forceNativeGC()
	forceNativeGC()

	gotSmall, err := getEmbedGatherScratch(256)
	if err != nil {
		t.Fatalf("get small embed-gather scratch again: %v", err)
	}
	defer putEmbedGatherScratch(gotSmall)
	if gotSmall != small {
		t.Fatal("embed-gather scratch pool evicted the small scratch after using a larger scratch")
	}

	gotLarge, err := getEmbedGatherScratch(512)
	if err != nil {
		t.Fatalf("get large embed-gather scratch again: %v", err)
	}
	defer putEmbedGatherScratch(gotLarge)
	if gotLarge != large {
		t.Fatal("embed-gather scratch pool evicted the large scratch after reusing the small scratch")
	}
}
