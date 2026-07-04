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

func TestEmbedGatherQuantBF16IntoUsesCallerBacking(t *testing.T) {
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

// TestEmbedGatherQuantParity gates the GPU embed-gather: EmbedGatherQuantBF16 must reproduce the host
// embedTokenQuant for a token's 4-bit embedding row (same f32 affine arithmetic, same bf16 round). This
// is the seam that lets the chained decode step compute the next input on-GPU (the submit-ahead pipeline).
func TestEmbedGatherQuantParity(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Skipf("device init: %v", err)
	}
	if !gpuHasGeluKernel() {
		t.Skip("custom kernel library not loaded")
	}
	const vocab, dModel, gs, bits = 256, 1536, 64, 4
	const scale = float32(0.5)
	packed, scales, biases := embedGatherQuantFixture(vocab, dModel, gs, bits)

	for _, tok := range []int32{0, 5, 42, 255} {
		ref, err := embedTokenQuant(packed, scales, biases, tok, vocab, dModel, gs, bits, scale)
		if err != nil {
			t.Fatalf("tok %d: embedTokenQuant: %v", tok, err)
		}
		got, err := EmbedGatherQuantBF16(tok, packed, scales, biases, dModel, gs, bits, scale)
		if err != nil {
			t.Fatalf("tok %d: EmbedGatherQuantBF16: %v", tok, err)
		}
		if cos := cosineBF16(got, ref); cos < 0.99999 {
			t.Fatalf("tok %d: GPU embed-gather cosine=%.7f vs host embedTokenQuant", tok, cos)
		}
	}
	t.Logf("GPU embed-gather matches host embedTokenQuant")
}
