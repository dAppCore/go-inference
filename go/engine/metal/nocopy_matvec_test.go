// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"testing"
	"unsafe"

	"github.com/tmc/apple/metal"
)

// TestMatVecBF16BufMatchesHost isolates MatVecBF16Buf from the shard/offset/decode machinery: it binds a
// FRESH buffer holding identical bytes at offset 0 — the exact thing MatVecBF16 does internally — so the two
// must be byte-for-byte equal. If this fails, MatVecBF16Buf's own dispatch differs from MatVecBF16; if it
// passes, any decode divergence is the shard binding (offset/alignment), not the kernel call.
func TestMatVecBF16BufMatchesHost(t *testing.T) {
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	outDim, inDim := 96, 256
	mat := make([]byte, outDim*inDim*bf16Size)
	vec := make([]byte, inDim*bf16Size)
	for i := range mat {
		mat[i] = byte((i*131 + 17) & 0x3f) // small deterministic bf16 bit patterns (no NaN/Inf)
	}
	for i := range vec {
		vec[i] = byte((i*97 + 5) & 0x3f)
	}
	host, err := MatVecBF16(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVecBF16: %v", err)
	}
	var res []byte
	withAutoreleasePool(func() {
		b := device.NewBufferWithBytesLengthOptions(unsafe.Pointer(&mat[0]), uint(len(mat)), metal.MTLResourceStorageModeShared)
		res, err = MatVecBF16Buf(bufView{buf: b, off: 0}, vec, outDim, inDim)
	})
	if err != nil {
		t.Fatalf("MatVecBF16Buf: %v", err)
	}
	for i := range host {
		if host[i] != res[i] {
			t.Fatalf("byte %d: host %d != resident %d — MatVecBF16Buf differs from MatVecBF16 with an identical buffer at offset 0", i, host[i], res[i])
		}
	}
	t.Logf("✓ MatVecBF16Buf == MatVecBF16 (%d out bytes, fresh buffer off 0)", len(host))
}

func TestResetResidentBufsForTestClearsCache(t *testing.T) {
	residentBufMu.Lock()
	residentBufs[uintptr(1)] = residentBuf{pin: []byte{1}}
	residentBufMu.Unlock()

	resetResidentBufsForTest()

	residentBufMu.Lock()
	defer residentBufMu.Unlock()
	if len(residentBufs) != 0 {
		t.Fatalf("residentBufs len after reset = %d, want 0", len(residentBufs))
	}
}

func TestMatVecBF16BufAllocationBudget(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const outDim, inDim = 128, 256
	mat := toBF16Bytes(syntheticFloat32(outDim*inDim, 3))
	vec := toBF16Bytes(syntheticFloat32(inDim, 5))
	matView := bufView{buf: residentBytes(mat), off: 0}
	if _, err := MatVecBF16Buf(matView, vec, outDim, inDim); err != nil {
		t.Fatalf("MatVecBF16Buf warmup: %v", err)
	}

	allocs := testing.AllocsPerRun(5, func() {
		if _, err := MatVecBF16Buf(matView, vec, outDim, inDim); err != nil {
			t.Fatalf("MatVecBF16Buf: %v", err)
		}
	})
	if allocs > 10 {
		t.Fatalf("MatVecBF16Buf allocations = %.0f, want <= 10", allocs)
	}
}

func TestMatVecBF16BufIntoReusesOutputBackingAndBypassesScratchOutput(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const outDim, inDim = 128, 256
	mat := toBF16Bytes(syntheticFloat32(outDim*inDim, 3))
	vec := toBF16Bytes(syntheticFloat32(inDim, 5))
	matView := bufView{buf: residentBytes(mat), off: 0}
	want, err := MatVecBF16Buf(matView, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVecBF16Buf reference: %v", err)
	}
	out := make([]byte, outDim*bf16Size)
	outPtr := unsafe.Pointer(&out[0])
	scratch, err := getQMVBF16Scratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch: %v", err)
	}
	sentinel := bytes.Repeat([]byte{0x71}, len(scratch.out.bytes))
	copy(scratch.out.bytes, sentinel)
	putQMVBF16Scratch(scratch)

	got, err := MatVecBF16BufInto(out, matView, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVecBF16BufInto: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MatVecBF16BufInto did not reuse caller-owned output backing")
	}
	eqBytes(t, "MatVecBF16BufInto", got, want)

	scratch, err = getQMVBF16Scratch(outDim, inDim)
	if err != nil {
		t.Fatalf("getQMVBF16Scratch after call: %v", err)
	}
	defer putQMVBF16Scratch(scratch)
	if !bytes.Equal(scratch.out.bytes, sentinel) {
		t.Fatal("MatVecBF16BufInto wrote through pooled scratch output instead of caller output")
	}
}

func TestMatVecBF16IntoReusesOutputBackingAndMatchesMatVecBF16(t *testing.T) {
	requireNativeRuntime(t)
	resetResidentBufsForTest()
	defer resetResidentBufsForTest()

	const outDim, inDim = 128, 256
	mat := toBF16Bytes(syntheticFloat32(outDim*inDim, 3))
	vec := toBF16Bytes(syntheticFloat32(inDim, 5))
	want, err := MatVecBF16(mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVecBF16 reference: %v", err)
	}
	out := bytes.Repeat([]byte{0xa5}, outDim*bf16Size)
	outPtr := unsafe.Pointer(&out[0])

	got, err := MatVecBF16Into(out, mat, vec, outDim, inDim)
	if err != nil {
		t.Fatalf("MatVecBF16Into: %v", err)
	}
	if len(got) != len(want) || unsafe.Pointer(&got[0]) != outPtr {
		t.Fatal("MatVecBF16Into did not reuse caller-owned output backing")
	}
	eqBytes(t, "MatVecBF16Into", got, want)
}
