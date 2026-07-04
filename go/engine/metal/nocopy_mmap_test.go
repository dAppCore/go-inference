// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"os"
	"syscall"
	"testing"
	"unsafe"

	coreio "dappco.re/go/io"
	"github.com/tmc/apple/kernel"
	"github.com/tmc/apple/metal"
)

// TestNoCopyMmapGPURead validates the keystone of the zero-copy weight path: a Metal no-copy
// buffer (bytesNoCopy) over FILE-BACKED mmap memory is correctly readable by the GPU. It
// maps a page-sized bf16 blob (the kernel returns a page-aligned base — what bytesNoCopy
// requires), wraps it no-copy, then runs the SAME kernel over the no-copy buffer and over a
// normal copied buffer and asserts identical GPU output. If Metal rejected the mapping or the
// GPU couldn't read it, the outputs would differ (or the buffer wouldn't be backed by the
// mmap). Proves the mmap → no-copy → GPU path before the assembler refactor commits to it.
// AX-11: no model load.
func TestNoCopyMmapGPURead(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		t.Fatal(err)
	}
	// One page of bf16 data — page-aligned base + page-multiple length keep bytesNoCopy happy.
	n := syscall.Getpagesize() / bf16Size
	data := make([]byte, n*bf16Size)
	addend := make([]byte, n*bf16Size)
	for i := 0; i < n; i++ {
		d := f32ToBF16(float32((i%17)-8) * 0.25) // clean finite bf16
		a := f32ToBF16(float32((i%5)+1) * 0.5)
		data[i*bf16Size], data[i*bf16Size+1] = byte(d), byte(d>>8)
		addend[i*bf16Size], addend[i*bf16Size+1] = byte(a), byte(a>>8)
	}
	path := t.TempDir() + "/raw.bf16"
	if err := coreio.Local.Write(path, string(data)); err != nil {
		t.Fatal(err)
	}
	fd, err := syscall.Open(path, syscall.O_RDONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Close(fd)
	mm, err := syscall.Mmap(fd, 0, len(data), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		t.Fatal(err)
	}
	defer syscall.Munmap(mm)

	var noCopyOut, copyOut []byte
	withAutoreleasePool(func() {
		noop := func(kernel.Pointer, uint64) {} // the mmap's lifetime is owned by the defer Munmap
		noCopy := device.NewBufferWithBytesNoCopyLengthOptionsDeallocator(unsafe.Pointer(&mm[0]), uint(len(mm)), metal.MTLResourceStorageModeShared, noop)
		if noCopy.Contents() != unsafe.Pointer(&mm[0]) {
			t.Fatalf("no-copy buffer not backed by the mmap (Contents=%p mmap=%p) — bytesNoCopy rejected the mapping", noCopy.Contents(), unsafe.Pointer(&mm[0]))
		}
		bBuf := sharedBytes(addend)
		out := scratchBF16(n)
		run := func(a metal.MTLBuffer) []byte {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if err := encAddBF16(enc, a, bBuf, out, n); err != nil {
				t.Fatal(err)
			}
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
			r := make([]byte, n*bf16Size)
			copy(r, unsafe.Slice((*byte)(out.Contents()), n*bf16Size))
			return r
		}
		noCopyOut = run(noCopy)
		copyOut = run(sharedBytes(data))
	})
	if !bytes.Equal(noCopyOut, copyOut) {
		t.Fatal("GPU output over the no-copy mmap buffer != over a copied buffer — the GPU did not read the mmap correctly")
	}
	t.Logf("no-copy mmap → GPU read OK: %d bf16 elems, page-aligned file-backed mmap, output matches the copied path", n)
}
