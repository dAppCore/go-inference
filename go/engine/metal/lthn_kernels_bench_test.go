// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"testing"
)

func BenchmarkMulScalarBF16_1024(b *testing.B) {
	requireNativeRuntime(b)
	if _, err := bf16MulScalarPipeline(); err != nil {
		b.Fatalf("bf16 scalar kernel unavailable: %v", err)
	}
	in := toBF16Bytes(syntheticFloat32(1024, 17))
	scalar := toBF16Bytes([]float32{0.375})
	if _, err := MulScalarBF16(in, scalar); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := MulScalarBF16(in, scalar); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkMulScalarBF16Into_1024(b *testing.B) {
	requireNativeRuntime(b)
	if _, err := bf16MulScalarPipeline(); err != nil {
		b.Fatalf("bf16 scalar kernel unavailable: %v", err)
	}
	in := toBF16Bytes(syntheticFloat32(1024, 17))
	scalar := toBF16Bytes([]float32{0.375})
	out := make([]byte, len(in))
	if err := MulScalarBF16Into(out, in, scalar); err != nil {
		b.Fatal(err)
	}
	b.SetBytes(int64(len(in)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := MulScalarBF16Into(out, in, scalar); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGeluGate{Composed,Fused} measure the gelu cost in ONE command buffer: the composed chain
// (~10 dispatches, each op rounded to bf16) vs the fused kernel (1 dispatch). The commit+wait is
// constant across both, so the delta isolates the dispatch-count cost — what the fused kernel saves
// on a host-bound decode. Synthetic (AX-11): no model load, dFF-sized buffers only.
func benchGeluGate(b *testing.B, fused bool) {
	if os.Getenv(MetallibPathEnv) == "" {
		b.Skip("metallib not set")
	}
	if err := ensureInit(); err != nil {
		b.Fatal(err)
	}
	if fused && !gpuHasGeluKernel() {
		b.Skip("fused gelu kernel not loaded")
	}
	const dFF = 8192
	zeros := make([]byte, dFF*bf16Size)
	withAutoreleasePool(func() {
		gBuf := sharedBytes(zeros)
		uBuf := sharedBytes(append([]byte(nil), zeros...))
		out := scratchBF16(dFF)
		x2, x3, x3s, inner := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		scaled, tnh, onePlus := scratchBF16(dFF), scratchBF16(dFF), scratchBF16(dFF)
		halfG, gelu := scratchBF16(dFF), scratchBF16(dFF)
		c044 := sharedBytes(bf16ConstBytes(dFF, 0.044715))
		c079 := sharedBytes(bf16ConstBytes(dFF, 0.7978845608028654))
		c1 := sharedBytes(bf16ConstBytes(dFF, 1.0))
		c05 := sharedBytes(bf16ConstBytes(dFF, 0.5))
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if fused {
				_ = encGeluGateMulFused(enc, gBuf, uBuf, out, dFF)
			} else {
				_ = encMulBF16(enc, gBuf, gBuf, x2, dFF)
				_ = encMulBF16(enc, x2, gBuf, x3, dFF)
				_ = encMulBF16(enc, x3, c044, x3s, dFF)
				_ = encAddBF16(enc, gBuf, x3s, inner, dFF)
				_ = encMulBF16(enc, inner, c079, scaled, dFF)
				_ = encTanhBF16(enc, scaled, tnh, dFF)
				_ = encAddBF16(enc, tnh, c1, onePlus, dFF)
				_ = encMulBF16(enc, gBuf, c05, halfG, dFF)
				_ = encMulBF16(enc, halfG, onePlus, gelu, dFF)
				_ = encMulBF16(enc, gelu, uBuf, out, dFF)
			}
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
	})
}

func BenchmarkGeluGateComposed(b *testing.B) { benchGeluGate(b, false) }
func BenchmarkGeluGateFused(b *testing.B)    { benchGeluGate(b, true) }
