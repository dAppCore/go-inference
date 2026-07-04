// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

func BenchmarkCastBF16F32Roundtrip1024(b *testing.B) {
	requireNativeRuntime(b)

	in := toBF16Bytes(syntheticFloat32(1024, 3))
	b.SetBytes(int64(len(in)))
	withAutoreleasePool(func() {
		bf := sharedBytes(in)
		f32 := scratch(len(in) / bf16Size)
		out := scratchBF16(len(in) / bf16Size)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			cb := queue.CommandBuffer()
			enc := cb.ComputeCommandEncoder()
			if err := encWidenBF16ToF32(enc, bf, f32, len(in)/bf16Size); err != nil {
				b.Fatal(err)
			}
			if err := encNarrowF32ToBF16(enc, f32, out, len(in)/bf16Size); err != nil {
				b.Fatal(err)
			}
			enc.EndEncoding()
			cb.Commit()
			cb.WaitUntilCompleted()
		}
	})
}
