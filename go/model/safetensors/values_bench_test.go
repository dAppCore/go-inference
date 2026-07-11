// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import "testing"

// The values benches baseline the whole-file float codec (AX-11): DecodeFloat32 upcasts a
// tensor's raw byte payload to []float32 per dtype at load — the per-element F16/BF16 upcast
// or F32 reinterpret, allocating the [elements]float32 result. EncodeFloat32 is the reverse
// F32 pack. These run once per weight at load (the simple whole-file path; index.go's
// DecodeFloatData is the streaming/unsafe sibling). Sized to a realistic per-tensor chunk
// (1M elements). Pure Go, synthetic bytes — no file read.

const benchDecodeElems = 1 << 20 // 1M elements — a realistic per-tensor decode chunk

// benchF16Bytes fills n F16 elements as little-endian raw bytes with a deterministic spread.
func benchF16Bytes(n int) []byte {
	raw := make([]byte, n*2)
	for i := 0; i < n; i++ {
		v := uint16(i * 137) // full pattern spread (finite, inf and nan all exercised)
		raw[2*i], raw[2*i+1] = byte(v), byte(v>>8)
	}
	return raw
}

func benchF32Slice(n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = float32((i*131)%4096-2048) * 0.001
	}
	return s
}

// BenchmarkDecodeFloat32_F16 — the F16 upcast: per-element Float16ToFloat32 into a fresh
// [elements]float32. The scalar whole-file path (contrast index.go's NEON DecodeFloatData).
func BenchmarkDecodeFloat32_F16(b *testing.B) {
	raw := benchF16Bytes(benchDecodeElems)
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeFloat32("F16", raw, benchDecodeElems); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkDecodeFloat32_BF16 — the BF16 upcast: per-element left-shift widen into a fresh
// result slice. The dtype most model weights ship in.
func BenchmarkDecodeFloat32_BF16(b *testing.B) {
	raw := benchF16Bytes(benchDecodeElems)
	b.SetBytes(int64(len(raw)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := DecodeFloat32("BF16", raw, benchDecodeElems); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkEncodeFloat32 — the F32 pack: per-element PutUint32 into a fresh [4·len] byte
// buffer, the on-disk layout WriteSafetensors expects.
func BenchmarkEncodeFloat32(b *testing.B) {
	vals := benchF32Slice(benchDecodeElems)
	b.SetBytes(int64(len(vals) * 4))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = EncodeFloat32(vals)
	}
}
