// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	core "dappco.re/go"
	"testing"
)

// The write benches baseline the hand-rolled header emitter (AX-11): subsetHeaderEncoded
// validates + sorts refs and emits the safetensors JSON header bytes directly (replacing a
// reflection-driven json.Marshal), sizing one output buffer up-front — run once per
// WriteSubset. appendJSONInt64 is its allocation-free base-10 emitter. Both are pure over
// synthetic refs — no file written. Sized to a realistic sharded checkpoint's tensor count.

func benchSubsetRefs(n int) []TensorRef {
	refs := make([]TensorRef, n)
	var off int64
	for i := 0; i < n; i++ {
		bytes := int64(4096 * 2048 * 2)
		refs[i] = TensorRef{
			Name: "model.layers." + core.Sprintf("%d", i) + ".self_attn.q_proj.weight",
			DType: "BF16", Shape: []uint64{4096, 2048}, ByteLen: bytes, DataStart: off,
		}
		off += bytes
	}
	return refs
}

// BenchmarkSubsetHeaderEncoded — encoding a 256-tensor subset header: the name sort + the
// per-entry JSON emit into the pre-sized buffer. The byName map + names slice + the output
// buffer are the allocation story a WriteSubset pays once.
func BenchmarkSubsetHeaderEncoded(b *testing.B) {
	refs := benchSubsetRefs(256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, err := subsetHeaderEncoded(refs); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkAppendJSONInt64 — the base-10 int emit in isolation: a digit-extraction unroll
// into a fixed stack buffer appended to dst, expected zero heap allocation.
func BenchmarkAppendJSONInt64(b *testing.B) {
	dst := make([]byte, 0, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		dst = appendJSONInt64(dst[:0], int64(9_223_372_036_854_775_807-i))
	}
}
