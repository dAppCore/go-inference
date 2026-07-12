// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// The quantize-writer benches baseline the pure header-preparation work (AX-11), run once per
// GGUF write: ggufQuantizeMetadata builds the []MetadataEntry (general.* + architecture.* keys)
// from the source descriptor, and assignGGUFTensorOffsets walks the quantised tensors assigning
// each a 32-byte-aligned data offset. Both are pure over in-memory descriptors — the file
// write itself (WriteFile) is I/O and not benched here. Synthetic — no file.

// BenchmarkGGUFQuantizeMetadata — assembling the GGUF metadata entries from a source + labels:
// the entry slice + the per-key value boxing, the once-per-write header prep.
func BenchmarkGGUFQuantizeMetadata(b *testing.B) {
	source := Source{
		Architecture: "gemma3", VocabSize: 262144, HiddenSize: 2048, NumLayers: 34, ContextLength: 131072,
	}
	labels := map[string]string{"provenance": "bench", "run": "b"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ggufQuantizeMetadata(source, QuantizeQ4_K, labels)
	}
}

// BenchmarkAssignGGUFTensorOffsets — walking 300 tensors to assign 32-byte-aligned data
// offsets: an in-place pass, no allocation. The per-write layout step.
func BenchmarkAssignGGUFTensorOffsets(b *testing.B) {
	tensors := make([]Tensor, 300)
	for i := range tensors {
		tensors[i] = Tensor{Name: "blk.weight", Data: make([]byte, 144*8)} // a Q4_K block-ish size
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		assignGGUFTensorOffsets(tensors, 32)
	}
}
