// SPDX-Licence-Identifier: EUPL-1.2

package filestore

import "testing"

// The record-codec benches baseline the on-disk record header path (AX-11): encodeRecordHeader
// writes the 24-byte header (magic + chunkID + payloadSize + metaSize) into a caller buffer —
// fired once per chunk written on every state save, so it is zero-alloc by design;
// decodeRecordHeader reads + validates it, run once per record at the rebuildIndex cold Open
// (10k-scale). These pin the per-record codec stays allocation-free. Pure Go — no file.

// BenchmarkEncodeRecordHeader — the per-chunk header write into a reused buffer: the magic
// copy + three little-endian puts, expected zero allocation.
func BenchmarkEncodeRecordHeader(b *testing.B) {
	buf := make([]byte, recordHeaderLen)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		encodeRecordHeader(buf, 42, 65536, 128)
	}
}

// BenchmarkDecodeRecordHeader — the per-record header read at index rebuild: a length check, a
// single-Uint32 magic compare, three little-endian reads, no allocation. The 10k-scale cold-open cost.
func BenchmarkDecodeRecordHeader(b *testing.B) {
	buf := make([]byte, recordHeaderLen)
	encodeRecordHeader(buf, 42, 65536, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := decodeRecordHeader(buf); err != nil {
			b.Fatal(err)
		}
	}
}
