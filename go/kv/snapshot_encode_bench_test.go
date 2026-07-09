// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for KV snapshot save/load + analysis primitives.
// Per AX-11 — Snapshot.Save fires per generation step (checkpointing);
// LoadWithOptions fires per session resume; Analyze runs on every
// resumed snapshot. The binary encoder (bytes / writeWithOptions)
// is the inner loop both Save and SaveStateBlocks hit.
//
// Run:    go test -bench='BenchmarkSnapshot|BenchmarkAnalyze|BenchmarkHash' -benchmem -run='^$' ./go/kv

package kv

import (
	"bytes"
	"testing"

	core "dappco.re/go"
)

func BenchmarkSnapshot_Save_512Tokens(b *testing.B) {
	dir := b.TempDir()
	snap := benchSnapshot(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkErr = snap.Save(core.JoinPath(dir, "snap.bin"))
	}
}

func BenchmarkSnapshot_Save_2048Tokens(b *testing.B) {
	dir := b.TempDir()
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkErr = snap.Save(core.JoinPath(dir, "snap.bin"))
	}
}

// --- Encoder hot path: bytes() in-memory (no disk IO) ---

func BenchmarkSnapshot_Bytes_512Tokens(b *testing.B) {
	snap := benchSnapshot(512)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes, benchSinkErr = snap.bytes()
	}
}

func BenchmarkSnapshot_Bytes_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes, benchSinkErr = snap.bytes()
	}
}

// --- writeWithOptions to a discarding writer (isolates the encoder
// from the alloc-the-return-slice cost in bytes()) ---

func BenchmarkSnapshot_WriteWithOptions_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	var buf bytes.Buffer
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf.Reset()
		benchSinkErr = snap.writeWithOptions(&buf, SaveOptions{})
	}
}

// --- Load (full roundtrip) ---
