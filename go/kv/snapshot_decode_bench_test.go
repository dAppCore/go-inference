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
	"testing"

	core "dappco.re/go"
)

func BenchmarkSnapshot_Load_512Tokens(b *testing.B) {
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.bin")
	if err := benchSnapshot(512).Save(path); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkSnapshot, benchSinkErr = Load(path)
	}
}

// --- Analyze ---
