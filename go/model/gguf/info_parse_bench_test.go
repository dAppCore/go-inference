// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the GGUF wire-parse hot loops. readGGUFString fires once
// per metadata key + once per string value on every header parse — these
// two benches moved here from the root inference package when the wire
// parser was unified into this package.
//
// Run:    go test -bench='BenchmarkInfoParse' -benchmem -run='^$' ./gguf

package gguf

import (
	"bytes"
	"encoding/binary"
	"testing"
)

// Sinks defeat compiler DCE.
var (
	infoParseSinkStr string
	infoParseSinkErr error
)

func BenchmarkInfoParse_readGGUFString_Short(b *testing.B) {
	payload := []byte("qwen3")
	header := make([]byte, 8)
	binary.LittleEndian.PutUint64(header, uint64(len(payload)))
	frame := append(header, payload...)
	scratch := make([]byte, 8)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		infoParseSinkStr, infoParseSinkErr = readGGUFString(bytes.NewReader(frame), scratch)
	}
}

func BenchmarkInfoParse_readGGUFString_Long(b *testing.B) {
	// Token strings can be up to a few hundred bytes (BPE merges).
	payload := bytes.Repeat([]byte("abcdef"), 64) // 384 bytes
	header := make([]byte, 8)
	binary.LittleEndian.PutUint64(header, uint64(len(payload)))
	frame := append(header, payload...)
	scratch := make([]byte, 8)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		infoParseSinkStr, infoParseSinkErr = readGGUFString(bytes.NewReader(frame), scratch)
	}
}
