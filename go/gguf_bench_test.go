// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the GGUF model-file primitives.
// Per AX-11 — ReadGGUFInfo is called once per model load; the
// metadata loop fires once per metadata entry, of which a typical
// GGUF has hundreds (every tensor name, vocab token, RoPE setting).
// readGGUFString is the per-entry hot loop the consumer pays.
//
// Run:    go test -bench='BenchmarkGGUF' -benchmem -run='^$' .

package inference

import (
	"bytes"
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE.
var (
	ggufSinkInfo GGUFInfo
	ggufSinkErr  error
	ggufSinkStr  string
)

// writeBenchGGUF builds a synthetic GGUF with the requested metadata
// shape — same wire format the production parser reads but built
// in-memory and written to a temp file via core.WriteFile so the
// bench harness can re-parse the same file many times.
func writeBenchGGUF(b *testing.B, metadata map[string]any) string {
	b.Helper()
	buf := core.NewBuffer()
	mustWrite := func(value any) {
		if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
			b.Fatal(err)
		}
	}
	writeString := func(value string) {
		mustWrite(uint64(len(value)))
		if _, err := buf.Write([]byte(value)); err != nil {
			b.Fatal(err)
		}
	}
	mustWrite(uint32(0x46554747))   // magic
	mustWrite(uint32(3))            // version
	mustWrite(uint64(0))            // tensor count
	mustWrite(uint64(len(metadata)))
	for key, value := range metadata {
		writeString(key)
		switch typed := value.(type) {
		case string:
			mustWrite(uint32(8))
			writeString(typed)
		case uint32:
			mustWrite(uint32(4))
			mustWrite(typed)
		default:
			b.Fatalf("unsupported metadata test value %T", value)
		}
	}
	path := core.JoinPath(b.TempDir(), "model.gguf")
	if r := core.WriteFile(path, buf.Bytes(), 0o644); !r.OK {
		b.Fatal(r.Value)
	}
	return path
}

// --- ReadGGUFInfo end-to-end (per-model load floor) ---

func BenchmarkGGUF_ReadInfo_Minimal(b *testing.B) {
	path := writeBenchGGUF(b, map[string]any{
		"general.architecture":  "qwen3",
		"general.file_type":     uint32(15),
		"qwen3.block_count":     uint32(28),
		"qwen3.context_length":  uint32(40960),
		"qwen3.embedding_length": uint32(2048),
	})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ggufSinkInfo, ggufSinkErr = ReadGGUFInfo(path)
	}
}

// BenchmarkGGUF_ReadInfo_VocabHeavy approximates a real model header
// — a few architecture fields plus a synthetic burst of metadata
// entries that mirrors the per-entry alloc cost of vocab string
// tables (which can have 256k+ entries on Gemma-class tokenisers).
func BenchmarkGGUF_ReadInfo_VocabHeavy(b *testing.B) {
	metadata := map[string]any{
		"general.architecture":   "qwen3",
		"general.file_type":      uint32(15),
		"qwen3.block_count":      uint32(28),
		"qwen3.context_length":   uint32(40960),
		"qwen3.embedding_length": uint32(2048),
	}
	// 200 synthetic metadata string entries — proxy for tokeniser
	// configuration + vocab marker strings.
	for i := 0; i < 200; i++ {
		metadata[core.Sprintf("synthetic.meta.%d", i)] = core.Sprintf("value-payload-%d", i)
	}
	path := writeBenchGGUF(b, metadata)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ggufSinkInfo, ggufSinkErr = ReadGGUFInfo(path)
	}
}

// --- readGGUFString in isolation (per-entry hot loop) ---

func BenchmarkGGUF_ReadString_Short(b *testing.B) {
	payload := []byte("qwen3")
	header := make([]byte, 8)
	binary.LittleEndian.PutUint64(header, uint64(len(payload)))
	frame := append(header, payload...)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ggufSinkStr, ggufSinkErr = readGGUFString(bytes.NewReader(frame))
	}
}

func BenchmarkGGUF_ReadString_Long(b *testing.B) {
	// Token strings can be up to a few hundred bytes (BPE merges).
	payload := bytes.Repeat([]byte("abcdef"), 64) // 384 bytes
	header := make([]byte, 8)
	binary.LittleEndian.PutUint64(header, uint64(len(payload)))
	frame := append(header, payload...)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ggufSinkStr, ggufSinkErr = readGGUFString(bytes.NewReader(frame))
	}
}
