// SPDX-Licence-Identifier: EUPL-1.2

package filestore

import "testing"

// The json-codec benches baseline the hand-rolled record JSON path (AX-11) — the allocation-
// lean encoder/parser the file store uses instead of reflection for its per-record metadata:
// extractRecordURI pulls the "uri" field out of a record's meta bytes without a full decode
// (run per record at index rebuild); appendJSONString / appendJSONField emit an escaped field
// into a reused buffer at write; jsonUnescape decodes an escaped string body. All are
// per-record on the state save/load path. Pure Go — no file.

// BenchmarkExtractRecordURI — pulling the uri span out of a record's meta JSON without a full
// unmarshal: the hand-rolled scan the index rebuild runs per record.
func BenchmarkExtractRecordURI(b *testing.B) {
	data := []byte(`{"uri":"lthn://state/9f3a1c/session-42/0","labels":["kv","decode"],"tags":{"role":"assistant"}}`)
	b.SetBytes(int64(len(data)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := extractRecordURI(data); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkAppendJSONString — the escaped-string emit into a reused buffer: the fast (no-escape)
// path plus the escape branch, the per-field write cost.
func BenchmarkAppendJSONString(b *testing.B) {
	buf := make([]byte, 0, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = appendJSONString(buf[:0], "lthn://state/9f3a1c/session-42/0")
	}
}

// BenchmarkAppendJSONField — one key:value field appended into a reused buffer: the comma +
// quoted key + quoted value emit, the per-field write.
func BenchmarkAppendJSONField(b *testing.B) {
	buf := make([]byte, 0, 128)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = appendJSONField(buf[:0], "uri", "lthn://state/9f3a1c/session-42/0", true)
	}
}

// BenchmarkJSONUnescape — decoding an escaped string body (\n, \t, \uXXXX) into a Go string:
// the per-escaped-value cost on the parse path.
func BenchmarkJSONUnescape(b *testing.B) {
	src := []byte(`line one\nline two\ttabbedAend`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := jsonUnescape(src); err != nil {
			b.Fatal(err)
		}
	}
}
