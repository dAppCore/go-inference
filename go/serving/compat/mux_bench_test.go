// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the compat mux's streaming wire encoders. Per AX-11 — these
// fire PER GENERATED TOKEN on every OpenAI/Ollama-compatible streamed answer:
// writeResponseDeltaFrame (/v1/responses), writeOllamaChatFrame (/api/chat)
// and writeOllamaGenerateFrame (/api/generate) each walk the fixed frame
// punctuation plus the HTML-safe JSON-escaped delta into a reused buffer, and
// appendJSONStringHTML is the escaper underneath them all. The buffer is
// reused across tokens by contract, so the target — and what these benches
// lock in against regression — is ZERO allocations per token. idWithPrefix
// runs once per request (the wire request-ID).
//
// Run:    go test -bench=. -benchmem -run='^$' ./serving/compat/
package compat

import (
	"io"
	"strings"
	"testing"
)

// Sinks defeat compiler DCE.
var (
	compatBenchSinkBuf []byte
	compatBenchSinkStr string
)

// benchDelta is a realistic streamed text delta — a short word run, the
// dominant per-token shape (all-safe bytes, the fast single-append path).
const benchDelta = "the quick brown fox "

// benchDeltaEscaped carries the HTML-meta and quote bytes a code/markup answer
// routinely streams, exercising the escape branches of appendJSONStringHTML.
const benchDeltaEscaped = `if a < b && c > d { return "<ok>" }`

// --- writeResponseDeltaFrame (/v1/responses per-token) -------------------

func BenchmarkWriteResponseDeltaFrame(b *testing.B) {
	buf := make([]byte, 0, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = writeResponseDeltaFrame(io.Discard, buf, benchDelta)
	}
	compatBenchSinkBuf = buf
}

func BenchmarkWriteResponseDeltaFrame_Escaped(b *testing.B) {
	buf := make([]byte, 0, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = writeResponseDeltaFrame(io.Discard, buf, benchDeltaEscaped)
	}
	compatBenchSinkBuf = buf
}

// --- writeOllamaChatFrame / writeOllamaGenerateFrame (per-token) ---------

func BenchmarkWriteOllamaChatFrame(b *testing.B) {
	buf := make([]byte, 0, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = writeOllamaChatFrame(io.Discard, buf, "gemma-4-31b", benchDelta)
	}
	compatBenchSinkBuf = buf
}

func BenchmarkWriteOllamaGenerateFrame(b *testing.B) {
	buf := make([]byte, 0, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = writeOllamaGenerateFrame(io.Discard, buf, "gemma-4-31b", benchDelta)
	}
	compatBenchSinkBuf = buf
}

// --- appendJSONStringHTML (the escaper the frames share) -----------------

// All-safe delta: the common path — one bulk copy, no escapes.
func BenchmarkAppendJSONStringHTML_Safe(b *testing.B) {
	buf := make([]byte, 0, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = appendJSONStringHTML(buf[:0], benchDelta)
	}
	compatBenchSinkBuf = buf
}

// Escaped delta: HTML-meta + quote bytes force the per-byte escape branches.
func BenchmarkAppendJSONStringHTML_Escaped(b *testing.B) {
	buf := make([]byte, 0, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = appendJSONStringHTML(buf[:0], benchDeltaEscaped)
	}
	compatBenchSinkBuf = buf
}

// Long safe run: a paragraph-sized delta, isolating the bulk-copy throughput.
func BenchmarkAppendJSONStringHTML_LongSafe(b *testing.B) {
	long := strings.Repeat("the quick brown fox jumps over the lazy dog ", 8)
	buf := make([]byte, 0, len(long)+16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf = appendJSONStringHTML(buf[:0], long)
	}
	compatBenchSinkBuf = buf
}

// --- idWithPrefix (per-request wire ID) ----------------------------------

func BenchmarkIdWithPrefix(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		compatBenchSinkStr = idWithPrefix("resp_")
	}
}
