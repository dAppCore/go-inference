// SPDX-Licence-Identifier: EUPL-1.2

package jsonenc

import (
	"strings"
	"testing"
)

// AX-11 baseline benchmarks for the jsonenc encoder surface. This is
// the per-response JSON encoding hot path — every adapter (anthropic,
// ollama, openai) builds its wire output through these helpers. A
// regression here scales 1×per-response across every backend.
//
// Caller-provided buf pattern means alloc-count should stay at zero
// for hot paths once the caller has pre-allocated a reasonable
// capacity. The fast-path scan in AppendJSONString gates the bulk
// copy; the escape-bearing slow path only fires when the input has
// special bytes.
//
// Run:
//   go test -bench=. -benchmem -benchtime=300ms ./jsonenc/...

// sink prevents the compiler from optimising the bench body away.
var jsonencBenchSink []byte

// --- AppendJSONString ---

// Fast path — typical adapter response text, no escapes, ~80 chars.
// The bulk-copy bytecount that lands in production response bodies.
func BenchmarkAppendJSONString_ShortNoEscape(b *testing.B) {
	buf := make([]byte, 0, 256)
	s := "The quick brown fox jumps over the lazy dog, on a bright morning"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendJSONString(buf, s)
	}
}

// Fast path at scale — 1 KiB ASCII body, no escapes. Catches the
// case where a fast-path scan that became O(n²) by accident would
// surface as a step-change in ns/op.
func BenchmarkAppendJSONString_LongNoEscape(b *testing.B) {
	buf := make([]byte, 0, 2048)
	s := strings.Repeat("abcdefghij", 102) + "abcd" // 1024 chars
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendJSONString(buf, s)
	}
}

// Slow path — mixed escapes (one quote, one backslash, one newline,
// one tab) in a 100-char body. Production: code snippets / JSON
// payloads nested in chat responses.
func BenchmarkAppendJSONString_WithEscapes(b *testing.B) {
	buf := make([]byte, 0, 256)
	s := `The string is "hello", with a path\to\file and a
newline and	tab break in the body — typical mixed content.`
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendJSONString(buf, s)
	}
}

// Worst case — every character requires an escape. Catches the
// per-byte switch-dispatch cost in appendJSONStringEscaped.
func BenchmarkAppendJSONString_AllEscapes(b *testing.B) {
	buf := make([]byte, 0, 1024)
	s := strings.Repeat("\"\\\b\f\n\r\t", 16) // 112 chars, all escapes
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendJSONString(buf, s)
	}
}

// Degenerate — empty string. Should be the cheapest call — just two
// quote bytes appended.
func BenchmarkAppendJSONString_Empty(b *testing.B) {
	buf := make([]byte, 0, 16)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendJSONString(buf, "")
	}
}

// --- AppendStringField (composes AppendJSONString) ---

// Typical KV pair — covers the common shape `"key":"value"` adapters
// emit for every response field.
func BenchmarkAppendStringField_Typical(b *testing.B) {
	buf := make([]byte, 0, 256)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendStringField(buf, "model", "qwen3-7b", false)
	}
}

// --- AppendIntField, AppendInt64Field, AppendBoolField ---

func BenchmarkAppendIntField_Typical(b *testing.B) {
	buf := make([]byte, 0, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendIntField(buf, "tokens", 4096, false)
	}
}

func BenchmarkAppendInt64Field_Typical(b *testing.B) {
	buf := make([]byte, 0, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendInt64Field(buf, "created", int64(1714291200), false)
	}
}

func BenchmarkAppendBoolField_Typical(b *testing.B) {
	buf := make([]byte, 0, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendBoolField(buf, "done", true, false)
	}
}

// --- AppendFloat32Field, AppendFloat32, AppendFloat64 ---

// Float encoding is the surprise-alloc surface — strconv.AppendFloat
// is the underlying primitive and is well-tuned, but worth a baseline.
func BenchmarkAppendFloat32Field_Typical(b *testing.B) {
	buf := make([]byte, 0, 64)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendFloat32Field(buf, "temperature", float32(0.72), false)
	}
}

func BenchmarkAppendFloat32_Typical(b *testing.B) {
	buf := make([]byte, 0, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendFloat32(buf, float32(0.72))
	}
}

func BenchmarkAppendFloat64_Typical(b *testing.B) {
	buf := make([]byte, 0, 32)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonencBenchSink = AppendFloat64(buf, float64(0.7253689))
	}
}

// AX-11: alloc budget for the encoder surface. Every public Append*
// function should stay at zero allocations on a pre-sized buffer —
// the caller-provided buf pattern is the whole point. Any regression
// that adds an alloc (e.g. switching to fmt.Sprintf, capturing a
// closure, escaping a temporary) fails this gate before propagating
// to every backend that uses the encoder.
//
// Run: go test -run TestAllocBudget . ./jsonenc/...
func TestAllocBudget_JSONEnc_AppendNoAllocs(t *testing.T) {
	cases := []struct {
		name string
		fn   func([]byte) []byte
	}{
		{"AppendJSONString_ShortNoEscape", func(buf []byte) []byte {
			return AppendJSONString(buf, "hello world this is typical text")
		}},
		{"AppendJSONString_Empty", func(buf []byte) []byte {
			return AppendJSONString(buf, "")
		}},
		{"AppendStringField", func(buf []byte) []byte {
			return AppendStringField(buf, "key", "value", false)
		}},
		{"AppendIntField", func(buf []byte) []byte {
			return AppendIntField(buf, "n", 42, false)
		}},
		{"AppendInt64Field", func(buf []byte) []byte {
			return AppendInt64Field(buf, "ts", int64(1714291200), false)
		}},
		{"AppendBoolField", func(buf []byte) []byte {
			return AppendBoolField(buf, "ok", true, false)
		}},
		{"AppendFloat32Field", func(buf []byte) []byte {
			return AppendFloat32Field(buf, "t", float32(0.5), false)
		}},
		{"AppendFloat32", func(buf []byte) []byte {
			return AppendFloat32(buf, float32(0.5))
		}},
		{"AppendFloat64", func(buf []byte) []byte {
			return AppendFloat64(buf, float64(0.5))
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// Pre-allocate generously so cap never grows mid-call.
			buf := make([]byte, 0, 1024)
			avg := testing.AllocsPerRun(5, func() {
				jsonencBenchSink = tc.fn(buf)
			})
			const budget = 0.0
			if avg > budget {
				t.Fatalf("%s alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
					"This is the per-response JSON encoder hot path — every adapter "+
					"pays this on every response field. Profile with: go test -bench=. "+
					"-benchmem -memprofile=/tmp/enc.mem && go tool pprof /tmp/enc.mem",
					tc.name, avg, budget)
			}
		})
	}
}
