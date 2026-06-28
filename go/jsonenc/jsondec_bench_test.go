// SPDX-Licence-Identifier: EUPL-1.2

package jsonenc_test

import (
	"testing"

	"dappco.re/go/inference/jsonenc"
)

// AX-11 baseline benchmarks for the jsonenc DECODER surface
// (jsondec.go). This is the per-request JSON decode hot path — every
// adapter (anthropic, ollama, openai) walks its inbound request body
// through these primitives during UnmarshalJSON. A regression here
// scales 1×per-request across every backend.
//
// One benchmark per exported decode function, realistic adapter
// inputs, ReportAllocs. Package-level sinks defeat dead-code
// elimination. Black-box (package jsonenc_test) — every decode
// primitive is exported.
//
// Run:
//   go test -bench=. -benchmem -benchtime=200ms -run='^$' ./jsonenc/...

// Sinks — one per returned type so the compiler cannot prove the
// result unused and elide the call.
var (
	sinkStrings []string
	sinkString  string
	sinkBytes   []byte
	sinkInt     int64
	sinkInt32   int
	sinkBool    bool
	sinkF32     float32
	sinkF64     float64
	sinkByte    byte
	sinkErr     error
)

// --- ParseJSONStringList ---

// Single-string stop value — the `"END"` shape openai/ollama accept
// for a scalar `stop` field.
func BenchmarkParseJSONStringList_Single(b *testing.B) {
	data := []byte(`"</s>"`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkStrings, sinkErr = jsonenc.ParseJSONStringList(data)
	}
}

// Multi-element stop list — the common `["A","B","C"]` array shape.
// Exercises the parseJSONStringArray append loop.
func BenchmarkParseJSONStringList_Array(b *testing.B) {
	data := []byte(`["END","</s>","\n\nUser:","STOP"]`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkStrings, sinkErr = jsonenc.ParseJSONStringList(data)
	}
}

// --- ParseJSONString ---

// Fast path — typical adapter content string, no escapes. Returns a
// fresh Go string (inherent copy).
func BenchmarkParseJSONString_NoEscape(b *testing.B) {
	data := []byte(`"The quick brown fox jumps over the lazy dog, bright morning"`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString, sinkInt32, sinkErr = jsonenc.ParseJSONString(data, 0)
	}
}

// Escape path — mixed escapes in a content body (code snippet style).
func BenchmarkParseJSONString_Escape(b *testing.B) {
	data := []byte(`"line1\nline2 with \"quotes\" and a \\ slash and\ttab"`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString, sinkInt32, sinkErr = jsonenc.ParseJSONString(data, 0)
	}
}

// --- ParseJSONStringRaw ---

// Fast path — no-copy slice into data, should be zero-alloc.
func BenchmarkParseJSONStringRaw_NoEscape(b *testing.B) {
	data := []byte(`"The quick brown fox jumps over the lazy dog, bright morning"`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBytes, sinkInt32, sinkErr = jsonenc.ParseJSONStringRaw(data, 0)
	}
}

// Escape path — must materialise a decoded buffer.
func BenchmarkParseJSONStringRaw_Escape(b *testing.B) {
	data := []byte(`"line1\nline2 with \"quotes\" and a \\ slash and\ttab"`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBytes, sinkInt32, sinkErr = jsonenc.ParseJSONStringRaw(data, 0)
	}
}

// --- SkipJSONWhitespace ---

func BenchmarkSkipJSONWhitespace(b *testing.B) {
	data := []byte("    \t\n\r  {")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt32 = jsonenc.SkipJSONWhitespace(data, 0)
	}
}

// --- ParseJSONInt ---

func BenchmarkParseJSONInt(b *testing.B) {
	data := []byte(`1714291200`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt, sinkInt32, sinkErr = jsonenc.ParseJSONInt(data, 0)
	}
}

// --- ParseJSONBool ---

func BenchmarkParseJSONBool(b *testing.B) {
	data := []byte(`true`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBool, sinkInt32, sinkErr = jsonenc.ParseJSONBool(data, 0)
	}
}

// --- IsJSONNull ---

func BenchmarkIsJSONNull(b *testing.B) {
	data := []byte(`null`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBool = jsonenc.IsJSONNull(data, 0)
	}
}

// --- SkipJSONValue ---

// Skip an unknown nested field value during single-pass dispatch —
// the load-bearing use: a request carries fields an adapter ignores.
func BenchmarkSkipJSONValue_Nested(b *testing.B) {
	data := []byte(`{"a":1,"b":[1,2,3,{"x":"y"}],"c":{"d":"e","f":null}}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt32, sinkErr = jsonenc.SkipJSONValue(data, 0)
	}
}

// --- SkipJSONString ---

func BenchmarkSkipJSONString(b *testing.B) {
	data := []byte(`"The quick brown fox jumps over the lazy dog"`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt32, sinkErr = jsonenc.SkipJSONString(data, 0)
	}
}

// --- MatchObjectStart / MatchArrayStart ---

func BenchmarkMatchObjectStart(b *testing.B) {
	data := []byte(`  {"model":"x"}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt32, sinkErr = jsonenc.MatchObjectStart(data, 0)
	}
}

func BenchmarkMatchArrayStart(b *testing.B) {
	data := []byte(`  [1,2,3]`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt32, sinkErr = jsonenc.MatchArrayStart(data, 0)
	}
}

// --- ParseJSONFloat32 / ParseJSONFloat64 ---

func BenchmarkParseJSONFloat32(b *testing.B) {
	data := []byte(`0.7253689`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkF32, sinkInt32, sinkErr = jsonenc.ParseJSONFloat32(data, 0)
	}
}

func BenchmarkParseJSONFloat64(b *testing.B) {
	data := []byte(`-1.5e2`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkF64, sinkInt32, sinkErr = jsonenc.ParseJSONFloat64(data, 0)
	}
}

// --- CountJSONArrayElements ---

func BenchmarkCountJSONArrayElements(b *testing.B) {
	data := []byte(`1,2,3,{"x":"y"},[4,5],"s"]`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkInt32 = jsonenc.CountJSONArrayElements(data, 0)
	}
}

// --- HexChar (encode side, exported, previously unbenched) ---

func BenchmarkHexChar(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkByte = jsonenc.HexChar(byte(i))
	}
}
