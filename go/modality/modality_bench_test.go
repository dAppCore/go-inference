// SPDX-Licence-Identifier: EUPL-1.2

package modality_test

import (
	"testing"

	"dappco.re/go/inference/modality"
)

// Package sinks — prevent the compiler eliminating the benchmarked work.
var (
	benchModalitySink []modality.Modality
	benchParseSink    modality.Modality
	benchMessageSink  modality.Message
	benchCountsSink   modality.OutputCounts
	benchPartSink     modality.ContentPart
	benchStringSink   string
	benchBoolSink     bool
)

// benchReqClean is a typical request modality set as the router hands it over:
// a small in-order set, no duplicates, clean lower-case wire strings — the
// common path through Requested (per request).
var benchReqClean = []modality.Modality{modality.Text, modality.Image}

// benchReqDirty exercises the dedup + tolerance branches: duplicates that must
// collapse keeping first-seen order, plus a whitespace/mixed-case entry.
var benchReqDirty = []modality.Modality{
	modality.Image, modality.Image, modality.Text, modality.Modality(" Audio "),
}

// benchParts is a realistic interleaved multimodal assistant output as a backend
// streams it back: text around an image and an audio clip, several text runs so
// the concatenation path (>=2 text parts) is exercised.
var benchParts = []modality.ContentPart{
	modality.TextPart("Here is the diagram you asked for: "),
	modality.ImagePart([]byte{0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a}, "image/png"),
	modality.TextPart(" and a short narration to go with it. "),
	modality.AudioPart([]byte{0x52, 0x49, 0x46, 0x46, 0x00, 0x00}, "audio/wav"),
	modality.TextPart("Let me know if you want changes."),
}

// BenchmarkParseModality_Clean is the zero-work path: a clean lower-case wire
// string (no trim, no case fold). Trim returns a sub-slice and Lower's ASCII
// fast path returns the input unchanged, so the happy path should be alloc-free.
func BenchmarkParseModality_Clean(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var m modality.Modality
	for i := 0; i < b.N; i++ {
		m, _ = modality.ParseModality("image")
	}
	benchParseSink = m
}

// BenchmarkParseModality_Fold exercises the case-folding path ("AUDIO"): here the
// strings.ToLower allocation is input-driven, not a missed win — kept distinct
// from the clean bench so the two cost profiles do not blur.
func BenchmarkParseModality_Fold(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var m modality.Modality
	for i := 0; i < b.N; i++ {
		m, _ = modality.ParseModality("AUDIO")
	}
	benchParseSink = m
}

// BenchmarkRequested_Clean normalises a typical clean request set (per request):
// parse each, dedup, return the output set.
func BenchmarkRequested_Clean(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var out []modality.Modality
	for i := 0; i < b.N; i++ {
		out, _ = modality.Requested(benchReqClean)
	}
	benchModalitySink = out
}

// BenchmarkRequested_Dirty drives the dedup + whitespace/case path: duplicates
// collapse, a padded mixed-case entry is folded.
func BenchmarkRequested_Dirty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var out []modality.Modality
	for i := 0; i < b.N; i++ {
		out, _ = modality.Requested(benchReqDirty)
	}
	benchModalitySink = out
}

// BenchmarkRequested_Empty is the nil-request implies-text path.
func BenchmarkRequested_Empty(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var out []modality.Modality
	for i := 0; i < b.N; i++ {
		out, _ = modality.Requested(nil)
	}
	benchModalitySink = out
}

// BenchmarkAssemble collects an interleaved multimodal output into one message:
// the text-part concatenation and the retained part list (per request).
func BenchmarkAssemble(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var msg modality.Message
	for i := 0; i < b.N; i++ {
		msg = modality.Assemble(benchParts)
	}
	benchMessageSink = msg
}

// BenchmarkCounts tallies an interleaved output by kind for usage accounting
// (per request). A value-struct accumulator over a slice should be alloc-free.
func BenchmarkCounts(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var c modality.OutputCounts
	for i := 0; i < b.N; i++ {
		c = modality.Counts(benchParts)
	}
	benchCountsSink = c
}

// BenchmarkTextPart / BenchmarkImagePart confirm the part constructors are plain
// value returns (the struct escapes only through the sink here).
func BenchmarkTextPart(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var p modality.ContentPart
	for i := 0; i < b.N; i++ {
		p = modality.TextPart("the answer is 42")
	}
	benchPartSink = p
}

func BenchmarkImagePart(b *testing.B) {
	data := []byte{0x89, 0x50, 0x4e, 0x47}
	b.ReportAllocs()
	b.ResetTimer()
	var p modality.ContentPart
	for i := 0; i < b.N; i++ {
		p = modality.ImagePart(data, "image/png")
	}
	benchPartSink = p
}

// BenchmarkString / BenchmarkValid / BenchmarkIsEmpty confirm the scalar
// accessors are alloc-free.
func BenchmarkString(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var s string
	for i := 0; i < b.N; i++ {
		s = modality.Image.String()
	}
	benchStringSink = s
}

func BenchmarkValid(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	var ok bool
	for i := 0; i < b.N; i++ {
		ok = modality.Audio.Valid()
	}
	benchBoolSink = ok
}

func BenchmarkIsEmpty(b *testing.B) {
	p := modality.ImagePart([]byte{1, 2}, "image/png")
	b.ReportAllocs()
	b.ResetTimer()
	var ok bool
	for i := 0; i < b.N; i++ {
		ok = p.IsEmpty()
	}
	benchBoolSink = ok
}
