// SPDX-Licence-Identifier: EUPL-1.2

// AX-11 allocation baselines for the safetensors header-parse hot path.
//
// safetensors_bench_test.go already benches ReadIndex end-to-end (file
// open + read + parse). These benches isolate the *parse* itself: they
// build the header JSON once in memory and drive ParseHeaderRefs and the
// first-pass counter countTensorsAndDims directly, so the reported
// allocs/op·B/op reflect the hand-rolled walker without file-I/O noise.
// ParseHeaderRefs fires once per shard on every model load; a Gemma-class
// model has ~200+ tensor refs, so this is a real load-path inner loop.
//
// Run:    go test -run='^$' -bench=BenchmarkHeaderParse -benchmem -benchtime=20x ./safetensors/
//
// Reuses stIntStr, stSinkIndex and stSinkErr from safetensors_bench_test.go
// (same package) — do not redeclare them here.

package safetensors

import (
	"testing"

	core "dappco.re/go"
)

// Sinks for the two-value countTensorsAndDims return — kept distinct from
// the shared stSink* set so this file owns them.
var (
	stSinkCountTensors int
	stSinkCountDims    int
)

// buildHeaderBytes assembles a safetensors JSON header (the bytes that sit
// after the 8-byte length prefix) for tensorCount U8 tensors plus, when
// withMetadata is set, a __metadata__ block. No file is written — the
// returned bytes feed ParseHeaderRefs / countTensorsAndDims directly so
// the bench measures parse cost in isolation. Built with core.JSONMarshal
// so the byte shape matches what a real writer emits; names are sorted to
// mirror on-disk ordering.
func buildHeaderBytes(b *testing.B, tensorCount int, withMetadata bool) []byte {
	b.Helper()
	header := map[string]HeaderEntry{}
	names := make([]string, 0, tensorCount)
	for i := range tensorCount {
		names = append(names, "model.layers."+stIntStr(i/4)+".self_attn.q_proj.weight."+stIntStr(i%4))
	}
	core.SliceSort(names)
	var offset int64
	for _, name := range names {
		header[name] = HeaderEntry{
			DType:       "U8",
			Shape:       []int64{16},
			DataOffsets: []int64{offset, offset + 16},
		}
		offset += 16
	}
	wrapped := map[string]any{}
	for k, v := range header {
		wrapped[k] = map[string]any{
			"dtype":        v.DType,
			"shape":        v.Shape,
			"data_offsets": v.DataOffsets,
		}
	}
	if withMetadata {
		wrapped["__metadata__"] = map[string]any{
			"format":  "pt",
			"version": "1",
			"extra":   "value with \"escapes\" and {braces} inside",
		}
	}
	encoded := core.JSONMarshal(wrapped)
	if !encoded.OK {
		b.Fatalf("JSONMarshal: %v", encoded.Value)
	}
	return encoded.Value.([]byte)
}

// buildMultiDimHeaderBytes assembles a header whose tensors carry 4-D
// shapes — the representative transformer weight shape (heads × layers ×
// dim × headdim) — so the shape-slab carving path is exercised with more
// than one dim per tensor.
func buildMultiDimHeaderBytes(b *testing.B, tensorCount int) []byte {
	b.Helper()
	names := make([]string, 0, tensorCount)
	for i := range tensorCount {
		names = append(names, "model.layers."+stIntStr(i)+".self_attn.q_proj.weight")
	}
	core.SliceSort(names)
	wrapped := map[string]any{}
	var offset int64
	const bytesPer = 4 * 28 * 64 * 64 * 2 // F16
	for _, name := range names {
		wrapped[name] = map[string]any{
			"dtype":        "F16",
			"shape":        []int64{4, 28, 64, 64},
			"data_offsets": []int64{offset, offset + bytesPer},
		}
		offset += bytesPer
	}
	encoded := core.JSONMarshal(wrapped)
	if !encoded.OK {
		b.Fatalf("JSONMarshal: %v", encoded.Value)
	}
	return encoded.Value.([]byte)
}

// --- ParseHeaderRefs — the parse path in isolation (no file I/O) ---

// Representative single-tensor-class header: 16 small tensors, the size a
// tiny adapter or a sharded model's smallest shard carries.
func BenchmarkHeaderParse_ParseHeaderRefs_Small(b *testing.B) {
	header := buildHeaderBytes(b, 16, false)
	dataStart := int64(8 + len(header))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkIndex, stSinkErr = ParseHeaderRefs("bench.safetensors", header, dataStart)
	}
}

// Typical per-shard header: ~200 tensors (≈ a 28-layer model, 7
// tensors/layer). This is the load-path representative case.
func BenchmarkHeaderParse_ParseHeaderRefs_Typical(b *testing.B) {
	header := buildHeaderBytes(b, 200, true)
	dataStart := int64(8 + len(header))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkIndex, stSinkErr = ParseHeaderRefs("bench.safetensors", header, dataStart)
	}
}

// Large many-tensor header: 1000 tensors, the stress shape a big dense
// checkpoint in one file would produce.
func BenchmarkHeaderParse_ParseHeaderRefs_Large(b *testing.B) {
	header := buildHeaderBytes(b, 1000, true)
	dataStart := int64(8 + len(header))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkIndex, stSinkErr = ParseHeaderRefs("bench.safetensors", header, dataStart)
	}
}

// Multi-dim header: 200 tensors each with a 4-D shape, exercising the
// shape-slab carve with several dims per tensor (vs the 1-D fast case).
func BenchmarkHeaderParse_ParseHeaderRefs_MultiDim(b *testing.B) {
	header := buildMultiDimHeaderBytes(b, 200)
	dataStart := int64(8 + len(header))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkIndex, stSinkErr = ParseHeaderRefs("bench.safetensors", header, dataStart)
	}
}

// --- countTensorsAndDims — the cheap first-pass sizer ---

// The first pass runs before ParseHeaderRefs's main walk to size the map,
// Names slice and shape slab in one allocation each. It is alloc-free by
// design (pure index walk over the bytes); this bench pins that baseline.
func BenchmarkHeaderParse_CountTensorsAndDims_Typical(b *testing.B) {
	header := buildHeaderBytes(b, 200, true)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkCountTensors, stSinkCountDims = countTensorsAndDims(header)
	}
}

func BenchmarkHeaderParse_CountTensorsAndDims_Large(b *testing.B) {
	header := buildHeaderBytes(b, 1000, true)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		stSinkCountTensors, stSinkCountDims = countTensorsAndDims(header)
	}
}
