// SPDX-Licence-Identifier: EUPL-1.2

// dtype + encoding variant benches.
//
// Encoding pathways exposed through SaveOptions.KVEncoding and the
// per-head/per-layer KeyDType / ValueDType fields drive different
// internal encode/decode legs. Existing benches only cover the default
// (float32) and EncodingNative-with-float32-values path. This file
// widens that surface against the four KV dtype legs we ship:
//
//   - float32           — base path, exercised by benchSnapshot()
//   - float16 (native)  — Apple MLX-Metal default for KV cache
//   - bfloat16 (native) — Gemma 4 / Qwen 3 default for compute dtype
//   - Q8 (kv-quantized) — memory-pressure cold path
//
// Coverage map (W7-F deepening pass):
//
//   - bytes() encode each variant @ 512 / 2048 tokens
//   - Load each variant @ 2048 tokens (the parse + decode leg)
//   - HashSnapshot each variant — the SaveStateBlocks per-block hash
//     fires per checkpoint × per block, encoding choice dictates the
//     stream-encoder branch (raw bytes vs. f32 stream vs. q8 quantize).
//
// Run: go test -bench='BenchmarkDtype' -benchmem -run='^$' ./go/kv

package kv

import (
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
)

// benchSnapshotF16 builds a fixture whose per-head K/V tensors carry
// native float16 KeyBytes / ValueBytes alongside the equivalent
// float32 values. Mirrors the shape go-mlx captures from Metal F16
// KV caches via CaptureOptions.RawKVOnly=true plus the float32 side
// for analyse paths.
func benchSnapshotF16(tokenCount int) *Snapshot {
	tokens := make([]int32, tokenCount)
	values := make([]float32, tokenCount)
	for i := range tokens {
		tokens[i] = int32(i + 1)
		values[i] = float32(i % 256)
	}
	keyBytes := make([]byte, tokenCount*2)
	valueBytes := make([]byte, tokenCount*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(keyBytes[i*2:i*2+2], float32ToFloat16(v))
		binary.LittleEndian.PutUint16(valueBytes[i*2:i*2+2], float32ToFloat16(v+1000))
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "qwen3",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        tokenCount,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{
			{Layer: 0, CacheIndex: 0, Heads: []HeadSnapshot{{Key: values, KeyDType: "float16", KeyBytes: keyBytes, Value: values, ValueDType: "float16", ValueBytes: valueBytes}}},
			{Layer: 1, CacheIndex: 1, Heads: []HeadSnapshot{{Key: values, KeyDType: "float16", KeyBytes: keyBytes, Value: values, ValueDType: "float16", ValueBytes: valueBytes}}},
		},
	}
}

// benchSnapshotBF16 — bfloat16 native dtype variant. Same shape as
// benchSnapshotF16; bfloat16 keeps the top 16 bits of the f32 bit
// pattern (no rounding required) — bench against the bfloat16 decode
// path which is byte-shift only vs. f16 ieee mantissa work.
func benchSnapshotBF16(tokenCount int) *Snapshot {
	tokens := make([]int32, tokenCount)
	values := make([]float32, tokenCount)
	for i := range tokens {
		tokens[i] = int32(i + 1)
		values[i] = float32(i % 256)
	}
	keyBytes := make([]byte, tokenCount*2)
	valueBytes := make([]byte, tokenCount*2)
	for i, v := range values {
		binary.LittleEndian.PutUint16(keyBytes[i*2:i*2+2], uint16(math.Float32bits(v)>>16))
		binary.LittleEndian.PutUint16(valueBytes[i*2:i*2+2], uint16(math.Float32bits(v+1000)>>16))
	}
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        tokens,
		TokenOffset:   tokenCount,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        tokenCount,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{
			{Layer: 0, CacheIndex: 0, Heads: []HeadSnapshot{{Key: values, KeyDType: "bfloat16", KeyBytes: keyBytes, Value: values, ValueDType: "bfloat16", ValueBytes: valueBytes}}},
			{Layer: 1, CacheIndex: 1, Heads: []HeadSnapshot{{Key: values, KeyDType: "bfloat16", KeyBytes: keyBytes, Value: values, ValueDType: "bfloat16", ValueBytes: valueBytes}}},
		},
	}
}

// --- bytes() encode per encoding ---

func BenchmarkDtype_Bytes_Float32_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes, benchSinkErr = snap.bytesWithOptions(SaveOptions{KVEncoding: KVSnapshotEncodingFloat32})
	}
}

func BenchmarkDtype_Bytes_NativeF16_2048Tokens(b *testing.B) {
	snap := benchSnapshotF16(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes, benchSinkErr = snap.bytesWithOptions(SaveOptions{KVEncoding: EncodingNative})
	}
}

func BenchmarkDtype_Bytes_NativeBF16_2048Tokens(b *testing.B) {
	snap := benchSnapshotBF16(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes, benchSinkErr = snap.bytesWithOptions(SaveOptions{KVEncoding: EncodingNative})
	}
}

func BenchmarkDtype_Bytes_Q8_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBytes, benchSinkErr = snap.bytesWithOptions(SaveOptions{KVEncoding: EncodingQ8})
	}
}

// --- Load parse + decode per encoding ---

func BenchmarkDtype_Load_Float32_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.bin")
	if err := snap.SaveWithOptions(path, SaveOptions{KVEncoding: KVSnapshotEncodingFloat32}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := Load(path)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

func BenchmarkDtype_Load_NativeF16_2048Tokens(b *testing.B) {
	snap := benchSnapshotF16(2048)
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.bin")
	if err := snap.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingNative}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// RawKVOnly=false to exercise the float16 → float32 decode
		// (math.Float16ToFloat32 per element) — the analyse-path leg.
		out, err := LoadWithOptions(path, LoadOptions{})
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

func BenchmarkDtype_Load_NativeF16_RawOnly_2048Tokens(b *testing.B) {
	snap := benchSnapshotF16(2048)
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.bin")
	if err := snap.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingNative}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		// RawKVOnly=true skips the float16→f32 decode — the cold
		// state-store wake path that re-warms a session for Metal
		// (Metal consumes the raw F16 bytes directly).
		out, err := LoadWithOptions(path, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

func BenchmarkDtype_Load_NativeBF16_RawOnly_2048Tokens(b *testing.B) {
	snap := benchSnapshotBF16(2048)
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.bin")
	if err := snap.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingNative}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := LoadWithOptions(path, LoadOptions{RawKVOnly: true})
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

func BenchmarkDtype_Load_Q8_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	dir := b.TempDir()
	path := core.JoinPath(dir, "snap.bin")
	if err := snap.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingQ8}); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, err := Load(path)
		if err != nil {
			b.Fatal(err)
		}
		benchSinkSnapshot = out
	}
}

// --- HashSnapshot per encoding — fires per checkpoint × per block ---

func BenchmarkDtype_HashSnapshot_Float32_2048Tokens(b *testing.B) {
	snap := benchSnapshot(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString, benchSinkErr = HashSnapshot(snap)
	}
}

func BenchmarkDtype_HashSnapshot_NativeF16_2048Tokens(b *testing.B) {
	snap := benchSnapshotF16(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString, benchSinkErr = HashSnapshot(snap)
	}
}

func BenchmarkDtype_HashSnapshot_NativeBF16_2048Tokens(b *testing.B) {
	snap := benchSnapshotBF16(2048)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkString, benchSinkErr = HashSnapshot(snap)
	}
}
