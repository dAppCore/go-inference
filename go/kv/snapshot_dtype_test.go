// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"bytes"
	"math"
	"testing"
)

func TestKVSnapshot_Q8ValidateBitTricks(t *testing.T) {
	// Bit-trick validate (NaN/Inf detect via exp mask + abs via bit-clear)
	// must produce maxAbs identical to the prior math.Abs walk and reject
	// the same NaN/Inf inputs as math.IsNaN/math.IsInf would.
	probes := []struct {
		name string
		vals []float32
		ok   bool
		max  float32
	}{
		{name: "positive", vals: []float32{0.5, 1.0, 1.5, 0.25}, ok: true, max: 1.5},
		{name: "negative", vals: []float32{-0.5, -1.0, -1.5, -0.25}, ok: true, max: 1.5},
		{name: "mixed", vals: []float32{-1.0, 2.0, -3.0, 0.5, -0.25, 0.75, 1.25, -1.5}, ok: true, max: 3.0},
		{name: "zero", vals: []float32{0, 0, 0, 0}, ok: true, max: 0},
		{name: "scalar-tail", vals: []float32{0.5, -0.5, 1.0}, ok: true, max: 1.0},
		{name: "nan-in-block", vals: []float32{1, 2, float32(math.NaN()), 3}, ok: false},
		{name: "nan-in-tail", vals: []float32{1, 2, 3, 4, float32(math.NaN())}, ok: false},
		{name: "posinf", vals: []float32{1, 2, float32(math.Inf(1))}, ok: false},
		{name: "neginf", vals: []float32{1, 2, float32(math.Inf(-1))}, ok: false},
	}
	for _, probe := range probes {
		maxAbs, ok := kvSnapshotQ8Validate(probe.vals)
		if ok != probe.ok {
			t.Fatalf("%s: ok = %v, want %v", probe.name, ok, probe.ok)
		}
		if ok && maxAbs != probe.max {
			t.Fatalf("%s: maxAbs = %v, want %v", probe.name, maxAbs, probe.max)
		}
	}
}

// TestKVSnapshot_Q8NativeRoundTrip pins the wire coherence of the raw q8 KV
// block dtype (#1846 state lane): kv carries KVNativeDTypeQ8 as an opaque native
// tensor (one byte per element), so a q8 layer's codes+scales survive
// encode→decode byte-exact and HashSnapshot does not choke on them. Pre-existing
// bf16 snapshots never carry the dtype, so this is purely additive.
func TestKVSnapshot_Q8NativeRoundTrip(t *testing.T) {
	kBlock := make([]byte, 30)
	vBlock := make([]byte, 30)
	for i := range kBlock {
		kBlock[i] = byte(i*7 + 1)
		vBlock[i] = byte(i*11 + 3)
	}
	snap := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2, 3},
		TokenOffset:   3,
		NumLayers:     1,
		NumHeads:      2,
		SeqLen:        3,
		HeadDim:       4,
		NumQueryHeads: 2,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			CacheMode:  "fixed",
			KeyDType:   KVNativeDTypeQ8,
			KeyBytes:   kBlock,
			KeyShape:   []int32{1, 2, 3, 4},
			ValueDType: KVNativeDTypeQ8,
			ValueBytes: vBlock,
			ValueShape: []int32{1, 2, 3, 4},
		}},
	}

	data, err := snap.bytesWithOptions(SaveOptions{KVEncoding: EncodingNative})
	if err != nil {
		t.Fatalf("encode q8-native snapshot: %v", err)
	}
	got, err := parseKVSnapshotWithOptions(data, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("decode q8-native snapshot: %v", err)
	}
	if len(got.Layers) != 1 {
		t.Fatalf("decoded layers = %d, want 1", len(got.Layers))
	}
	layer := got.Layers[0]
	if layer.KeyDType != KVNativeDTypeQ8 || layer.ValueDType != KVNativeDTypeQ8 {
		t.Fatalf("decoded dtype key=%q value=%q, want %q", layer.KeyDType, layer.ValueDType, KVNativeDTypeQ8)
	}
	if !bytes.Equal(layer.KeyBytes, kBlock) {
		t.Fatalf("decoded K block diverges: got %v want %v", layer.KeyBytes, kBlock)
	}
	if !bytes.Equal(layer.ValueBytes, vBlock) {
		t.Fatalf("decoded V block diverges: got %v want %v", layer.ValueBytes, vBlock)
	}

	if _, err := HashSnapshot(snap); err != nil {
		t.Fatalf("HashSnapshot(q8-native): %v", err)
	}

	// Validation guards: element count is the byte count, so a mismatch rejects.
	if _, err := validateKVSnapshotNativeTensor(KVNativeDTypeQ8, kBlock, len(kBlock)+1); err == nil {
		t.Fatal("validateKVSnapshotNativeTensor(q8-native length mismatch) error = nil")
	}
	if _, _, _, _, err := kvSnapshotNativeTensorInfo([]float32{1}, KVNativeDTypeQ8, kBlock); err == nil {
		t.Fatal("kvSnapshotNativeTensorInfo(q8-native with float values) error = nil")
	}
}

func TestKVSnapshot_NativeTensorValidationGuards(t *testing.T) {
	if _, err := validateKVSnapshotNativeTensor("int4", []byte{1}, 1); err == nil {
		t.Fatal("validateKVSnapshotNativeTensor(bad dtype) error = nil")
	}
	if _, err := validateKVSnapshotNativeTensor("float16", []byte{1}, 1); err == nil {
		t.Fatal("validateKVSnapshotNativeTensor(length mismatch) error = nil")
	}
	if _, err := decodeKVSnapshotNativeTensor("float16", []byte{1}, 1); err == nil {
		t.Fatal("decodeKVSnapshotNativeTensor(length mismatch) error = nil")
	}
	if _, _, _, _, err := kvSnapshotNativeTensorInfo([]float32{1, 2}, "float16", []byte{1, 2}); err == nil {
		t.Fatal("kvSnapshotNativeTensorInfo(element mismatch) error = nil")
	}
	if got := appendKVEncodedF32s(nil, []float32{1, 2}, KVSnapshotEncodingFloat32); len(got) == 0 {
		t.Fatal("appendKVEncodedF32s() returned empty encoding")
	}
}

// TestKVSnapshot_DecodeNativeFloat32_Good drives decodeKVSnapshotNativeTensor's
// float32 reinterpret-cast arm (snapshot.go:1347-1351), which the existing
// validation-error test never reaches (it only feeds mismatched lengths).
func TestKVSnapshot_DecodeNativeFloat32Path(t *testing.T) {
	raw := appendKVF32Raw(nil, []float32{1.5, -2.25})
	values, err := decodeKVSnapshotNativeTensor("float32", raw, 2)
	if err != nil || len(values) != 2 || values[0] != 1.5 || values[1] != -2.25 {
		t.Fatalf("decodeKVSnapshotNativeTensor(float32) = %v/%v, want [1.5 -2.25]", values, err)
	}
}

// TestSnapshot_QuantizeKVSnapshotQ8_Good covers the validate+quantise wrapper
// quantizeKVSnapshotQ8 (snapshot.go), which computes maxAbs then forwards to
// quantizeKVSnapshotQ8WithMaxAbs. The returned scale is maxAbs/127 and the
// largest-magnitude value must dequantise back to ~itself.
func TestSnapshot_QuantizeKVSnapshotQ8Path(t *testing.T) {
	values := []float32{0, 1.27, -1.27, 0.635}

	scale, quant := quantizeKVSnapshotQ8(values)
	if len(quant) != len(values) {
		t.Fatalf("quant len = %d, want %d", len(quant), len(values))
	}
	wantScale := float32(1.27) / 127
	if scale != wantScale {
		t.Fatalf("scale = %v, want %v", scale, wantScale)
	}
	// The +1.27 peak quantises to +127 and the -1.27 to -127.
	if int8(quant[1]) != 127 || int8(quant[2]) != -127 {
		t.Fatalf("quant peaks = %d/%d, want 127/-127", int8(quant[1]), int8(quant[2]))
	}

	// All-zero input keeps scale 1 (no divide-by-zero) and zeroed output.
	zScale, zQuant := quantizeKVSnapshotQ8([]float32{0, 0, 0})
	if zScale != 1 {
		t.Fatalf("all-zero scale = %v, want 1", zScale)
	}
	for i, b := range zQuant {
		if b != 0 {
			t.Fatalf("all-zero quant[%d] = %d, want 0", i, b)
		}
	}
}
