// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"bytes"
	"errors"
	"testing"
)

// snapshotBadLayerRaw returns a snapshot whose single layer carries a
// native-only raw Key tensor with an unsupported dtype. Under EncodingNative
// the encoder rejects it (errUnsupportedNativeTensor) the moment it touches
// the layer key, which is the failure all three encode surfaces share.
func snapshotBadLayerRaw() *Snapshot {
	s := testSnapshot()
	s.Version = SnapshotVersion
	s.Layers = []LayerSnapshot{{
		Layer:      0,
		CacheIndex: 0,
		KeyBytes:   []byte{0, 0, 0, 0},
		KeyDType:   "nonsense", // unrecognised → errUnsupportedNativeTensor
		KeyShape:   []int32{1, 1, 2, 2},
	}}
	return s
}

// snapshotBadLayerValueRaw is the value-side mirror: a valid layer key but a
// value raw tensor with an unsupported dtype, so the encoder gets past the
// key and trips on the layer value.
func snapshotBadLayerValueRaw() *Snapshot {
	s := testSnapshot()
	s.Version = SnapshotVersion
	s.Layers = []LayerSnapshot{{
		Layer:      0,
		CacheIndex: 0,
		KeyBytes:   []byte{1, 2},
		KeyDType:   "float16",
		KeyShape:   []int32{1, 1, 1, 1},
		ValueBytes: []byte{0, 0, 0, 0},
		ValueDType: "nonsense", // unrecognised → errUnsupportedNativeTensor
		ValueShape: []int32{1, 1, 2, 2},
	}}
	return s
}

// snapshotBadHeadRaw returns a snapshot whose head carries a raw Key tensor
// with an unsupported dtype (and no layer-raw tensors), so the per-head
// encode arm is the one that fails.
func snapshotBadHeadRaw() *Snapshot {
	s := testSnapshot()
	s.Version = SnapshotVersion
	s.Layers = []LayerSnapshot{{
		Layer:      0,
		CacheIndex: 0,
		Heads: []HeadSnapshot{{
			KeyBytes: []byte{0, 0},
			KeyDType: "nonsense", // unrecognised → errUnsupportedNativeTensor
		}},
	}}
	return s
}

// snapshotBadHeadValueRaw is the head value-side mirror.
func snapshotBadHeadValueRaw() *Snapshot {
	s := testSnapshot()
	s.Version = SnapshotVersion
	s.Layers = []LayerSnapshot{{
		Layer:      0,
		CacheIndex: 0,
		Heads: []HeadSnapshot{{
			Key:        []float32{1, 2},
			ValueBytes: []byte{0, 0},
			ValueDType: "nonsense", // unrecognised → errUnsupportedNativeTensor
		}},
	}}
	return s
}

// TestSnapshotEncodeCover_LayerEncodeErrors drives the layer key/value encode
// error arms across all three encode surfaces (encodedSizeWithOptions,
// bytesWithOptions, writeWithOptions) with EncodingNative — the only encoding
// that walks raw layer tensors.
func TestSnapshotEncodeCover_LayerEncodeErrors(t *testing.T) {
	opts := SaveOptions{KVEncoding: EncodingNative}

	for name, build := range map[string]func() *Snapshot{
		"layer-key":   snapshotBadLayerRaw,
		"layer-value": snapshotBadLayerValueRaw,
	} {
		t.Run(name, func(t *testing.T) {
			s := build()
			if _, err := s.encodedSizeWithOptions(opts); err == nil {
				t.Fatal("encodedSizeWithOptions error = nil, want native tensor error")
			}
			if _, err := s.bytesWithOptions(opts); err == nil {
				t.Fatal("bytesWithOptions error = nil, want native tensor error")
			}
			var buf bytes.Buffer
			if err := s.writeWithOptions(&buf, opts); err == nil {
				t.Fatal("writeWithOptions error = nil, want native tensor error")
			}
		})
	}
}

// TestSnapshotEncodeCover_HeadEncodeErrors drives the per-head key/value
// encode error arms across the three encode surfaces.
func TestSnapshotEncodeCover_HeadEncodeErrors(t *testing.T) {
	opts := SaveOptions{KVEncoding: EncodingNative}

	for name, build := range map[string]func() *Snapshot{
		"head-key":   snapshotBadHeadRaw,
		"head-value": snapshotBadHeadValueRaw,
	} {
		t.Run(name, func(t *testing.T) {
			s := build()
			if _, err := s.encodedSizeWithOptions(opts); err == nil {
				t.Fatal("encodedSizeWithOptions error = nil, want native tensor error")
			}
			if _, err := s.bytesWithOptions(opts); err == nil {
				t.Fatal("bytesWithOptions error = nil, want native tensor error")
			}
			var buf bytes.Buffer
			if err := s.writeWithOptions(&buf, opts); err == nil {
				t.Fatal("writeWithOptions error = nil, want native tensor error")
			}
		})
	}
}

// TestSnapshotEncodeCover_WriteEarlyValidation drives the three early-exit
// guards of writeWithOptions that the size-pass siblings cannot reach because
// writeWithOptions validates independently of encodedSizeWithOptions: a bad
// encoding, a compressed-payload mismatch, and an out-of-range version.
func TestSnapshotEncodeCover_WriteEarlyValidation(t *testing.T) {
	var buf bytes.Buffer

	// Bad encoding → normalizeKVSnapshotEncoding error (guard at 213).
	if err := testSnapshot().writeWithOptions(&buf, SaveOptions{KVEncoding: Encoding("nope")}); err == nil {
		t.Fatal("writeWithOptions(bad encoding) error = nil, want encoding error")
	}

	// TurboQuant payloads without the matching cache mode → the compressed-
	// payload validator rejects it (guard at 216).
	badPayload := testSnapshot()
	badPayload.Layers[0].TurboQuantPayloads = [][]byte{{1, 2, 3}}
	badPayload.Layers[0].CacheMode = "" // not "turboquant"
	if err := badPayload.writeWithOptions(&buf, SaveOptions{}); err == nil {
		t.Fatal("writeWithOptions(payload mode mismatch) error = nil, want payload error")
	}

	// Version beyond SnapshotVersion → the version-range guard at 225.
	badVersion := testSnapshot()
	badVersion.Version = SnapshotVersion + 1
	if err := badVersion.writeWithOptions(&buf, SaveOptions{}); err == nil {
		t.Fatal("writeWithOptions(version too high) error = nil, want version error")
	}
}

// TestSnapshotEncodeCover_StreamRawNeedsNative drives the stream encoder's
// errRawTensorNeedsNative arm: a head with a raw payload but no float32
// values, serialised under a non-native (Q8) encoding via writeWithOptions
// (which does not pre-validate via encodedSizeWithOptions).
func TestSnapshotEncodeCover_StreamRawNeedsNative(t *testing.T) {
	rawOnly := testSnapshot()
	rawOnly.Version = SnapshotVersion
	rawOnly.Layers = []LayerSnapshot{{
		Layer:      0,
		CacheIndex: 0,
		Heads: []HeadSnapshot{{
			KeyBytes: cvtRawF16(2, 2),
			KeyDType: "float16",
		}},
	}}
	var buf bytes.Buffer
	if err := rawOnly.writeWithOptions(&buf, SaveOptions{KVEncoding: EncodingQ8}); !errors.Is(err, errRawTensorNeedsNative) {
		t.Fatalf("writeWithOptions(raw-only Q8) error = %v, want errRawTensorNeedsNative", err)
	}
}

// failingWriter fails after acceptN successful writes so the stream writer's
// error guard short-circuits the rest of the encode.
type failingWriter struct {
	acceptN int
	count   int
}

func (w *failingWriter) Write(p []byte) (int, error) {
	if w.count >= w.acceptN {
		return 0, errors.New("forced write failure")
	}
	w.count++
	return len(p), nil
}

// TestSnapshotEncodeCover_StreamWriteError drives the stream writer's error
// propagation: once the underlying writer fails, writeWithOptions returns the
// stream error rather than completing.
func TestSnapshotEncodeCover_StreamWriteError(t *testing.T) {
	s := testSnapshot()
	// Accept the magic write then fail — exercises the w.err guard threading
	// through the subsequent u32/bytes calls.
	w := &failingWriter{acceptN: 1}
	if err := s.writeWithOptions(w, SaveOptions{}); err == nil {
		t.Fatal("writeWithOptions(failing writer) error = nil, want the forced failure")
	}
}

// shortWriter reports fewer bytes written than handed to it, which the stream
// writer turns into io.ErrShortWrite.
type shortWriter struct{}

func (shortWriter) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	return len(p) - 1, nil
}

// TestSnapshotEncodeCover_StreamShortWrite drives the n != len(data) branch of
// the stream writer's bytes() helper.
func TestSnapshotEncodeCover_StreamShortWrite(t *testing.T) {
	s := testSnapshot()
	if err := s.writeWithOptions(shortWriter{}, SaveOptions{}); err == nil {
		t.Fatal("writeWithOptions(short writer) error = nil, want ErrShortWrite")
	}
}

// TestSnapshotEncodeCover_Q8Quantize drives the Q8 quantise path through the
// fused encode arm (appendKVEncodedTensor). The lower -127 clamp inside
// quantizeKVSnapshotQ8WithMaxAbs is mathematically unreachable when maxAbs is
// the honest max-abs (value/scale stays within [-127,+127]), so it is left
// uncovered deliberately.
func TestSnapshotEncodeCover_Q8Quantize(t *testing.T) {
	values := []float32{254, -254, 1, -1}
	scale, quantized := quantizeKVSnapshotQ8(values)
	if scale <= 0 {
		t.Fatalf("quantizeKVSnapshotQ8 scale = %v, want > 0", scale)
	}
	if len(quantized) != len(values) {
		t.Fatalf("quantised len = %d, want %d", len(quantized), len(values))
	}

	// Drive the fused encode arm (appendKVEncodedTensor, Q8) end to end.
	out, err := appendKVEncodedTensor(nil, values, "", nil, EncodingQ8)
	if err != nil {
		t.Fatalf("appendKVEncodedTensor(Q8) error = %v", err)
	}
	if len(out) == 0 {
		t.Fatal("appendKVEncodedTensor(Q8) returned no bytes")
	}
}

// TestSnapshotEncodeCover_AppendEncodedTensorErrors drives the two encode
// error arms reachable by calling appendKVEncodedTensor directly: a native
// raw tensor with an unsupported dtype, and a raw-only tensor under a
// non-native encoding (errRawTensorNeedsNative).
func TestSnapshotEncodeCover_AppendEncodedTensorErrors(t *testing.T) {
	// Native + raw with an unrecognised dtype → kvSnapshotNativeTensorInfo
	// surfaces errUnsupportedNativeTensor.
	if _, err := appendKVEncodedTensor(nil, nil, "nonsense", []byte{1, 2}, EncodingNative); err == nil {
		t.Fatal("appendKVEncodedTensor(native bad dtype) error = nil, want native tensor error")
	}
	// Raw-only tensor under Q8 (non-native) → errRawTensorNeedsNative.
	if _, err := appendKVEncodedTensor(nil, nil, "float16", []byte{1, 2}, EncodingQ8); !errors.Is(err, errRawTensorNeedsNative) {
		t.Fatalf("appendKVEncodedTensor(raw-only Q8) error = %v, want errRawTensorNeedsNative", err)
	}
}

// TestSnapshotEncodeCover_StreamEncodedTensorPaths drives the stream encoder's
// native-raw fast path and Q8 quantise path via writeWithOptions: one head
// carries a valid native raw tensor (Native encoding), another snapshot's head
// carries plain f32 values under Q8.
func TestSnapshotEncodeCover_StreamEncodedTensorPaths(t *testing.T) {
	// Native raw head → stream.encodedTensor takes the raw fast path.
	nativeRaw := testSnapshot()
	nativeRaw.Version = SnapshotVersion
	nativeRaw.Layers = []LayerSnapshot{{
		Layer:      0,
		CacheIndex: 0,
		Heads: []HeadSnapshot{{
			KeyBytes:   cvtRawF16(2, 2),
			KeyDType:   "float16",
			ValueBytes: cvtRawF16(2, 2),
			ValueDType: "float16",
		}},
	}}
	var buf bytes.Buffer
	if err := nativeRaw.writeWithOptions(&buf, SaveOptions{KVEncoding: EncodingNative}); err != nil {
		t.Fatalf("writeWithOptions(native raw) error = %v", err)
	}
	if buf.Len() == 0 {
		t.Fatal("writeWithOptions(native raw) wrote nothing")
	}

	// Q8 head → stream.encodedTensor takes the quantise path.
	q8 := testSnapshot()
	var q8buf bytes.Buffer
	if err := q8.writeWithOptions(&q8buf, SaveOptions{KVEncoding: EncodingQ8}); err != nil {
		t.Fatalf("writeWithOptions(Q8) error = %v", err)
	}
	if q8buf.Len() == 0 {
		t.Fatal("writeWithOptions(Q8) wrote nothing")
	}
}

// TestSnapshotEncodeCover_LegacyVersionHeads drives the pre-v3 head encode
// arms (the float32-list `else` branches) of encodedSizeWithOptions,
// bytesWithOptions and writeWithOptions, plus the tokenOffset==0 fallback.
func TestSnapshotEncodeCover_LegacyVersionHeads(t *testing.T) {
	// Version 1 keeps effectiveVersion below 3 (float32 encoding, no native /
	// compressed / max-size layer features), so heads serialise as plain
	// float32 lists. TokenOffset 0 forces the len(Tokens) fallback (v>=2 only,
	// so use version 2 for that sub-check).
	legacy := &Snapshot{
		Version:      1,
		Architecture: "gemma4_text",
		Tokens:       []int32{1, 2},
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       2,
		HeadDim:      2,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{1, 0, 0, 1},
				Value: []float32{0, 1, 1, 0},
			}},
		}},
	}
	if _, err := legacy.encodedSizeWithOptions(SaveOptions{}); err != nil {
		t.Fatalf("encodedSizeWithOptions(v1) error = %v", err)
	}
	data, err := legacy.bytesWithOptions(SaveOptions{})
	if err != nil {
		t.Fatalf("bytesWithOptions(v1) error = %v", err)
	}
	if len(data) == 0 {
		t.Fatal("bytesWithOptions(v1) returned no bytes")
	}
	var buf bytes.Buffer
	if err := legacy.writeWithOptions(&buf, SaveOptions{}); err != nil {
		t.Fatalf("writeWithOptions(v1) error = %v", err)
	}

	// Version 2 with TokenOffset 0 exercises the tokenOffset = len(Tokens)
	// fallback in bytesWithOptions/writeWithOptions.
	v2 := &Snapshot{
		Version:      2,
		Architecture: "gemma4_text",
		Tokens:       []int32{1, 2, 3},
		TokenOffset:  0,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       1,
		HeadDim:      2,
		Layers:       []LayerSnapshot{{Layer: 0, CacheIndex: 0}},
	}
	if _, err := v2.bytesWithOptions(SaveOptions{}); err != nil {
		t.Fatalf("bytesWithOptions(v2 tokenOffset 0) error = %v", err)
	}
}

// TestSnapshotEncodeCover_AppendEncodedF32s covers the appendKVEncodedF32s
// happy path (it forwards to appendKVEncodedTensor with no raw payload).
func TestSnapshotEncodeCover_AppendEncodedF32s(t *testing.T) {
	out := appendKVEncodedF32s(nil, []float32{1, 2, 3}, KVSnapshotEncodingFloat32)
	if len(out) == 0 {
		t.Fatal("appendKVEncodedF32s returned no bytes")
	}
	// Native encoding takes the stream-float32 fast path.
	if got := appendKVEncodedF32s(nil, []float32{4, 5}, EncodingNative); len(got) == 0 {
		t.Fatal("appendKVEncodedF32s(native) returned no bytes")
	}
}
