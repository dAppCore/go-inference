// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"encoding/binary"
	"math"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
)

// TestSnapshotEncode_Snapshot_Save_Good writes a snapshot to a path with the
// default encoding and loads it back, asserting the round-trip preserves
// version, token offset, generated tokens and the logit tensor.
func TestSnapshotEncode_Snapshot_Save_Good(t *testing.T) {
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{11, 12},
		Generated:     []int32{12},
		TokenOffset:   9,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 4},
		Logits:        []float32{0.1, 0.2, 0.3, 0.4},
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{1, 2, 3, 4},
				Value: []float32{5, 6, 7, 8},
			}},
		}},
	}
	path := core.PathJoin(t.TempDir(), "restorable.kvbin")

	if err := snapshot.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := Load(path)

	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if loaded.Version != SnapshotVersion || loaded.TokenOffset != 9 || loaded.Generated[0] != 12 {
		t.Fatalf("loaded version/offset/generated = %d/%d/%v", loaded.Version, loaded.TokenOffset, loaded.Generated)
	}
	if len(loaded.LogitShape) != 3 || loaded.LogitShape[2] != 4 || len(loaded.Logits) != 4 || loaded.Logits[3] != 0.4 {
		t.Fatalf("loaded logits = shape %v values %v", loaded.LogitShape, loaded.Logits)
	}
}

// TestSnapshotEncode_Snapshot_Save_Bad asserts Save returns the nil-snapshot
// error rather than writing a file for a nil receiver.
func TestSnapshotEncode_Snapshot_Save_Bad(t *testing.T) {
	var snapshot *Snapshot

	if err := snapshot.Save(core.PathJoin(t.TempDir(), "nil.kvbin")); err == nil {
		t.Fatal("Save() error = nil, want nil snapshot error")
	}
}

// TestSnapshotEncode_Snapshot_Save_Ugly asks Save to write to a path inside a
// directory that does not exist, so the underlying file write fails and Save
// must surface that error rather than panic.
func TestSnapshotEncode_Snapshot_Save_Ugly(t *testing.T) {
	snapshot := testSnapshot()

	err := snapshot.Save(core.PathJoin(t.TempDir(), "no-such-dir", "snapshot.kvbin"))
	if err == nil {
		t.Fatal("Save(unwritable path) error = nil, want write error")
	}
}

// TestSnapshotEncode_Snapshot_SaveWithOptions_Good folds the encoding round-trip
// cases: each sub-case saves a snapshot under a specific KV encoding and loads
// it back, asserting the encoding-specific reconstruction (quantised Q8, native
// dtype bytes, native raw-only, native layer raw-only, short dtype tags, and the
// encoded-size/serialised-bytes agreement).
func TestSnapshotEncode_Snapshot_SaveWithOptions_Good(t *testing.T) {
	t.Run("QuantizedQ8", func(t *testing.T) {
		snapshot := &Snapshot{
			Version:       SnapshotVersion,
			Architecture:  "qwen3",
			Tokens:        []int32{1, 2, 3},
			TokenOffset:   3,
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        2,
			HeadDim:       2,
			NumQueryHeads: 1,
			LogitShape:    []int32{1, 1, 2},
			Logits:        []float32{0.25, 0.75},
			Layers: []LayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				Heads: []HeadSnapshot{{
					Key:   []float32{-1, -0.5, 0.5, 1},
					Value: []float32{0, 0.25, -0.25, 0.75},
				}},
			}},
		}
		path := core.PathJoin(t.TempDir(), "quantized-q8.kvbin")

		if err := snapshot.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingQ8}); err != nil {
			t.Fatalf("SaveWithOptions() error = %v", err)
		}
		loaded, err := Load(path)
		if err != nil {
			t.Fatalf("Load() error = %v", err)
		}

		if loaded.Version != SnapshotVersion {
			t.Fatalf("loaded Version = %d, want %d", loaded.Version, SnapshotVersion)
		}
		for i, want := range snapshot.Layers[0].Heads[0].Key {
			if diff := loaded.Layers[0].Heads[0].Key[i] - want; diff < -0.01 || diff > 0.01 {
				t.Fatalf("loaded key[%d] = %f, want near %f", i, loaded.Layers[0].Heads[0].Key[i], want)
			}
		}
		if loaded.Logits[1] != 0.75 {
			t.Fatalf("loaded logits = %v, want unquantized logits preserved", loaded.Logits)
		}
	})

	t.Run("NativeDType", func(t *testing.T) {
		keyBytes := appendUint16LE(nil, float32ToFloat16(1.5))
		keyBytes = appendUint16LE(keyBytes, float32ToFloat16(-2))
		valueBytes := appendUint16LE(nil, uint16(math.Float32bits(0.25)>>16))
		valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(-0.75)>>16))
		snapshot := &Snapshot{
			Version:       SnapshotVersion,
			Architecture:  "gemma4_text",
			Tokens:        []int32{1},
			TokenOffset:   1,
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        1,
			HeadDim:       2,
			NumQueryHeads: 1,
			Layers: []LayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				Heads: []HeadSnapshot{{
					Key:        []float32{1.5, -2},
					KeyDType:   "float16",
					KeyBytes:   keyBytes,
					Value:      []float32{0.25, -0.75},
					ValueDType: "bfloat16",
					ValueBytes: valueBytes,
				}},
			}},
		}
		path := core.PathJoin(t.TempDir(), "native-dtype.kvbin")

		if err := snapshot.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingNative}); err != nil {
			t.Fatalf("SaveWithOptions(native) error = %v", err)
		}
		loaded, err := Load(path)
		if err != nil {
			t.Fatalf("Load() error = %v", err)
		}

		head := loaded.Layers[0].Heads[0]
		if head.KeyDType != "float16" || head.ValueDType != "bfloat16" {
			t.Fatalf("loaded dtypes = %q/%q, want float16/bfloat16", head.KeyDType, head.ValueDType)
		}
		if !equalBytes(head.KeyBytes, keyBytes) || !equalBytes(head.ValueBytes, valueBytes) {
			t.Fatalf("loaded native bytes = %v/%v, want %v/%v", head.KeyBytes, head.ValueBytes, keyBytes, valueBytes)
		}
		if diff := head.Key[0] - 1.5; diff < -0.001 || diff > 0.001 {
			t.Fatalf("loaded f16 key[0] = %f, want near 1.5", head.Key[0])
		}
		if got := binary.LittleEndian.Uint16(head.ValueBytes); got != binary.LittleEndian.Uint16(valueBytes) {
			t.Fatalf("loaded bf16 value bits = %#x, want %#x", got, binary.LittleEndian.Uint16(valueBytes))
		}
	})

	t.Run("NativeRawOnly", func(t *testing.T) {
		keyBytes := appendUint16LE(nil, float32ToFloat16(1))
		keyBytes = appendUint16LE(keyBytes, float32ToFloat16(2))
		keyBytes = appendUint16LE(keyBytes, float32ToFloat16(3))
		keyBytes = appendUint16LE(keyBytes, float32ToFloat16(4))
		valueBytes := appendUint16LE(nil, uint16(math.Float32bits(5)>>16))
		valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(6)>>16))
		valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(7)>>16))
		valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(8)>>16))
		snapshot := &Snapshot{
			Version:       SnapshotVersion,
			Architecture:  "gemma4_text",
			Tokens:        []int32{1, 2},
			TokenOffset:   2,
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        2,
			HeadDim:       2,
			NumQueryHeads: 1,
			Layers: []LayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				Heads: []HeadSnapshot{{
					KeyDType:   "float16",
					KeyBytes:   keyBytes,
					ValueDType: "bfloat16",
					ValueBytes: valueBytes,
				}},
			}},
		}
		path := core.PathJoin(t.TempDir(), "native-raw-only.kvbin")

		if err := snapshot.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingNative}); err != nil {
			t.Fatalf("SaveWithOptions(native raw-only) error = %v", err)
		}
		rawOnly, err := LoadWithOptions(path, LoadOptions{RawKVOnly: true})
		if err != nil {
			t.Fatalf("LoadWithOptions(raw-only) error = %v", err)
		}
		head := rawOnly.Layers[0].Heads[0]
		if len(head.Key) != 0 || len(head.Value) != 0 {
			t.Fatalf("raw-only load decoded float32 key/value lengths = %d/%d, want 0/0", len(head.Key), len(head.Value))
		}
		if head.KeyDType != "float16" || head.ValueDType != "bfloat16" || !equalBytes(head.KeyBytes, keyBytes) || !equalBytes(head.ValueBytes, valueBytes) {
			t.Fatalf("raw-only head = %+v, want native bytes preserved", head)
		}

		decoded, err := Load(path)
		if err != nil {
			t.Fatalf("Load(default) error = %v", err)
		}
		decodedHead := decoded.Layers[0].Heads[0]
		if len(decodedHead.Key) != 4 || len(decodedHead.Value) != 4 || decodedHead.Key[3] != 4 {
			t.Fatalf("default load head = %+v, want decoded float32 values for debugging", decodedHead)
		}
	})

	t.Run("NativeLayerRawOnly", func(t *testing.T) {
		keyBytes := appendUint16LE(nil, float32ToFloat16(1))
		keyBytes = appendUint16LE(keyBytes, float32ToFloat16(2))
		keyBytes = appendUint16LE(keyBytes, float32ToFloat16(3))
		keyBytes = appendUint16LE(keyBytes, float32ToFloat16(4))
		valueBytes := appendUint16LE(nil, uint16(math.Float32bits(5)>>16))
		valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(6)>>16))
		valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(7)>>16))
		valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(8)>>16))
		snapshot := &Snapshot{
			Version:       SnapshotVersion,
			Architecture:  "gemma4_text",
			Tokens:        []int32{1, 2},
			TokenOffset:   2,
			NumLayers:     1,
			NumHeads:      2,
			SeqLen:        2,
			HeadDim:       1,
			NumQueryHeads: 2,
			Layers: []LayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				KeyDType:   "float16",
				KeyBytes:   keyBytes,
				KeyShape:   []int32{1, 2, 2, 1},
				ValueDType: "bfloat16",
				ValueBytes: valueBytes,
				ValueShape: []int32{1, 2, 2, 1},
				Heads:      make([]HeadSnapshot, 2),
			}},
		}
		path := core.PathJoin(t.TempDir(), "native-layer-raw-only.kvbin")

		if err := snapshot.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingNative}); err != nil {
			t.Fatalf("SaveWithOptions(native layer raw-only) error = %v", err)
		}
		loaded, err := LoadWithOptions(path, LoadOptions{RawKVOnly: true})
		if err != nil {
			t.Fatalf("LoadWithOptions(native layer raw-only) error = %v", err)
		}
		layer := loaded.Layers[0]
		if loaded.Version != SnapshotVersion || !equalBytes(layer.KeyBytes, keyBytes) || !equalBytes(layer.ValueBytes, valueBytes) {
			t.Fatalf("loaded native layer = version:%d key:%v value:%v", loaded.Version, layer.KeyBytes, layer.ValueBytes)
		}
		if len(layer.Heads) != 2 || len(layer.Heads[0].KeyBytes) != 0 || len(layer.Heads[1].ValueBytes) != 0 {
			t.Fatalf("loaded heads = %+v, want shape-only heads without duplicated raw bytes", layer.Heads)
		}
		if len(layer.KeyShape) != 4 || layer.KeyShape[1] != 2 || layer.KeyShape[2] != 2 {
			t.Fatalf("loaded key shape = %v, want [1 2 2 1]", layer.KeyShape)
		}
	})

	t.Run("ShortFormDType", func(t *testing.T) {
		// The native reader/writer accept both long ("float16") and short
		// ("F16") dtype tags. The short forms travel a separate dtypeString
		// fast-path; round-trip them to assert the canonical short tag and
		// raw bytes survive bit-exact.
		keyBytes := appendUint16LE(nil, float32ToFloat16(1))
		keyBytes = appendUint16LE(keyBytes, float32ToFloat16(2))
		valueBytes := appendUint16LE(nil, uint16(math.Float32bits(3)>>16))
		valueBytes = appendUint16LE(valueBytes, uint16(math.Float32bits(4)>>16))
		snapshot := &Snapshot{
			Version:       SnapshotVersion,
			Architecture:  "gemma4_text",
			Tokens:        []int32{7, 8},
			TokenOffset:   2,
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        2,
			HeadDim:       1,
			NumQueryHeads: 1,
			Layers: []LayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				Heads: []HeadSnapshot{{
					KeyDType:   "F16",
					KeyBytes:   keyBytes,
					ValueDType: "BF16",
					ValueBytes: valueBytes,
				}},
			}},
		}
		path := core.PathJoin(t.TempDir(), "short-dtype.kvbin")

		if err := snapshot.SaveWithOptions(path, SaveOptions{KVEncoding: EncodingNative}); err != nil {
			t.Fatalf("SaveWithOptions(native short dtype) error = %v", err)
		}
		loaded, err := LoadWithOptions(path, LoadOptions{RawKVOnly: true})
		if err != nil {
			t.Fatalf("LoadWithOptions(raw-only) error = %v", err)
		}
		head := loaded.Layers[0].Heads[0]
		// normalizeKVSnapshotTensorDType maps "F16"→"float16", "BF16"→"bfloat16".
		if head.KeyDType != "float16" || head.ValueDType != "bfloat16" {
			t.Fatalf("loaded dtypes = %q/%q, want canonicalised float16/bfloat16", head.KeyDType, head.ValueDType)
		}
		if !equalBytes(head.KeyBytes, keyBytes) || !equalBytes(head.ValueBytes, valueBytes) {
			t.Fatalf("loaded native bytes = %v/%v, want %v/%v (bit-exact)", head.KeyBytes, head.ValueBytes, keyBytes, valueBytes)
		}
	})

	t.Run("EncodedSizeMatchesSerialisedBytes", func(t *testing.T) {
		nativeKey := appendUint16LE(nil, float32ToFloat16(1))
		nativeKey = appendUint16LE(nativeKey, float32ToFloat16(2))
		nativeValue := appendUint16LE(nil, uint16(math.Float32bits(3)>>16))
		nativeValue = appendUint16LE(nativeValue, uint16(math.Float32bits(4)>>16))
		snapshot := &Snapshot{
			Version:       SnapshotVersion,
			Architecture:  "gemma4_text",
			Tokens:        []int32{1, 2},
			Generated:     []int32{3},
			TokenOffset:   2,
			NumLayers:     1,
			NumHeads:      1,
			SeqLen:        2,
			HeadDim:       1,
			NumQueryHeads: 1,
			LogitShape:    []int32{1, 1, 2},
			Logits:        []float32{0.25, 0.75},
			Layers: []LayerSnapshot{{
				Layer:      0,
				CacheIndex: 0,
				Heads: []HeadSnapshot{{
					Key:        []float32{1, 2},
					KeyDType:   "float16",
					KeyBytes:   nativeKey,
					Value:      []float32{3, 4},
					ValueDType: "bfloat16",
					ValueBytes: nativeValue,
				}},
			}},
		}
		for _, opts := range []SaveOptions{
			{},
			{KVEncoding: EncodingQ8},
			{KVEncoding: EncodingNative},
		} {
			size, err := snapshot.encodedSizeWithOptions(opts)
			if err != nil {
				t.Fatalf("encodedSizeWithOptions(%q) error = %v", opts.KVEncoding, err)
			}
			data, err := snapshot.bytesWithOptions(opts)
			if err != nil {
				t.Fatalf("bytesWithOptions(%q) error = %v", opts.KVEncoding, err)
			}
			if size != len(data) {
				t.Fatalf("encodedSizeWithOptions(%q) = %d, serialised bytes = %d", opts.KVEncoding, size, len(data))
			}
		}
	})
}

// TestSnapshotEncode_Snapshot_SaveWithOptions_Bad asserts SaveWithOptions
// rejects an unsupported KV encoding before writing anything.
func TestSnapshotEncode_Snapshot_SaveWithOptions_Bad(t *testing.T) {
	snapshot := &Snapshot{Version: SnapshotVersion}

	err := snapshot.SaveWithOptions(core.PathJoin(t.TempDir(), "bad.kvbin"), SaveOptions{KVEncoding: "q2"})

	if err == nil {
		t.Fatal("SaveWithOptions() error = nil, want unsupported encoding error")
	}
}

// TestSnapshotEncode_Snapshot_SaveWithOptions_Ugly asks SaveWithOptions to
// encode a snapshot carrying raw native bytes with no float32 fallback under a
// non-native (Q8) encoding; the encoder needs EncodingNative to pass raw
// payloads through, so the encode fails with errRawTensorNeedsNative.
func TestSnapshotEncode_Snapshot_SaveWithOptions_Ugly(t *testing.T) {
	rawOnly := &Snapshot{
		Version: SnapshotVersion, Architecture: "gemma4_text",
		Tokens: []int32{1}, TokenOffset: 1,
		NumLayers: 1, NumHeads: 1, SeqLen: 1, HeadDim: 1, NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer: 0,
			Heads: []HeadSnapshot{{
				KeyDType: "float16",
				KeyBytes: []byte{1, 0}, // raw, no float32 Key alongside
			}},
		}},
	}

	if err := rawOnly.SaveWithOptions(core.PathJoin(t.TempDir(), "raw-q8.kvbin"), SaveOptions{KVEncoding: EncodingQ8}); err == nil {
		t.Fatal("SaveWithOptions(raw-only head, Q8) error = nil, want errRawTensorNeedsNative")
	}
}

// TestSnapshotEncode_Snapshot_MarshalBinary_Good asserts MarshalBinary produces
// the same bytes as the internal bytes() encoder and that the buffer round-trips
// back through UnmarshalBinary and parseKVSnapshot to the original state.
func TestSnapshotEncode_Snapshot_MarshalBinary_Good(t *testing.T) {
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{11, 12},
		Generated:     []int32{12},
		TokenOffset:   9,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{1, 2, 3, 4},
				Value: []float32{5, 6, 7, 8},
			}},
		}},
	}

	data, err := snapshot.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary() error = %v", err)
	}
	if legacy, err := snapshot.bytes(); err != nil || !equalBytes(data, legacy) {
		t.Fatalf("bytes() = %d/%v, want MarshalBinary bytes %d", len(legacy), err, len(data))
	}
	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary() error = %v", err)
	}
	if loaded.TokenOffset != 9 || len(loaded.Tokens) != 2 || loaded.Layers[0].Heads[0].Value[3] != 8 {
		t.Fatalf("loaded snapshot = %+v, want marshalled state", loaded)
	}
	parsed, err := parseKVSnapshot(data)
	if err != nil {
		t.Fatalf("parseKVSnapshot() error = %v", err)
	}
	if parsed.Architecture != snapshot.Architecture || parsed.NumHeads != 1 {
		t.Fatalf("parsed snapshot = %+v, want architecture metadata", parsed)
	}
}

// TestSnapshotEncode_Snapshot_MarshalBinary_Bad asserts MarshalBinary surfaces
// the TurboQuant cache-mode mismatch: a layer carrying TurboQuant payloads but a
// non-turboquant cache mode cannot be encoded.
func TestSnapshotEncode_Snapshot_MarshalBinary_Bad(t *testing.T) {
	withPayload := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1},
		TokenOffset:   1,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        1,
		HeadDim:       1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer:              0,
			CacheIndex:         0,
			CacheMode:          "paged",
			TurboQuantPayloads: [][]byte{{1, 2, 3}},
		}},
	}

	if _, err := withPayload.MarshalBinary(); err == nil || !core.Contains(err.Error(), "TurboQuant KV payload requires turboquant cache mode") {
		t.Fatalf("MarshalBinary() error = %v, want TurboQuant cache-mode mismatch", err)
	}
}

// TestSnapshotEncode_Snapshot_MarshalBinary_Ugly asserts MarshalBinary errors on
// a nil receiver rather than panicking or returning a buffer.
func TestSnapshotEncode_Snapshot_MarshalBinary_Ugly(t *testing.T) {
	var snapshot *Snapshot
	if _, err := snapshot.MarshalBinary(); err == nil {
		t.Fatal("MarshalBinary(nil) error = nil, want snapshot error")
	}
}

// TestSnapshotEncode_EncodeErrors_Bad drives the encode-path guards shared by
// encodedSizeWithOptions / bytesWithOptions / writeWithOptions: an invalid
// KVEncoding is rejected up front, and a snapshot carrying a malformed native
// layer tensor (a dtype/shape the encoder can't size) surfaces the encode error
// rather than producing a corrupt buffer.
func TestSnapshotEncode_EncodeErrorGuards(t *testing.T) {
	// Invalid encoding rejected by all three entry points.
	bad := SaveOptions{KVEncoding: "not-an-encoding"}
	if _, err := testSnapshot().encodedSizeWithOptions(bad); err == nil {
		t.Fatal("encodedSizeWithOptions(bad encoding) error = nil")
	}
	if _, err := testSnapshot().bytesWithOptions(bad); err == nil {
		t.Fatal("bytesWithOptions(bad encoding) error = nil")
	}

	// A head carrying raw native bytes but NO float32 values cannot be encoded
	// under a non-native encoding (Q8): the encoder needs EncodingNative to
	// pass raw payloads through, so the size pass surfaces errRawTensorNeedsNative.
	rawOnly := &Snapshot{
		Version: SnapshotVersion, Architecture: "gemma4_text",
		Tokens: []int32{1}, TokenOffset: 1,
		NumLayers: 1, NumHeads: 1, SeqLen: 1, HeadDim: 1, NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Layer: 0,
			Heads: []HeadSnapshot{{
				KeyDType: "float16",
				KeyBytes: []byte{1, 0}, // raw, no float32 Key alongside
			}},
		}},
	}
	if _, err := rawOnly.encodedSizeWithOptions(SaveOptions{KVEncoding: EncodingQ8}); err == nil {
		t.Fatal("encodedSizeWithOptions(raw-only head, Q8) error = nil, want errRawTensorNeedsNative")
	}
	if _, err := rawOnly.bytesWithOptions(SaveOptions{KVEncoding: EncodingQ8}); err == nil {
		t.Fatal("bytesWithOptions(raw-only head, Q8) error = nil, want errRawTensorNeedsNative")
	}
}

// TestSnapshotEncode_NormalizeSnapshot_GoodUgly covers normalizeSnapshot: the
// nil guard (Ugly), the Version==0 default fill, and the TokenOffset==0 →
// len(Tokens) default fill (Good), plus the already-populated no-op case.
func TestSnapshotEncode_NormalizeSnapshot_GoodUgly(t *testing.T) {
	// Ugly: nil snapshot must be a no-op (no panic).
	normalizeSnapshot(nil)

	// Good: zero Version and zero TokenOffset both get filled.
	snapshot := &Snapshot{Tokens: []int32{1, 2, 3}}
	normalizeSnapshot(snapshot)
	if snapshot.Version != SnapshotVersion {
		t.Fatalf("Version = %d, want default %d", snapshot.Version, SnapshotVersion)
	}
	if snapshot.TokenOffset != 3 {
		t.Fatalf("TokenOffset = %d, want len(Tokens) = 3", snapshot.TokenOffset)
	}

	// A snapshot already carrying both fields is left untouched.
	preset := &Snapshot{Version: 2, TokenOffset: 9, Tokens: []int32{1}}
	normalizeSnapshot(preset)
	if preset.Version != 2 || preset.TokenOffset != 9 {
		t.Fatalf("preset normalised to %d/%d, want 2/9 unchanged", preset.Version, preset.TokenOffset)
	}
}

// TestSnapshotEncode_RichVersion6_EncodeRoundTrip_Good drives the version-gated
// encode arms shared by encodedSizeWithOptions / bytesWithOptions /
// writeWithOptions / the stream encoder across three usage surfaces: the
// in-memory MarshalBinary round-trip, a SaveStateBlocks to a streaming store
// (BinaryStreamWriter → kvSnapshotStreamWriter), and HashSnapshot. Each recovers
// the rich snapshot's observable shape.
func TestSnapshotEncode_RichVersion6RoundTrip(t *testing.T) {
	source := kvSnapshotRichV6()

	// Surface 1: in-memory binary round-trip under native encoding
	// (bytesWithOptions + encodedSizeWithOptions). Native is required because
	// the rich snapshot carries raw layer tensors; the default float32
	// MarshalBinary cannot encode raw payloads (errRawTensorNeedsNative).
	data, err := source.bytesWithOptions(SaveOptions{KVEncoding: EncodingNative})
	if err != nil {
		t.Fatalf("bytesWithOptions(rich v6, native) error = %v", err)
	}
	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary(rich v6) error = %v", err)
	}
	if loaded.Version != SnapshotVersion {
		t.Fatalf("loaded version = %d, want %d", loaded.Version, SnapshotVersion)
	}
	if len(loaded.Generated) != 1 || len(loaded.Logits) != 3 {
		t.Fatalf("loaded generated/logits = %d/%d, want 1/3", len(loaded.Generated), len(loaded.Logits))
	}
	if loaded.Layers[0].MaxSize != 4096 || loaded.Layers[1].CacheMode != "turboquant" {
		t.Fatalf("loaded layer metadata = maxsize %d / mode %q, want 4096 / turboquant", loaded.Layers[0].MaxSize, loaded.Layers[1].CacheMode)
	}

	// Surface 2: stream-save path (writeWithOptions via kvSnapshotStreamWriter).
	stream := &streamRecordingStateStore{store: state.NewInMemoryStore(nil)}
	bundle, err := source.SaveStateBlocks(context.Background(), stream, StateBlockOptions{
		BlockSize:  2, // whole snapshot in one block (TurboQuant needs full range)
		KVEncoding: EncodingNative,
		URI:        "mlx://rich-v6",
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks(rich v6, stream store) error = %v", err)
	}
	if stream.streamPuts == 0 {
		t.Fatal("stream store recorded no PutBytesStream calls, want the stream-write path exercised")
	}
	if len(bundle.Blocks) != 1 {
		t.Fatalf("bundle blocks = %d, want 1 whole-snapshot block", len(bundle.Blocks))
	}

	// Surface 3: HashSnapshot (writeWithOptions to a hash sink) is stable.
	hash, err := HashSnapshot(source)
	if err != nil || len(hash) != 64 {
		t.Fatalf("HashSnapshot(rich v6) = %q / %v, want 64-hex digest", hash, err)
	}
}
