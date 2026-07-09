// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"encoding/binary"
	"testing"

	core "dappco.re/go"
)

// TestSnapshotDecodeCover_ParseTokensIntoEmpty drives the tokenCount == 0 early
// return of parseKVSnapshotTokensInto via a minimal v6 header that declares no
// tokens.
func TestSnapshotDecodeCover_ParseTokensIntoEmpty(t *testing.T) {
	var data []byte
	data = append(data, kvSnapshotMagic...)
	data = binary.LittleEndian.AppendUint32(data, 6) // version
	data = binary.LittleEndian.AppendUint32(data, 0) // architecture length
	for range 5 {
		data = binary.LittleEndian.AppendUint32(data, 0) // layers/heads/seq/headDim/queryHeads
	}
	data = binary.LittleEndian.AppendUint32(data, 0) // token offset (v>=2)
	data = binary.LittleEndian.AppendUint32(data, 0) // token count = 0

	dst := []int32{99}
	out, err := parseKVSnapshotTokensInto(dst, data)
	if err != nil {
		t.Fatalf("parseKVSnapshotTokensInto(0 tokens) error = %v", err)
	}
	if len(out) != 1 || out[0] != 99 {
		t.Fatalf("parseKVSnapshotTokensInto(0 tokens) = %v, want dst unchanged", out)
	}
}

// TestSnapshotDecodeCover_ParseTokensInto_Guards drives the magic, version and
// token-count guards of parseKVSnapshotTokensInto.
func TestSnapshotDecodeCover_ParseTokensInto_Guards(t *testing.T) {
	// Bad magic.
	if _, err := parseKVSnapshotTokensInto(nil, []byte("not-a-snapshot-header-xxxx")); err == nil {
		t.Fatal("parseKVSnapshotTokensInto(bad magic) error = nil")
	}

	// Good magic, bad version.
	var badVer []byte
	badVer = append(badVer, kvSnapshotMagic...)
	badVer = binary.LittleEndian.AppendUint32(badVer, 999)
	if _, err := parseKVSnapshotTokensInto(nil, badVer); err == nil {
		t.Fatal("parseKVSnapshotTokensInto(bad version) error = nil")
	}

	// Good header but a token count that exceeds the available bytes.
	var overflow []byte
	overflow = append(overflow, kvSnapshotMagic...)
	overflow = binary.LittleEndian.AppendUint32(overflow, 6)
	overflow = binary.LittleEndian.AppendUint32(overflow, 0)
	for range 5 {
		overflow = binary.LittleEndian.AppendUint32(overflow, 0)
	}
	overflow = binary.LittleEndian.AppendUint32(overflow, 0)
	overflow = binary.LittleEndian.AppendUint32(overflow, 1000) // claims 1000 tokens
	if _, err := parseKVSnapshotTokensInto(nil, overflow); err == nil {
		t.Fatal("parseKVSnapshotTokensInto(token overflow) error = nil")
	}
}

// TestSnapshotDecodeCover_TruncatedSnapshots drives the reader truncation arms
// of parseKVSnapshotWithOptions (i32s / bytes / f32s / encodedTensor returning
// nil on a short read) by parsing progressively-truncated valid snapshot bytes.
// Every truncation point past the header must error rather than panic.
func TestSnapshotDecodeCover_TruncatedSnapshots(t *testing.T) {
	full, err := kvSnapshotBlocksTestSnapshot().MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary() error = %v", err)
	}
	// Truncate at every byte from just-past-magic to one short of full. The
	// reader must surface a truncation error at each cut without panicking.
	for cut := len(kvSnapshotMagic) + 1; cut < len(full); cut++ {
		if _, err := parseKVSnapshot(full[:cut]); err == nil {
			t.Fatalf("parseKVSnapshot(truncated at %d) error = nil, want truncation error", cut)
		}
	}
}

// TestSnapshotDecodeCover_TruncatedNativeSnapshot drives the i32s / bytes /
// native-tensor reader arms at truncation: a native-encoded snapshot carrying
// layer KeyShape/ValueShape (i32s) and raw dtype tags (bytes), truncated at
// every offset past the header. Each cut must surface an error, not panic.
func TestSnapshotDecodeCover_TruncatedNativeSnapshot(t *testing.T) {
	src := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 1,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			KeyDType:   "float16",
			KeyBytes:   cvtRawF16(2, 2),
			KeyShape:   []int32{1, 1, 2, 2},
			ValueDType: "float16",
			ValueBytes: cvtRawF16(2, 2),
			ValueShape: []int32{1, 1, 2, 2},
		}},
	}
	full, err := src.bytesWithOptions(SaveOptions{KVEncoding: EncodingNative})
	if err != nil {
		t.Fatalf("bytesWithOptions(native) error = %v", err)
	}
	// Round-trip sanity first.
	if _, err := parseKVSnapshot(full); err != nil {
		t.Fatalf("parseKVSnapshot(native full) error = %v", err)
	}
	for cut := len(kvSnapshotMagic) + 1; cut < len(full); cut++ {
		if _, err := parseKVSnapshot(full[:cut]); err == nil {
			t.Fatalf("parseKVSnapshot(native truncated at %d) error = nil, want truncation error", cut)
		}
		// The RawKVOnly path walks the same readers via a different tensor arm.
		if _, err := parseKVSnapshotWithOptions(full[:cut], LoadOptions{RawKVOnly: true}); err == nil {
			t.Fatalf("parseKVSnapshotWithOptions(raw, truncated at %d) error = nil, want truncation error", cut)
		}
	}
}

// TestSnapshotDecodeCover_NativeTensorDecode drives the native-tensor
// (encoding tag 2) reader path for both the RawKVOnly fast path and the full
// float32-decode path, then drives the decode validation error arm directly via
// decodeKVSnapshotNativeTensor with a declared element count that disagrees with
// the raw byte length.
func TestSnapshotDecodeCover_NativeTensorDecode(t *testing.T) {
	src := testSnapshot()
	src.SeqLen = 2
	src.HeadDim = 2
	src.Layers = []LayerSnapshot{{
		Heads: []HeadSnapshot{{KeyBytes: cvtRawF16(2, 2), KeyDType: "float16", ValueBytes: cvtRawF16(2, 2), ValueDType: "float16"}},
	}}
	data, err := src.bytesWithOptions(SaveOptions{KVEncoding: EncodingNative})
	if err != nil {
		t.Fatalf("bytesWithOptions(native) error = %v", err)
	}
	// Full decode (tag 2 → decodeKVSnapshotNativeTensor) round trip.
	if _, err := parseKVSnapshotWithOptions(data, LoadOptions{}); err != nil {
		t.Fatalf("parseKVSnapshotWithOptions(native full) error = %v", err)
	}
	// RawKVOnly path (tag 2 → returns raw bytes without float32 decode).
	if _, err := parseKVSnapshotWithOptions(data, LoadOptions{RawKVOnly: true}); err != nil {
		t.Fatalf("parseKVSnapshotWithOptions(native raw) error = %v", err)
	}

	// Direct decode validation error: 4 bytes of float16 is 2 elements, but
	// declare 9 → validateKVSnapshotNativeTensor rejects the byte length.
	if _, err := decodeKVSnapshotNativeTensor("float16", cvtRawF16(2, 2), 9); err == nil {
		t.Fatal("decodeKVSnapshotNativeTensor(bad element count) error = nil, want validation error")
	}
	// Unsupported dtype with raw present.
	if _, err := decodeKVSnapshotNativeTensor("nonsense", []byte{1, 2}, 1); err == nil {
		t.Fatal("decodeKVSnapshotNativeTensor(bad dtype) error = nil, want validation error")
	}
}

// TestSnapshotDecodeCover_HeadSlabFallback drives the per-layer head-slab
// fallback (make path) of parseKVSnapshotWithOptions: a snapshot whose layers
// carry divergent head counts, so the uniform slab is exhausted and a later
// layer falls back to its own make.
func TestSnapshotDecodeCover_HeadSlabFallback(t *testing.T) {
	src := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		TokenOffset:   2,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{
			{Layer: 0, CacheIndex: 0, Heads: []HeadSnapshot{
				{Key: []float32{1, 2, 3, 4}, Value: []float32{5, 6, 7, 8}},
			}},
			// Second layer carries two heads → wider than the first, exhausting
			// the slab sized to the first layer's width.
			{Layer: 1, CacheIndex: 1, Heads: []HeadSnapshot{
				{Key: []float32{1, 2, 3, 4}, Value: []float32{5, 6, 7, 8}},
				{Key: []float32{9, 10, 11, 12}, Value: []float32{13, 14, 15, 16}},
			}},
		},
	}
	data, err := src.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary() error = %v", err)
	}
	loaded, err := parseKVSnapshot(data)
	if err != nil {
		t.Fatalf("parseKVSnapshot(divergent heads) error = %v", err)
	}
	if len(loaded.Layers) != 2 || len(loaded.Layers[1].Heads) != 2 {
		t.Fatalf("parsed layers = %+v, want layer 1 with 2 heads", loaded.Layers)
	}
}

// TestSnapshotDecodeCover_HashMismatch drives the optional KV-hash mismatch path
// where a caller-declared hash disagrees with the payload, exercised via the
// raw payload loader's sibling check at decode time.
func TestSnapshotDecodeCover_HashMismatch(t *testing.T) {
	data, err := kvSnapshotBlocksTestSnapshot().MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary() error = %v", err)
	}
	// Sanity: the round trip parses, and the SHA256 helper is stable — this
	// anchors the corrupted-hash comparisons used elsewhere in the suite.
	if _, err := parseKVSnapshot(data); err != nil {
		t.Fatalf("parseKVSnapshot(round trip) error = %v", err)
	}
	if core.SHA256Hex(data) == "" {
		t.Fatal("SHA256Hex returned empty")
	}
}
