// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"testing"

	core "dappco.re/go"
)

// TestSnapshot_Clone_Good asserts Clone returns a deep copy: mutating the clone's
// token, generated, logit and head slices must not touch the source snapshot.
func TestSnapshot_Clone_Good(t *testing.T) {
	snapshot := &Snapshot{
		Version:      SnapshotVersion,
		Tokens:       []int32{1, 2},
		Generated:    []int32{2},
		TokenOffset:  4,
		Architecture: "gemma4_text",
		LogitShape:   []int32{1, 1, 3},
		Logits:       []float32{0.1, 0.2, 0.7},
		Layers: []LayerSnapshot{{
			Layer: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{1, 2},
				Value: []float32{3, 4},
			}},
		}},
	}

	cloned := snapshot.Clone()
	cloned.Tokens[0] = 99
	cloned.Generated[0] = 88
	cloned.Logits[0] = 0.9
	cloned.LogitShape[0] = 9
	cloned.Layers[0].Heads[0].Key[0] = 88

	if snapshot.Tokens[0] != 1 || snapshot.Generated[0] != 2 || snapshot.Logits[0] != 0.1 || snapshot.LogitShape[0] != 1 || snapshot.Layers[0].Heads[0].Key[0] != 1 {
		t.Fatal("Clone() returned aliased snapshot data")
	}
}

func kvSnapshotTurboQuantNoPayloadBytes() []byte {
	var data []byte
	data = append(data, kvSnapshotMagic...)
	data = appendKVU32(data, SnapshotVersion)
	data = appendKVBytes(data, core.AsBytes("gemma4_text"))
	data = appendKVU32(data, 1) // layers
	data = appendKVU32(data, 0) // heads
	data = appendKVU32(data, 0) // seq len
	data = appendKVU32(data, 0) // head dim
	data = appendKVU32(data, 0) // query heads
	data = appendKVU32(data, 0) // token offset
	data = appendKVU32(data, 0) // tokens
	data = appendKVU32(data, 0) // generated
	data = appendKVU32(data, 1) // layer count
	data = appendKVI32(data, 0)
	data = appendKVI32(data, 0)
	data = appendKVU32(data, 0) // head count
	data = appendKVBytes(data, core.AsBytes("turboquant"))
	data = appendKVU32(data, 0) // TurboQuant payload count
	data = appendKVU32(data, 0) // max size (v6)
	data = appendKVI32s(data, nil)
	data = appendKVU32(data, 0) // key tensor encoding
	data = appendKVU32(data, 0) // key tensor values
	data = appendKVI32s(data, nil)
	data = appendKVU32(data, 0) // value tensor encoding
	data = appendKVU32(data, 0) // value tensor values
	data = appendKVU32(data, 0) // logit shape
	data = appendKVF32s(data, nil)
	return data
}

// TestSnapshot_DropFloat32_Good asserts DropFloat32 clears the float32 Key/Value
// slices on a head that also carries raw KeyBytes/ValueBytes, leaving the raw
// bytes intact.
func TestSnapshot_DropFloat32_Good(t *testing.T) {
	snapshot := &Snapshot{Layers: []LayerSnapshot{{
		Heads: []HeadSnapshot{{
			Key:        []float32{1},
			KeyBytes:   []byte{1, 2},
			Value:      []float32{2},
			ValueBytes: []byte{3, 4},
		}},
	}}}

	DropFloat32(snapshot)

	head := snapshot.Layers[0].Heads[0]
	if len(head.Key) != 0 || len(head.Value) != 0 || len(head.KeyBytes) != 2 || len(head.ValueBytes) != 2 {
		t.Fatalf("DropFloat32() head = %+v, want raw bytes retained and float32 dropped", head)
	}
}

// TestSnapshot_DropFloat32_Bad asserts DropFloat32 does NOT drop the float32
// slices when there are no raw bytes to fall back to — dropping them would lose
// the only copy of the tensor, so the guard (len(KeyBytes)>0) must keep them.
func TestSnapshot_DropFloat32_Bad(t *testing.T) {
	snapshot := &Snapshot{Layers: []LayerSnapshot{{
		Heads: []HeadSnapshot{{
			Key:   []float32{1, 2},
			Value: []float32{3, 4},
		}},
	}}}

	DropFloat32(snapshot)

	head := snapshot.Layers[0].Heads[0]
	if len(head.Key) != 2 || len(head.Value) != 2 {
		t.Fatalf("DropFloat32(no raw bytes) head = %+v, want float32 retained (nothing to fall back to)", head)
	}
}

// TestSnapshot_DropFloat32_Ugly asserts DropFloat32 is a safe no-op on
// degenerate inputs: a nil snapshot and a snapshot whose layers carry no heads
// must both pass through without panicking and leave the (absent) data alone.
func TestSnapshot_DropFloat32_Ugly(t *testing.T) {
	DropFloat32(nil)

	empty := &Snapshot{Layers: []LayerSnapshot{{Layer: 0}}}
	DropFloat32(empty)
	if len(empty.Layers) != 1 || empty.Layers[0].Heads != nil {
		t.Fatalf("DropFloat32(no heads) = %+v, want untouched empty layer", empty.Layers)
	}
}

// TestSnapshot_Head_Good asserts Head returns a defensive copy of an existing
// layer/head: the returned tensors carry the stored values and mutating the
// copy does not touch the source snapshot.
func TestSnapshot_Head_Good(t *testing.T) {
	snapshot := &Snapshot{
		Layers: []LayerSnapshot{{
			Layer: 0,
			Heads: []HeadSnapshot{{
				Key:   []float32{1, 2},
				Value: []float32{3, 4},
			}},
		}},
	}

	head, ok := snapshot.Head(0, 0)
	if !ok {
		t.Fatal("Head(0, 0) ok = false, want true")
	}
	if head.Key[0] != 1 || head.Value[1] != 4 {
		t.Fatalf("Head(0, 0) = %+v, want stored key/value", head)
	}
	head.Key[0] = 99
	if snapshot.Layers[0].Heads[0].Key[0] != 1 {
		t.Fatal("Head() returned an aliased key slice, want defensive copy")
	}
}

// TestSnapshot_Head_Bad asserts Head reports ok = false for a layer that is
// present in the slice but whose head index is out of range.
func TestSnapshot_Head_Bad(t *testing.T) {
	snapshot := &Snapshot{
		Layers: []LayerSnapshot{{
			Layer: 0,
			Heads: []HeadSnapshot{{Key: []float32{1}, Value: []float32{2}}},
		}},
	}

	if _, ok := snapshot.Head(0, 5); ok {
		t.Fatal("Head(0, out-of-range head) ok = true, want false")
	}
	if _, ok := snapshot.Head(3, 0); ok {
		t.Fatal("Head(missing layer) ok = true, want false")
	}
}

// TestSnapshot_Head_Ugly drives the guard branches Head's happy path never
// reaches: a sparse layer whose index does not match its slot, a nil receiver,
// and negative layer/head indices.
func TestSnapshot_Head_Ugly(t *testing.T) {
	snapshot := &Snapshot{
		Layers: []LayerSnapshot{{
			Layer: 7,
			Heads: []HeadSnapshot{{
				Key:   []float32{1},
				Value: []float32{2},
			}},
		}},
	}

	if _, ok := snapshot.Head(0, 0); ok {
		t.Fatal("Head(0, 0) ok = true for sparse layer 7")
	}
	if head, ok := snapshot.Head(7, 0); !ok || head.Key[0] != 1 || head.Value[0] != 2 {
		t.Fatalf("Head(7, 0) = %+v/%v, want sparse layer data", head, ok)
	}

	// Guard branches: nil receiver, negative indices, and a head index past
	// the layer's head slice must all report ok = false.
	var nilSnapshot *Snapshot
	if _, ok := nilSnapshot.Head(0, 0); ok {
		t.Fatal("Head(nil receiver) ok = true, want false")
	}
	if _, ok := snapshot.Head(-1, 0); ok {
		t.Fatal("Head(negative layer) ok = true, want false")
	}
	if _, ok := snapshot.Head(7, -1); ok {
		t.Fatal("Head(negative head) ok = true, want false")
	}
	if _, ok := snapshot.Head(7, 5); ok {
		t.Fatal("Head(out-of-range head) ok = true, want false")
	}
}

// TestSnapshot_ResultError_Good asserts ResultError passes an error value in a
// Result straight through unchanged.
func TestSnapshot_ResultError_Good(t *testing.T) {
	sentinel := core.NewError("boom")
	if got := ResultError(core.Result{Value: sentinel}); got != sentinel {
		t.Fatalf("ResultError(error) = %v, want passthrough of %v", got, sentinel)
	}
}

// TestSnapshot_ResultError_Bad asserts ResultError wraps a string value into an
// error carrying that text.
func TestSnapshot_ResultError_Bad(t *testing.T) {
	if got := ResultError(core.Result{Value: "text failure"}); got == nil || got.Error() != "text failure" {
		t.Fatalf("ResultError(string) = %v, want wrapped error", got)
	}
}

// TestSnapshot_ResultError_Ugly asserts ResultError falls back to the unknown
// filesystem sentinel when the Result value is neither an error nor a string.
func TestSnapshot_ResultError_Ugly(t *testing.T) {
	if got := ResultError(core.Result{Value: 42}); got == nil {
		t.Fatal("ResultError(unknown type) = nil, want fallback error")
	}
}

// TestKVSnapshot_EffectiveSeqLen_GoodBadUgly covers the three branches: a
// populated SeqLen (Good), a nil snapshot (Bad), and a zero SeqLen that falls
// back to the token count (Ugly).
func TestKVSnapshot_EffectiveSeqLen_GoodBadUgly(t *testing.T) {
	if got := EffectiveSeqLen(&Snapshot{SeqLen: 9}); got != 9 {
		t.Fatalf("EffectiveSeqLen(SeqLen=9) = %d, want 9", got)
	}
	if got := EffectiveSeqLen(nil); got != 0 {
		t.Fatalf("EffectiveSeqLen(nil) = %d, want 0", got)
	}
	if got := EffectiveSeqLen(&Snapshot{Tokens: []int32{1, 2, 3}}); got != 3 {
		t.Fatalf("EffectiveSeqLen(zero SeqLen) = %d, want token count 3", got)
	}
}

// TestSnapshot_HashSnapshot_Good asserts HashSnapshot is deterministic: hashing
// the same float32 snapshot twice yields the same non-empty digest.
func TestSnapshot_HashSnapshot_Good(t *testing.T) {
	snapshot := testSnapshot()
	hash, err := HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("HashSnapshot() error = %v", err)
	}
	again, err := HashSnapshot(snapshot)
	if err != nil || hash == "" || hash != again {
		t.Fatalf("HashSnapshot() = %q/%q, want stable non-empty hash", hash, again)
	}
}

// TestSnapshot_HashSnapshot_Bad asserts HashSnapshot errors on a nil snapshot.
func TestSnapshot_HashSnapshot_Bad(t *testing.T) {
	if _, err := HashSnapshot(nil); err == nil {
		t.Fatal("HashSnapshot(nil) error = nil, want snapshot error")
	}
}

// TestSnapshot_HashSnapshot_Ugly hashes a raw-native-only snapshot (float32
// Value dropped, only ValueBytes present) which drives HashSnapshot down the
// requiresNativeEncoding branch — it must still produce a non-empty digest.
func TestSnapshot_HashSnapshot_Ugly(t *testing.T) {
	native := testSnapshot()
	head := &native.Layers[0].Heads[0]
	for _, value := range head.Value {
		head.ValueBytes = appendUint16LE(head.ValueBytes, float32ToFloat16(value))
	}
	head.Value = nil
	head.ValueDType = "float16"
	nativeHash, err := HashSnapshot(native)
	if err != nil || nativeHash == "" {
		t.Fatalf("HashSnapshot(native) = %q, err = %v, want non-empty hash", nativeHash, err)
	}
}

// TestSnapshot_Clone_Bad asserts Clone returns nil for a nil receiver rather
// than dereferencing it.
func TestSnapshot_Clone_Bad(t *testing.T) {
	var snapshot *Snapshot

	if snapshot.Clone() != nil {
		t.Fatal("Clone() on nil snapshot returned non-nil")
	}
}

// TestSnapshot_Clone_Ugly asserts Clone preserves a sparse layer's metadata
// (a layer whose index does not match its slot, with no heads).
func TestSnapshot_Clone_Ugly(t *testing.T) {
	snapshot := &Snapshot{
		Layers: []LayerSnapshot{{Layer: 7}},
	}

	cloned := snapshot.Clone()

	if len(cloned.Layers) != 1 || cloned.Layers[0].Layer != 7 || cloned.Layers[0].Heads != nil {
		t.Fatalf("Clone() sparse layer = %+v, want preserved sparse metadata", cloned.Layers)
	}
}

// TestKVSnapshot_TokenOffsetDefault_Ugly loads a v1 buffer that omits the token
// offset field, so the parser's trailing `TokenOffset == 0 → len(Tokens)`
// fixup fires (snapshot.go:639). v1 has no per-tensor encoding header, so the
// head goes through the same f32s path as the v2 case.
func TestKVSnapshot_TokenOffsetDefaultV1Parse(t *testing.T) {
	var data []byte
	data = append(data, kvSnapshotMagic...)
	data = appendKVU32(data, 1) // version 1 (no TokenOffset/Generated/Logits fields)
	data = appendKVBytes(data, core.AsBytes("gemma4_text"))
	data = appendKVU32(data, 1) // NumLayers
	data = appendKVU32(data, 1) // NumHeads
	data = appendKVU32(data, 2) // SeqLen
	data = appendKVU32(data, 2) // HeadDim
	data = appendKVU32(data, 1) // NumQueryHeads
	data = appendKVI32s(data, []int32{3, 4})
	data = appendKVU32(data, 1) // layer count
	data = appendKVI32(data, 0) // Layer
	data = appendKVI32(data, 0) // CacheIndex
	data = appendKVU32(data, 1) // head count
	data = appendKVF32s(data, []float32{1, 2, 3, 4})
	data = appendKVF32s(data, []float32{4, 3, 2, 1})

	var loaded Snapshot
	if err := loaded.UnmarshalBinary(data); err != nil {
		t.Fatalf("UnmarshalBinary(v1) error = %v", err)
	}
	if loaded.TokenOffset != 2 {
		t.Fatalf("loaded v1 TokenOffset = %d, want default to token count 2", loaded.TokenOffset)
	}
}

// TestKVSnapshot_NativeTensorInfo_Bad covers the two early-return error arms of
// kvSnapshotNativeTensorInfo: an unknown dtype with raw bytes present
// (snapshot.go:862) and a raw length that is not a whole number of elements for
// the dtype (snapshot.go:865).
func TestKVSnapshot_NativeTensorInfoGuards(t *testing.T) {
	if _, _, _, ok, err := kvSnapshotNativeTensorInfo(nil, "int8", []byte{1, 2}); ok || err == nil {
		t.Fatalf("kvSnapshotNativeTensorInfo(unknown dtype) = ok %v/err %v, want false + error", ok, err)
	}
	// float16 = 2 bytes/value; 3 raw bytes is not a whole number of elements.
	if _, _, _, ok, err := kvSnapshotNativeTensorInfo(nil, "float16", []byte{1, 2, 3}); ok || err == nil {
		t.Fatalf("kvSnapshotNativeTensorInfo(odd length) = ok %v/err %v, want false + error", ok, err)
	}
}

// TestKVSnapshot_EncodedTensorSize_GoodBadUgly covers kvSnapshotEncodedTensorSize:
// a native tensor with an unknown dtype surfaces the info error (snapshot.go:843,
// Bad); empty values with raw bytes under a non-native encoding hits the
// raw-requires-native guard (snapshot.go:850, Ugly); a plain float32 tensor
// returns the 8+4N size (Good).
func TestKVSnapshot_EncodedTensorSize_GoodBadUgly(t *testing.T) {
	if _, err := kvSnapshotEncodedTensorSize(nil, "int8", []byte{1, 2}, EncodingNative); err == nil {
		t.Fatal("kvSnapshotEncodedTensorSize(native bad dtype) error = nil, want native-info error")
	}
	if _, err := kvSnapshotEncodedTensorSize(nil, "", []byte{1, 2, 3}, KVSnapshotEncodingFloat32); err == nil {
		t.Fatal("kvSnapshotEncodedTensorSize(raw without native) error = nil, want raw-needs-native error")
	}
	size, err := kvSnapshotEncodedTensorSize([]float32{1, 2}, "", nil, KVSnapshotEncodingFloat32)
	if err != nil || size != 8+2*4 {
		t.Fatalf("kvSnapshotEncodedTensorSize(float32) = %d/%v, want %d", size, err, 8+2*4)
	}
}

// TestKVSnapshot_NilPredicates_Bad exercises the nil-snapshot guards that the
// happy-path tests never reach: validateKVSnapshotLayerSchemes
// (snapshot.go:1482), requiresNativeEncoding (1498), and
// snapshotHasLayerNativeTensors (1518). cloneKVLayers(nil) covers the empty
// guard at 1367.
func TestKVSnapshot_NilPredicateGuards(t *testing.T) {
	if err := validateKVSnapshotLayerSchemes(nil); err == nil {
		t.Fatal("validateKVSnapshotLayerSchemes(nil) error = nil, want snapshot-nil error")
	}
	if requiresNativeEncoding(nil) {
		t.Fatal("requiresNativeEncoding(nil) = true, want false")
	}
	if snapshotHasLayerNativeTensors(nil) {
		t.Fatal("snapshotHasLayerNativeTensors(nil) = true, want false")
	}
	if cloneKVLayers(nil) != nil {
		t.Fatal("cloneKVLayers(nil) != nil, want nil")
	}
}

// TestKVSnapshot_LayerNativeTensors_Good drives the positive arms of
// snapshotHasLayerNativeTensors (layer.KeyBytes present, snapshot.go:1522) and
// requiresNativeEncoding (which short-circuits true through it, 1501), plus
// cloneKVLayers over a fully-populated layer (the per-layer clone body, 1376).
func TestKVSnapshot_LayerNativeTensorArms(t *testing.T) {
	snapshot := &Snapshot{
		Layers: []LayerSnapshot{{
			Layer:      3,
			CacheIndex: 1,
			KeyDType:   "float16",
			KeyBytes:   []byte{1, 2},
			KeyShape:   []int32{1, 1},
		}},
	}
	if !snapshotHasLayerNativeTensors(snapshot) {
		t.Fatal("snapshotHasLayerNativeTensors(layer bytes) = false, want true")
	}
	if !requiresNativeEncoding(snapshot) {
		t.Fatal("requiresNativeEncoding(layer bytes) = false, want true")
	}
	cloned := cloneKVLayers(snapshot.Layers)
	if len(cloned) != 1 || cloned[0].Layer != 3 || !equalBytes(cloned[0].KeyBytes, []byte{1, 2}) {
		t.Fatalf("cloneKVLayers(populated) = %+v, want deep copy with KeyBytes", cloned)
	}
	// requiresNativeEncoding's head-bytes arm (snapshot.go:1506/1509): a head
	// with ValueBytes but no float32 Value, no layer-level native bytes.
	headOnly := &Snapshot{Layers: []LayerSnapshot{{Heads: []HeadSnapshot{{
		ValueBytes: []byte{9, 9},
		ValueDType: "float16",
	}}}}}
	if !requiresNativeEncoding(headOnly) {
		t.Fatal("requiresNativeEncoding(head bytes) = false, want true")
	}
}

// TestKVSnapshot_FirstNonEmpty_GoodBadUgly covers firstNonEmpty: a real value
// is returned (Good), all-empty inputs fall through to "" (snapshot.go:1466,
// Bad), and a whitespace-only value is skipped in favour of a later real one
// via the core.Trim branch (Ugly).
func TestKVSnapshot_FirstNonEmpty_GoodBadUgly(t *testing.T) {
	if got := firstNonEmpty("", "real"); got != "real" {
		t.Fatalf("firstNonEmpty(empty, real) = %q, want \"real\"", got)
	}
	if got := firstNonEmpty("", ""); got != "" {
		t.Fatalf("firstNonEmpty(all empty) = %q, want empty string", got)
	}
	if got := firstNonEmpty("   ", "kept"); got != "kept" {
		t.Fatalf("firstNonEmpty(whitespace, kept) = %q, want \"kept\"", got)
	}
}

// TestKVSnapshot_HashSnapshotNativeError_Bad drives HashSnapshot's
// writeWithOptions error arm (snapshot.go:1546): a head carrying KeyBytes with
// an empty dtype forces requiresNativeEncoding true, so HashSnapshot selects
// native encoding, and the native encoder rejects the unknown dtype mid-write.
func TestKVSnapshot_HashSnapshotNativeEncodeError(t *testing.T) {
	snapshot := &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		NumLayers:     1,
		NumHeads:      1,
		NumQueryHeads: 1,
		Layers: []LayerSnapshot{{
			Heads: []HeadSnapshot{{
				KeyBytes: []byte{1, 2, 3}, // raw bytes, empty dtype → native encode fails
			}},
		}},
	}

	if _, err := HashSnapshot(snapshot); err == nil {
		t.Fatal("HashSnapshot(native bad dtype) error = nil, want native-encode error")
	}
}

func equalBytes(left, right []byte) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

// kvSnapshotRichV6 builds a version-6 snapshot exercising every version-gated
// encode arm: Generated tokens (v2), per-head float32 K/V (v3), a native layer
// raw tensor (v4), a TurboQuant compressed layer (v5), a MaxSize window clamp
// (v6), and LogitShape/Logits. SeqLen 2 so a single block holds it whole (the
// TurboQuant payload requires a full-range block).
func kvSnapshotRichV6() *Snapshot {
	keyBytes := appendUint16LE(nil, float32ToFloat16(1.5))
	keyBytes = appendUint16LE(keyBytes, float32ToFloat16(-2))
	valueBytes := appendUint16LE(nil, float32ToFloat16(0.25))
	valueBytes = appendUint16LE(valueBytes, float32ToFloat16(-0.75))
	return &Snapshot{
		Version:       SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{2},
		TokenOffset:   2,
		NumLayers:     2,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       1,
		NumQueryHeads: 1,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []LayerSnapshot{
			{
				// Native layer raw tensor (v4) + MaxSize clamp (v6).
				Layer:      0,
				CacheIndex: 0,
				MaxSize:    4096,
				KeyDType:   "float16",
				KeyBytes:   keyBytes,
				KeyShape:   []int32{1, 1, 2, 1},
				ValueDType: "float16",
				ValueBytes: valueBytes,
				ValueShape: []int32{1, 1, 2, 1},
				Heads:      make([]HeadSnapshot, 1),
			},
			{
				// TurboQuant compressed layer (v5) — requires the turboquant
				// cache mode and at least one payload.
				Layer:              1,
				CacheIndex:         1,
				MaxSize:            4096,
				CacheMode:          "turboquant",
				TurboQuantPayloads: [][]byte{{1, 2, 3, 4}},
				Heads:              make([]HeadSnapshot, 1),
			},
		},
	}
}
