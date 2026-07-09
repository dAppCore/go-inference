// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/model/state"
)

// TestStateStore_Snapshot_SaveState_Good writes a snapshot with SaveState and
// reads it back, asserting the envelope carries the KV kind/encoding and that
// the round-trip preserves architecture, token offset, layer count and head
// tensor shapes.
func TestStateStore_Snapshot_SaveState_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()

	ref, err := snapshot.SaveState(context.Background(), store, StateOptions{
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/test",
		Title:      "test session",
		Labels:     []string{"session-kv"},
	})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}
	if ref.ChunkID == 0 || ref.Codec != state.CodecMemory {
		t.Fatalf("State ref = %+v, want in-memory chunk ref", ref)
	}
	chunk, err := state.Resolve(context.Background(), store, ref.ChunkID)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	if !core.Contains(chunk.Text, `"kind":"`+KVSnapshotStateKind+`"`) || !core.Contains(chunk.Text, `"binary_encoding":"base64"`) {
		t.Fatalf("State payload = %s, want KV envelope", chunk.Text)
	}

	loaded, err := LoadFromState(context.Background(), store, ref)
	if err != nil {
		t.Fatalf("LoadFromState() error = %v", err)
	}
	if loaded.Architecture != snapshot.Architecture || loaded.TokenOffset != snapshot.TokenOffset || loaded.NumLayers != snapshot.NumLayers {
		t.Fatalf("loaded metadata = %+v, want %+v", loaded, snapshot)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head(0, 0) ok = false, want true")
	}
	if len(head.Key) != len(snapshot.Layers[0].Heads[0].Key) || len(head.Value) != len(snapshot.Layers[0].Heads[0].Value) {
		t.Fatalf("loaded head = %+v, want same tensor sizes", head)
	}
}

func TestKVSnapshotState_Bad_LoadRejectsHashMismatch(t *testing.T) {
	store := state.NewInMemoryStore(map[int]string{
		1: `{"version":1,"kind":"` + KVSnapshotStateKind + `","binary_encoding":"base64","kv_hash":"sha256:not-it","data":"` + core.Base64Encode([]byte(kvSnapshotMagic)) + `"}`,
	})

	_, err := LoadFromState(context.Background(), store, state.ChunkRef{ChunkID: 1})

	if err == nil {
		t.Fatal("LoadFromState() error = nil, want hash mismatch")
	}
}

// TestStateStore_Snapshot_SaveState_Bad drives every guard arm of SaveState:
// nil snapshot, nil store, an unsupported KV encoding, and a writer whose Put
// fails. Each must return a non-nil error rather than a chunk ref.
func TestStateStore_Snapshot_SaveState_Bad(t *testing.T) {
	var snapshot *Snapshot
	if _, err := snapshot.SaveState(context.Background(), state.NewInMemoryStore(nil), StateOptions{}); err == nil {
		t.Fatal("SaveState(nil snapshot) error = nil")
	}
	if _, err := testSnapshot().SaveState(context.Background(), nil, StateOptions{}); err == nil {
		t.Fatal("SaveState(nil store) error = nil")
	}
	if _, err := testSnapshot().SaveState(context.Background(), state.NewInMemoryStore(nil), StateOptions{KVEncoding: "q2"}); err == nil {
		t.Fatal("SaveState(bad encoding) error = nil")
	}
	if _, err := testSnapshot().SaveState(nil, failingStateWriter{}, StateOptions{}); err == nil {
		t.Fatal("SaveState(write failure) error = nil")
	}
}

// TestStateStore_LoadFromState_Bad drives LoadFromState's guard and decode
// failure arms: nil store, corrupt envelope JSON, and the five
// decodeKVSnapshotStateEnvelope rejection cases (bad version, wrong kind,
// non-base64 binary encoding, undecodable data, payload-length mismatch). A
// valid envelope is decoded last to prove the rejections are specific.
func TestStateStore_LoadFromState_Bad(t *testing.T) {
	if _, err := LoadFromState(context.Background(), nil, state.ChunkRef{ChunkID: 1}); err == nil {
		t.Fatal("LoadFromState(nil store) error = nil")
	}
	store := state.NewInMemoryStore(map[int]string{1: "{"})
	if _, err := LoadFromState(nil, store, state.ChunkRef{ChunkID: 1}); err == nil {
		t.Fatal("LoadFromState(corrupt JSON) error = nil")
	}

	for _, envelope := range []kvSnapshotStateEnvelope{
		{Version: KVSnapshotStateVersion + 1, Kind: KVSnapshotStateKind, BinaryEncoding: "base64"},
		{Version: KVSnapshotStateVersion, Kind: "wrong", BinaryEncoding: "base64"},
		{Version: KVSnapshotStateVersion, Kind: KVSnapshotStateKind, BinaryEncoding: "hex"},
		{Version: KVSnapshotStateVersion, Kind: KVSnapshotStateKind, BinaryEncoding: "base64", Data: "not base64"},
		{Version: KVSnapshotStateVersion, Kind: KVSnapshotStateKind, BinaryEncoding: "base64", Data: core.Base64Encode([]byte("x")), PayloadByteCount: 2},
	} {
		if _, err := decodeKVSnapshotStateEnvelope(envelope); err == nil {
			t.Fatalf("decodeKVSnapshotStateEnvelope(%+v) error = nil", envelope)
		}
	}
	if data, err := decodeKVSnapshotStateEnvelope(kvSnapshotStateEnvelope{
		Version:        KVSnapshotStateVersion,
		Kind:           KVSnapshotStateKind,
		BinaryEncoding: "base64",
		Data:           core.Base64Encode([]byte("x")),
	}); err != nil || string(data) != "x" {
		t.Fatalf("decodeKVSnapshotStateEnvelope(valid) = %q/%v, want x/nil", string(data), err)
	}
}

func TestKVSnapshotStateHelpers_Good(t *testing.T) {
	snapshot := testSnapshot()
	snapshot.Version = 0
	opts := kvSnapshotStatePutOptions(snapshot, StateOptions{
		Kind:   "custom-kind",
		Track:  "custom-track",
		URI:    "mlx://custom",
		Title:  "custom title",
		Tags:   map[string]string{"caller": "yes"},
		Labels: []string{"caller-label"},
	}, kvSnapshotStateEnvelope{
		KVHash:           "hash",
		KVEncoding:       string(EncodingNative),
		Architecture:     "gemma4_text",
		TokenCount:       2,
		PayloadByteCount: 32,
	})
	if opts.Kind != "custom-kind" || opts.Track != "custom-track" || opts.URI != "mlx://custom" || opts.Title != "custom title" {
		t.Fatalf("put options = %+v, want caller metadata", opts)
	}
	if opts.Tags["caller"] != "yes" || opts.Tags["kv_hash"] != "hash" || opts.Tags["payload_bytes"] != "32" {
		t.Fatalf("put option tags = %+v, want caller and KV tags", opts.Tags)
	}
	if got := effectiveVersion(snapshot, EncodingQ8); got != SnapshotVersion {
		t.Fatalf("effectiveVersion(q8) = %d, want %d", got, SnapshotVersion)
	}
	if got := EffectiveTokenOffset(&Snapshot{Tokens: []int32{1, 2, 3}}); got != 3 {
		t.Fatalf("EffectiveTokenOffset(default) = %d, want token length", got)
	}
	if got := EffectiveTokenOffset(nil); got != 0 {
		t.Fatalf("EffectiveTokenOffset(nil) = %d, want 0", got)
	}
	sourceTags := map[string]string{"a": "b"}
	tags := cloneKVSnapshotStateTags(sourceTags)
	tags["a"] = "changed"
	if sourceTags["a"] != "b" {
		t.Fatalf("source tags were mutated: %+v", sourceTags)
	}
}

type failingStateWriter struct{}

func (failingStateWriter) Put(context.Context, string, state.PutOptions) (state.ChunkRef, error) {
	return state.ChunkRef{}, core.NewError("put failed")
}

// TestStateStore_Snapshot_SaveMemvid_Good asserts the deprecated SaveMemvid
// alias writes a chunk that the canonical LoadFromState path decodes back to the
// same KV state — the alias must be a transparent forward to SaveState.
func TestStateStore_Snapshot_SaveMemvid_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()

	ref, err := snapshot.SaveMemvid(context.Background(), store, MemvidOptions{
		KVEncoding: EncodingQ8,
		URI:        "mlx://session/memvid",
		Title:      "memvid session",
	})
	if err != nil {
		t.Fatalf("SaveMemvid() error = %v", err)
	}
	if ref.ChunkID == 0 {
		t.Fatalf("SaveMemvid() ref = %+v, want a written chunk", ref)
	}

	loaded, err := LoadFromState(context.Background(), store, ref)
	if err != nil {
		t.Fatalf("LoadFromState() error = %v", err)
	}
	if loaded.Architecture != snapshot.Architecture || loaded.NumLayers != snapshot.NumLayers {
		t.Fatalf("loaded metadata = %+v, want %+v", loaded, snapshot)
	}
}

// TestStateStore_LoadFromMemvid_Good asserts the deprecated LoadFromMemvid alias
// decodes a chunk written by the canonical SaveState path.
func TestStateStore_LoadFromMemvid_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, StateOptions{KVEncoding: EncodingQ8})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}

	loaded, err := LoadFromMemvid(context.Background(), store, ref)
	if err != nil {
		t.Fatalf("LoadFromMemvid() error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset || loaded.NumHeads != snapshot.NumHeads {
		t.Fatalf("loaded metadata = %+v, want %+v", loaded, snapshot)
	}
}

// TestStateStore_LoadFromMemvidWithOptions_Good asserts the deprecated
// LoadFromMemvidWithOptions alias forwards decode options to
// LoadFromStateWithOptions: RawKVOnly skips float32 reconstruction so the loaded
// head exposes raw bytes rather than decoded values.
func TestStateStore_LoadFromMemvidWithOptions_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, StateOptions{KVEncoding: EncodingNative})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}

	loaded, err := LoadFromMemvidWithOptions(context.Background(), store, ref, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromMemvidWithOptions() error = %v", err)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("loaded Head(0, 0) ok = false, want true")
	}
	if len(head.KeyBytes) == 0 {
		t.Fatalf("loaded head = %+v, want raw key bytes retained under RawKVOnly", head)
	}
}

// TestStateStore_LoadFromMemvid_Bad asserts the deprecated load aliases surface
// the same guard errors as the canonical path (nil store, missing chunk).
func TestStateStore_LoadFromMemvid_Bad(t *testing.T) {
	if _, err := LoadFromMemvid(context.Background(), nil, state.ChunkRef{ChunkID: 1}); err == nil {
		t.Fatal("LoadFromMemvid(nil store) error = nil, want store error")
	}
	if _, err := LoadFromMemvidWithOptions(context.Background(), nil, state.ChunkRef{ChunkID: 1}, LoadOptions{}); err == nil {
		t.Fatal("LoadFromMemvidWithOptions(nil store) error = nil, want store error")
	}
	store := state.NewInMemoryStore(nil)
	if _, err := LoadFromMemvid(context.Background(), store, state.ChunkRef{ChunkID: 999}); err == nil {
		t.Fatal("LoadFromMemvid(missing chunk) error = nil, want resolve error")
	}
}

// TestStateStore_Snapshot_SaveState_Ugly covers SaveState's bytesWithOptions
// error path (state_store.go:95-97). A snapshot carrying a Version above
// SnapshotVersion is encoded with a valid encoding; the binary encoder's
// version guard (snapshot.go) rejects it, and SaveState must surface that error
// rather than panic or write a chunk.
func TestStateStore_Snapshot_SaveState_Ugly(t *testing.T) {
	snapshot := testSnapshot()
	snapshot.Version = SnapshotVersion + 1

	ref, err := snapshot.SaveState(context.Background(), state.NewInMemoryStore(nil), StateOptions{KVEncoding: EncodingQ8})
	if err == nil {
		t.Fatalf("SaveState(bumped version) error = nil, ref = %+v; want encode error", ref)
	}
}

// TestStateStore_EffectiveVersionBumps_Good covers the four version-bump arms
// of effectiveVersion (state_store.go:258,261,264,267). Each sub-case starts
// from a low base Version so the relevant `version < N` guard can fire, then
// sets the layer/encoding trigger and asserts the bumped result.
func TestStateStore_EffectiveVersionBumps_Good(t *testing.T) {
	// 258: non-float32 encoding with base version < 3 bumps to 3.
	base := &Snapshot{Version: 1, Layers: []LayerSnapshot{{Layer: 0}}}
	if got := effectiveVersion(base, EncodingQ8); got != 3 {
		t.Fatalf("effectiveVersion(v1, q8) = %d, want 3 (non-float32 bump)", got)
	}
	// Float32 encoding leaves a low version unbumped by the 258 arm.
	if got := effectiveVersion(&Snapshot{Version: 1, Layers: []LayerSnapshot{{Layer: 0}}}, KVSnapshotEncodingFloat32); got != 1 {
		t.Fatalf("effectiveVersion(v1, float32) = %d, want 1 (no non-float32 bump)", got)
	}

	// 261: a layer carrying native tensor bytes bumps to >= 4.
	native := &Snapshot{Version: 1, Layers: []LayerSnapshot{{Layer: 0, KeyBytes: []byte{1, 2}}}}
	if got := effectiveVersion(native, KVSnapshotEncodingFloat32); got < 4 {
		t.Fatalf("effectiveVersion(native tensors) = %d, want >= 4", got)
	}

	// 264: a layer carrying a compressed cache mode bumps to >= 5.
	compressed := &Snapshot{Version: 1, Layers: []LayerSnapshot{{Layer: 0, CacheMode: "turboquant"}}}
	if got := effectiveVersion(compressed, KVSnapshotEncodingFloat32); got < 5 {
		t.Fatalf("effectiveVersion(compressed payloads) = %d, want >= 5", got)
	}

	// 267: a layer carrying a MaxSize window clamp bumps to >= 6.
	clamped := &Snapshot{Version: 1, Layers: []LayerSnapshot{{Layer: 0, MaxSize: 4096}}}
	if got := effectiveVersion(clamped, KVSnapshotEncodingFloat32); got < 6 {
		t.Fatalf("effectiveVersion(max size) = %d, want >= 6", got)
	}
}

// TestStateStore_SnapshotHasLayerMaxSize_GoodBadUgly covers
// snapshotHasLayerMaxSize: the nil-snapshot guard (state_store.go:274) and the
// MaxSize>0 true arm (state_store.go:278), plus the all-zero false case.
func TestStateStore_SnapshotHasLayerMaxSize_GoodBadUgly(t *testing.T) {
	// Ugly: nil snapshot returns false (274).
	if snapshotHasLayerMaxSize(nil) {
		t.Fatal("snapshotHasLayerMaxSize(nil) = true, want false")
	}
	// Good: a layer with a positive MaxSize returns true (278).
	if !snapshotHasLayerMaxSize(&Snapshot{Layers: []LayerSnapshot{{Layer: 0, MaxSize: 8}}}) {
		t.Fatal("snapshotHasLayerMaxSize(MaxSize>0) = false, want true")
	}
	// Bad: layers present but all MaxSize zero returns false.
	if snapshotHasLayerMaxSize(&Snapshot{Layers: []LayerSnapshot{{Layer: 0}}}) {
		t.Fatal("snapshotHasLayerMaxSize(no clamp) = true, want false")
	}
}

// TestStateStore_SnapshotHasLayerCompressedPayloads_Ugly covers the
// nil-snapshot guard of snapshotHasLayerCompressedPayloads (state_store.go:286).
func TestStateStore_SnapshotHasLayerCompressedPayloads_Ugly(t *testing.T) {
	if snapshotHasLayerCompressedPayloads(nil) {
		t.Fatal("snapshotHasLayerCompressedPayloads(nil) = true, want false")
	}
}

// TestStateStore_Snapshot_SaveMemvid_Bad asserts the deprecated SaveMemvid alias
// surfaces the same guard errors as SaveState: a nil snapshot and a nil store
// both fail without writing a chunk.
func TestStateStore_Snapshot_SaveMemvid_Bad(t *testing.T) {
	var snapshot *Snapshot
	if _, err := snapshot.SaveMemvid(context.Background(), state.NewInMemoryStore(nil), MemvidOptions{}); err == nil {
		t.Fatal("SaveMemvid(nil snapshot) error = nil, want snapshot error")
	}
	if _, err := testSnapshot().SaveMemvid(context.Background(), nil, MemvidOptions{}); err == nil {
		t.Fatal("SaveMemvid(nil store) error = nil, want store error")
	}
}

// TestStateStore_Snapshot_SaveMemvid_Ugly covers SaveMemvid's forwarded encode
// error path: a snapshot whose Version exceeds SnapshotVersion is rejected by
// the binary encoder, so the alias must surface that error rather than write a
// chunk.
func TestStateStore_Snapshot_SaveMemvid_Ugly(t *testing.T) {
	snapshot := testSnapshot()
	snapshot.Version = SnapshotVersion + 1

	ref, err := snapshot.SaveMemvid(context.Background(), state.NewInMemoryStore(nil), MemvidOptions{KVEncoding: EncodingQ8})
	if err == nil {
		t.Fatalf("SaveMemvid(bumped version) error = nil, ref = %+v; want encode error", ref)
	}
}

// TestStateStore_LoadFromState_Good writes a snapshot then reads it back through
// LoadFromState specifically, asserting the decoded snapshot recovers the token
// stream and head tensors independently of the SaveState round-trip test.
func TestStateStore_LoadFromState_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, StateOptions{KVEncoding: EncodingQ8})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}

	loaded, err := LoadFromState(context.Background(), store, ref)
	if err != nil {
		t.Fatalf("LoadFromState() error = %v", err)
	}
	if len(loaded.Tokens) != len(snapshot.Tokens) || loaded.NumQueryHeads != snapshot.NumQueryHeads {
		t.Fatalf("LoadFromState() metadata = %+v, want token/head match with %+v", loaded, snapshot)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("LoadFromState() Head(0, 0) ok = false, want true")
	}
	if len(head.Key) != len(snapshot.Layers[0].Heads[0].Key) {
		t.Fatalf("LoadFromState() head key len = %d, want %d", len(head.Key), len(snapshot.Layers[0].Heads[0].Key))
	}
}

// TestStateStore_LoadFromState_Ugly feeds LoadFromState a structurally valid
// envelope whose inner KV payload is base64-correct but not a parsable snapshot
// (hash omitted so the bytes reach the inner parser). LoadFromState must surface
// the inner parse failure rather than panic.
func TestStateStore_LoadFromState_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(map[int]string{
		1: `{"version":1,"kind":"` + KVSnapshotStateKind + `","binary_encoding":"base64","data":"` + core.Base64Encode([]byte("not-a-kv-snapshot")) + `"}`,
	})

	_, err := LoadFromState(context.Background(), store, state.ChunkRef{ChunkID: 1})
	if err == nil {
		t.Fatal("LoadFromState(garbage payload) error = nil, want inner parse error")
	}
}

// TestStateStore_LoadFromStateWithOptions_Good asserts LoadFromStateWithOptions
// honours decode options: under RawKVOnly the loaded head retains raw key bytes
// instead of reconstructed float32 values.
func TestStateStore_LoadFromStateWithOptions_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := testSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, StateOptions{KVEncoding: EncodingNative})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}

	loaded, err := LoadFromStateWithOptions(context.Background(), store, ref, LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadFromStateWithOptions() error = %v", err)
	}
	head, ok := loaded.Head(0, 0)
	if !ok {
		t.Fatal("LoadFromStateWithOptions() Head(0, 0) ok = false, want true")
	}
	if len(head.KeyBytes) == 0 {
		t.Fatalf("LoadFromStateWithOptions() head = %+v, want raw key bytes under RawKVOnly", head)
	}
}

// TestStateStore_LoadFromStateWithOptions_Bad drives LoadFromStateWithOptions'
// guard arms directly: a nil store and a nil context paired with a corrupt
// chunk both fail.
func TestStateStore_LoadFromStateWithOptions_Bad(t *testing.T) {
	if _, err := LoadFromStateWithOptions(context.Background(), nil, state.ChunkRef{ChunkID: 1}, LoadOptions{}); err == nil {
		t.Fatal("LoadFromStateWithOptions(nil store) error = nil, want store error")
	}
	store := state.NewInMemoryStore(map[int]string{1: "{"})
	if _, err := LoadFromStateWithOptions(nil, store, state.ChunkRef{ChunkID: 1}, LoadOptions{}); err == nil {
		t.Fatal("LoadFromStateWithOptions(corrupt JSON) error = nil, want parse error")
	}
}

// TestStateStore_LoadFromStateWithOptions_Ugly asks LoadFromStateWithOptions to
// resolve a chunk ID that is not present in the store; the resolve step must
// fail and the error propagate rather than returning a zero snapshot.
func TestStateStore_LoadFromStateWithOptions_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)

	snap, err := LoadFromStateWithOptions(context.Background(), store, state.ChunkRef{ChunkID: 12345}, LoadOptions{})
	if err == nil {
		t.Fatalf("LoadFromStateWithOptions(missing chunk) error = nil, snap = %+v; want resolve error", snap)
	}
}

// TestStateStore_LoadFromMemvid_Ugly asks the deprecated LoadFromMemvid alias to
// resolve a missing chunk; the forwarded resolve must fail.
func TestStateStore_LoadFromMemvid_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)

	if _, err := LoadFromMemvid(context.Background(), store, state.ChunkRef{ChunkID: 7777}); err == nil {
		t.Fatal("LoadFromMemvid(missing chunk) error = nil, want resolve error")
	}
}

// TestStateStore_LoadFromMemvidWithOptions_Bad asserts the deprecated
// LoadFromMemvidWithOptions alias surfaces the nil-store guard error from the
// canonical path it forwards to.
func TestStateStore_LoadFromMemvidWithOptions_Bad(t *testing.T) {
	if _, err := LoadFromMemvidWithOptions(context.Background(), nil, state.ChunkRef{ChunkID: 1}, LoadOptions{}); err == nil {
		t.Fatal("LoadFromMemvidWithOptions(nil store) error = nil, want store error")
	}
}

// TestStateStore_LoadFromMemvidWithOptions_Ugly asks LoadFromMemvidWithOptions
// to resolve a missing chunk; the forwarded resolve must fail.
func TestStateStore_LoadFromMemvidWithOptions_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)

	if _, err := LoadFromMemvidWithOptions(context.Background(), store, state.ChunkRef{ChunkID: 8888}, LoadOptions{}); err == nil {
		t.Fatal("LoadFromMemvidWithOptions(missing chunk) error = nil, want resolve error")
	}
}

// TestStateStore_EffectiveTokenOffset_Good asserts EffectiveTokenOffset returns
// the explicit TokenOffset when it is set on the snapshot.
func TestStateStore_EffectiveTokenOffset_Good(t *testing.T) {
	if got := EffectiveTokenOffset(&Snapshot{TokenOffset: 17, Tokens: []int32{1, 2}}); got != 17 {
		t.Fatalf("EffectiveTokenOffset(explicit) = %d, want 17", got)
	}
}

// TestStateStore_EffectiveTokenOffset_Bad asserts EffectiveTokenOffset falls
// back to the token count when TokenOffset is zero (the default-derivation arm).
func TestStateStore_EffectiveTokenOffset_Bad(t *testing.T) {
	if got := EffectiveTokenOffset(&Snapshot{Tokens: []int32{1, 2, 3, 4}}); got != 4 {
		t.Fatalf("EffectiveTokenOffset(zero offset) = %d, want token length 4", got)
	}
}

// TestStateStore_EffectiveTokenOffset_Ugly asserts EffectiveTokenOffset returns
// 0 for a nil snapshot rather than panicking.
func TestStateStore_EffectiveTokenOffset_Ugly(t *testing.T) {
	if got := EffectiveTokenOffset(nil); got != 0 {
		t.Fatalf("EffectiveTokenOffset(nil) = %d, want 0", got)
	}
}
