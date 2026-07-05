// SPDX-Licence-Identifier: EUPL-1.2

package bundle

import (
	"math"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	state "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/train/lora"
)

// covSnapshot mirrors bundleTestSnapshot but leaves Version and TokenOffset
// at their zero values so New must fill them in — exercising the
// snap.Version==0 and snap.TokenOffset==0 normalisation branches that the
// fully-populated bundleTestSnapshot skips.
func covSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		// Version intentionally 0 → New sets it to kv.SnapshotVersion.
		Architecture: "gemma4_text",
		Tokens:       []int32{1, 2},
		Generated:    []int32{2},
		// TokenOffset intentionally 0 → New sets it to len(Tokens).
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []kv.LayerSnapshot{{
			Layer: 0,
			Heads: []kv.HeadSnapshot{{Key: []float32{1, 0, 0, 1}, Value: []float32{0, 1, 1, 0}}},
		}},
	}
}

// TestBundle_New_NormalisesZeroVersionAndOffset covers the two New()
// normalisation branches: a snapshot with Version==0 gets the current
// SnapshotVersion, and TokenOffset==0 is backfilled from the token count.
func TestBundle_New_NormalisesZeroVersionAndOffset(t *testing.T) {
	b, err := New(covSnapshot(), Options{Source: ModelInfo{Architecture: "gemma4_text"}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if b.KV.Version != kv.SnapshotVersion {
		t.Fatalf("KV.Version = %d, want %d (backfilled)", b.KV.Version, kv.SnapshotVersion)
	}
	if b.KV.TokenOffset != 2 {
		t.Fatalf("KV.TokenOffset = %d, want 2 (len(Tokens) backfill)", b.KV.TokenOffset)
	}
	if b.Prompt.TokenOffset != 2 {
		t.Fatalf("Prompt.TokenOffset = %d, want 2", b.Prompt.TokenOffset)
	}
}

// hashFailingSnapshot returns an otherwise-valid snapshot whose single layer
// carries TurboQuant payloads without the matching "turboquant" CacheMode.
// kv.HashSnapshot streams the snapshot through the encoder, which rejects
// that inconsistency up front, so any code path that hashes this snapshot
// fails deterministically. Snapshot.Clone preserves CacheMode and the
// payloads, so the failure survives New's defensive clone.
func hashFailingSnapshot() *kv.Snapshot {
	s := bundleTestSnapshot()
	s.Layers[0].TurboQuantPayloads = [][]byte{{0x01}}
	s.Layers[0].CacheMode = "" // not "turboquant" → encoder rejects
	return s
}

// TestBundle_New_HashError covers New's kv.HashSnapshot failure branch: a
// snapshot the encoder rejects makes the hash computation error before the
// bundle is assembled, so New returns that error.
func TestBundle_New_HashError(t *testing.T) {
	if _, err := New(hashFailingSnapshot(), Options{Source: ModelInfo{Architecture: "gemma4_text"}}); err == nil {
		t.Fatal("New(hash-failing snapshot) error = nil, want hash error")
	}
}

// TestBundle_Validate_HashComputeError covers Validate's branch where an
// inline KV snapshot is present with a non-empty KVHash but hashing it fails
// outright (encoder rejection), distinct from a clean hash that merely
// mismatches.
func TestBundle_Validate_HashComputeError(t *testing.T) {
	b := &Bundle{
		Version: Version, Kind: Kind,
		KV:     hashFailingSnapshot(),
		KVHash: "any-non-empty-hash",
	}
	if err := b.Validate(); err == nil {
		t.Fatal("Validate(hash-compute error) error = nil, want hash error")
	}
}

// TestBundle_Save_MarshalError covers the marshal-failure branch in Save and
// SaveCompact. A bundle built by New passes Validate (version/kind/KV-hash
// all consistent), but injecting a NaN into the SAMI summary makes
// encoding/json reject the value at marshal time — Validate does not inspect
// SAMI, so the marshal-error path is the only thing that fires.
func TestBundle_Save_MarshalError(t *testing.T) {
	b, err := New(bundleTestSnapshot(), Options{Source: ModelInfo{Architecture: "gemma4_text"}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if b.SAMI == nil {
		t.Fatal("expected New to populate SAMI")
	}
	b.SAMI.MeanCoherence = math.NaN() // json.Marshal rejects NaN

	path := core.PathJoin(t.TempDir(), "nan.bundle.json")
	if err := b.Save(path); err == nil {
		t.Fatal("Save(NaN SAMI) error = nil, want marshal error")
	}
	if err := b.SaveCompact(path); err == nil {
		t.Fatal("SaveCompact(NaN SAMI) error = nil, want marshal error")
	}
}

// TestBundle_Load_InvalidAfterParse covers Load's post-parse Validate branch:
// the file is well-formed JSON (so read + parse both succeed) but encodes a
// bundle that fails Validate — here an unsupported version — so Load must
// surface the validation error rather than return the bundle.
func TestBundle_Load_InvalidAfterParse(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "invalid.bundle.json")
	// version 0 is below the supported floor → Validate rejects it, but the
	// JSON itself parses cleanly.
	body := `{"version":0,"kind":"` + Kind + `","kv_path":"/x"}`
	if result := core.WriteFile(path, []byte(body), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}
	if _, err := Load(path); err == nil {
		t.Fatal("Load(parses but invalid) error = nil, want validate error")
	}
}

// TestBundle_Snapshot_LoadError covers the kv.Load failure branch inside
// Snapshot: KVPath points at a file that is not a decodable snapshot, so the
// embedded loader returns an error that Snapshot propagates.
func TestBundle_Snapshot_LoadError(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "notasnapshot.bin")
	if result := core.WriteFile(path, []byte("not a snapshot"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}
	b := &Bundle{Version: Version, Kind: Kind, KVPath: path}
	if _, err := b.Snapshot(); err == nil {
		t.Fatal("Snapshot(undecodable KVPath) error = nil, want load error")
	}
}

// TestBundle_Snapshot_HashMismatchAfterLoad covers Snapshot's KVHash branch
// on the disk-load path: a valid snapshot is saved to disk and referenced by
// KVPath, but the bundle carries a deliberately wrong KVHash, so the rehydrated
// snapshot's hash disagrees and Snapshot returns the mismatch error.
func TestBundle_Snapshot_HashMismatchAfterLoad(t *testing.T) {
	snapshot := bundleTestSnapshot()
	path := core.PathJoin(t.TempDir(), "good.snapshot.json")
	if err := snapshot.Save(path); err != nil {
		t.Fatalf("snapshot.Save() error = %v", err)
	}
	b := &Bundle{Version: Version, Kind: Kind, KVPath: path, KVHash: "deadbeef-not-the-real-hash"}
	if _, err := b.Snapshot(); err == nil {
		t.Fatal("Snapshot(wrong KVHash) error = nil, want hash mismatch")
	}
}

// TestBundle_SnapshotFromState_LoadError covers the kv.LoadFromState failure
// branch: a State ref that does not resolve in the store makes LoadFromState
// error, and SnapshotFromState propagates it.
func TestBundle_SnapshotFromState_LoadError(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	// A ref whose chunk was never written → LoadFromState cannot resolve it.
	b := &Bundle{
		Version: Version, Kind: Kind,
		Refs: []Ref{{Kind: RefState, State: state.ChunkRef{ChunkID: 999999}}},
	}
	if _, err := b.SnapshotFromState(t.Context(), store); err == nil {
		t.Fatal("SnapshotFromState(unresolvable ref) error = nil, want load error")
	}
}

// TestBundle_stateRef_NilReceiver covers the nil-receiver guard in stateRef
// via the exported Validate path that calls it; a nil *Bundle returns the
// nil error from Validate, but the helper itself is also reachable directly.
func TestBundle_stateRef_NilReceiver(t *testing.T) {
	if _, ok := (*Bundle)(nil).stateRef(); ok {
		t.Fatal("stateRef(nil) ok = true, want false")
	}
}

// TestBundle_stateRef_MemvidFallbackInStateRef covers the RefState case where
// the typed State field is zero but the legacy Memvid field carries the chunk
// id — a migrated bundle — so stateRef falls back to Memvid.
func TestBundle_stateRef_MemvidFallbackInStateRef(t *testing.T) {
	b := &Bundle{
		Version: Version, Kind: Kind,
		Refs: []Ref{{
			Kind:   RefState,
			State:  state.ChunkRef{}, // zero → skip typed branch
			Memvid: state.ChunkRef{ChunkID: 7, Segment: "seg"},
		}},
	}
	ref, ok := b.stateRef()
	if !ok {
		t.Fatal("stateRef() ok = false, want true via Memvid fallback")
	}
	if ref.ChunkID != 7 {
		t.Fatalf("stateRef() ChunkID = %d, want 7 (Memvid fallback)", ref.ChunkID)
	}
}

// TestBundle_Validate_StateRefOnlyOK covers Validate's "no KV, no KVPath, but
// a resolvable State ref" success branch — the bundle is valid purely on the
// strength of its cold-storage reference and Validate returns nil.
func TestBundle_Validate_StateRefOnlyOK(t *testing.T) {
	b := &Bundle{
		Version: Version, Kind: Kind,
		Refs: []Ref{{Kind: RefState, State: state.ChunkRef{ChunkID: 3}}},
	}
	if err := b.Validate(); err != nil {
		t.Fatalf("Validate(state-ref only) error = %v, want nil", err)
	}
}

// TestBundle_Validate_KVHashMismatchInline covers Validate's inline KV-hash
// branch: an in-memory KV snapshot present with a KVHash that does not match
// its content makes Validate return the mismatch error.
func TestBundle_Validate_KVHashMismatchInline(t *testing.T) {
	b := &Bundle{
		Version: Version, Kind: Kind,
		KV:     bundleTestSnapshot(),
		KVHash: "not-the-real-hash",
	}
	if err := b.Validate(); err == nil {
		t.Fatal("Validate(inline KV hash mismatch) error = nil, want mismatch")
	}
}

// TestBundle_CheckCompatibility_ValidateError covers CheckCompatibility's
// early Validate-failure branch: a structurally invalid bundle (bad kind)
// must fail before any architecture/layer comparison.
func TestBundle_CheckCompatibility_ValidateError(t *testing.T) {
	b := &Bundle{Version: Version, Kind: "wrong-kind", KV: bundleTestSnapshot()}
	if err := CheckCompatibility(ModelInfo{Architecture: "gemma4_text"}, b); err == nil {
		t.Fatal("CheckCompatibility(invalid bundle) error = nil, want validate error")
	}
}

// TestBundle_FileHash_StatErrorMissing covers FileHash's open-failure branch:
// a path that does not exist cannot be opened, so FileHash returns the open
// error before any stat/read.
func TestBundle_FileHash_OpenErrorMissing(t *testing.T) {
	if _, err := FileHash(core.PathJoin(t.TempDir(), "does-not-exist.bin")); err == nil {
		t.Fatal("FileHash(missing) error = nil, want open error")
	}
}

// TestBundle_FileHash_ReadErrorOnDirectory covers FileHash's small-file
// ReadFull-failure branch: opening a directory succeeds and its reported size
// is below the streaming threshold, so FileHash takes the buffer path, but
// reading bytes from a directory descriptor fails (EISDIR).
func TestBundle_FileHash_ReadErrorOnDirectory(t *testing.T) {
	dir := t.TempDir()
	if _, err := FileHash(dir); err == nil {
		t.Fatal("FileHash(directory) error = nil, want read error")
	}
}

// TestBundle_buildModel_OversizedBuffer covers buildModel's heap-fallback
// branch: when Name+Path+Architecture exceed the 256-byte stack scratch, the
// hash payload is built in a heap-allocated buffer instead. The resulting
// bundle must still hash and round-trip.
func TestBundle_buildModel_OversizedBuffer(t *testing.T) {
	long := strings.Repeat("m", 300)
	model := buildModel(bundleTestSnapshot(), Options{
		Model:     long,
		ModelPath: long,
		Source:    ModelInfo{Architecture: long},
	})
	if model.Hash == "" {
		t.Fatal("buildModel(oversized) Hash = empty, want a computed hash")
	}
	if model.Name != long {
		t.Fatalf("buildModel(oversized) Name length = %d, want 300", len(model.Name))
	}
}

// TestBundle_buildAdapter_OversizedBuffer covers buildAdapter's heap-fallback
// branch: a target-key set large enough to push the hash payload past the
// 256-byte stack scratch forces the make([]byte) path.
func TestBundle_buildAdapter_OversizedBuffer(t *testing.T) {
	keys := make([]string, 0, 40)
	for range 40 {
		keys = append(keys, "self_attn.q_proj") // 40×16 chars ≫ 256 bytes
	}
	adapter := buildAdapter(Adapter{
		Name:       "big",
		Path:       "/adapters/big",
		Rank:       8,
		Alpha:      16,
		Scale:      2,
		TargetKeys: keys,
	}, "", lora.AdapterInfo{})
	if adapter.Hash == "" {
		t.Fatal("buildAdapter(oversized) Hash = empty, want a computed hash")
	}
	if len(adapter.TargetKeys) != 40 {
		t.Fatalf("buildAdapter(oversized) TargetKeys = %d, want 40", len(adapter.TargetKeys))
	}
}

// TestBundle_joinChunkRefs covers all four branches of joinChunkRefs: both
// empty (nil), fallback-only (alias fallback), primary-only (alias primary),
// and both present (fresh joined allocation in primary-then-fallback order).
func TestBundle_joinChunkRefs(t *testing.T) {
	a := []state.ChunkRef{{ChunkID: 1}}
	b := []state.ChunkRef{{ChunkID: 2}, {ChunkID: 3}}

	if got := joinChunkRefs(nil, nil); got != nil {
		t.Fatalf("joinChunkRefs(nil,nil) = %v, want nil", got)
	}
	if got := joinChunkRefs(nil, b); len(got) != 2 || got[0].ChunkID != 2 {
		t.Fatalf("joinChunkRefs(nil,b) = %v, want fallback aliased", got)
	}
	if got := joinChunkRefs(a, nil); len(got) != 1 || got[0].ChunkID != 1 {
		t.Fatalf("joinChunkRefs(a,nil) = %v, want primary aliased", got)
	}
	got := joinChunkRefs(a, b)
	if len(got) != 3 || got[0].ChunkID != 1 || got[1].ChunkID != 2 || got[2].ChunkID != 3 {
		t.Fatalf("joinChunkRefs(a,b) = %v, want [1 2 3]", got)
	}
}

// TestBundle_resultError covers every branch of the resultError adapter: an
// OK result maps to nil, an error value passes through, a string value wraps
// into a new error, and any other value yields the generic fallback error.
func TestBundle_resultError(t *testing.T) {
	if err := resultError(core.Result{OK: true}); err != nil {
		t.Fatalf("resultError(OK) = %v, want nil", err)
	}
	wrapped := core.NewError("boom")
	if err := resultError(core.Result{Value: wrapped, OK: false}); err != wrapped {
		t.Fatalf("resultError(error value) = %v, want the wrapped error", err)
	}
	if err := resultError(core.Result{Value: "text failure", OK: false}); err == nil || !strings.Contains(err.Error(), "text failure") {
		t.Fatalf("resultError(string value) = %v, want NewError(text)", err)
	}
	if err := resultError(core.Result{Value: 42, OK: false}); err == nil {
		t.Fatal("resultError(other value) = nil, want generic fallback error")
	}
}
