// SPDX-Licence-Identifier: EUPL-1.2

package bundle

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/lora"
	state "dappco.re/go/inference/state"
)

func bundleTestSnapshot() *kv.Snapshot {
	return &kv.Snapshot{
		Version:       kv.SnapshotVersion,
		Architecture:  "gemma4_text",
		Tokens:        []int32{1, 2},
		Generated:     []int32{2},
		TokenOffset:   2,
		NumLayers:     1,
		NumHeads:      1,
		SeqLen:        2,
		HeadDim:       2,
		NumQueryHeads: 8,
		LogitShape:    []int32{1, 1, 3},
		Logits:        []float32{0.1, 0.2, 0.7},
		Layers: []kv.LayerSnapshot{{
			Layer:      0,
			CacheIndex: 0,
			Heads: []kv.HeadSnapshot{{
				Key:   []float32{1, 0, 0, 1},
				Value: []float32{0, 1, 1, 0},
			}},
		}},
	}
}

// TestBundle_New_Good is the end-to-end happy path: New assembles a
// bundle from a snapshot + full Options, defensively clones the
// snapshot (mutating the source after New must not leak in), Save
// round-trips through Load, and every field survives the trip.
func TestBundle_New_Good(t *testing.T) {
	snapshot := bundleTestSnapshot()
	tokenizerPath := core.PathJoin(t.TempDir(), "tokenizer.json")
	if result := core.WriteFile(tokenizerPath, []byte(`{"model":{"type":"BPE","vocab":{},"merges":[]}}`), 0o600); !result.OK {
		t.Fatalf("WriteFile tokenizer: %s", result.Error())
	}
	tokenizerHash, err := FileHash(tokenizerPath)
	if err != nil {
		t.Fatalf("FileHash() error = %v", err)
	}
	b, err := New(snapshot, Options{
		Model:     "gemma4-e4b",
		ModelPath: "/models/gemma4",
		Source: ModelInfo{
			Architecture:  "gemma4_text",
			NumLayers:     1,
			VocabSize:     262144,
			QuantBits:     4,
			ContextLength: 131072,
		},
		Prompt: "stable context",
		Tokenizer: Tokenizer{
			Kind: "hf-tokenizer-json", Path: tokenizerPath, Version: "tokenizers-v1",
			Hash: tokenizerHash, VocabSize: 262144, BOS: 2, EOS: 1,
			ChatTemplate: "<start_of_turn>model\n",
		},
		Runtime: Runtime{Name: "go-mlx", Version: "dev", Platform: "darwin/arm64"},
		Adapter: Adapter{
			Name: "domain-lora", Path: "/adapters/domain",
			Rank: 8, Alpha: 16, TargetKeys: []string{"q_proj", "v_proj"},
		},
		Sampler: Sampler{MaxTokens: 32, Temperature: 0.2, TopK: 4, RepeatPenalty: 1.1},
		StateRefs: []state.ChunkRef{{
			ChunkID: 42, FrameOffset: 7, HasFrameOffset: true,
			Codec: state.CodecQRVideo, Segment: "/tmp/trace.mp4",
		}},
		Refs: []Ref{{Kind: "kv", URI: "file:///tmp/session.kvbin", Hash: "sha256:kv"}},
		Meta: map[string]string{"suite": "beta"},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	snapshot.Tokens[0] = 99
	path := core.PathJoin(t.TempDir(), "state.bundle.json")
	if err := b.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if loaded.Version != Version || loaded.Kind != Kind {
		t.Fatalf("loaded version/kind = %d/%q", loaded.Version, loaded.Kind)
	}
	if loaded.Model.Name != "gemma4-e4b" || loaded.Model.Architecture != "gemma4_text" {
		t.Fatalf("loaded model = %+v", loaded.Model)
	}
	if loaded.Model.VocabSize != 262144 || loaded.Model.QuantBits != 4 || loaded.Model.ContextLength != 131072 {
		t.Fatalf("loaded model metadata = %+v", loaded.Model)
	}
	if loaded.Prompt.Text != "stable context" || loaded.Prompt.Hash == "" {
		t.Fatalf("loaded prompt = %+v", loaded.Prompt)
	}
	if loaded.Tokenizer.Path != tokenizerPath || loaded.Tokenizer.Hash != tokenizerHash || loaded.Tokenizer.ChatTemplateHash == "" {
		t.Fatalf("loaded tokenizer = %+v", loaded.Tokenizer)
	}
	if loaded.Runtime.Name != "go-mlx" || loaded.Runtime.Version != "dev" {
		t.Fatalf("loaded runtime = %+v", loaded.Runtime)
	}
	if loaded.Adapter.Name != "domain-lora" || loaded.Adapter.Hash == "" || loaded.Adapter.Rank != 8 {
		t.Fatalf("loaded adapter = %+v", loaded.Adapter)
	}
	if loaded.Sampler.MaxTokens != 32 || loaded.Sampler.TopK != 4 {
		t.Fatalf("loaded sampler = %+v", loaded.Sampler)
	}
	if loaded.KV == nil || loaded.KV.Tokens[0] != 1 || loaded.KVHash == "" {
		t.Fatalf("loaded KV = %+v hash=%q", loaded.KV, loaded.KVHash)
	}
	if loaded.Analysis == nil || loaded.SAMI == nil || loaded.SAMI.Architecture != "gemma4_text" {
		t.Fatalf("loaded analysis/SAMI = %+v/%+v", loaded.Analysis, loaded.SAMI)
	}
	if len(loaded.Refs) != 2 || loaded.Refs[1].Kind != RefState || loaded.Refs[1].State.ChunkID != 42 {
		t.Fatalf("loaded refs = %+v", loaded.Refs)
	}
	if loaded.Meta["suite"] != "beta" {
		t.Fatalf("loaded meta = %+v", loaded.Meta)
	}
}

// TestBundle_New_Bad asserts the one failure New can return — a nil
// snapshot is rejected rather than producing a half-built bundle.
func TestBundle_New_Bad(t *testing.T) {
	if _, err := New(nil, Options{}); err == nil {
		t.Fatal("New(nil) error = nil, want nil snapshot error")
	}
}

// TestBundle_New_Ugly drives New's adapter-from-Source edge case: the
// caller supplies no explicit Adapter but a populated Source.Adapter,
// so New must lift the active adapter identity AND defensively clone
// its TargetKeys (mutating the caller's slice afterwards must not leak
// into the bundle).
func TestBundle_New_Ugly(t *testing.T) {
	info := ModelInfo{
		Adapter: lora.AdapterInfo{
			Name: "active", Path: "/adapters/active", Hash: "active-hash",
			Rank: 4, Alpha: 8, Scale: 2, TargetKeys: []string{"q_proj"},
		},
	}
	b, err := New(bundleTestSnapshot(), Options{Source: info})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	info.Adapter.TargetKeys[0] = "mutated"
	if b.Adapter.Name != "active" || b.Adapter.Path != "/adapters/active" || b.Adapter.Hash != "active-hash" {
		t.Fatalf("bundle adapter = %+v, want active adapter identity", b.Adapter)
	}
	if len(b.Adapter.TargetKeys) != 1 || b.Adapter.TargetKeys[0] != "q_proj" {
		t.Fatalf("bundle adapter targets = %v, want defensive copy", b.Adapter.TargetKeys)
	}
}

// TestBundle_Save_Good writes a freshly-built bundle with Save and
// confirms the file is valid indented JSON that Load round-trips, and
// that the human-debug indent contract holds (two-space indent present).
func TestBundle_Save_Good(t *testing.T) {
	b, err := New(bundleTestSnapshot(), Options{
		Model:  "gemma4-e2b",
		Source: ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	path := core.PathJoin(t.TempDir(), "state.bundle.json")
	if err := b.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	read := core.ReadFile(path)
	if !read.OK {
		t.Fatalf("ReadFile() error = %v", read.Value)
	}
	data := string(read.Value.([]byte))
	if !core.Contains(data, "\n  \"version\": 1") {
		t.Fatalf("Save did not emit two-space indented JSON: %q", data[:min(80, len(data))])
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load(saved) error = %v", err)
	}
	if loaded.Model.Name != "gemma4-e2b" || loaded.KVHash != b.KVHash {
		t.Fatalf("Save→Load lost fidelity: loaded = %+v", loaded.Model)
	}
}

// TestBundle_Save_Bad confirms Save refuses an invalid bundle — the
// Validate gate fires before any bytes hit disk, so a bundle with a
// bad version never writes a corrupt artifact.
func TestBundle_Save_Bad(t *testing.T) {
	b := &Bundle{Version: 0, Kind: Kind}
	path := core.PathJoin(t.TempDir(), "bad.bundle.json")
	if err := b.Save(path); err == nil {
		t.Fatal("Save(invalid) error = nil, want validate error")
	}
	if core.Stat(path).OK {
		t.Fatal("Save(invalid) wrote a file despite failing validation")
	}
}

// TestBundle_Save_Ugly drives Save to an unwritable path (a file nested
// under a path component that is itself a regular file). Save must
// surface the write error rather than panic.
func TestBundle_Save_Ugly(t *testing.T) {
	b, err := New(bundleTestSnapshot(), Options{Source: ModelInfo{Architecture: "gemma4_text"}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	notDir := core.PathJoin(t.TempDir(), "afile")
	if result := core.WriteFile(notDir, []byte("x"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}
	// afile/child treats a regular file as a directory — guaranteed write failure.
	if err := b.Save(core.PathJoin(notDir, "child.json")); err == nil {
		t.Fatal("Save(unwritable path) error = nil, want write error")
	}
}

// TestBundle_SaveCompact_Good verifies SaveCompact emits wire-identical
// content to Save (after whitespace strip), Load handles both, and the
// loaded bundles are structurally identical. Compact must also be
// materially smaller on disk.
//
// Uses a realistic (512-token / 8-layer) snapshot rather than the tiny
// 2-token bundleTestSnapshot — the whitespace-ratio gate only holds on
// shapes large enough to swamp the fixed-cost JSON header. The 2-token
// shape gets ~35% reduction (mostly header), the 512/8 shape gets ~90%
// which matches the W10-AG forward note's 75.7% expectation comfortably.
func TestBundle_SaveCompact_Good(t *testing.T) {
	// Build a representative snapshot: 512 tokens × 8 layers — the
	// "typical" Save benchmark shape. This isolates Save's per-element
	// whitespace overhead from the fixed JSON envelope.
	tokenCount, numLayers := 512, 8
	tokens := make([]int32, tokenCount)
	headKey := make([]float32, tokenCount)
	headValue := make([]float32, tokenCount)
	for i := range tokenCount {
		tokens[i] = int32(i + 1)
		headKey[i] = float32(i)
		headValue[i] = float32(i + 1000)
	}
	layers := make([]kv.LayerSnapshot, numLayers)
	for i := range layers {
		layers[i] = kv.LayerSnapshot{
			Layer: i, CacheIndex: i,
			Heads: []kv.HeadSnapshot{{Key: headKey, Value: headValue}},
		}
	}
	snapshot := &kv.Snapshot{
		Version: kv.SnapshotVersion, Architecture: "qwen3",
		Tokens: tokens, TokenOffset: tokenCount,
		NumLayers: numLayers, NumHeads: 1, SeqLen: tokenCount,
		HeadDim: 1, NumQueryHeads: 1, Layers: layers,
	}
	b, err := New(snapshot, Options{
		Model:     "qwen3",
		ModelPath: "/models/qwen3",
		Source: ModelInfo{
			Architecture: "qwen3", NumLayers: numLayers,
			VocabSize: 1000, QuantBits: 4, ContextLength: 40960,
		},
		Prompt:  "stable context",
		Runtime: Runtime{Name: "go-mlx", Version: "dev", Platform: "darwin/arm64"},
		Sampler: Sampler{MaxTokens: 32, Temperature: 0.2, TopK: 4, RepeatPenalty: 1.1},
		Meta:    map[string]string{"suite": "beta"},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	dir := t.TempDir()
	indentedPath := core.PathJoin(dir, "indented.bundle.json")
	compactPath := core.PathJoin(dir, "compact.bundle.json")
	if err := b.Save(indentedPath); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	if err := b.SaveCompact(compactPath); err != nil {
		t.Fatalf("SaveCompact() error = %v", err)
	}
	// Disk size: compact must be materially smaller. Gate at 70%
	// reduction — W10-AG observed 75.7% from MarshalIndent's
	// `appendNewline`. Below 70% on a realistic-shape bundle means
	// either the shape regressed or compact isn't actually compact.
	indentedBytes := core.ReadFile(indentedPath)
	if !indentedBytes.OK {
		t.Fatalf("ReadFile(indented) error = %v", indentedBytes.Value)
	}
	compactBytes := core.ReadFile(compactPath)
	if !compactBytes.OK {
		t.Fatalf("ReadFile(compact) error = %v", compactBytes.Value)
	}
	indentedSize := len(indentedBytes.Value.([]byte))
	compactSize := len(compactBytes.Value.([]byte))
	if compactSize >= indentedSize {
		t.Fatalf("SaveCompact size = %d, Save size = %d — compact must be smaller", compactSize, indentedSize)
	}
	saved := float64(indentedSize-compactSize) / float64(indentedSize) * 100
	if saved < 70 {
		t.Fatalf("SaveCompact saved %.1f%% (%d → %d bytes) — gate is 70%% on realistic shape", saved, indentedSize, compactSize)
	}
	t.Logf("SaveCompact saved %.1f%% (%d → %d bytes)", saved, indentedSize, compactSize)

	// Both forms must Load cleanly to structurally identical bundles.
	loadedIndented, err := Load(indentedPath)
	if err != nil {
		t.Fatalf("Load(indented) error = %v", err)
	}
	loadedCompact, err := Load(compactPath)
	if err != nil {
		t.Fatalf("Load(compact) error = %v", err)
	}
	if loadedIndented.KVHash != loadedCompact.KVHash {
		t.Fatalf("KVHash mismatch: indented=%q compact=%q", loadedIndented.KVHash, loadedCompact.KVHash)
	}
	if loadedIndented.Version != loadedCompact.Version || loadedIndented.Kind != loadedCompact.Kind {
		t.Fatalf("version/kind mismatch: indented=%d/%q compact=%d/%q",
			loadedIndented.Version, loadedIndented.Kind,
			loadedCompact.Version, loadedCompact.Kind)
	}
	if loadedIndented.Model.Hash != loadedCompact.Model.Hash {
		t.Fatalf("Model.Hash mismatch: indented=%q compact=%q", loadedIndented.Model.Hash, loadedCompact.Model.Hash)
	}
	if loadedIndented.Meta["suite"] != loadedCompact.Meta["suite"] {
		t.Fatalf("Meta mismatch: indented=%v compact=%v", loadedIndented.Meta, loadedCompact.Meta)
	}
	// Wire parity — re-marshalling both forms compact must produce the same
	// bytes. This locks in the "same wire shape, just no whitespace" claim.
	reIndented := core.JSONMarshal(loadedIndented)
	if !reIndented.OK {
		t.Fatalf("re-marshal(indented) error = %v", reIndented.Value)
	}
	reCompact := core.JSONMarshal(loadedCompact)
	if !reCompact.OK {
		t.Fatalf("re-marshal(compact) error = %v", reCompact.Value)
	}
	if string(reIndented.Value.([]byte)) != string(reCompact.Value.([]byte)) {
		t.Fatal("indented and compact round-trips produced divergent wire bytes")
	}
}

// TestBundle_SaveCompact_Bad ensures SaveCompact applies the same
// Validate gate as Save (no path that bypasses bundle integrity).
func TestBundle_SaveCompact_Bad(t *testing.T) {
	b := &Bundle{Version: 0, Kind: Kind}
	if err := b.SaveCompact(core.PathJoin(t.TempDir(), "bad.json")); err == nil {
		t.Fatal("SaveCompact(bad) error = nil, want validate error")
	}
}

// TestBundle_SaveCompact_Ugly drives SaveCompact to an unwritable path:
// a valid bundle but a destination nested under a regular file. The
// write error must surface rather than panic.
func TestBundle_SaveCompact_Ugly(t *testing.T) {
	b, err := New(bundleTestSnapshot(), Options{Source: ModelInfo{Architecture: "gemma4_text"}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	notDir := core.PathJoin(t.TempDir(), "afile")
	if result := core.WriteFile(notDir, []byte("x"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}
	if err := b.SaveCompact(core.PathJoin(notDir, "child.json")); err == nil {
		t.Fatal("SaveCompact(unwritable path) error = nil, want write error")
	}
}

// TestBundle_Load_Good round-trips a bundle through Save then Load and
// asserts the loaded artifact carries the same identity the saved one
// did — the canonical reader half of the Save contract.
func TestBundle_Load_Good(t *testing.T) {
	b, err := New(bundleTestSnapshot(), Options{
		Model:  "gemma4-e2b",
		Source: ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
		Meta:   map[string]string{"suite": "load"},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	path := core.PathJoin(t.TempDir(), "state.bundle.json")
	if err := b.Save(path); err != nil {
		t.Fatalf("Save() error = %v", err)
	}
	loaded, err := Load(path)
	if err != nil {
		t.Fatalf("Load() error = %v", err)
	}
	if loaded.Model.Name != "gemma4-e2b" || loaded.Kind != Kind || loaded.KVHash == "" {
		t.Fatalf("Load() returned = %+v", loaded.Model)
	}
	if loaded.Meta["suite"] != "load" {
		t.Fatalf("Load() meta = %+v", loaded.Meta)
	}
}

// TestBundle_Load_Bad confirms Load surfaces an error for a path that
// does not exist rather than returning a zero bundle.
func TestBundle_Load_Bad(t *testing.T) {
	if _, err := Load(core.PathJoin(t.TempDir(), "missing.bundle.json")); err == nil {
		t.Fatal("Load(missing) error = nil, want read error")
	}
}

// TestBundle_Load_Ugly feeds Load a truncated JSON document — the parse
// step must fail cleanly rather than panic or return a partial bundle.
func TestBundle_Load_Ugly(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "broken.bundle.json")
	if result := core.WriteFile(path, []byte("{"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}
	if _, err := Load(path); err == nil {
		t.Fatal("Load() error = nil, want corrupt bundle error")
	}
}

// TestBundle_Snapshot_Good asserts Snapshot returns a defensive clone of
// the embedded KV (mutating one result must not affect the next) and
// also loads from KVPath when the bundle has no inline KV.
func TestBundle_Snapshot_Good(t *testing.T) {
	snapshot := bundleTestSnapshot()
	b, err := New(snapshot, Options{Prompt: "persisted"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	first, err := b.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot() error = %v", err)
	}
	first.Tokens[0] = 99
	second, err := b.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot() second error = %v", err)
	}
	if second.Tokens[0] != 1 {
		t.Fatalf("Snapshot() returned shared tokens = %v, want defensive clone", second.Tokens)
	}
	kvPath := core.PathJoin(t.TempDir(), "state.kvbin")
	if err := snapshot.Save(kvPath); err != nil {
		t.Fatalf("kv.Snapshot.Save() error = %v", err)
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	pathBundle := &Bundle{Version: Version, Kind: Kind, KVPath: kvPath, KVHash: hash}
	loaded, err := pathBundle.Snapshot()
	if err != nil {
		t.Fatalf("Snapshot(KVPath) error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset || len(loaded.Tokens) != len(snapshot.Tokens) {
		t.Fatalf("loaded path snapshot = %+v, want %+v", loaded, snapshot)
	}
}

// TestBundle_Snapshot_Bad covers Snapshot's failure paths: a nil bundle,
// a bundle with no KV at all, and a KVPath whose on-disk hash disagrees
// with the recorded KVHash.
func TestBundle_Snapshot_Bad(t *testing.T) {
	if _, err := (*Bundle)(nil).Snapshot(); err == nil {
		t.Fatal("Snapshot(nil bundle) error = nil")
	}
	if _, err := (&Bundle{Version: Version, Kind: Kind}).Snapshot(); err == nil {
		t.Fatal("Snapshot(no KV) error = nil")
	}
	snapshot := bundleTestSnapshot()
	kvPath := core.PathJoin(t.TempDir(), "state.kvbin")
	if err := snapshot.Save(kvPath); err != nil {
		t.Fatalf("kv.Snapshot.Save() error = %v", err)
	}
	bad := &Bundle{Version: Version, Kind: Kind, KVPath: kvPath, KVHash: "bad-hash"}
	if _, err := bad.Snapshot(); err == nil {
		t.Fatal("Snapshot(KVPath hash mismatch) error = nil")
	}
}

// TestBundle_Snapshot_Ugly drives the empty-KVPath boundary: a bundle
// whose KV is nil and whose KVPath is the empty string must report the
// no-snapshot sentinel rather than attempting to load "".
func TestBundle_Snapshot_Ugly(t *testing.T) {
	b := &Bundle{Version: Version, Kind: Kind, KVPath: ""}
	if _, err := b.Snapshot(); err == nil {
		t.Fatal("Snapshot(empty KVPath, nil KV) error = nil, want no-snapshot error")
	}
}

// TestBundle_SnapshotFromState_Good resolves a State-backed KV snapshot:
// save the snapshot into an in-memory State store, reference it from a
// bundle, and confirm SnapshotFromState rehydrates the same identity.
func TestBundle_SnapshotFromState_Good(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := bundleTestSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, kv.StateOptions{})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	b := &Bundle{
		Version: Version, Kind: Kind, KVHash: hash,
		Refs: []Ref{{Kind: RefState, URI: StateURI(ref), State: ref}},
	}
	loaded, err := b.SnapshotFromState(context.Background(), store)
	if err != nil {
		t.Fatalf("SnapshotFromState() error = %v", err)
	}
	if loaded.Architecture != snapshot.Architecture || loaded.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("loaded snapshot = %+v, want %+v", loaded, snapshot)
	}
}

// TestBundle_SnapshotFromState_Bad covers SnapshotFromState's failure
// paths: a nil bundle, a bundle with no State ref to resolve, and a
// State-backed ref whose rehydrated hash disagrees with KVHash.
func TestBundle_SnapshotFromState_Bad(t *testing.T) {
	if _, err := (*Bundle)(nil).SnapshotFromState(context.Background(), state.NewInMemoryStore(nil)); err == nil {
		t.Fatal("SnapshotFromState(nil bundle) error = nil")
	}
	if _, err := (&Bundle{Version: Version, Kind: Kind}).SnapshotFromState(nil, state.NewInMemoryStore(nil)); err == nil {
		t.Fatal("SnapshotFromState(no ref) error = nil")
	}
	store := state.NewInMemoryStore(nil)
	ref, err := bundleTestSnapshot().SaveState(context.Background(), store, kv.StateOptions{})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}
	b := &Bundle{
		Version: Version, Kind: Kind, KVHash: "bad-hash",
		Refs: []Ref{{Kind: RefState, State: ref}},
	}
	if _, err := b.SnapshotFromState(context.Background(), store); err == nil {
		t.Fatal("SnapshotFromState(hash mismatch) error = nil")
	}
}

// TestBundle_SnapshotFromState_Ugly passes a nil context: SnapshotFromState
// must default it to context.Background() internally and still resolve
// the State ref rather than panic on the nil.
func TestBundle_SnapshotFromState_Ugly(t *testing.T) {
	store := state.NewInMemoryStore(nil)
	snapshot := bundleTestSnapshot()
	ref, err := snapshot.SaveState(context.Background(), store, kv.StateOptions{})
	if err != nil {
		t.Fatalf("SaveState() error = %v", err)
	}
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	b := &Bundle{
		Version: Version, Kind: Kind, KVHash: hash,
		Refs: []Ref{{Kind: RefState, State: ref}},
	}
	//nolint:staticcheck // SA1012: nil ctx is the boundary under test — must be defaulted, not panic.
	loaded, err := b.SnapshotFromState(nil, store)
	if err != nil {
		t.Fatalf("SnapshotFromState(nil ctx) error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("loaded token offset = %d, want %d", loaded.TokenOffset, snapshot.TokenOffset)
	}
}

// TestBundle_SnapshotFromMemvid_Good drives the deprecated memvid alias
// through a frame-zero ref — the legacy path must still resolve a
// snapshot via the same State machinery SnapshotFromState uses.
func TestBundle_SnapshotFromMemvid_Good(t *testing.T) {
	source := state.NewInMemoryStore(nil)
	snapshot := bundleTestSnapshot()
	ref, err := snapshot.SaveMemvid(context.Background(), source, kv.MemvidOptions{})
	if err != nil {
		t.Fatalf("SaveMemvid() error = %v", err)
	}
	chunk, err := state.Resolve(context.Background(), source, ref.ChunkID)
	if err != nil {
		t.Fatalf("Resolve() error = %v", err)
	}
	store := state.NewInMemoryStoreWithManifest(map[int]string{0: chunk.Text}, map[int]state.ChunkRef{0: {
		ChunkID: 0, FrameOffset: 0, HasFrameOffset: true,
		Codec: state.CodecQRVideo, Segment: "/tmp/session.mp4",
	}})
	hash, err := kv.HashSnapshot(snapshot)
	if err != nil {
		t.Fatalf("kv.HashSnapshot() error = %v", err)
	}
	b := &Bundle{
		Version: Version, Kind: Kind, KVHash: hash,
		Refs: []Ref{{
			Kind: RefMemvid, URI: "memvid:///tmp/session.mp4#chunk=0",
			Memvid: state.ChunkRef{
				ChunkID: 0, FrameOffset: 0, HasFrameOffset: true,
				Codec: state.CodecQRVideo, Segment: "/tmp/session.mp4",
			},
		}},
	}
	loaded, err := b.SnapshotFromMemvid(context.Background(), store)
	if err != nil {
		t.Fatalf("SnapshotFromMemvid(frame zero) error = %v", err)
	}
	if loaded.TokenOffset != snapshot.TokenOffset {
		t.Fatalf("loaded token offset = %d, want %d", loaded.TokenOffset, snapshot.TokenOffset)
	}
}

// TestBundle_SnapshotFromMemvid_Bad confirms the deprecated alias shares
// SnapshotFromState's guards: a nil bundle is rejected.
func TestBundle_SnapshotFromMemvid_Bad(t *testing.T) {
	if _, err := (*Bundle)(nil).SnapshotFromMemvid(context.Background(), state.NewInMemoryStore(nil)); err == nil {
		t.Fatal("SnapshotFromMemvid(nil bundle) error = nil")
	}
}

// TestBundle_SnapshotFromMemvid_Ugly drives the alias with an inline KV
// already present — SnapshotFromMemvid must short-circuit to the embedded
// snapshot path rather than consult any memvid ref.
func TestBundle_SnapshotFromMemvid_Ugly(t *testing.T) {
	b, err := New(bundleTestSnapshot(), Options{Model: "gemma4-e2b"})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	loaded, err := b.SnapshotFromMemvid(context.Background(), nil)
	if err != nil {
		t.Fatalf("SnapshotFromMemvid(inline KV) error = %v", err)
	}
	if loaded.Architecture != "gemma4_text" || len(loaded.Tokens) != 2 {
		t.Fatalf("SnapshotFromMemvid(inline KV) = %+v, want embedded snapshot", loaded)
	}
}

// TestBundle_Validate_Good asserts a well-formed bundle (built by New)
// passes Validate, and that an embedded KV whose hash matches KVHash is
// accepted.
func TestBundle_Validate_Good(t *testing.T) {
	b, err := New(bundleTestSnapshot(), Options{
		Model:  "gemma4-e2b",
		Source: ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := b.Validate(); err != nil {
		t.Fatalf("Validate(well-formed) error = %v", err)
	}
}

// TestBundle_Validate_Bad walks every rejection Validate owns: nil
// bundle, unsupported version, wrong kind, no snapshot at all, and an
// embedded KV whose hash disagrees with the recorded KVHash.
func TestBundle_Validate_Bad(t *testing.T) {
	snapshot := bundleTestSnapshot()
	if err := (*Bundle)(nil).Validate(); err == nil {
		t.Fatal("Validate(nil) error = nil")
	}
	for name, bad := range map[string]*Bundle{
		"nil kv":  {Version: Version, Kind: Kind},
		"version": {Version: Version + 1, Kind: Kind, KV: snapshot.Clone()},
		"kind":    {Version: Version, Kind: "wrong", KV: snapshot.Clone()},
	} {
		if err := bad.Validate(); err == nil {
			t.Fatalf("%s Validate() error = nil", name)
		}
	}
	b, err := New(snapshot, Options{Source: ModelInfo{Architecture: "gemma4_text", NumLayers: 1}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	hashMismatch := *b
	hashMismatch.KV = b.KV.Clone()
	hashMismatch.KV.Tokens[0] = 99
	if err := hashMismatch.Validate(); err == nil {
		t.Fatal("Validate(hash mismatch) error = nil")
	}
}

// TestBundle_Validate_Ugly drives the version-zero boundary directly:
// Version 0 is below the valid range and must be rejected as an
// unsupported-version error.
func TestBundle_Validate_Ugly(t *testing.T) {
	b := &Bundle{Version: 0, Kind: Kind, KV: bundleTestSnapshot()}
	if err := b.Validate(); err == nil {
		t.Fatal("Validate(version 0) error = nil, want unsupported-version error")
	}
}

// TestBundle_CheckCompatibility_Good confirms a bundle validates against
// a loaded model whose architecture, layer count, and adapter identity
// all match the bundle's recorded expectations.
func TestBundle_CheckCompatibility_Good(t *testing.T) {
	b, err := New(bundleTestSnapshot(), Options{
		Source: ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
		Adapter: Adapter{
			Name: "domain", Path: "/adapters/domain", Hash: "adapter-hash",
			Rank: 8, Alpha: 16,
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := CheckCompatibility(ModelInfo{
		Architecture: "gemma4_text", NumLayers: 1,
		Adapter: lora.AdapterInfo{Name: "domain", Path: "/adapters/domain", Hash: "adapter-hash", Rank: 8, Alpha: 16},
	}, b); err != nil {
		t.Fatalf("CheckCompatibility(good) error = %v", err)
	}
}

// TestBundle_CheckCompatibility_Bad walks every incompatibility:
// architecture mismatch, layer-count mismatch, a missing adapter when one
// is required, and each adapter-field divergence (hash/path/rank/alpha).
func TestBundle_CheckCompatibility_Bad(t *testing.T) {
	b, err := New(bundleTestSnapshot(), Options{
		Source: ModelInfo{Architecture: "gemma4_text", NumLayers: 1},
		Adapter: Adapter{
			Name: "domain", Path: "/adapters/domain", Hash: "adapter-hash",
			Rank: 8, Alpha: 16,
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := CheckCompatibility(ModelInfo{Architecture: "llama", NumLayers: 1}, b); err == nil {
		t.Fatal("CheckCompatibility(architecture mismatch) error = nil")
	}
	if err := CheckCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 2}, b); err == nil {
		t.Fatal("CheckCompatibility(layer mismatch) error = nil")
	}
	if err := CheckCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 1}, b); err == nil {
		t.Fatal("CheckCompatibility(missing adapter) error = nil")
	}
	for name, adapter := range map[string]lora.AdapterInfo{
		"hash":  {Path: "/adapters/domain", Hash: "wrong", Rank: 8, Alpha: 16},
		"path":  {Path: "/other/domain", Rank: 8, Alpha: 16},
		"rank":  {Path: "/adapters/domain", Rank: 4, Alpha: 16},
		"alpha": {Path: "/adapters/domain", Rank: 8, Alpha: 8},
	} {
		if err := CheckCompatibility(ModelInfo{Architecture: "gemma4_text", NumLayers: 1, Adapter: adapter}, b); err == nil {
			t.Fatalf("CheckCompatibility(%s mismatch) error = nil", name)
		}
	}
}

// TestBundle_CheckCompatibility_Ugly drives the nil-bundle boundary and
// the all-zero-info case: a nil bundle is rejected, while a bundle with
// no adapter expectation is compatible with any model that omits one.
func TestBundle_CheckCompatibility_Ugly(t *testing.T) {
	if err := CheckCompatibility(ModelInfo{}, nil); err == nil {
		t.Fatal("CheckCompatibility(nil bundle) error = nil")
	}
	b, err := New(bundleTestSnapshot(), Options{Source: ModelInfo{Architecture: "gemma4_text", NumLayers: 1}})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}
	if err := CheckCompatibility(ModelInfo{}, b); err != nil {
		t.Fatalf("CheckCompatibility(zero info, no adapter) error = %v, want compatible", err)
	}
}

// TestBundle_FileHash_Good confirms FileHash is stable (two calls on the
// same file agree) and non-empty for real content.
func TestBundle_FileHash_Good(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "f.txt")
	if result := core.WriteFile(path, []byte("hello"), 0o600); !result.OK {
		t.Fatalf("WriteFile: %s", result.Error())
	}
	h1, err := FileHash(path)
	if err != nil {
		t.Fatalf("FileHash() error = %v", err)
	}
	h2, err := FileHash(path)
	if err != nil {
		t.Fatalf("FileHash() second error = %v", err)
	}
	if h1 != h2 || h1 == "" {
		t.Fatalf("FileHash not stable: %q vs %q", h1, h2)
	}
}

// TestBundle_FileHash_Bad confirms FileHash surfaces an error for a path
// that does not exist rather than returning an empty hash.
func TestBundle_FileHash_Bad(t *testing.T) {
	if _, err := FileHash(core.PathJoin(t.TempDir(), "missing")); err == nil {
		t.Fatal("FileHash(missing) error = nil")
	}
}

// TestBundle_FileHash_Ugly is the bit-exact parity check against the
// legacy `core.ReadFile + core.SHA256Hex` path across boundary sizes
// (empty, sub-block, exactly the stdlib io.Copy scratch, straddling it,
// multi-MB). FileHash's small-buffer and streaming branches MUST produce
// the same digest for any file content, otherwise bundle metadata
// round-trips silently regress across the version that flipped the impl.
func TestBundle_FileHash_Ugly(t *testing.T) {
	sizes := []int{
		0,               // empty file — boundary
		1,               // single byte — sub-block
		63,              // sub-SHA256-block
		64,              // exactly one SHA256 block
		65,              // one block + remainder
		1024,            // 1KB — small tokenizer
		32*1024 - 1,     // just under stdlib io.Copy default scratch
		32 * 1024,       // exactly stdlib io.Copy default scratch
		32*1024 + 1,     // straddle stdlib scratch boundary
		256 * 1024,      // 256KB
		1024 * 1024,     // 1MB — representative tokenizer.json
		3*1024*1024 + 7, // 3MB + 7 — non-aligned LoRA-scale
	}
	for _, n := range sizes {
		path := core.PathJoin(t.TempDir(), "f.bin")
		data := make([]byte, n)
		for i := range data {
			data[i] = byte(i * 31)
		}
		if result := core.WriteFile(path, data, 0o600); !result.OK {
			t.Fatalf("WriteFile(%d): %s", n, result.Error())
		}
		streamed, err := FileHash(path)
		if err != nil {
			t.Fatalf("FileHash(%d): %v", n, err)
		}
		expected := core.SHA256Hex(data)
		if streamed != expected {
			t.Fatalf("FileHash(%d) parity mismatch:\n  stream=%q\n  buffer=%q", n, streamed, expected)
		}
	}
}

// TestBundle_NormaliseTokenizer_Good confirms NormaliseTokenizer fills
// the missing Hash and ChatTemplateHash from Path and ChatTemplate.
func TestBundle_NormaliseTokenizer_Good(t *testing.T) {
	in := Tokenizer{Path: "/tok.json", ChatTemplate: "<bos>"}
	out := NormaliseTokenizer(in)
	if out.Hash == "" || out.ChatTemplateHash == "" {
		t.Fatalf("NormaliseTokenizer left hashes empty: %+v", out)
	}
}

// TestBundle_NormaliseTokenizer_Bad confirms NormaliseTokenizer does not
// fabricate hashes when the source fields are absent — an all-empty
// tokenizer stays all-empty (no Path → no Hash, no template → no hash).
func TestBundle_NormaliseTokenizer_Bad(t *testing.T) {
	out := NormaliseTokenizer(Tokenizer{})
	if out.Hash != "" || out.ChatTemplateHash != "" {
		t.Fatalf("NormaliseTokenizer(empty) fabricated hashes: %+v", out)
	}
}

// TestBundle_NormaliseTokenizer_Ugly confirms a caller-supplied Hash is
// preserved (not recomputed): when Hash is already set, the function
// must leave it untouched even though Path is present.
func TestBundle_NormaliseTokenizer_Ugly(t *testing.T) {
	out := NormaliseTokenizer(Tokenizer{Path: "/tok.json", Hash: "preset-hash"})
	if out.Hash != "preset-hash" {
		t.Fatalf("NormaliseTokenizer overwrote preset Hash = %q", out.Hash)
	}
}

// TestBundle_AdapterEmpty_Good confirms AdapterEmpty reports true for the
// zero adapter — the canonical "nothing set" case.
func TestBundle_AdapterEmpty_Good(t *testing.T) {
	if !AdapterEmpty(Adapter{}) {
		t.Fatal("AdapterEmpty(zero) = false")
	}
}

// TestBundle_AdapterEmpty_Bad confirms AdapterEmpty reports false as soon
// as any meaningful field is set (a name, or target keys).
func TestBundle_AdapterEmpty_Bad(t *testing.T) {
	if AdapterEmpty(Adapter{Name: "x"}) {
		t.Fatal("AdapterEmpty(name set) = true")
	}
	if AdapterEmpty(Adapter{TargetKeys: []string{"q_proj"}}) {
		t.Fatal("AdapterEmpty(targets set) = true")
	}
}

// TestBundle_AdapterEmpty_Ugly drives the single-numeric-field boundary:
// an adapter with only a non-zero Scale (and no name/path/keys) is still
// non-empty — the predicate must not ignore the float fields.
func TestBundle_AdapterEmpty_Ugly(t *testing.T) {
	if AdapterEmpty(Adapter{Scale: 1}) {
		t.Fatal("AdapterEmpty(scale only) = true, want false")
	}
	if AdapterEmpty(Adapter{Alpha: 0.0001}) {
		t.Fatal("AdapterEmpty(tiny alpha) = true, want false")
	}
}

// TestBundle_AdapterFromInfo_Good confirms AdapterFromInfo copies every
// field and defensively clones TargetKeys (mutating the source slice
// afterwards must not leak into the produced Adapter).
func TestBundle_AdapterFromInfo_Good(t *testing.T) {
	src := lora.AdapterInfo{
		Name: "v1", Path: "/v1.safetensors", Hash: "abc",
		Rank: 8, Alpha: 16, Scale: 2, TargetKeys: []string{"q_proj", "v_proj"},
	}
	adapter := AdapterFromInfo(src)
	if adapter.Name != src.Name || adapter.Path != src.Path || adapter.Hash != src.Hash ||
		adapter.Rank != src.Rank || adapter.Alpha != src.Alpha || adapter.Scale != src.Scale {
		t.Fatalf("AdapterFromInfo = %+v, want %+v", adapter, src)
	}
	if len(adapter.TargetKeys) != 2 || adapter.TargetKeys[1] != "v_proj" {
		t.Fatalf("AdapterFromInfo targets = %v", adapter.TargetKeys)
	}
	src.TargetKeys[0] = "mutated"
	if adapter.TargetKeys[0] == "mutated" {
		t.Fatal("AdapterFromInfo did not clone TargetKeys")
	}
}

// TestBundle_AdapterFromInfo_Bad confirms AdapterFromInfo lifts an empty
// AdapterInfo to an empty Adapter rather than inventing values.
func TestBundle_AdapterFromInfo_Bad(t *testing.T) {
	adapter := AdapterFromInfo(lora.AdapterInfo{})
	if !AdapterEmpty(adapter) {
		t.Fatalf("AdapterFromInfo(empty) = %+v, want empty adapter", adapter)
	}
}

// TestBundle_AdapterFromInfo_Ugly drives the nil-TargetKeys boundary: a
// SliceClone of nil must stay nil (not become a non-nil empty slice that
// would marshal differently downstream).
func TestBundle_AdapterFromInfo_Ugly(t *testing.T) {
	adapter := AdapterFromInfo(lora.AdapterInfo{Name: "x", TargetKeys: nil})
	if adapter.TargetKeys != nil {
		t.Fatalf("AdapterFromInfo(nil keys) TargetKeys = %v, want nil", adapter.TargetKeys)
	}
}

// TestBundle_AdapterToInfo_Good confirms AdapterToInfo round-trips an
// Adapter back to a lora.AdapterInfo carrying the same fields, and
// defensively clones TargetKeys.
func TestBundle_AdapterToInfo_Good(t *testing.T) {
	adapter := Adapter{
		Name: "v1", Path: "/v1.safetensors", Hash: "abc",
		Rank: 8, Alpha: 16, Scale: 2, TargetKeys: []string{"q_proj", "v_proj"},
	}
	info := AdapterToInfo(adapter)
	if info.Name != adapter.Name || info.Rank != adapter.Rank ||
		len(info.TargetKeys) != 2 || info.TargetKeys[1] != "v_proj" {
		t.Fatalf("AdapterToInfo = %+v, want %+v", info, adapter)
	}
	adapter.TargetKeys[0] = "mutated"
	if info.TargetKeys[0] == "mutated" {
		t.Fatal("AdapterToInfo did not clone TargetKeys")
	}
}

// TestBundle_AdapterToInfo_Bad confirms AdapterToInfo lowers an empty
// Adapter to an empty AdapterInfo (IsEmpty), not a populated one.
func TestBundle_AdapterToInfo_Bad(t *testing.T) {
	info := AdapterToInfo(Adapter{})
	if !info.IsEmpty() {
		t.Fatalf("AdapterToInfo(empty) = %+v, want empty info", info)
	}
}

// TestBundle_AdapterToInfo_Ugly drives the nil-TargetKeys boundary: the
// produced AdapterInfo keeps nil keys nil rather than allocating an
// empty slice.
func TestBundle_AdapterToInfo_Ugly(t *testing.T) {
	info := AdapterToInfo(Adapter{Name: "x", TargetKeys: nil})
	if info.TargetKeys != nil {
		t.Fatalf("AdapterToInfo(nil keys) TargetKeys = %v, want nil", info.TargetKeys)
	}
}

// TestBundle_HashString_Good confirms HashString returns a 64-char hex
// SHA-256 digest for non-empty input, stable across calls.
func TestBundle_HashString_Good(t *testing.T) {
	h := HashString("gemma4")
	if len(h) != 64 {
		t.Fatalf("HashString len = %d, want 64", len(h))
	}
	if h != HashString("gemma4") {
		t.Fatal("HashString not deterministic")
	}
}

// TestBundle_HashString_Bad confirms distinct inputs hash to distinct
// digests — HashString is not a constant.
func TestBundle_HashString_Bad(t *testing.T) {
	if HashString("a") == HashString("b") {
		t.Fatal("HashString collided on distinct inputs")
	}
}

// TestBundle_HashString_Ugly drives the empty-input boundary: HashString
// returns "" for "" (the omitempty contract), and remains non-empty for
// the single-byte case just above the boundary.
func TestBundle_HashString_Ugly(t *testing.T) {
	if HashString("") != "" {
		t.Fatal("HashString(\"\") returned non-empty")
	}
	if HashString("x") == "" {
		t.Fatal("HashString(single byte) returned empty")
	}
}

// TestBundle_StateURI_Good confirms both URI shapes: a segment-bearing
// ref renders state://<segment>#chunk=<id>, a bare ref renders
// state://chunk/<id>.
func TestBundle_StateURI_Good(t *testing.T) {
	withSeg := StateURI(state.ChunkRef{ChunkID: 5, Segment: "/tmp/x.mp4"})
	withoutSeg := StateURI(state.ChunkRef{ChunkID: 7})
	if withSeg != "state:///tmp/x.mp4#chunk=5" {
		t.Fatalf("with-segment URI = %q", withSeg)
	}
	if withoutSeg != "state://chunk/7" {
		t.Fatalf("without-segment URI = %q", withoutSeg)
	}
}

// TestBundle_StateURI_Bad confirms the rendered URI tracks the ChunkID —
// distinct chunk IDs produce distinct URIs (the id is not dropped).
func TestBundle_StateURI_Bad(t *testing.T) {
	if StateURI(state.ChunkRef{ChunkID: 1}) == StateURI(state.ChunkRef{ChunkID: 2}) {
		t.Fatal("StateURI collided across distinct chunk IDs")
	}
}

// TestBundle_StateURI_Ugly drives the zero-ChunkID / empty-segment
// boundary: the bare path must still render a well-formed chunk URI
// (state://chunk/0) rather than an empty or malformed string.
func TestBundle_StateURI_Ugly(t *testing.T) {
	got := StateURI(state.ChunkRef{})
	if got != "state://chunk/0" {
		t.Fatalf("StateURI(zero) = %q, want state://chunk/0", got)
	}
}

// TestBundle_MemvidURI_Good confirms the deprecated memvid alias renders
// both URI shapes with the memvid:// scheme: segment-bearing and bare.
func TestBundle_MemvidURI_Good(t *testing.T) {
	withSeg := MemvidURI(state.ChunkRef{ChunkID: 5, Segment: "session.mp4"})
	withoutSeg := MemvidURI(state.ChunkRef{ChunkID: 7})
	if withSeg != "memvid://session.mp4#chunk=5" {
		t.Fatalf("with-segment URI = %q", withSeg)
	}
	if withoutSeg != "memvid://chunk/7" {
		t.Fatalf("without-segment URI = %q", withoutSeg)
	}
}

// TestBundle_MemvidURI_Bad confirms the rendered URI tracks the ChunkID —
// distinct IDs produce distinct memvid URIs.
func TestBundle_MemvidURI_Bad(t *testing.T) {
	if MemvidURI(state.ChunkRef{ChunkID: 1}) == MemvidURI(state.ChunkRef{ChunkID: 2}) {
		t.Fatal("MemvidURI collided across distinct chunk IDs")
	}
}

// TestBundle_MemvidURI_Ugly drives the zero-value boundary: a zero ref
// renders memvid://chunk/0 rather than an empty or malformed string.
func TestBundle_MemvidURI_Ugly(t *testing.T) {
	got := MemvidURI(state.ChunkRef{})
	if got != "memvid://chunk/0" {
		t.Fatalf("MemvidURI(zero) = %q, want memvid://chunk/0", got)
	}
}
