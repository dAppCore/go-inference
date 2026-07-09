// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"context"
	"strconv"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/kv/memory"
	pkgbundle "dappco.re/go/inference/model/bundle"
	memvid "dappco.re/go/inference/model/state"
)

func TestKVSnapshotStateIndex_Good_PartialPrefixFromFullBundle(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	blk, err := snapshot.SaveStateBlocks(ctx, store, kv.StateBlockOptions{
		BlockSize:  2,
		KVEncoding: kv.EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}
	if _, err := kv.SaveStateBlockBundle(ctx, store, blk, "mlx://book/full/bundle"); err != nil {
		t.Fatalf("kv.SaveStateBlockBundle() error = %v", err)
	}
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://book/full/bundle",
		Title:     "full book",
		Model:     "demo",
		ModelInfo: memory.ModelInfo{
			Architecture:  "gemma4_text",
			NumLayers:     1,
			QuantBits:     4,
			ContextLength: 8,
		},
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		Entries: []StateIndexEntry{
			{
				URI:        "mlx://book/chapter-1",
				Title:      "Chapter 1",
				TokenStart: 0,
				TokenCount: 2,
				ByteStart:  0,
				ByteCount:  128,
				Labels:     []string{"chapter"},
				Meta:       map[string]string{"ordinal": "1"},
			},
			{
				URI:        "mlx://book/chapter-2",
				Title:      "Chapter 2",
				TokenStart: 2,
				TokenCount: 2,
				ByteStart:  128,
				ByteCount:  128,
				Labels:     []string{"chapter"},
				Meta:       map[string]string{"ordinal": "2"},
			},
		},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if index.Hash == "" || index.RequiredContextLength() != 4 {
		t.Fatalf("index hash/required = %q/%d, want hash and full required context", index.Hash, index.RequiredContextLength())
	}
	if err := CheckStateIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}, pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}, index); err != nil {
		t.Fatalf("CheckStateIndexCompatibility() error = %v", err)
	}
	if _, err := SaveStateIndex(ctx, store, index, "mlx://book/index"); err != nil {
		t.Fatalf("SaveStateIndex() error = %v", err)
	}
	loadedIndex, err := LoadStateIndex(ctx, store, "mlx://book/index")
	if err != nil {
		t.Fatalf("LoadStateIndex() error = %v", err)
	}
	loadedIndex.Entries[0].Labels[0] = "mutated"
	entry, ok := index.Entry("mlx://book/chapter-1")
	if !ok {
		t.Fatal("Entry(chapter-1) ok = false")
	}
	if entry.Labels[0] != "chapter" || entry.ByteStart != 0 || entry.ByteCount != 128 {
		t.Fatalf("entry clone = %+v, want original labels and byte span", entry)
	}

	recording := &indexRecordingMemvidStore{store: store}
	prefix, loadedEntry, err := LoadPrefixFromStateIndex(ctx, recording, index, "mlx://book/chapter-1", kv.LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadPrefixFromStateIndex() error = %v", err)
	}
	if loadedEntry.URI != "mlx://book/chapter-1" || loadedEntry.PrefixTokens() != 2 {
		t.Fatalf("loaded entry = %+v, want chapter-1 two-token prefix", loadedEntry)
	}
	if len(prefix.Tokens) != 2 || prefix.Tokens[0] != 1 || prefix.Tokens[1] != 2 {
		t.Fatalf("prefix tokens = %v, want first two tokens", prefix.Tokens)
	}
	if len(prefix.Logits) != 0 {
		t.Fatalf("prefix logits = %v, want terminal state cleared for partial prefix", prefix.Logits)
	}
	if len(recording.resolvedURIs) != 1 || recording.resolvedURIs[0] != "mlx://book/full/bundle" {
		t.Fatalf("resolved URIs = %v, want bundle manifest URI", recording.resolvedURIs)
	}
	if len(recording.resolved) != 1 {
		t.Fatalf("resolved chunks = %v, want one covering block", recording.resolved)
	}
}

func TestKVSnapshotMemvidBundleIndex_Good_DefaultFullEntry(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()

	index, err := NewMemvidIndex(blk, MemvidIndexOptions{BundleURI: "mlx://bundle"})

	if err != nil {
		t.Fatalf("NewMemvidIndex(default) error = %v", err)
	}
	if len(index.Entries) != 1 || index.Entries[0].TokenCount != blk.TokenCount || index.Entries[0].BundleURI != "mlx://bundle" {
		t.Fatalf("default entries = %+v, want full bundle entry", index.Entries)
	}
}

func TestKVSnapshotMemvidBundleIndex_Good_DerivesEntryByteSpan(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	blk.Blocks = []kv.MemvidBlockRef{
		{
			Index:            0,
			TokenStart:       0,
			TokenCount:       2,
			PayloadByteCount: 100,
			Memvid:           memvid.ChunkRef{ChunkID: 1, FrameOffset: 64, HasFrameOffset: true},
		},
		{
			Index:            1,
			TokenStart:       2,
			TokenCount:       2,
			PayloadByteCount: 300,
			Memvid:           memvid.ChunkRef{ChunkID: 2, FrameOffset: 256, HasFrameOffset: true},
		},
	}

	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://book/full/bundle",
		Entries: []MemvidIndexEntry{
			{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2},
			{URI: "mlx://book/chapter-2", TokenStart: 2, TokenCount: 2},
			{URI: "mlx://book/cross-block", TokenStart: 1, TokenCount: 2},
		},
	})

	if err != nil {
		t.Fatalf("NewMemvidIndex(byte span) error = %v", err)
	}
	chapter1, _ := index.Entry("mlx://book/chapter-1")
	if chapter1.ByteStart != 64 || chapter1.ByteCount != 100 {
		t.Fatalf("chapter-1 byte span = %d/%d, want 64/100", chapter1.ByteStart, chapter1.ByteCount)
	}
	chapter2, _ := index.Entry("mlx://book/chapter-2")
	if chapter2.ByteStart != 256 || chapter2.ByteCount != 300 {
		t.Fatalf("chapter-2 byte span = %d/%d, want 256/300", chapter2.ByteStart, chapter2.ByteCount)
	}
	cross, _ := index.Entry("mlx://book/cross-block")
	if cross.ByteStart != 64 || cross.ByteCount != 400 {
		t.Fatalf("cross-block byte span = %d/%d, want first frame offset and summed payload bytes 64/400", cross.ByteStart, cross.ByteCount)
	}
}

// TestKVSnapshotMemvidBundleIndex_Ugly_UnsortedBlocksByteSpan forces the
// linear (unsorted) fillIndexEntryByteSpan path: when bundle blocks are not
// in ascending TokenStart order, NewStateIndex falls back from the binary
// search to the full scan. The derived byte spans must match the sorted
// path's result regardless of physical block order.
func TestKVSnapshotMemvidBundleIndex_Ugly_UnsortedBlocksByteSpan(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	// Block for tokens 2-3 is listed BEFORE the block for tokens 0-1, so
	// stateBlockRefsSortedByTokenStart returns false.
	blk.Blocks = []kv.MemvidBlockRef{
		{
			Index:            1,
			TokenStart:       2,
			TokenCount:       2,
			PayloadByteCount: 300,
			Memvid:           memvid.ChunkRef{ChunkID: 2, FrameOffset: 256, HasFrameOffset: true},
		},
		{
			Index:            0,
			TokenStart:       0,
			TokenCount:       2,
			PayloadByteCount: 100,
			Memvid:           memvid.ChunkRef{ChunkID: 1, FrameOffset: 64, HasFrameOffset: true},
		},
	}

	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://book/full/bundle",
		Entries: []MemvidIndexEntry{
			{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2},
			{URI: "mlx://book/chapter-2", TokenStart: 2, TokenCount: 2},
			{URI: "mlx://book/cross-block", TokenStart: 1, TokenCount: 2},
		},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex(unsorted) error = %v", err)
	}
	chapter1, _ := index.Entry("mlx://book/chapter-1")
	if chapter1.ByteStart != 64 || chapter1.ByteCount != 100 {
		t.Fatalf("chapter-1 byte span = %d/%d, want 64/100 from unsorted scan", chapter1.ByteStart, chapter1.ByteCount)
	}
	chapter2, _ := index.Entry("mlx://book/chapter-2")
	if chapter2.ByteStart != 256 || chapter2.ByteCount != 300 {
		t.Fatalf("chapter-2 byte span = %d/%d, want 256/300 from unsorted scan", chapter2.ByteStart, chapter2.ByteCount)
	}
	// The unsorted scan walks blocks in physical (slice) order, so the
	// first frame offset it encounters for the cross-block span (tokens
	// 1-2, overlapping both blocks) is the physically-first block's
	// offset — here the token-2-3 block at 256, listed first. The byte
	// count is still the sum of both overlapping payloads. This contrasts
	// with the sorted path, which would report 64 (the token-ordered
	// first block).
	cross, _ := index.Entry("mlx://book/cross-block")
	if cross.ByteStart != 256 || cross.ByteCount != 400 {
		t.Fatalf("cross-block byte span = %d/%d, want 256/400 from unsorted (physical-order) scan", cross.ByteStart, cross.ByteCount)
	}
}

func TestKVSnapshotMemvidBundleIndex_Bad_ValidationAndCompatibility(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4},
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a"},
		Entries: []MemvidIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	for _, tc := range []struct {
		name  string
		index MemvidIndex
	}{
		{name: "bad kind", index: func() MemvidIndex {
			bad := *index
			bad.Kind = "bad"
			return bad
		}()},
		{name: "bad hash", index: func() MemvidIndex {
			bad := *index
			bad.Hash = "bad"
			return bad
		}()},
		{name: "duplicate uri", index: func() MemvidIndex {
			bad := *index
			bad.Entries = append(cloneIndexEntries(index.Entries), index.Entries[0])
			bad.Hash = indexHash(&bad)
			return bad
		}()},
		{name: "entry exceeds bundle", index: func() MemvidIndex {
			bad := *index
			bad.Entries = cloneIndexEntries(index.Entries)
			bad.Entries[0].TokenCount = 99
			bad.Entries[0].Hash = indexEntryHash(&bad.Entries[0])
			bad.Hash = indexHash(&bad)
			return bad
		}()},
		{name: "entry hash", index: func() MemvidIndex {
			bad := *index
			bad.Entries = cloneIndexEntries(index.Entries)
			bad.Entries[0].Hash = "bad"
			bad.Hash = ""
			return bad
		}()},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.index.Validate(); err == nil {
				t.Fatal("Validate() error = nil")
			}
		})
	}

	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "qwen3", NumLayers: 2, QuantBits: 4, ContextLength: 4}, pkgbundle.Tokenizer{Hash: "tok-a"}, index); err == nil {
		t.Fatal("expected architecture mismatch")
	}
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 4}, pkgbundle.Tokenizer{Hash: "tok-a"}, index); err == nil {
		t.Fatal("expected layer mismatch")
	}
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 8, ContextLength: 4}, pkgbundle.Tokenizer{Hash: "tok-a"}, index); err == nil {
		t.Fatal("expected quantization mismatch")
	}
	hashIndex, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4},
		Entries: []MemvidIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex(hash) error = %v", err)
	}
	hashIndex.Model.Hash = "different-model-hash"
	hashIndex.Hash = indexHash(hashIndex)
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4}, pkgbundle.Tokenizer{}, hashIndex); err == nil {
		t.Fatal("expected model hash mismatch")
	}
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 0}, pkgbundle.Tokenizer{Hash: "tok-b"}, index); err == nil {
		t.Fatal("expected tokenizer mismatch")
	}
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 0}, pkgbundle.Tokenizer{Hash: "tok-a"}, index); err != nil {
		t.Fatalf("zero context should skip context compatibility, got %v", err)
	}
}

func TestKVSnapshotMemvidBundleIndex_Bad_LoadAndStoreErrors(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		Entries: []MemvidIndexEntry{{
			URI:        "mlx://chapter",
			TokenStart: 0,
			TokenCount: 1,
		}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	if _, err := SaveMemvidIndex(ctx, nil, index, "mlx://index"); err == nil {
		t.Fatal("SaveMemvidIndex(nil store) error = nil")
	}
	if _, err := SaveMemvidIndex(ctx, store, index, ""); err == nil {
		t.Fatal("SaveMemvidIndex(empty URI) error = nil")
	}
	if _, err := LoadMemvidIndex(ctx, nil, "mlx://index"); err == nil {
		t.Fatal("LoadMemvidIndex(nil store) error = nil")
	}
	if _, err := LoadMemvidIndex(ctx, store, ""); err == nil {
		t.Fatal("LoadMemvidIndex(empty URI) error = nil")
	}
	if _, _, err := LoadPrefixFromMemvidIndex(ctx, nil, index, "mlx://chapter", kv.LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromMemvidIndex(nil store) error = nil")
	}
	if _, _, err := LoadPrefixFromMemvidIndex(ctx, store, index, "mlx://missing", kv.LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromMemvidIndex(missing entry) error = nil")
	}
	if _, _, err := LoadPrefixFromMemvidIndex(ctx, store, index, "mlx://chapter", kv.LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromMemvidIndex(missing bundle) error = nil")
	}
	corrupt := core.JSONMarshalString(map[string]any{"version": 1, "kind": MemvidIndexKind})
	if _, err := store.Put(ctx, corrupt, memvid.PutOptions{URI: "mlx://bad-index"}); err != nil {
		t.Fatalf("write corrupt index: %v", err)
	}
	if _, err := LoadMemvidIndex(ctx, store, "mlx://bad-index"); err == nil {
		t.Fatal("LoadMemvidIndex(corrupt) error = nil")
	}
}

// TestStateIndex_validate_Bad_DirectGuards exercises the top-level validate
// guards that NewStateIndex's happy path never reaches: a nil index, an
// out-of-range version, and a kind-valid index emptied of entries. Each is
// built as a literal (not via NewStateIndex, which would reject it earlier)
// so the specific sentinel return is reached.
func TestStateIndex_validate_Bad_DirectGuards(t *testing.T) {
	for _, tc := range []struct {
		name  string
		index *StateIndex
		want  error
	}{
		{name: "nil index", index: nil, want: errStateIndexNil},
		{
			name:  "version zero",
			index: &StateIndex{Version: 0, Kind: StateIndexKind, TokenCount: 4, Entries: []StateIndexEntry{{URI: "mlx://a", TokenCount: 1}}},
			want:  errStateIndexUnsupportedVersion,
		},
		{
			name:  "version above current",
			index: &StateIndex{Version: KVSnapshotStateBundleIndexVersion + 1, Kind: StateIndexKind, TokenCount: 4, Entries: []StateIndexEntry{{URI: "mlx://a", TokenCount: 1}}},
			want:  errStateIndexUnsupportedVersion,
		},
		{
			name:  "wrong kind",
			index: &StateIndex{Version: 1, Kind: "not-an-index", TokenCount: 4, Entries: []StateIndexEntry{{URI: "mlx://a", TokenCount: 1}}},
			want:  errStateIndexInvalidKind,
		},
		{
			name:  "zero token count",
			index: &StateIndex{Version: 1, Kind: StateIndexKind, TokenCount: 0, Entries: []StateIndexEntry{{URI: "mlx://a", TokenCount: 1}}},
			want:  errStateIndexEmptyTokenCount,
		},
		{
			name:  "no entries",
			index: &StateIndex{Version: 1, Kind: StateIndexKind, BundleURI: "mlx://b", TokenCount: 4},
			want:  errStateIndexNoEntries,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if err := tc.index.Validate(); !core.Is(err, tc.want) {
				t.Fatalf("Validate() error = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestStateIndex_validateEntry_Bad_Guards exercises each per-entry guard in
// validateEntry. The index carries a bundle URI so the entry-level
// bundle-required guard only fires for the dedicated case (an index with no
// bundle URI and an entry with no bundle URI).
func TestStateIndex_validateEntry_Bad_Guards(t *testing.T) {
	base := func(entry StateIndexEntry) *StateIndex {
		return &StateIndex{Version: 1, Kind: StateIndexKind, BundleURI: "mlx://b", TokenCount: 4, Entries: []StateIndexEntry{entry}}
	}
	for _, tc := range []struct {
		name  string
		index *StateIndex
		want  error
	}{
		{name: "empty entry uri", index: base(StateIndexEntry{URI: "  ", TokenCount: 1}), want: errStateIndexEntryURIRequired},
		{
			name:  "entry bundle required when index bundle empty",
			index: &StateIndex{Version: 1, Kind: StateIndexKind, TokenCount: 4, Entries: []StateIndexEntry{{URI: "mlx://a", TokenCount: 1}}},
			want:  errStateIndexEntryBundleRequired,
		},
		{name: "negative token start", index: base(StateIndexEntry{URI: "mlx://a", TokenStart: -1, TokenCount: 1}), want: errStateIndexEntryTokenStart},
		{name: "zero token count", index: base(StateIndexEntry{URI: "mlx://a", TokenStart: 0, TokenCount: 0}), want: errStateIndexEntryTokenCount},
		{name: "entry exceeds bundle", index: base(StateIndexEntry{URI: "mlx://a", TokenStart: 0, TokenCount: 99}), want: errStateIndexEntryExceedsBundle},
		{name: "negative byte start", index: base(StateIndexEntry{URI: "mlx://a", TokenStart: 0, TokenCount: 1, ByteStart: -1}), want: errStateIndexEntryByteSpan},
		{name: "negative byte count", index: base(StateIndexEntry{URI: "mlx://a", TokenStart: 0, TokenCount: 1, ByteCount: -1}), want: errStateIndexEntryByteSpan},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// validate(false) skips the hash check so each guard above is the
			// first failure reached.
			if err := tc.index.validate(false); !core.Is(err, tc.want) {
				t.Fatalf("validate() error = %v, want %v", err, tc.want)
			}
		})
	}
}

// TestStateIndex_validate_Ugly_LargeIndexPooledMapPath drives the pooled
// membership-set branch in validate that only engages above
// validateLinearScanThreshold (32). The first index has 40 distinct-URI
// entries and must validate clean; the second injects a duplicate URI into
// the large set to hit the map-hit duplicate-return inside that same branch.
func TestStateIndex_validate_Ugly_LargeIndexPooledMapPath(t *testing.T) {
	const n = validateLinearScanThreshold + 8 // 40, comfortably over the threshold
	makeLarge := func(dupURI bool) *StateIndex {
		entries := make([]StateIndexEntry, 0, n)
		for i := range n {
			uri := "mlx://span/" + strconv.Itoa(i)
			if dupURI && i == n-1 {
				uri = "mlx://span/0" // collide with the first entry
			}
			entries = append(entries, StateIndexEntry{URI: uri, TokenStart: 0, TokenCount: 1})
		}
		return &StateIndex{Version: 1, Kind: StateIndexKind, BundleURI: "mlx://b", TokenCount: 4, Entries: entries}
	}

	if err := makeLarge(false).validate(false); err != nil {
		t.Fatalf("validate(large distinct) error = %v, want nil", err)
	}
	if err := makeLarge(true).validate(false); !core.Is(err, errStateIndexDuplicateURI) {
		t.Fatalf("validate(large with dup) error = %v, want errStateIndexDuplicateURI", err)
	}
}

// TestStateIndex_Entry_Bad_NilReceiver and RequiredContextLength on a nil
// receiver take the early-return guards no constructed index reaches.
func TestStateIndex_Entry_Bad_NilReceiver(t *testing.T) {
	var index *StateIndex
	if _, ok := index.Entry("mlx://anything"); ok {
		t.Fatal("Entry(nil receiver) ok = true, want false")
	}
	if got := index.RequiredContextLength(); got != 0 {
		t.Fatalf("RequiredContextLength(nil receiver) = %d, want 0", got)
	}
}

// TestModelHashComparable_Bad_MissingFields covers every false-return in
// modelHashComparable through its public caller CheckStateIndexCompatibility:
// when the index carries a model hash but the runtime ModelInfo is missing a
// field the hash was computed over, the hashes are not comparable and the
// model-hash mismatch check is skipped (so compatibility passes despite
// differing hashes). Each case zeroes exactly one comparable field.
func TestModelHashComparable_Bad_MissingFields(t *testing.T) {
	// Hash-bearing index with no Name/Path so the modelHashComparable gate in
	// CheckStateIndexCompatibility is reached. Architecture is left empty so
	// the architecture mismatch guard never short-circuits these cases.
	hashIndex := func() *StateIndex {
		idx := &StateIndex{
			Version:    1,
			Kind:       StateIndexKind,
			BundleURI:  "mlx://b",
			TokenCount: 4,
			Model:      pkgbundle.Model{VocabSize: 1000, NumLayers: 2, QuantBits: 4, ContextLength: 8},
			Entries:    []StateIndexEntry{{URI: "mlx://a", TokenStart: 0, TokenCount: 1}},
		}
		idx.Model.Hash = "model-hash-that-differs-from-runtime"
		idx.Hash = indexHash(idx)
		return idx
	}
	for _, tc := range []struct {
		name string
		info memory.ModelInfo
	}{
		// Each info matches the layer/quant guards (so they pass) but zeroes a
		// single field the index model sets, tripping a not-comparable return.
		{name: "missing vocab", info: memory.ModelInfo{VocabSize: 0, NumLayers: 2, QuantBits: 4, ContextLength: 8}},
		{name: "missing layers", info: memory.ModelInfo{VocabSize: 1000, NumLayers: 0, QuantBits: 4, ContextLength: 8}},
		{name: "missing quant", info: memory.ModelInfo{VocabSize: 1000, NumLayers: 2, QuantBits: 0, ContextLength: 8}},
		{name: "missing context", info: memory.ModelInfo{VocabSize: 1000, NumLayers: 2, QuantBits: 4, ContextLength: 0}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Not comparable → model-hash check skipped → compatibility passes
			// even though the stored hash differs from any runtime hash.
			if err := CheckStateIndexCompatibility(tc.info, pkgbundle.Tokenizer{}, hashIndex()); err != nil {
				t.Fatalf("CheckStateIndexCompatibility() error = %v, want nil (hash not comparable, check skipped)", err)
			}
		})
	}

	// Architecture branch: an index model that names an architecture against a
	// runtime ModelInfo that does not. CheckStateIndexCompatibility's own
	// architecture guard only fires when BOTH are set, so an empty runtime
	// architecture slips through to modelHashComparable, which returns
	// not-comparable — the model-hash check is again skipped.
	t.Run("missing architecture", func(t *testing.T) {
		idx := &StateIndex{
			Version:    1,
			Kind:       StateIndexKind,
			BundleURI:  "mlx://b",
			TokenCount: 4,
			Model:      pkgbundle.Model{Architecture: "gemma4_text"},
			Entries:    []StateIndexEntry{{URI: "mlx://a", TokenStart: 0, TokenCount: 1}},
		}
		idx.Model.Hash = "model-hash-that-differs-from-runtime"
		idx.Hash = indexHash(idx)
		if err := CheckStateIndexCompatibility(memory.ModelInfo{}, pkgbundle.Tokenizer{}, idx); err != nil {
			t.Fatalf("CheckStateIndexCompatibility(no runtime arch) error = %v, want nil", err)
		}
	})
}

// TestIndexHashEquals_Ugly_NonHexExpected forces the hex.Decode failure
// branch in both indexHashEquals and indexEntryHashEquals: a 64-character
// string that passes the length gate but is not valid hex. Validate surfaces
// it as a hash mismatch rather than panicking on the decode error.
func TestIndexHashEquals_Ugly_NonHexExpected(t *testing.T) {
	nonHex := strings.Repeat("g", 64) // correct length, invalid hex digits

	index := &StateIndex{
		Version:    1,
		Kind:       StateIndexKind,
		BundleURI:  "mlx://b",
		TokenCount: 4,
		Entries:    []StateIndexEntry{{URI: "mlx://a", TokenStart: 0, TokenCount: 1}},
	}
	if indexHashEquals(index, nonHex) {
		t.Fatal("indexHashEquals(non-hex) = true, want false on decode error")
	}
	entry := &index.Entries[0]
	if indexEntryHashEquals(entry, nonHex) {
		t.Fatal("indexEntryHashEquals(non-hex) = true, want false on decode error")
	}

	// Through Validate: a stored hash of correct length but non-hex content
	// must be rejected, exercising the same decode-failure branch on the
	// check-hashes path.
	index.Hash = nonHex
	if err := index.Validate(); !core.Is(err, errStateIndexHashMismatch) {
		t.Fatalf("Validate(non-hex index hash) error = %v, want errStateIndexHashMismatch", err)
	}
}

// TestIndexHash_Bad_NilIndex covers the nil-index early returns of indexHash
// and indexHashBytes (the empty-string contract the hash wrappers rely on).
func TestIndexHash_Bad_NilIndex(t *testing.T) {
	if got := indexHash(nil); got != "" {
		t.Fatalf("indexHash(nil) = %q, want empty string", got)
	}
}

// TestStateBlockRefsSorted_Ugly_EqualStartDescendingIndex covers the
// tie-break false-return in stateBlockRefsSortedByTokenStart: two blocks with
// the same TokenStart but a descending Index are NOT considered sorted, so
// NewStateIndex falls back to the linear byte-span fill. The derived span
// must still be correct.
func TestStateBlockRefsSorted_Ugly_EqualStartDescendingIndex(t *testing.T) {
	if stateBlockRefsSortedByTokenStart([]kv.StateBlockRef{
		{Index: 1, TokenStart: 0, TokenCount: 2},
		{Index: 0, TokenStart: 0, TokenCount: 2}, // same start, lower index after — not sorted
	}) {
		t.Fatal("stateBlockRefsSortedByTokenStart(equal start, descending index) = true, want false")
	}
}

func kvSnapshotIndexTestBundle() *kv.MemvidBlockBundle {
	return &kv.MemvidBlockBundle{
		Version:      kv.MemvidBlockVersion,
		Kind:         kv.MemvidBlockBundleKind,
		SnapshotHash: "snapshot",
		KVEncoding:   kv.EncodingNative,
		Architecture: "gemma4_text",
		TokenCount:   4,
		TokenOffset:  4,
		BlockSize:    2,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       4,
		HeadDim:      2,
		Blocks: []kv.MemvidBlockRef{{
			Index:      0,
			TokenStart: 0,
			TokenCount: 2,
			Memvid:     memvid.ChunkRef{ChunkID: 1},
		}},
	}
}

type indexRecordingMemvidStore struct {
	store        memvid.Store
	resolved     []int
	resolvedURIs []string
}

func (s *indexRecordingMemvidStore) Get(ctx context.Context, chunkID int) (string, error) {
	s.resolved = append(s.resolved, chunkID)
	return s.store.Get(ctx, chunkID)
}

func (s *indexRecordingMemvidStore) Resolve(ctx context.Context, chunkID int) (memvid.Chunk, error) {
	s.resolved = append(s.resolved, chunkID)
	return memvid.Resolve(ctx, s.store, chunkID)
}

func (s *indexRecordingMemvidStore) ResolveBytes(ctx context.Context, chunkID int) (memvid.Chunk, error) {
	s.resolved = append(s.resolved, chunkID)
	return memvid.ResolveBytes(ctx, s.store, chunkID)
}

func (s *indexRecordingMemvidStore) ResolveURI(ctx context.Context, uri string) (memvid.Chunk, error) {
	s.resolvedURIs = append(s.resolvedURIs, uri)
	return memvid.ResolveURI(ctx, s.store, uri)
}

// =====================================================================
// Canonical AX-7 triplets (Test<Index>_<Symbol>_{Good,Bad,Ugly}).
//
// The scenario tests above (TestKVSnapshot*) exercise the same symbols
// through end-to-end flows; these per-symbol triplets pin each public
// symbol to its own Good/Bad/Ugly cases for the file-aware audit. The
// deprecated Memvid wrappers each get their own triplet (they forward
// to the State versions, so the triplet proves the forward holds).
// =====================================================================

// indexTestStoreBundle saves the shared 4-token synthetic snapshot into an
// in-memory State store under bundleURI and returns the store plus the
// resulting block bundle, so the index triplets that need a real saved
// bundle (Save/Load round-trips, prefix loads) have one without touching
// Metal or a model file.
func indexTestStoreBundle(t *testing.T, bundleURI string) (memvid.Store, *kv.StateBlockBundle) {
	t.Helper()
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	blk, err := snapshot.SaveStateBlocks(ctx, store, kv.StateBlockOptions{BlockSize: 2, KVEncoding: kv.EncodingNative})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}
	if _, err := kv.SaveStateBlockBundle(ctx, store, blk, bundleURI); err != nil {
		t.Fatalf("SaveStateBlockBundle() error = %v", err)
	}
	return store, blk
}

// --- NewStateIndex --------------------------------------------------------

func TestIndex_NewStateIndex_Good(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Title:     "full book",
		Model:     "demo",
		Entries: []StateIndexEntry{
			{URI: "mlx://book/chapter-1", Title: "Chapter 1", TokenStart: 0, TokenCount: 2},
			{URI: "mlx://book/chapter-2", Title: "Chapter 2", TokenStart: 2, TokenCount: 2},
		},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if index.Kind != StateIndexKind || index.Version != KVSnapshotStateBundleIndexVersion {
		t.Fatalf("index kind/version = %q/%d, want canonical values", index.Kind, index.Version)
	}
	if len(index.Entries) != 2 || index.Hash == "" {
		t.Fatalf("index entries/hash = %d/%q, want two entries and a hash", len(index.Entries), index.Hash)
	}
	if index.RequiredContextLength() != 4 {
		t.Fatalf("required context = %d, want 4", index.RequiredContextLength())
	}
}

func TestIndex_NewStateIndex_Bad(t *testing.T) {
	// A nil bundle fails the up-front ValidateStateBlockBundle guard before
	// any field is read.
	if _, err := NewStateIndex(nil, StateIndexOptions{BundleURI: "mlx://book/bundle"}); err == nil {
		t.Fatal("NewStateIndex(nil bundle) error = nil")
	}
	// A non-nil but invalid bundle (zero version, no blocks) is rejected too.
	if _, err := NewStateIndex(&kv.StateBlockBundle{}, StateIndexOptions{BundleURI: "mlx://book/bundle"}); err == nil {
		t.Fatal("NewStateIndex(invalid bundle) error = nil")
	}
}

func TestIndex_NewStateIndex_Ugly(t *testing.T) {
	// A supplied entry hash that does not match the canonical recomputation
	// is rejected with errStateIndexEntryHashMismatch — the construction
	// path recomputes and compares rather than trusting the caller's hash.
	blk := kvSnapshotIndexTestBundle()
	if _, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://book/bundle",
		Entries: []StateIndexEntry{
			{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2, Hash: "deadbeef"},
		},
	}); !core.Is(err, errStateIndexEntryHashMismatch) {
		t.Fatalf("NewStateIndex(bad entry hash) error = %v, want errStateIndexEntryHashMismatch", err)
	}
	// No explicit entries → a single full-bundle entry is synthesised
	// covering every token, with the bundle URI as the default entry URI.
	index, err := NewStateIndex(blk, StateIndexOptions{BundleURI: "mlx://book/bundle"})
	if err != nil {
		t.Fatalf("NewStateIndex(default entry) error = %v", err)
	}
	if len(index.Entries) != 1 || index.Entries[0].TokenCount != blk.TokenCount || index.Entries[0].URI != "mlx://book/bundle" {
		t.Fatalf("default entry = %+v, want one full-bundle entry", index.Entries)
	}
}

// --- NewMemvidIndex (deprecated wrapper) ----------------------------------

func TestIndex_NewMemvidIndex_Good(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{BundleURI: "mlx://bundle"})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	// Forwards to NewStateIndex, so the synthesised full-bundle entry and the
	// canonical kind match the State constructor's output exactly.
	if index.Kind != MemvidIndexKind || len(index.Entries) != 1 || index.Entries[0].TokenCount != blk.TokenCount {
		t.Fatalf("NewMemvidIndex result = %+v, want full-bundle entry with canonical kind", index)
	}
}

func TestIndex_NewMemvidIndex_Bad(t *testing.T) {
	if _, err := NewMemvidIndex(nil, MemvidIndexOptions{BundleURI: "mlx://bundle"}); err == nil {
		t.Fatal("NewMemvidIndex(nil bundle) error = nil")
	}
}

func TestIndex_NewMemvidIndex_Ugly(t *testing.T) {
	// The wrapper produces a byte-identical hash to NewStateIndex over the
	// same inputs — proving it is a pure alias and not a divergent path.
	blk := kvSnapshotIndexTestBundle()
	opts := MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []MemvidIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	}
	viaMemvid, err := NewMemvidIndex(blk, opts)
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	viaState, err := NewStateIndex(blk, StateIndexOptions(opts))
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if viaMemvid.Hash != viaState.Hash {
		t.Fatalf("hash divergence: memvid %q vs state %q", viaMemvid.Hash, viaState.Hash)
	}
}

// --- SaveStateIndex -------------------------------------------------------

func TestIndex_SaveStateIndex_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if _, err := SaveStateIndex(ctx, store, index, "mlx://index"); err != nil {
		t.Fatalf("SaveStateIndex() error = %v", err)
	}
	// The save persisted the index under its URI: reload proves it round-trips
	// (a failed Put would surface here as a resolve error) and the recovered
	// entry span matches what was written.
	loaded, err := LoadStateIndex(ctx, store, "mlx://index")
	if err != nil {
		t.Fatalf("reload after SaveStateIndex error = %v", err)
	}
	if loaded.Entries[0].URI != "mlx://chapter" || loaded.Hash != index.Hash {
		t.Fatalf("reloaded index = %+v, want round-trip equal to saved", loaded)
	}
}

func TestIndex_SaveStateIndex_Bad(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if _, err := SaveStateIndex(ctx, nil, index, "mlx://index"); !core.Is(err, errStateStoreNil) {
		t.Fatalf("SaveStateIndex(nil store) error = %v, want errStateStoreNil", err)
	}
	if _, err := SaveStateIndex(ctx, store, index, "  "); !core.Is(err, errStateIndexURIRequired) {
		t.Fatalf("SaveStateIndex(blank URI) error = %v, want errStateIndexURIRequired", err)
	}
}

func TestIndex_SaveStateIndex_Ugly(t *testing.T) {
	// An index that fails validation (tampered kind) is rejected by Save
	// before any store write happens — Save validates first.
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	index.Kind = "tampered"
	if _, err := SaveStateIndex(ctx, store, index, "mlx://index"); !core.Is(err, errStateIndexInvalidKind) {
		t.Fatalf("SaveStateIndex(invalid index) error = %v, want errStateIndexInvalidKind", err)
	}
}

// --- SaveMemvidIndex (deprecated wrapper) ---------------------------------

func TestIndex_SaveMemvidIndex_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []MemvidIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	if _, err := SaveMemvidIndex(ctx, store, index, "mlx://index"); err != nil {
		t.Fatalf("SaveMemvidIndex() error = %v", err)
	}
	if _, err := LoadMemvidIndex(ctx, store, "mlx://index"); err != nil {
		t.Fatalf("reload after SaveMemvidIndex error = %v", err)
	}
}

func TestIndex_SaveMemvidIndex_Bad(t *testing.T) {
	ctx := context.Background()
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []MemvidIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	if _, err := SaveMemvidIndex(ctx, nil, index, "mlx://index"); !core.Is(err, errStateStoreNil) {
		t.Fatalf("SaveMemvidIndex(nil store) error = %v, want errStateStoreNil", err)
	}
}

func TestIndex_SaveMemvidIndex_Ugly(t *testing.T) {
	// Blank URI is rejected by the forwarded SaveStateIndex guard.
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []MemvidIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	if _, err := SaveMemvidIndex(ctx, store, index, ""); !core.Is(err, errStateIndexURIRequired) {
		t.Fatalf("SaveMemvidIndex(empty URI) error = %v, want errStateIndexURIRequired", err)
	}
}

// --- LoadStateIndex -------------------------------------------------------

func TestIndex_LoadStateIndex_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://chapter", Title: "Chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if _, err := SaveStateIndex(ctx, store, index, "mlx://index"); err != nil {
		t.Fatalf("SaveStateIndex() error = %v", err)
	}
	loaded, err := LoadStateIndex(ctx, store, "mlx://index")
	if err != nil {
		t.Fatalf("LoadStateIndex() error = %v", err)
	}
	if loaded.Hash != index.Hash || loaded.Entries[0].URI != "mlx://chapter" {
		t.Fatalf("loaded = %+v, want round-trip equal to saved index", loaded)
	}
}

func TestIndex_LoadStateIndex_Bad(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	if _, err := LoadStateIndex(ctx, nil, "mlx://index"); !core.Is(err, errStateStoreNil) {
		t.Fatalf("LoadStateIndex(nil store) error = %v, want errStateStoreNil", err)
	}
	if _, err := LoadStateIndex(ctx, store, ""); !core.Is(err, errStateIndexURIRequired) {
		t.Fatalf("LoadStateIndex(empty URI) error = %v, want errStateIndexURIRequired", err)
	}
}

func TestIndex_LoadStateIndex_Ugly(t *testing.T) {
	// A stored payload that parses as JSON but fails index validation (here a
	// header-only doc with no entries) is rejected by the post-parse Validate
	// rather than returned as a half-built index.
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	corrupt := core.JSONMarshalString(map[string]any{"version": 1, "kind": StateIndexKind})
	if _, err := store.Put(ctx, corrupt, memvid.PutOptions{URI: "mlx://bad-index"}); err != nil {
		t.Fatalf("write corrupt index: %v", err)
	}
	if _, err := LoadStateIndex(ctx, store, "mlx://bad-index"); err == nil {
		t.Fatal("LoadStateIndex(header-only) error = nil, want validation failure")
	}
}

// --- LoadMemvidIndex (deprecated wrapper) ---------------------------------

func TestIndex_LoadMemvidIndex_Good(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []MemvidIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	if _, err := SaveMemvidIndex(ctx, store, index, "mlx://index"); err != nil {
		t.Fatalf("SaveMemvidIndex() error = %v", err)
	}
	loaded, err := LoadMemvidIndex(ctx, store, "mlx://index")
	if err != nil {
		t.Fatalf("LoadMemvidIndex() error = %v", err)
	}
	if loaded.Hash != index.Hash {
		t.Fatalf("loaded hash = %q, want %q", loaded.Hash, index.Hash)
	}
}

func TestIndex_LoadMemvidIndex_Bad(t *testing.T) {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	if _, err := LoadMemvidIndex(ctx, nil, "mlx://index"); !core.Is(err, errStateStoreNil) {
		t.Fatalf("LoadMemvidIndex(nil store) error = %v, want errStateStoreNil", err)
	}
	if _, err := LoadMemvidIndex(ctx, store, "   "); !core.Is(err, errStateIndexURIRequired) {
		t.Fatalf("LoadMemvidIndex(blank URI) error = %v, want errStateIndexURIRequired", err)
	}
}

func TestIndex_LoadMemvidIndex_Ugly(t *testing.T) {
	// Resolving a URI that was never stored surfaces the resolve error
	// through the forwarded LoadStateIndex.
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	if _, err := LoadMemvidIndex(ctx, store, "mlx://never-written"); err == nil {
		t.Fatal("LoadMemvidIndex(missing URI) error = nil, want resolve failure")
	}
}

// --- LoadPrefixFromStateIndex ---------------------------------------------

func TestIndex_LoadPrefixFromStateIndex_Good(t *testing.T) {
	const bundleURI = "mlx://book/bundle"
	store, blk := indexTestStoreBundle(t, bundleURI)
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: bundleURI,
		Entries:   []StateIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	prefix, entry, err := LoadPrefixFromStateIndex(context.Background(), store, index, "mlx://book/chapter-1", kv.LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadPrefixFromStateIndex() error = %v", err)
	}
	if entry.URI != "mlx://book/chapter-1" || entry.PrefixTokens() != 2 {
		t.Fatalf("entry = %+v, want chapter-1 two-token prefix", entry)
	}
	if len(prefix.Tokens) != 2 || prefix.Tokens[0] != 1 || prefix.Tokens[1] != 2 {
		t.Fatalf("prefix tokens = %v, want first two synthetic tokens", prefix.Tokens)
	}
}

func TestIndex_LoadPrefixFromStateIndex_Bad(t *testing.T) {
	const bundleURI = "mlx://book/bundle"
	store, blk := indexTestStoreBundle(t, bundleURI)
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: bundleURI,
		Entries:   []StateIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if _, _, err := LoadPrefixFromStateIndex(context.Background(), nil, index, "mlx://book/chapter-1", kv.LoadOptions{}); !core.Is(err, errStateStoreNil) {
		t.Fatalf("LoadPrefixFromStateIndex(nil store) error = %v, want errStateStoreNil", err)
	}
	if _, _, err := LoadPrefixFromStateIndex(context.Background(), store, index, "mlx://book/missing", kv.LoadOptions{}); !core.Is(err, errStateIndexEntryNotFound) {
		t.Fatalf("LoadPrefixFromStateIndex(missing entry) error = %v, want errStateIndexEntryNotFound", err)
	}
}

func TestIndex_LoadPrefixFromStateIndex_Ugly(t *testing.T) {
	// The entry resolves but its referenced bundle was never saved, so the
	// bundle load fails after the entry lookup succeeds — a different failure
	// point from the missing-entry and nil-store cases.
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://book/unsaved-bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	store := memvid.NewInMemoryStore(nil) // empty: bundle URI never written
	if _, _, err := LoadPrefixFromStateIndex(context.Background(), store, index, "mlx://book/chapter-1", kv.LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromStateIndex(unsaved bundle) error = nil, want bundle load failure")
	}
}

// --- LoadPrefixFromMemvidIndex (deprecated wrapper) -----------------------

func TestIndex_LoadPrefixFromMemvidIndex_Good(t *testing.T) {
	const bundleURI = "mlx://book/bundle"
	store, blk := indexTestStoreBundle(t, bundleURI)
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: bundleURI,
		Entries:   []MemvidIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	prefix, entry, err := LoadPrefixFromMemvidIndex(context.Background(), store, index, "mlx://book/chapter-1", kv.LoadOptions{RawKVOnly: true})
	if err != nil {
		t.Fatalf("LoadPrefixFromMemvidIndex() error = %v", err)
	}
	if entry.URI != "mlx://book/chapter-1" || len(prefix.Tokens) != 2 {
		t.Fatalf("entry/prefix = %+v/%v, want chapter-1 two-token prefix", entry, prefix.Tokens)
	}
}

func TestIndex_LoadPrefixFromMemvidIndex_Bad(t *testing.T) {
	const bundleURI = "mlx://book/bundle"
	store, blk := indexTestStoreBundle(t, bundleURI)
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: bundleURI,
		Entries:   []MemvidIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	if _, _, err := LoadPrefixFromMemvidIndex(context.Background(), nil, index, "mlx://book/chapter-1", kv.LoadOptions{}); !core.Is(err, errStateStoreNil) {
		t.Fatalf("LoadPrefixFromMemvidIndex(nil store) error = %v, want errStateStoreNil", err)
	}
	if _, _, err := LoadPrefixFromMemvidIndex(context.Background(), store, index, "mlx://book/missing", kv.LoadOptions{}); !core.Is(err, errStateIndexEntryNotFound) {
		t.Fatalf("LoadPrefixFromMemvidIndex(missing entry) error = %v, want errStateIndexEntryNotFound", err)
	}
}

func TestIndex_LoadPrefixFromMemvidIndex_Ugly(t *testing.T) {
	// Entry resolves but the bundle URI was never written → bundle load fails
	// after a successful entry lookup, exercising the forwarded path's later
	// failure point.
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://book/unsaved-bundle",
		Entries:   []MemvidIndexEntry{{URI: "mlx://book/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	store := memvid.NewInMemoryStore(nil)
	if _, _, err := LoadPrefixFromMemvidIndex(context.Background(), store, index, "mlx://book/chapter-1", kv.LoadOptions{}); err == nil {
		t.Fatal("LoadPrefixFromMemvidIndex(unsaved bundle) error = nil, want bundle load failure")
	}
}

// --- CheckStateIndexCompatibility -----------------------------------------

func TestIndex_CheckStateIndexCompatibility_Good(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	info := memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	tok := pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: info,
		Tokenizer: tok,
		Entries:   []StateIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if err := CheckStateIndexCompatibility(info, tok, index); err != nil {
		t.Fatalf("CheckStateIndexCompatibility(matching) error = %v, want nil", err)
	}
}

func TestIndex_CheckStateIndexCompatibility_Bad(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4},
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a"},
		Entries:   []StateIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 1}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if err := CheckStateIndexCompatibility(memory.ModelInfo{Architecture: "qwen3", NumLayers: 2, QuantBits: 4, ContextLength: 4}, pkgbundle.Tokenizer{Hash: "tok-a"}, index); !core.Is(err, errStateIndexArchitectureMismatch) {
		t.Fatalf("arch mismatch error = %v, want errStateIndexArchitectureMismatch", err)
	}
	if err := CheckStateIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4}, pkgbundle.Tokenizer{Hash: "tok-b"}, index); !core.Is(err, errStateIndexTokenizerMismatch) {
		t.Fatalf("tokenizer mismatch error = %v, want errStateIndexTokenizerMismatch", err)
	}
}

func TestIndex_CheckStateIndexCompatibility_Ugly(t *testing.T) {
	// Zero-valued runtime fields disable the corresponding guard rather than
	// being treated as mismatches: an index that names an architecture,
	// layers, quant, and context still passes against an all-zero ModelInfo
	// and empty tokenizer, because every comparison is gated on BOTH sides
	// being set. A nil index, by contrast, fails its up-front Validate.
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4},
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		Entries:   []StateIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 1}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if err := CheckStateIndexCompatibility(memory.ModelInfo{}, pkgbundle.Tokenizer{}, index); err != nil {
		t.Fatalf("CheckStateIndexCompatibility(all-zero runtime) error = %v, want nil", err)
	}
	if err := CheckStateIndexCompatibility(memory.ModelInfo{}, pkgbundle.Tokenizer{}, nil); !core.Is(err, errStateIndexNil) {
		t.Fatalf("CheckStateIndexCompatibility(nil index) error = %v, want errStateIndexNil", err)
	}
}

// --- CheckMemvidIndexCompatibility (deprecated wrapper) -------------------

func TestIndex_CheckMemvidIndexCompatibility_Good(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	info := memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	tok := pkgbundle.Tokenizer{Hash: "tok-a"}
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: info,
		Tokenizer: tok,
		Entries:   []MemvidIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	if err := CheckMemvidIndexCompatibility(info, tok, index); err != nil {
		t.Fatalf("CheckMemvidIndexCompatibility(matching) error = %v, want nil", err)
	}
}

func TestIndex_CheckMemvidIndexCompatibility_Bad(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4},
		Entries:   []MemvidIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 1}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 4}, pkgbundle.Tokenizer{}, index); !core.Is(err, errStateIndexLayerMismatch) {
		t.Fatalf("layer mismatch error = %v, want errStateIndexLayerMismatch", err)
	}
}

func TestIndex_CheckMemvidIndexCompatibility_Ugly(t *testing.T) {
	// Quantisation mismatch fires only when both sides set QuantBits; a
	// zero runtime QuantBits disables the check. The wrapper forwards to
	// CheckStateIndexCompatibility, so both behaviours hold identically.
	blk := kvSnapshotIndexTestBundle()
	index, err := NewMemvidIndex(blk, MemvidIndexOptions{
		BundleURI: "mlx://bundle",
		ModelInfo: memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 4, ContextLength: 4},
		Entries:   []MemvidIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 1}},
	})
	if err != nil {
		t.Fatalf("NewMemvidIndex() error = %v", err)
	}
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 8, ContextLength: 4}, pkgbundle.Tokenizer{}, index); !core.Is(err, errStateIndexQuantMismatch) {
		t.Fatalf("quant mismatch error = %v, want errStateIndexQuantMismatch", err)
	}
	if err := CheckMemvidIndexCompatibility(memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 2, QuantBits: 0, ContextLength: 4}, pkgbundle.Tokenizer{}, index); err != nil {
		t.Fatalf("CheckMemvidIndexCompatibility(zero quant) error = %v, want nil (check disabled)", err)
	}
}

// --- (*StateIndex) Validate -----------------------------------------------

func TestIndex_StateIndex_Validate_Good(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	// A freshly constructed index — with its hash computed by the
	// constructor — validates including the hash check.
	if err := index.Validate(); err != nil {
		t.Fatalf("Validate() error = %v, want nil for well-formed index", err)
	}
}

func TestIndex_StateIndex_Validate_Bad(t *testing.T) {
	// A literal index with a wrong kind never passes the schema guard. Built
	// directly (not via NewStateIndex, which would reject it) so the specific
	// sentinel is reached.
	index := &StateIndex{
		Version:    KVSnapshotStateBundleIndexVersion,
		Kind:       "not-an-index",
		BundleURI:  "mlx://bundle",
		TokenCount: 4,
		Entries:    []StateIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 1}},
	}
	if err := index.Validate(); !core.Is(err, errStateIndexInvalidKind) {
		t.Fatalf("Validate(wrong kind) error = %v, want errStateIndexInvalidKind", err)
	}
}

func TestIndex_StateIndex_Validate_Ugly(t *testing.T) {
	// A constructor-built index whose stored hash is then tampered with must
	// fail the check-hashes tail of Validate (the constructor's hash no
	// longer matches the recomputation). This is distinct from the
	// schema-level Bad case: every field is structurally valid; only the
	// integrity hash is wrong.
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	index.Hash = "0000000000000000000000000000000000000000000000000000000000000000"
	if err := index.Validate(); !core.Is(err, errStateIndexHashMismatch) {
		t.Fatalf("Validate(tampered hash) error = %v, want errStateIndexHashMismatch", err)
	}
}

// --- (*StateIndex) Entry --------------------------------------------------

func TestIndex_StateIndex_Entry_Good(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://chapter", Title: "Chapter", TokenStart: 0, TokenCount: 2, Labels: []string{"chapter"}}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	entry, ok := index.Entry("mlx://chapter")
	if !ok {
		t.Fatal("Entry(mlx://chapter) ok = false, want true")
	}
	if entry.Title != "Chapter" || entry.TokenCount != 2 {
		t.Fatalf("entry = %+v, want the named chapter span", entry)
	}
	// The returned entry is a defensive copy: mutating its Labels must not
	// touch the index's stored entry.
	entry.Labels[0] = "mutated"
	again, _ := index.Entry("mlx://chapter")
	if again.Labels[0] != "chapter" {
		t.Fatalf("index entry mutated through returned copy: %q", again.Labels[0])
	}
}

func TestIndex_StateIndex_Entry_Bad(t *testing.T) {
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		Entries:   []StateIndexEntry{{URI: "mlx://chapter", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if entry, ok := index.Entry("mlx://does-not-exist"); ok || entry.URI != "" {
		t.Fatalf("Entry(unknown) = %+v/%v, want zero entry and false", entry, ok)
	}
}

func TestIndex_StateIndex_Entry_Ugly(t *testing.T) {
	// A nil receiver returns the zero entry and false rather than panicking —
	// the early-return guard no constructed index reaches.
	var index *StateIndex
	if entry, ok := index.Entry("mlx://anything"); ok || entry.URI != "" {
		t.Fatalf("Entry(nil receiver) = %+v/%v, want zero entry and false", entry, ok)
	}
}

// --- (*StateIndex) RequiredContextLength ----------------------------------

func TestIndex_StateIndex_RequiredContextLength_Good(t *testing.T) {
	// Two spans, the longer ending at token 4: required context is the
	// largest PrefixTokens across all entries.
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: "mlx://bundle",
		Entries: []StateIndexEntry{
			{URI: "mlx://chapter-1", TokenStart: 0, TokenCount: 2},
			{URI: "mlx://chapter-2", TokenStart: 2, TokenCount: 2},
		},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if got := index.RequiredContextLength(); got != 4 {
		t.Fatalf("RequiredContextLength(two spans) = %d, want 4", got)
	}
}

func TestIndex_StateIndex_RequiredContextLength_Bad(t *testing.T) {
	// A nil receiver reports 0 rather than panicking.
	var index *StateIndex
	if got := index.RequiredContextLength(); got != 0 {
		t.Fatalf("RequiredContextLength(nil receiver) = %d, want 0", got)
	}
}

func TestIndex_StateIndex_RequiredContextLength_Ugly(t *testing.T) {
	// A single full-bundle span: required context equals the whole token
	// count — a different (boundary) input from the multi-span Good case.
	blk := kvSnapshotIndexTestBundle()
	index, err := NewStateIndex(blk, StateIndexOptions{BundleURI: "mlx://bundle"})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	if got := index.RequiredContextLength(); got != blk.TokenCount {
		t.Fatalf("RequiredContextLength(full bundle) = %d, want %d", got, blk.TokenCount)
	}
}

// --- (StateIndexEntry) PrefixTokens ---------------------------------------

func TestIndex_StateIndexEntry_PrefixTokens_Good(t *testing.T) {
	// A mid-bundle span starting at 2, length 2: the prefix that must be
	// restored runs through token 4.
	entry := StateIndexEntry{TokenStart: 2, TokenCount: 2}
	if got := entry.PrefixTokens(); got != 4 {
		t.Fatalf("PrefixTokens(2+2) = %d, want 4", got)
	}
}

func TestIndex_StateIndexEntry_PrefixTokens_Bad(t *testing.T) {
	// A zero-value entry (no span) needs a zero-length prefix — the
	// arithmetic floor, distinct from the populated Good input.
	entry := StateIndexEntry{}
	if got := entry.PrefixTokens(); got != 0 {
		t.Fatalf("PrefixTokens(zero entry) = %d, want 0", got)
	}
}

func TestIndex_StateIndexEntry_PrefixTokens_Ugly(t *testing.T) {
	// A leading span (start 0): the prefix is exactly the span length —
	// another distinct input that proves PrefixTokens is TokenStart+TokenCount
	// with no off-by-one at the bundle head.
	entry := StateIndexEntry{TokenStart: 0, TokenCount: 3}
	if got := entry.PrefixTokens(); got != 3 {
		t.Fatalf("PrefixTokens(0+3) = %d, want 3", got)
	}
}
