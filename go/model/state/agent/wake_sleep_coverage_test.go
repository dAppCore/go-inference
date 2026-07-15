// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/kv/memory"
	pkgbundle "dappco.re/go/inference/model/bundle"
	state "dappco.re/go/inference/model/state"
)

// --- LoadWakeSnapshot: block-load error after a successful plan ------------

// TestWakeSleep_LoadWakeSnapshot_Ugly_BlockLoadError drives the second error
// return of LoadWakeSnapshot: PlanWake succeeds (manifest + index resolve
// cleanly) but the block-chunk read fails, so LoadPrefixFromStateBlocks
// surfaces an error while plan/report were already built.
func TestWakeSleep_LoadWakeSnapshot_Ugly_BlockLoadError(t *testing.T) {
	const bundleURI = "mlx://agent/session-1/bundle"
	inner, blk := wakeSleepTestBundle(t, bundleURI)
	idx := planWakeIndex(t, blk, bundleURI)
	fault := newFaultStore(inner.(*state.InMemoryStore))
	fault.getErr = core.NewError("block read boom") // manifest resolves; block read fails
	snapshot, report, err := LoadWakeSnapshot(context.Background(), fault, WakeOptions{
		Index:       idx,
		EntryURI:    "mlx://agent/session-1/chapter-1",
		Tokenizer:   pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		LoadOptions: kv.LoadOptions{RawKVOnly: true},
	}, memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8})
	if err == nil {
		t.Fatal("LoadWakeSnapshot(block read error) error = nil, want block-load failure")
	}
	if snapshot != nil || report != nil {
		t.Fatalf("LoadWakeSnapshot(block read error) = %+v/%+v, want nil/nil on error", snapshot, report)
	}
}

// --- PlanWake: bundle-resolve error + invalid prefix ----------------------

// TestWakeSleep_PlanWake_Ugly_BundleResolveError makes the bundle manifest
// fail to resolve, hitting PlanWake's LoadStateBlockBundle error branch (the
// index itself is the in-memory opts.Index, so only the manifest URI faults).
func TestWakeSleep_PlanWake_Ugly_BundleResolveError(t *testing.T) {
	const bundleURI = "mlx://agent/session-1/bundle"
	inner, blk := wakeSleepTestBundle(t, bundleURI)
	idx := planWakeIndex(t, blk, bundleURI)
	fault := newFaultStore(inner.(*state.InMemoryStore))
	fault.resolveErr = core.NewError("manifest resolve boom")
	fault.resolveURIMatch = bundleURI // only the bundle manifest faults
	if _, err := PlanWake(context.Background(), fault, WakeOptions{
		Index:     idx,
		EntryURI:  "mlx://agent/session-1/chapter-1",
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
	}, memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}); err == nil {
		t.Fatal("PlanWake(bundle resolve error) error = nil, want manifest resolve failure")
	}
}

// TestWakeSleep_PlanWake_Ugly_InvalidPrefix drives PlanWake's
// prefix-out-of-range guard: the index claims a 4-token entry but the bundle
// actually stored at the entry's URI only holds 2 tokens, so prefixTokens (4)
// exceeds bundle.TokenCount (2).
func TestWakeSleep_PlanWake_Ugly_InvalidPrefix(t *testing.T) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	full := kvSnapshotBlocksTestSnapshot() // 4 tokens
	fullBlk, err := full.SaveStateBlocks(ctx, store, kv.StateBlockOptions{BlockSize: 2, KVEncoding: kv.EncodingNative})
	if err != nil {
		t.Fatalf("SaveStateBlocks(full) error = %v", err)
	}
	idx, err := NewStateIndex(fullBlk, StateIndexOptions{
		BundleURI: "mlx://agent/session-1/bundle",
		ModelInfo: memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		Entries:   []StateIndexEntry{{URI: "mlx://agent/session-1/all", TokenStart: 0, TokenCount: 4}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex(full) error = %v", err)
	}

	// Overwrite the bundle manifest with a shorter (2-token) bundle.
	short := shortTwoTokenSnapshot()
	shortBlk, err := short.SaveStateBlocks(ctx, store, kv.StateBlockOptions{BlockSize: 2, KVEncoding: kv.EncodingNative})
	if err != nil {
		t.Fatalf("SaveStateBlocks(short) error = %v", err)
	}
	if _, err := kv.SaveStateBlockBundle(ctx, store, shortBlk, "mlx://agent/session-1/bundle"); err != nil {
		t.Fatalf("SaveStateBlockBundle(short) error = %v", err)
	}

	if _, err := PlanWake(ctx, store, WakeOptions{
		Index:                  idx,
		EntryURI:               "mlx://agent/session-1/all",
		SkipCompatibilityCheck: true, // the entry exceeds the model context; skip the compat gate so the prefix guard is reached
	}, memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}); !core.Is(err, errStateIndexPrefixInvalid) {
		t.Fatalf("PlanWake(prefix > bundle) error = %v, want errStateIndexPrefixInvalid", err)
	}
}

// --- loadIndex: in-memory index validate error + IndexURI load path -------

// TestWakeSleep_PlanWake_Ugly_InMemoryIndexValidateError covers loadIndex's
// in-memory-index validation branch. That branch only validates opts.Index
// when mustValidate is true, and PlanWake passes SkipCompatibilityCheck AS
// mustValidate — so the validation only runs when SkipCompatibilityCheck is
// true. An invalid in-memory index then surfaces its validation error.
func TestWakeSleep_PlanWake_Ugly_InMemoryIndexValidateError(t *testing.T) {
	store, _ := wakeSleepTestBundle(t, "mlx://agent/session-1/bundle")
	bad := &StateIndex{Version: 1, Kind: StateIndexKind, BundleURI: "mlx://agent/session-1/bundle", TokenCount: 0}
	if _, err := PlanWake(context.Background(), store, WakeOptions{
		Index:                  bad,
		SkipCompatibilityCheck: true, // routes mustValidate=true into loadIndex
	}, memory.ModelInfo{}); !core.Is(err, errStateIndexEmptyTokenCount) {
		t.Fatalf("PlanWake(invalid in-memory index) error = %v, want errStateIndexEmptyTokenCount", err)
	}
}

// TestWakeSleep_PlanWake_Ugly_IndexURILoadPath covers loadIndex's
// LoadStateIndex branch — no in-memory opts.Index, an IndexURI that resolves
// to a stored index. This drives the final `return LoadStateIndex(...)` line
// that the opts.Index path never reaches.
func TestWakeSleep_PlanWake_Ugly_IndexURILoadPath(t *testing.T) {
	ctx := context.Background()
	const bundleURI = "mlx://agent/session-1/bundle"
	const indexURI = "mlx://agent/session-1/index"
	store, blk := wakeSleepTestBundle(t, bundleURI)
	idx := planWakeIndex(t, blk, bundleURI)
	if _, err := SaveStateIndex(ctx, store.(state.Writer), idx, indexURI); err != nil {
		t.Fatalf("SaveStateIndex() error = %v", err)
	}
	plan, err := PlanWake(ctx, store, WakeOptions{
		IndexURI:  indexURI, // no in-memory Index → loadIndex takes the LoadStateIndex path
		EntryURI:  "mlx://agent/session-1/chapter-1",
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
	}, memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8})
	if err != nil {
		t.Fatalf("PlanWake(IndexURI load path) error = %v", err)
	}
	if plan.Entry.URI != "mlx://agent/session-1/chapter-1" || plan.Report.IndexURI != indexURI {
		t.Fatalf("plan = entry %q index %q, want chapter-1 + %q", plan.Entry.URI, plan.Report.IndexURI, indexURI)
	}
}

// --- sleepEntryMeta: parent_index_uri allocates the meta map ---------------

// TestWakeSleep_sleepEntryMeta_Ugly_ParentIndexOnlyAllocatesMap covers the
// nil-map allocation branch guarded by ParentIndexURI. With no caller Meta and
// only a ParentIndexURI set, sleepEntryMeta must allocate the map there (the
// parent_entry/parent_bundle branches allocate it in the existing cases).
func TestWakeSleep_sleepEntryMeta_Ugly_ParentIndexOnlyAllocatesMap(t *testing.T) {
	meta := sleepEntryMeta(SleepOptions{ParentIndexURI: "mlx://agent/parent/index"})
	if meta == nil {
		t.Fatal("sleepEntryMeta(parent index only) = nil, want allocated map")
	}
	if meta["parent_index_uri"] != "mlx://agent/parent/index" {
		t.Fatalf("meta = %v, want parent_index_uri set", meta)
	}
	if _, ok := meta["parent_entry_uri"]; ok {
		t.Fatalf("meta = %v, want only parent_index_uri", meta)
	}
}

// --- blocksNeededForPrefix: nil bundle / non-positive prefix --------------

// TestWakeSleep_blocksNeededForPrefix_Bad_Guards covers the early-return guard
// (nil bundle or prefixTokens <= 0 → 0 blocks) directly.
func TestWakeSleep_blocksNeededForPrefix_Bad_Guards(t *testing.T) {
	if got := blocksNeededForPrefix(nil, 4); got != 0 {
		t.Fatalf("blocksNeededForPrefix(nil bundle) = %d, want 0", got)
	}
	blk := kvSnapshotIndexTestBundle()
	if got := blocksNeededForPrefix(blk, 0); got != 0 {
		t.Fatalf("blocksNeededForPrefix(zero prefix) = %d, want 0", got)
	}
	if got := blocksNeededForPrefix(blk, -1); got != 0 {
		t.Fatalf("blocksNeededForPrefix(negative prefix) = %d, want 0", got)
	}
}

// TestWakeSleep_blocksNeededForPrefix_Ugly_EarlyStartBreak drives the loop's
// "next block starts at/after the prefix" break (the early-exit distinct from
// the coverage-complete break). Blocks are (start 0,count 1) and (start 2,
// count 2); a prefix of 2 counts the first block (its span 0..1 does not yet
// cover the prefix, so the coverage break does not fire) then breaks on the
// second block because its TokenStart (2) is already >= the prefix (2).
func TestWakeSleep_blocksNeededForPrefix_Ugly_EarlyStartBreak(t *testing.T) {
	bundle := &kv.StateBlockBundle{
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 1},
			{Index: 1, TokenStart: 2, TokenCount: 2},
		},
	}
	if got := blocksNeededForPrefix(bundle, 2); got != 1 {
		t.Fatalf("blocksNeededForPrefix(start-break) = %d, want 1 (first block counted, second starts at prefix)", got)
	}
}
