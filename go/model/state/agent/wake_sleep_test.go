// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"context"
	"errors"
	"testing"

	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/kv/memory"
	pkgbundle "dappco.re/go/inference/model/bundle"
	state "dappco.re/go/inference/model/state"
)

// wakeSleepTestBundle saves the shared 4-token synthetic snapshot into an
// in-memory state store and returns the resulting block bundle plus the
// store, so the wake-path tests have a real (tiny) bundle to plan/load
// against without touching Metal or a model file.
func wakeSleepTestBundle(t *testing.T, bundleURI string) (state.Store, *kv.StateBlockBundle) {
	t.Helper()
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	blk, err := snapshot.SaveStateBlocks(ctx, store, kv.StateBlockOptions{
		BlockSize:  2,
		KVEncoding: kv.EncodingNative,
	})
	if err != nil {
		t.Fatalf("SaveStateBlocks() error = %v", err)
	}
	if _, err := kv.SaveStateBlockBundle(ctx, store, blk, bundleURI); err != nil {
		t.Fatalf("SaveStateBlockBundle() error = %v", err)
	}
	return store, blk
}

// --- SleepURIs ------------------------------------------------------------

func TestWakeSleep_SleepURIs_Good(t *testing.T) {
	entryURI, bundleURI, indexURI, err := SleepURIs(SleepOptions{
		EntryURI:  "mlx://agent/session-1",
		BundleURI: "mlx://agent/session-1/bundle",
		IndexURI:  "mlx://agent/session-1/index",
	})
	if err != nil {
		t.Fatalf("SleepURIs() error = %v", err)
	}
	if entryURI != "mlx://agent/session-1" || bundleURI != "mlx://agent/session-1/bundle" || indexURI != "mlx://agent/session-1/index" {
		t.Fatalf("SleepURIs() = %q/%q/%q, want all-set values preserved", entryURI, bundleURI, indexURI)
	}
}

func TestWakeSleep_SleepURIs_Bad(t *testing.T) {
	// Only EntryURI supplied: bundle/index must derive from it rather
	// than error. This is the closest thing to a "bad"/incomplete input
	// SleepURIs accepts — the trailing errStateURIRequired branch is
	// unreachable because EntryURI always defaults to a non-empty value
	// (see the skipped-sentinel note in the test report).
	entryURI, bundleURI, indexURI, err := SleepURIs(SleepOptions{EntryURI: "mlx://agent/only-entry"})
	if err != nil {
		t.Fatalf("SleepURIs(only entry) error = %v", err)
	}
	if entryURI != "mlx://agent/only-entry" {
		t.Fatalf("entryURI = %q, want supplied value", entryURI)
	}
	if bundleURI != "mlx://agent/only-entry/bundle" {
		t.Fatalf("bundleURI = %q, want derived /bundle suffix", bundleURI)
	}
	if indexURI != "mlx://agent/only-entry/index" {
		t.Fatalf("indexURI = %q, want derived /index suffix", indexURI)
	}
}

func TestWakeSleep_SleepURIs_Ugly(t *testing.T) {
	// Nothing set + whitespace-only entry: must fall through to the
	// default "mlx://state/latest" anchor and derive both children from
	// it. Trimming means a blank/space EntryURI is treated as empty.
	for _, entry := range []string{"", "   "} {
		entryURI, bundleURI, indexURI, err := SleepURIs(SleepOptions{EntryURI: entry})
		if err != nil {
			t.Fatalf("SleepURIs(%q) error = %v", entry, err)
		}
		if entryURI != "mlx://state/latest" {
			t.Fatalf("entryURI = %q, want default anchor for input %q", entryURI, entry)
		}
		if bundleURI != "mlx://state/latest/bundle" || indexURI != "mlx://state/latest/index" {
			t.Fatalf("derived = %q/%q, want children of the default anchor", bundleURI, indexURI)
		}
	}
	// Bundle-only set: entryURI must adopt the bundle URI, index derives
	// from it.
	entryURI, bundleURI, indexURI, err := SleepURIs(SleepOptions{BundleURI: "mlx://agent/bundle-only"})
	if err != nil {
		t.Fatalf("SleepURIs(bundle only) error = %v", err)
	}
	if entryURI != "mlx://agent/bundle-only" || bundleURI != "mlx://agent/bundle-only" || indexURI != "mlx://agent/bundle-only/index" {
		t.Fatalf("bundle-only = %q/%q/%q, want entry adopts bundle", entryURI, bundleURI, indexURI)
	}
	// Index-only set: entryURI adopts the index URI; bundle derives from
	// it.
	entryURI, bundleURI, _, err = SleepURIs(SleepOptions{IndexURI: "mlx://agent/index-only"})
	if err != nil {
		t.Fatalf("SleepURIs(index only) error = %v", err)
	}
	if entryURI != "mlx://agent/index-only" || bundleURI != "mlx://agent/index-only/bundle" {
		t.Fatalf("index-only = %q/%q, want entry adopts index", entryURI, bundleURI)
	}
}

// --- SleepBlockOptions ----------------------------------------------------

func TestWakeSleep_SleepBlockOptions_Good(t *testing.T) {
	// Empty BlockOptions: KVEncoding/URI/Title default and the canonical
	// "state" label is appended. (The trusted-flag plumb is covered in
	// wake_sleep_trusted_test.go and is not duplicated here.)
	blockOpts := SleepBlockOptions(SleepOptions{Title: "session-1"}, "mlx://agent/session-1/bundle")
	if blockOpts.KVEncoding != kv.EncodingNative {
		t.Fatalf("KVEncoding = %q, want default %q", blockOpts.KVEncoding, kv.EncodingNative)
	}
	if blockOpts.URI != "mlx://agent/session-1/bundle/blocks" {
		t.Fatalf("URI = %q, want derived /blocks suffix", blockOpts.URI)
	}
	if blockOpts.Title != "session-1" {
		t.Fatalf("Title = %q, want SleepOptions title", blockOpts.Title)
	}
	if len(blockOpts.Labels) != 1 || blockOpts.Labels[0] != "state" {
		t.Fatalf("Labels = %v, want [state] appended", blockOpts.Labels)
	}
}

func TestWakeSleep_SleepBlockOptions_Bad(t *testing.T) {
	// No title anywhere: Title falls back to the package default rather
	// than staying empty.
	blockOpts := SleepBlockOptions(SleepOptions{}, "mlx://agent/x/bundle")
	if blockOpts.Title != "go-mlx State" {
		t.Fatalf("Title = %q, want %q fallback", blockOpts.Title, "go-mlx State")
	}
}

func TestWakeSleep_SleepBlockOptions_Ugly(t *testing.T) {
	// Pre-seeded labels + explicit URI/encoding must be preserved, with
	// "state" appended without mutating the caller's backing array.
	callerLabels := []string{"agent", "preset"}
	opts := SleepOptions{
		BlockOptions: kv.StateBlockOptions{
			KVEncoding: kv.EncodingNative,
			URI:        "mlx://explicit/blocks",
			Title:      "explicit",
			Labels:     callerLabels,
		},
	}
	blockOpts := SleepBlockOptions(opts, "mlx://agent/ignored/bundle")
	if blockOpts.URI != "mlx://explicit/blocks" || blockOpts.Title != "explicit" {
		t.Fatalf("explicit URI/Title overwritten: %q/%q", blockOpts.URI, blockOpts.Title)
	}
	if len(blockOpts.Labels) != 3 || blockOpts.Labels[2] != "state" {
		t.Fatalf("Labels = %v, want preset labels + state", blockOpts.Labels)
	}
	if len(callerLabels) != 2 {
		t.Fatalf("caller label slice mutated: %v", callerLabels)
	}

	// The trusted flag must reach the block options — the continuity lane's
	// declaration rides SleepOptions into kv.StateBlockOptions. (Folded in
	// from the former wake_sleep_trusted_test.go scenario test.)
	t.Run("TrustedFlagPlumbs", func(t *testing.T) {
		blockOpts := SleepBlockOptions(SleepOptions{ReuseParentPrefixTrusted: true}, "mlx://bundle")
		if !blockOpts.ReusePrefixTrusted {
			t.Fatal("ReusePrefixTrusted did not plumb through SleepBlockOptions")
		}
		if SleepBlockOptions(SleepOptions{}, "mlx://bundle").ReusePrefixTrusted {
			t.Fatal("ReusePrefixTrusted set without the SleepOptions declaration")
		}
	})
}

// --- sleepEntryMeta -------------------------------------------------------

func TestWakeSleep_sleepEntryMeta_Good(t *testing.T) {
	// All parent URIs set + caller meta: result carries both the cloned
	// caller keys and the three parent_* keys.
	meta := sleepEntryMeta(SleepOptions{
		ParentEntryURI:  "mlx://agent/session-0",
		ParentBundleURI: "mlx://agent/session-0/bundle",
		ParentIndexURI:  "mlx://agent/session-0/index",
		Meta:            map[string]string{"session_id": "s-1"},
	})
	if meta["session_id"] != "s-1" {
		t.Fatalf("session_id = %q, want cloned caller meta", meta["session_id"])
	}
	if meta["parent_entry_uri"] != "mlx://agent/session-0" ||
		meta["parent_bundle_uri"] != "mlx://agent/session-0/bundle" ||
		meta["parent_index_uri"] != "mlx://agent/session-0/index" {
		t.Fatalf("parent keys = %+v, want all three parent_* keys", meta)
	}
}

func TestWakeSleep_sleepEntryMeta_Bad(t *testing.T) {
	// No parents and no caller meta: nil map (nothing to seed).
	if meta := sleepEntryMeta(SleepOptions{Title: "bare"}); meta != nil {
		t.Fatalf("sleepEntryMeta(bare) = %+v, want nil", meta)
	}
}

func TestWakeSleep_sleepEntryMeta_Ugly(t *testing.T) {
	// Only one parent URI set, no caller meta: map is created lazily and
	// holds exactly that one key.
	meta := sleepEntryMeta(SleepOptions{ParentBundleURI: "mlx://agent/p/bundle"})
	if len(meta) != 1 || meta["parent_bundle_uri"] != "mlx://agent/p/bundle" {
		t.Fatalf("meta = %+v, want single parent_bundle_uri key", meta)
	}
}

// --- NewSleepIndex --------------------------------------------------------

func TestWakeSleep_NewSleepIndex_Good(t *testing.T) {
	_, blk := wakeSleepTestBundle(t, "mlx://agent/session-1/bundle")
	idx, err := NewSleepIndex(blk, SleepOptions{
		Title:          "session-1",
		Model:          "qwen3-7b",
		ParentEntryURI: "mlx://agent/session-0",
		Labels:         []string{"agent", "checkpoint"},
		Meta:           map[string]string{"session_id": "s-1"},
	}, "mlx://agent/session-1", "mlx://agent/session-1/bundle")
	if err != nil {
		t.Fatalf("NewSleepIndex() error = %v", err)
	}
	if len(idx.Entries) != 1 {
		t.Fatalf("entries = %d, want single sleep entry", len(idx.Entries))
	}
	entry := idx.Entries[0]
	if entry.URI != "mlx://agent/session-1" || entry.BundleURI != "mlx://agent/session-1/bundle" {
		t.Fatalf("entry URIs = %q/%q, want sleep URIs", entry.URI, entry.BundleURI)
	}
	if entry.TokenCount != blk.TokenCount {
		t.Fatalf("entry token count = %d, want bundle %d", entry.TokenCount, blk.TokenCount)
	}
	if entry.Meta["parent_entry_uri"] != "mlx://agent/session-0" {
		t.Fatalf("entry meta missing parent: %+v", entry.Meta)
	}
	if err := idx.Validate(); err != nil {
		t.Fatalf("Validate() error = %v", err)
	}
}

func TestWakeSleep_NewSleepIndex_Bad(t *testing.T) {
	// A nil bundle must return errBundleNil (via the up-front
	// ValidateStateBlockBundle guard) rather than panicking on the
	// bundle.TokenCount dereference.
	if _, err := NewSleepIndex(nil, SleepOptions{}, "mlx://agent/x", "mlx://agent/x/bundle"); err == nil {
		t.Fatal("NewSleepIndex(nil bundle) error = nil")
	}
	// A non-nil but invalid bundle (no blocks / zero version) is likewise
	// rejected.
	bad := &kv.StateBlockBundle{}
	if _, err := NewSleepIndex(bad, SleepOptions{}, "mlx://agent/x", "mlx://agent/x/bundle"); err == nil {
		t.Fatal("NewSleepIndex(invalid bundle) error = nil")
	}
}

func TestWakeSleep_NewSleepIndex_Ugly(t *testing.T) {
	// Empty title: the entry title defaults to "State" rather than being
	// left blank.
	_, blk := wakeSleepTestBundle(t, "mlx://agent/untitled/bundle")
	idx, err := NewSleepIndex(blk, SleepOptions{}, "mlx://agent/untitled", "mlx://agent/untitled/bundle")
	if err != nil {
		t.Fatalf("NewSleepIndex() error = %v", err)
	}
	if idx.Entries[0].Title != "State" {
		t.Fatalf("entry title = %q, want default %q", idx.Entries[0].Title, "State")
	}
}

// --- NewSleepReport -------------------------------------------------------

func TestWakeSleep_NewSleepReport_Good(t *testing.T) {
	_, blk := wakeSleepTestBundle(t, "mlx://agent/session-1/bundle")
	opts := SleepOptions{
		Title:           "session-1",
		ParentEntryURI:  "mlx://agent/session-0",
		ParentBundleURI: "mlx://agent/session-0/bundle",
		ParentIndexURI:  "mlx://agent/session-0/index",
	}
	idx, err := NewSleepIndex(blk, opts, "mlx://agent/session-1", "mlx://agent/session-1/bundle")
	if err != nil {
		t.Fatalf("NewSleepIndex() error = %v", err)
	}
	bundleRef := state.ChunkRef{ChunkID: 1, FrameOffset: 64, HasFrameOffset: true}
	indexRef := state.ChunkRef{ChunkID: 2, FrameOffset: 256, HasFrameOffset: true}
	report := NewSleepReport(idx, blk, opts, "mlx://agent/session-1", "mlx://agent/session-1/bundle", "mlx://agent/session-1/index", bundleRef, indexRef)
	if report.IndexURI != "mlx://agent/session-1/index" || report.EntryURI != "mlx://agent/session-1" || report.BundleURI != "mlx://agent/session-1/bundle" {
		t.Fatalf("report URIs = %q/%q/%q, want supplied values", report.IndexURI, report.EntryURI, report.BundleURI)
	}
	if report.ParentEntryURI != "mlx://agent/session-0" {
		t.Fatalf("report parent entry = %q, want from opts", report.ParentEntryURI)
	}
	if report.TokenCount != blk.TokenCount || report.BlockSize != blk.BlockSize {
		t.Fatalf("report token/block = %d/%d, want bundle %d/%d", report.TokenCount, report.BlockSize, blk.TokenCount, blk.BlockSize)
	}
	if report.BlocksWritten != len(blk.Blocks) {
		t.Fatalf("report blocks written = %d, want %d", report.BlocksWritten, len(blk.Blocks))
	}
	if report.IndexHash != idx.Hash {
		t.Fatalf("report index hash = %q, want %q", report.IndexHash, idx.Hash)
	}
	if report.BundleRef != bundleRef || report.IndexRef != indexRef {
		t.Fatalf("report refs = %+v/%+v, want supplied refs", report.BundleRef, report.IndexRef)
	}
}

func TestWakeSleep_NewSleepReport_Bad(t *testing.T) {
	// No parents and an empty title: the report still assembles, but the
	// parent_* fields stay empty and the carried token/block counts come
	// straight off the bundle. NewSleepReport has no error return — the
	// "bad" shape is a minimal-options report that must not invent values.
	_, blk := wakeSleepTestBundle(t, "mlx://agent/bare/bundle")
	opts := SleepOptions{}
	idx, err := NewSleepIndex(blk, opts, "mlx://agent/bare", "mlx://agent/bare/bundle")
	if err != nil {
		t.Fatalf("NewSleepIndex() error = %v", err)
	}
	report := NewSleepReport(idx, blk, opts, "mlx://agent/bare", "mlx://agent/bare/bundle", "mlx://agent/bare/index", state.ChunkRef{}, state.ChunkRef{})
	if report.ParentEntryURI != "" || report.ParentBundleURI != "" || report.ParentIndexURI != "" {
		t.Fatalf("report parents = %q/%q/%q, want all empty for parentless sleep", report.ParentEntryURI, report.ParentBundleURI, report.ParentIndexURI)
	}
	if report.Title != "" {
		t.Fatalf("report title = %q, want empty (NewSleepReport copies opts.Title verbatim)", report.Title)
	}
	if report.TokenCount != blk.TokenCount {
		t.Fatalf("report token count = %d, want bundle %d", report.TokenCount, blk.TokenCount)
	}
	if (report.BundleRef != state.ChunkRef{}) || (report.IndexRef != state.ChunkRef{}) {
		t.Fatalf("report refs = %+v/%+v, want zero refs", report.BundleRef, report.IndexRef)
	}
}

func TestWakeSleep_NewSleepReport_Ugly(t *testing.T) {
	// Reused-block bundle: BlocksReused is carried verbatim from the bundle
	// even when it equals or exceeds BlocksWritten, and KVEncoding flows
	// through. This exercises the report assembling over a grafted-prefix
	// bundle rather than a freshly captured one.
	_, blk := wakeSleepTestBundle(t, "mlx://agent/reused/bundle")
	blk.ReusedBlocks = len(blk.Blocks)
	opts := SleepOptions{Title: "reused-session"}
	idx, err := NewSleepIndex(blk, opts, "mlx://agent/reused", "mlx://agent/reused/bundle")
	if err != nil {
		t.Fatalf("NewSleepIndex() error = %v", err)
	}
	report := NewSleepReport(idx, blk, opts, "mlx://agent/reused", "mlx://agent/reused/bundle", "mlx://agent/reused/index", state.ChunkRef{ChunkID: 7}, state.ChunkRef{ChunkID: 8})
	if report.BlocksReused != len(blk.Blocks) {
		t.Fatalf("report blocks reused = %d, want %d carried from bundle", report.BlocksReused, len(blk.Blocks))
	}
	if report.BlocksWritten != len(blk.Blocks) {
		t.Fatalf("report blocks written = %d, want %d", report.BlocksWritten, len(blk.Blocks))
	}
	if report.KVEncoding != blk.KVEncoding {
		t.Fatalf("report KV encoding = %q, want bundle %q", report.KVEncoding, blk.KVEncoding)
	}
	if report.SnapshotHash != blk.SnapshotHash {
		t.Fatalf("report snapshot hash = %q, want bundle %q", report.SnapshotHash, blk.SnapshotHash)
	}
}

// --- WakeReportFromSleep --------------------------------------------------

func TestWakeSleep_WakeReportFromSleep_Good(t *testing.T) {
	sleep := &SleepReport{
		IndexURI:     "mlx://agent/session-1/index",
		EntryURI:     "mlx://agent/session-1",
		BundleURI:    "mlx://agent/session-1/bundle",
		Title:        "session-1",
		TokenCount:   2048,
		BlockSize:    512,
		IndexHash:    "deadbeef",
		SnapshotHash: "feed1234",
	}
	wake := WakeReportFromSleep(sleep)
	if wake.PrefixTokens != 2048 || wake.BundleTokens != 2048 {
		t.Fatalf("wake tokens = %d/%d, want sleep token count for both", wake.PrefixTokens, wake.BundleTokens)
	}
	if wake.BlocksRead != 0 {
		t.Fatalf("wake blocks read = %d, want 0 (state still resident after sleep)", wake.BlocksRead)
	}
	if wake.IndexHash != "deadbeef" || wake.SnapshotHash != "feed1234" {
		t.Fatalf("wake hashes = %q/%q, want carried from sleep", wake.IndexHash, wake.SnapshotHash)
	}
}

func TestWakeSleep_WakeReportFromSleep_Bad(t *testing.T) {
	// Nil sleep report yields a nil wake report (no panic).
	if wake := WakeReportFromSleep(nil); wake != nil {
		t.Fatalf("WakeReportFromSleep(nil) = %+v, want nil", wake)
	}
}

func TestWakeSleep_WakeReportFromSleep_Ugly(t *testing.T) {
	// A zero-token sleep report still converts cleanly: PrefixTokens and
	// BundleTokens both mirror the (zero) token count, BlocksRead is forced
	// to 0, and the URI/title fields carry through even when the spans are
	// degenerate. Exercises the field-by-field mapping on a boundary input
	// distinct from the populated Good case.
	sleep := &SleepReport{
		IndexURI:     "mlx://agent/empty/index",
		EntryURI:     "mlx://agent/empty",
		BundleURI:    "mlx://agent/empty/bundle",
		Title:        "empty",
		TokenCount:   0,
		BlockSize:    0,
		IndexHash:    "",
		SnapshotHash: "",
	}
	wake := WakeReportFromSleep(sleep)
	if wake == nil {
		t.Fatal("WakeReportFromSleep(zero-token) = nil, want non-nil report")
	}
	if wake.PrefixTokens != 0 || wake.BundleTokens != 0 {
		t.Fatalf("wake tokens = %d/%d, want 0/0 mirrored from sleep", wake.PrefixTokens, wake.BundleTokens)
	}
	if wake.BlocksRead != 0 {
		t.Fatalf("wake blocks read = %d, want 0", wake.BlocksRead)
	}
	if wake.IndexURI != "mlx://agent/empty/index" || wake.EntryURI != "mlx://agent/empty" || wake.BundleURI != "mlx://agent/empty/bundle" {
		t.Fatalf("wake URIs = %q/%q/%q, want carried from sleep", wake.IndexURI, wake.EntryURI, wake.BundleURI)
	}
	if wake.Title != "empty" {
		t.Fatalf("wake title = %q, want carried from sleep", wake.Title)
	}
}

// --- CloneWakeReport ------------------------------------------------------

func TestWakeSleep_CloneWakeReport_Good(t *testing.T) {
	original := &WakeReport{
		IndexURI:     "mlx://agent/session-1/index",
		EntryURI:     "mlx://agent/session-1",
		Title:        "session-1",
		PrefixTokens: 2048,
		BlocksRead:   8,
	}
	clone := CloneWakeReport(original)
	if clone == original {
		t.Fatal("CloneWakeReport returned the same pointer")
	}
	if *clone != *original {
		t.Fatalf("clone = %+v, want equal value to original", clone)
	}
	clone.Title = "mutated"
	if original.Title != "session-1" {
		t.Fatalf("mutating clone changed original: %q", original.Title)
	}
}

func TestWakeSleep_CloneWakeReport_Bad(t *testing.T) {
	if clone := CloneWakeReport(nil); clone != nil {
		t.Fatalf("CloneWakeReport(nil) = %+v, want nil", clone)
	}
}

func TestWakeSleep_CloneWakeReport_Ugly(t *testing.T) {
	// Cloning a zero-value (non-nil) report must return a distinct pointer
	// that compares equal to the empty original — the clone path is a flat
	// struct copy, so even an all-zero report round-trips to an independent
	// equal value rather than aliasing the input.
	original := &WakeReport{}
	clone := CloneWakeReport(original)
	if clone == original {
		t.Fatal("CloneWakeReport(zero) returned the same pointer")
	}
	if *clone != *original {
		t.Fatalf("clone = %+v, want equal to zero original", clone)
	}
	// Mutating the clone leaves the zero original untouched.
	clone.IndexURI = "mlx://agent/mutated"
	if original.IndexURI != "" {
		t.Fatalf("mutating clone changed zero original: %q", original.IndexURI)
	}
}

// --- PlanWake -------------------------------------------------------------

func planWakeIndex(t *testing.T, blk *kv.StateBlockBundle, bundleURI string) *StateIndex {
	t.Helper()
	idx, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: bundleURI,
		Title:     "session-1",
		Model:     "demo",
		ModelInfo: memory.ModelInfo{
			Architecture:  "gemma4_text",
			NumLayers:     1,
			QuantBits:     4,
			ContextLength: 8,
		},
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		Entries: []StateIndexEntry{{
			URI:        "mlx://agent/session-1/chapter-1",
			TokenStart: 0,
			TokenCount: 2,
		}},
	})
	if err != nil {
		t.Fatalf("NewStateIndex() error = %v", err)
	}
	return idx
}

func TestWakeSleep_PlanWake_Good(t *testing.T) {
	const bundleURI = "mlx://agent/session-1/bundle"
	store, blk := wakeSleepTestBundle(t, bundleURI)
	idx := planWakeIndex(t, blk, bundleURI)
	plan, err := PlanWake(context.Background(), store, WakeOptions{
		Index:     idx,
		EntryURI:  "mlx://agent/session-1/chapter-1",
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
	}, memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8})
	if err != nil {
		t.Fatalf("PlanWake() error = %v", err)
	}
	if plan.Entry.URI != "mlx://agent/session-1/chapter-1" {
		t.Fatalf("plan entry = %q, want chapter-1", plan.Entry.URI)
	}
	if plan.Report.PrefixTokens != 2 {
		t.Fatalf("plan prefix tokens = %d, want 2", plan.Report.PrefixTokens)
	}
	if plan.Report.BundleURI != bundleURI {
		t.Fatalf("plan bundle URI = %q, want %q", plan.Report.BundleURI, bundleURI)
	}
	if plan.Report.BlocksRead <= 0 {
		t.Fatalf("plan blocks read = %d, want > 0", plan.Report.BlocksRead)
	}
}

func TestWakeSleep_PlanWake_Bad(t *testing.T) {
	const bundleURI = "mlx://agent/session-1/bundle"
	store, blk := wakeSleepTestBundle(t, bundleURI)
	idx := planWakeIndex(t, blk, bundleURI)
	info := memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}
	tok := pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"}

	// Nil store.
	if _, err := PlanWake(context.Background(), nil, WakeOptions{Index: idx}, info); !errors.Is(err, errStateStoreNil) {
		t.Fatalf("PlanWake(nil store) error = %v, want errStateStoreNil", err)
	}
	// No index + no IndexURI: loadIndex returns errStateIndexURIRequired.
	if _, err := PlanWake(context.Background(), store, WakeOptions{Tokenizer: tok}, info); !errors.Is(err, errStateIndexURIRequired) {
		t.Fatalf("PlanWake(no index) error = %v, want errStateIndexURIRequired", err)
	}
	// Unknown entry URI.
	if _, err := PlanWake(context.Background(), store, WakeOptions{Index: idx, EntryURI: "mlx://agent/session-1/missing", Tokenizer: tok}, info); !errors.Is(err, errStateIndexEntryNotFound) {
		t.Fatalf("PlanWake(missing entry) error = %v, want errStateIndexEntryNotFound", err)
	}
	// Compatibility check failure (architecture mismatch).
	if _, err := PlanWake(context.Background(), store, WakeOptions{Index: idx, EntryURI: "mlx://agent/session-1/chapter-1", Tokenizer: tok}, memory.ModelInfo{Architecture: "qwen3", NumLayers: 1, QuantBits: 4, ContextLength: 8}); !errors.Is(err, errStateIndexArchitectureMismatch) {
		t.Fatalf("PlanWake(arch mismatch) error = %v, want errStateIndexArchitectureMismatch", err)
	}
}

func TestWakeSleep_PlanWake_Ugly(t *testing.T) {
	// SkipCompatibilityCheck true: a mismatching ModelInfo is tolerated
	// because the compat gate is bypassed, and a nil ctx defaults to
	// Background. Default entry selection (empty EntryURI) picks the
	// first index entry.
	const bundleURI = "mlx://agent/session-1/bundle"
	store, blk := wakeSleepTestBundle(t, bundleURI)
	idx := planWakeIndex(t, blk, bundleURI)
	plan, err := PlanWake(nil, store, WakeOptions{ //nolint:staticcheck // exercising the nil-ctx default branch
		Index:                  idx,
		SkipCompatibilityCheck: true,
	}, memory.ModelInfo{Architecture: "totally-different", NumLayers: 99, QuantBits: 8, ContextLength: 1})
	if err != nil {
		t.Fatalf("PlanWake(skip compat, nil ctx) error = %v", err)
	}
	if plan.Entry.URI != "mlx://agent/session-1/chapter-1" {
		t.Fatalf("default entry = %q, want first index entry", plan.Entry.URI)
	}
}

// --- LoadWakeSnapshot -----------------------------------------------------

func TestWakeSleep_LoadWakeSnapshot_Good(t *testing.T) {
	const bundleURI = "mlx://agent/session-1/bundle"
	store, blk := wakeSleepTestBundle(t, bundleURI)
	idx := planWakeIndex(t, blk, bundleURI)
	snapshot, report, err := LoadWakeSnapshot(context.Background(), store, WakeOptions{
		Index:       idx,
		EntryURI:    "mlx://agent/session-1/chapter-1",
		Tokenizer:   pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		LoadOptions: kv.LoadOptions{RawKVOnly: true},
	}, memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8})
	if err != nil {
		t.Fatalf("LoadWakeSnapshot() error = %v", err)
	}
	if report.PrefixTokens != 2 {
		t.Fatalf("report prefix tokens = %d, want 2", report.PrefixTokens)
	}
	// The 4-token fixture restores the first two tokens for the chapter-1
	// prefix.
	if len(snapshot.Tokens) != 2 || snapshot.Tokens[0] != 1 || snapshot.Tokens[1] != 2 {
		t.Fatalf("snapshot tokens = %v, want first two synthetic tokens", snapshot.Tokens)
	}
}

func TestWakeSleep_LoadWakeSnapshot_Bad(t *testing.T) {
	// PlanWake failure (nil store) propagates without a snapshot.
	snapshot, report, err := LoadWakeSnapshot(context.Background(), nil, WakeOptions{}, memory.ModelInfo{})
	if err == nil {
		t.Fatal("LoadWakeSnapshot(nil store) error = nil")
	}
	if snapshot != nil || report != nil {
		t.Fatalf("LoadWakeSnapshot(nil store) = %+v/%+v, want nil/nil on error", snapshot, report)
	}
}

func TestWakeSleep_LoadWakeSnapshot_Ugly(t *testing.T) {
	// SkipCompatibilityCheck bypasses the model-identity gate, so a wildly
	// mismatching ModelInfo still wakes; an empty EntryURI defaults to the
	// first index entry. This drives the full plan-then-load path on the
	// awkward (compat-skipped, default-entry) inputs rather than the clean
	// explicit-entry Good path.
	const bundleURI = "mlx://agent/session-1/bundle"
	store, blk := wakeSleepTestBundle(t, bundleURI)
	idx := planWakeIndex(t, blk, bundleURI)
	snapshot, report, err := LoadWakeSnapshot(context.Background(), store, WakeOptions{
		Index:                  idx,
		SkipCompatibilityCheck: true,
		LoadOptions:            kv.LoadOptions{RawKVOnly: true},
	}, memory.ModelInfo{Architecture: "totally-different", NumLayers: 99, QuantBits: 8, ContextLength: 1})
	if err != nil {
		t.Fatalf("LoadWakeSnapshot(skip compat, default entry) error = %v", err)
	}
	if report.EntryURI != "mlx://agent/session-1/chapter-1" {
		t.Fatalf("report entry = %q, want first index entry by default", report.EntryURI)
	}
	if len(snapshot.Tokens) != 2 || snapshot.Tokens[0] != 1 || snapshot.Tokens[1] != 2 {
		t.Fatalf("snapshot tokens = %v, want first two synthetic tokens", snapshot.Tokens)
	}
}

// --- blocksNeededForPrefix ------------------------------------------------

func TestWakeSleep_blocksNeededForPrefix_Good(t *testing.T) {
	_, blk := wakeSleepTestBundle(t, "mlx://agent/blocks/bundle")
	// The synthetic bundle has a 2-token block size over 4 tokens.
	if n := blocksNeededForPrefix(blk, blk.TokenCount); n != len(blk.Blocks) {
		t.Fatalf("blocksNeededForPrefix(all) = %d, want %d", n, len(blk.Blocks))
	}
	if n := blocksNeededForPrefix(blk, 1); n != 1 {
		t.Fatalf("blocksNeededForPrefix(1) = %d, want 1", n)
	}
}

func TestWakeSleep_blocksNeededForPrefix_Bad(t *testing.T) {
	_, blk := wakeSleepTestBundle(t, "mlx://agent/blocks2/bundle")
	if n := blocksNeededForPrefix(nil, 4); n != 0 {
		t.Fatalf("blocksNeededForPrefix(nil bundle) = %d, want 0", n)
	}
	if n := blocksNeededForPrefix(blk, 0); n != 0 {
		t.Fatalf("blocksNeededForPrefix(zero prefix) = %d, want 0", n)
	}
}
