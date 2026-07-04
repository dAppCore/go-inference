// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"context"
	"fmt"

	pkgbundle "dappco.re/go/inference/bundle"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/memory"
	state "dappco.re/go/inference/state"
)

// exampleWakeStore saves the shared 4-token synthetic snapshot into an
// in-memory State store under bundleURI and returns the store plus the
// block bundle, so the wake-path examples below have a real (tiny) bundle
// to plan and load against without touching Metal or a model file.
func exampleWakeStore(bundleURI string) (state.Store, *kv.StateBlockBundle) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	snapshot := kvSnapshotBlocksTestSnapshot()
	blk, err := snapshot.SaveStateBlocks(ctx, store, kv.StateBlockOptions{BlockSize: 2, KVEncoding: kv.EncodingNative})
	if err != nil {
		panic(err)
	}
	if _, err := kv.SaveStateBlockBundle(ctx, store, blk, bundleURI); err != nil {
		panic(err)
	}
	return store, blk
}

// exampleWakeIndex builds a single-chapter StateIndex over blk for the
// wake examples. The chapter names the first two tokens of the bundle.
func exampleWakeIndex(blk *kv.StateBlockBundle, bundleURI string) *StateIndex {
	idx, err := NewStateIndex(blk, StateIndexOptions{
		BundleURI: bundleURI,
		Title:     "session-1",
		Model:     "demo",
		ModelInfo: memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8},
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		Entries:   []StateIndexEntry{{URI: "mlx://agent/session-1/chapter-1", TokenStart: 0, TokenCount: 2}},
	})
	if err != nil {
		panic(err)
	}
	return idx
}

// ExampleSleepURIs shows how a single EntryURI is expanded into the
// derived bundle and index URIs used by a sleep round.
func ExampleSleepURIs() {
	entryURI, bundleURI, indexURI, err := SleepURIs(SleepOptions{EntryURI: "mlx://agent/session-1"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(entryURI)
	fmt.Println(bundleURI)
	fmt.Println(indexURI)
	// Output:
	// mlx://agent/session-1
	// mlx://agent/session-1/bundle
	// mlx://agent/session-1/index
}

// ExampleSleepBlockOptions shows the defaulting applied to an empty
// BlockOptions: native KV encoding, a derived blocks URI, and the
// canonical "state" label appended.
func ExampleSleepBlockOptions() {
	blockOpts := SleepBlockOptions(SleepOptions{Title: "session-1"}, "mlx://agent/session-1/bundle")
	fmt.Println(blockOpts.KVEncoding)
	fmt.Println(blockOpts.URI)
	fmt.Println(blockOpts.Labels)
	// Output:
	// native
	// mlx://agent/session-1/bundle/blocks
	// [state]
}

// ExampleWakeReportFromSleep shows converting a SleepReport into the
// WakeReport a caller continues with in-process (no reload needed).
func ExampleWakeReportFromSleep() {
	wake := WakeReportFromSleep(&SleepReport{
		EntryURI:   "mlx://agent/session-1",
		Title:      "session-1",
		TokenCount: 2048,
		BlockSize:  512,
	})
	fmt.Println(wake.EntryURI)
	fmt.Println(wake.PrefixTokens)
	fmt.Println(wake.BlocksRead)
	// Output:
	// mlx://agent/session-1
	// 2048
	// 0
}

// ExampleCloneWakeReport shows that the clone is an independent copy:
// mutating it leaves the original untouched.
func ExampleCloneWakeReport() {
	original := &WakeReport{Title: "session-1", PrefixTokens: 2048}
	clone := CloneWakeReport(original)
	clone.Title = "mutated"
	fmt.Println(original.Title)
	fmt.Println(clone.Title)
	// Output:
	// session-1
	// mutated
}

// ExampleNewSleepReport assembles a durable sleep report from a freshly
// built index and bundle, printing the load-bearing scalar fields.
func ExampleNewSleepReport() {
	bundle := &kv.StateBlockBundle{
		Version:      kv.MemvidBlockVersion,
		Kind:         kv.MemvidBlockBundleKind,
		SnapshotHash: "snap",
		KVEncoding:   kv.EncodingNative,
		Architecture: "qwen3",
		TokenCount:   4,
		BlockSize:    2,
		NumLayers:    1,
		NumHeads:     1,
		SeqLen:       4,
		HeadDim:      2,
		Blocks: []kv.StateBlockRef{
			{Index: 0, TokenStart: 0, TokenCount: 2},
			{Index: 1, TokenStart: 2, TokenCount: 2},
		},
	}
	opts := SleepOptions{Title: "session-1"}
	idx, err := NewSleepIndex(bundle, opts, "mlx://agent/session-1", "mlx://agent/session-1/bundle")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	report := NewSleepReport(idx, bundle, opts,
		"mlx://agent/session-1", "mlx://agent/session-1/bundle", "mlx://agent/session-1/index",
		state.ChunkRef{}, state.ChunkRef{})
	fmt.Println(report.EntryURI)
	fmt.Println(report.TokenCount)
	fmt.Println(report.BlocksWritten)
	// Output:
	// mlx://agent/session-1
	// 4
	// 2
}

// ExampleNewSleepIndex builds the durable index a sleep round writes: one
// entry covering the whole bundle, carrying the parent link in its meta.
func ExampleNewSleepIndex() {
	_, blk := exampleWakeStore("mlx://agent/session-1/bundle")
	idx, err := NewSleepIndex(blk, SleepOptions{
		Title:          "session-1",
		Model:          "demo",
		ParentEntryURI: "mlx://agent/session-0",
	}, "mlx://agent/session-1", "mlx://agent/session-1/bundle")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(len(idx.Entries))
	fmt.Println(idx.Entries[0].URI)
	fmt.Println(idx.Entries[0].TokenCount)
	fmt.Println(idx.Entries[0].Meta["parent_entry_uri"])
	// Output:
	// 1
	// mlx://agent/session-1
	// 4
	// mlx://agent/session-0
}

// ExamplePlanWake resolves a named chapter through an index and reports the
// prefix length and bundle it would restore — without loading any KV data.
func ExamplePlanWake() {
	const bundleURI = "mlx://agent/session-1/bundle"
	store, blk := exampleWakeStore(bundleURI)
	idx := exampleWakeIndex(blk, bundleURI)
	plan, err := PlanWake(context.Background(), store, WakeOptions{
		Index:     idx,
		EntryURI:  "mlx://agent/session-1/chapter-1",
		Tokenizer: pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
	}, memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(plan.Entry.URI)
	fmt.Println(plan.Report.PrefixTokens)
	fmt.Println(plan.Report.BundleURI)
	// Output:
	// mlx://agent/session-1/chapter-1
	// 2
	// mlx://agent/session-1/bundle
}

// ExampleLoadWakeSnapshot restores the KV prefix a named chapter needs — the
// first two tokens of the four-token synthetic bundle.
func ExampleLoadWakeSnapshot() {
	const bundleURI = "mlx://agent/session-1/bundle"
	store, blk := exampleWakeStore(bundleURI)
	idx := exampleWakeIndex(blk, bundleURI)
	snapshot, report, err := LoadWakeSnapshot(context.Background(), store, WakeOptions{
		Index:       idx,
		EntryURI:    "mlx://agent/session-1/chapter-1",
		Tokenizer:   pkgbundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		LoadOptions: kv.LoadOptions{RawKVOnly: true},
	}, memory.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(report.PrefixTokens)
	fmt.Println(len(snapshot.Tokens))
	// Output:
	// 2
	// 2
}
