// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for wake/sleep orchestration scaffolding. These are the
// pure-data shape transformations the agent runtime does on every
// session resume + checkpoint round — URI resolution, block-options
// shaping, plan construction, report cloning. The Metal-side KV
// load/save path is not benched here; that's the kv package.
//
// Per AX-11 — Sleep is invoked at minimum once per session shutdown,
// often more (checkpointing during long generation runs). Wake is
// once per session resume. SleepURIs + SleepBlockOptions + NewSleepIndex
// fire on every Sleep.
//
// Run:    go test -bench='BenchmarkWakeSleep' -benchmem -run='^$' ./go/agent

package agent

import (
	"context"
	"testing"

	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/kv/memory"
	"dappco.re/go/inference/model/bundle"
	state "dappco.re/go/inference/model/state"
)

// Sinks defeat compiler DCE.
var (
	wakeSleepBenchSinkEntryURI  string
	wakeSleepBenchSinkBundleURI string
	wakeSleepBenchSinkIndexURI  string
	wakeSleepBenchSinkErr       error
	wakeSleepBenchSinkOpts      kv.StateBlockOptions
	wakeSleepBenchSinkIndex     *StateIndex
	wakeSleepBenchSinkReport    *SleepReport
	wakeSleepBenchSinkWake      *WakeReport
	wakeSleepBenchSinkPlan      *WakePlan
	wakeSleepBenchSinkInt       int
)

// benchSleepOptions returns a populated SleepOptions value used by
// the sleep-side benches.
func benchSleepOptions() SleepOptions {
	return SleepOptions{
		EntryURI:        "mlx://agent/session-1",
		BundleURI:       "mlx://agent/session-1/bundle",
		IndexURI:        "mlx://agent/session-1/index",
		ParentEntryURI:  "mlx://agent/session-0",
		ParentBundleURI: "mlx://agent/session-0/bundle",
		ParentIndexURI:  "mlx://agent/session-0/index",
		Title:           "session-1",
		Model:           "qwen3-7b",
		ModelPath:       "/models/qwen3-7b",
		ModelInfo: memory.ModelInfo{
			Architecture:  "qwen3",
			NumLayers:     28,
			QuantBits:     4,
			ContextLength: 40960,
		},
		Tokenizer: bundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		Labels:    []string{"agent", "checkpoint"},
		Meta:      map[string]string{"session_id": "s-1", "agent": "cladius"},
	}
}

// --- SleepURIs — URI defaulting + validation. Pure string-ops; hit
// once per Sleep but cheap.

func BenchmarkWakeSleep_SleepURIs_AllSet(b *testing.B) {
	opts := benchSleepOptions()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkEntryURI, wakeSleepBenchSinkBundleURI, wakeSleepBenchSinkIndexURI, wakeSleepBenchSinkErr = SleepURIs(opts)
	}
}

func BenchmarkWakeSleep_SleepURIs_OnlyEntry(b *testing.B) {
	// Only EntryURI set — exercises the bundleURI/indexURI derivation
	// branch.
	opts := SleepOptions{EntryURI: "mlx://agent/session-only-entry"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkEntryURI, wakeSleepBenchSinkBundleURI, wakeSleepBenchSinkIndexURI, wakeSleepBenchSinkErr = SleepURIs(opts)
	}
}

func BenchmarkWakeSleep_SleepURIs_EmptyDefaults(b *testing.B) {
	// Nothing set — exercises the core.FirstNonBlank fallback chain
	// and the default "mlx://state/latest" fall-through.
	opts := SleepOptions{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkEntryURI, wakeSleepBenchSinkBundleURI, wakeSleepBenchSinkIndexURI, wakeSleepBenchSinkErr = SleepURIs(opts)
	}
}

// --- SleepBlockOptions — defensive label clone + KV encoding default.
// Hit once per Sleep.

func BenchmarkWakeSleep_SleepBlockOptions_FreshShape(b *testing.B) {
	opts := benchSleepOptions()
	const bundleURI = "mlx://agent/session-1/bundle"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkOpts = SleepBlockOptions(opts, bundleURI)
	}
}

func BenchmarkWakeSleep_SleepBlockOptions_PreSeededLabels(b *testing.B) {
	opts := benchSleepOptions()
	opts.BlockOptions = kv.StateBlockOptions{
		BlockSize:  512,
		KVEncoding: kv.EncodingNative,
		Labels:     []string{"agent", "preset"},
	}
	const bundleURI = "mlx://agent/session-1/bundle"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkOpts = SleepBlockOptions(opts, bundleURI)
	}
}

// --- NewSleepIndex — wraps NewStateIndex with the sleep-side entry
// metadata derivation (sleepEntryMeta).

func BenchmarkWakeSleep_NewSleepIndex_3Blocks(b *testing.B) {
	blk := benchIndexBundle(b, 3)
	opts := benchSleepOptions()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkIndex, wakeSleepBenchSinkErr = NewSleepIndex(blk, opts, "mlx://agent/session-1", "mlx://agent/session-1/bundle")
	}
}

func BenchmarkWakeSleep_NewSleepIndex_100Blocks(b *testing.B) {
	blk := benchIndexBundle(b, 100)
	opts := benchSleepOptions()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkIndex, wakeSleepBenchSinkErr = NewSleepIndex(blk, opts, "mlx://agent/session-1", "mlx://agent/session-1/bundle")
	}
}

// --- NewSleepReport — stamped report struct, fired once per Sleep.

func BenchmarkWakeSleep_NewSleepReport(b *testing.B) {
	blk := benchIndexBundle(b, 10)
	opts := benchSleepOptions()
	idx, err := NewSleepIndex(blk, opts, "mlx://agent/session-1", "mlx://agent/session-1/bundle")
	if err != nil {
		b.Fatalf("NewSleepIndex: %v", err)
	}
	bundleRef := state.ChunkRef{ChunkID: 1, FrameOffset: 64, HasFrameOffset: true}
	indexRef := state.ChunkRef{ChunkID: 2, FrameOffset: 256, HasFrameOffset: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkReport = NewSleepReport(idx, blk, opts, "mlx://agent/session-1", "mlx://agent/session-1/bundle", "mlx://agent/session-1/index", bundleRef, indexRef)
	}
}

// --- WakeReportFromSleep — converts SleepReport back into a WakeReport
// (used after a successful sleep when the caller wants to continue
// in-process without going through the LoadStateIndex round-trip).

func BenchmarkWakeSleep_WakeReportFromSleep(b *testing.B) {
	report := &SleepReport{
		IndexURI:     "mlx://agent/session-1/index",
		EntryURI:     "mlx://agent/session-1",
		BundleURI:    "mlx://agent/session-1/bundle",
		Title:        "session-1",
		TokenCount:   2048,
		BlockSize:    512,
		KVEncoding:   kv.EncodingNative,
		IndexHash:    "deadbeef",
		SnapshotHash: "feed1234",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkWake = WakeReportFromSleep(report)
	}
}

// --- CloneWakeReport — defensive copy used by callers that want to
// retain a stable snapshot of the report after the runtime continues
// mutating state.

func BenchmarkWakeSleep_CloneWakeReport_Populated(b *testing.B) {
	report := &WakeReport{
		IndexURI:     "mlx://agent/session-1/index",
		EntryURI:     "mlx://agent/session-1",
		BundleURI:    "mlx://agent/session-1/bundle",
		Title:        "session-1",
		PrefixTokens: 2048,
		BundleTokens: 4096,
		BlockSize:    512,
		BlocksRead:   8,
		IndexHash:    "deadbeef",
		SnapshotHash: "feed1234",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkWake = CloneWakeReport(report)
	}
}

func BenchmarkWakeSleep_CloneWakeReport_Nil(b *testing.B) {
	var report *WakeReport
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkWake = CloneWakeReport(report)
	}
}

// --- sleepEntryMeta — pure data shape. Hit once per Sleep. The
// branches that conditionally seed the parent_* keys are worth
// timing separately.

func BenchmarkWakeSleep_SleepEntryMeta_AllParentsSet(b *testing.B) {
	opts := benchSleepOptions()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkPlan = nil // keep wakeSleepBenchSinkPlan referenced
		_ = sleepEntryMeta(opts)
	}
}

func BenchmarkWakeSleep_SleepEntryMeta_NoParents(b *testing.B) {
	opts := benchSleepOptions()
	opts.ParentEntryURI = ""
	opts.ParentBundleURI = ""
	opts.ParentIndexURI = ""
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sleepEntryMeta(opts)
	}
}

func BenchmarkWakeSleep_SleepEntryMeta_NoMeta(b *testing.B) {
	// No meta map + no parents — exercises the all-nil path.
	opts := SleepOptions{Title: "bare"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = sleepEntryMeta(opts)
	}
}

// --- blocksNeededForPrefix — block walk by token boundary. Fires
// inside PlanWake; cost scales with block count up to the prefix.

func BenchmarkWakeSleep_BlocksNeededForPrefix_AllBlocks(b *testing.B) {
	blk := benchIndexBundle(b, 100)
	prefix := blk.TokenCount
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkInt = blocksNeededForPrefix(blk, prefix)
	}
}

func BenchmarkWakeSleep_BlocksNeededForPrefix_FirstBlock(b *testing.B) {
	blk := benchIndexBundle(b, 100)
	prefix := 1
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkInt = blocksNeededForPrefix(blk, prefix)
	}
}

func BenchmarkWakeSleep_BlocksNeededForPrefix_HalfWay(b *testing.B) {
	blk := benchIndexBundle(b, 100)
	prefix := blk.TokenCount / 2
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkInt = blocksNeededForPrefix(blk, prefix)
	}
}

// --- PlanWake — full plan-only path (no KV load). Hit on every
// LoadWakeSnapshot before the heavy block load.
// The bundle + index live in an in-memory state store seeded once;
// each iteration walks PlanWake's full flow.

func BenchmarkWakeSleep_PlanWake_SmallIndex(b *testing.B) {
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)
	blk := benchIndexBundle(b, 3)
	if _, err := kv.SaveStateBlockBundle(ctx, store, blk, "mlx://bench/bundle"); err != nil {
		b.Fatalf("SaveStateBlockBundle: %v", err)
	}
	idx, err := NewStateIndex(blk, benchIndexOptions("mlx://bench/bundle", benchIndexEntries(3)))
	if err != nil {
		b.Fatalf("NewStateIndex: %v", err)
	}
	opts := WakeOptions{
		Index:                  idx,
		EntryURI:               idx.Entries[0].URI,
		Tokenizer:              bundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"},
		SkipCompatibilityCheck: false,
	}
	info := memory.ModelInfo{Architecture: "qwen3", NumLayers: 28, QuantBits: 4, ContextLength: 40960}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		wakeSleepBenchSinkPlan, wakeSleepBenchSinkErr = PlanWake(ctx, store, opts, info)
	}
}
