// SPDX-Licence-Identifier: EUPL-1.2

package blockcache

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	state "dappco.re/go/inference/state"
)

// recordingStateWriter is a test stub that returns a fixed ChunkRef and records
// the last payload it received. It lets the State cold-store success path be
// driven with a ChunkRef whose optional fields (Codec/Segment/FrameOffset) are
// all populated, exercising the withStateLabels label-emission branches that the
// in-memory store leaves empty.
type recordingStateWriter struct {
	ref state.ChunkRef
}

func (w recordingStateWriter) Put(_ context.Context, _ string, _ state.PutOptions) (state.ChunkRef, error) {
	return w.ref, nil
}

// ---------------------------------------------------------------------------
// Unexported helper branch coverage — pure logic, no filesystem.
// ---------------------------------------------------------------------------

func TestBlockCacheHelpers_Branches(t *testing.T) {
	// stateStore on a nil receiver returns nil rather than dereferencing the
	// nil *Service (the early-return guard the public callers never hit because
	// they reject nil first).
	if (*Service)(nil).stateStore() != nil {
		t.Fatal("stateStore(nil service) != nil")
	}
	if (*Service)(nil).stateStoreEnabled() {
		t.Fatal("stateStoreEnabled(nil service) = true")
	}
	if (*Service)(nil).diskEnabled() {
		t.Fatal("diskEnabled(nil service) = true")
	}

	// blockRefs clamps a non-positive BlockSize to DefaultBlockSize. A Service
	// literal with BlockSize 0 (bypassing New's clamp) chunks a short token run
	// into a single default-sized block.
	zeroSizeService := &Service{cfg: Config{}, blocks: map[string]inference.CacheBlockRef{}}
	refs := zeroSizeService.blockRefs(inference.CacheWarmRequest{}, []int32{1, 2, 3}, nil)
	if len(refs) != 1 || refs[0].TokenCount != 3 {
		t.Fatalf("blockRefs(BlockSize=0) = %+v, want one default-sized block", refs)
	}

	// diskRecordCompatible rejects an empty-ID record outright, and rejects a
	// record whose adapter hash mismatches the configured identity.
	service := &Service{cfg: Config{
		ModelHash:     "sha256:model",
		AdapterHash:   "sha256:adapter",
		TokenizerHash: "sha256:tokenizer",
	}}
	if service.diskRecordCompatible(diskRecord{}) {
		t.Fatal("diskRecordCompatible(empty ID) = true")
	}
	if service.diskRecordCompatible(diskRecord{Ref: inference.CacheBlockRef{
		ID:          "x",
		ModelHash:   "sha256:model",
		AdapterHash: "sha256:other-adapter",
	}}) {
		t.Fatal("diskRecordCompatible(adapter mismatch) = true")
	}

	// blockRefMatchesLabels rejects an adapter_hash mismatch and a
	// tokenizer_hash mismatch (the two switch arms the existing Good test does
	// not flip individually).
	ref := inference.CacheBlockRef{ModelHash: "m", AdapterHash: "a", TokenizerHash: "t"}
	if blockRefMatchesLabels(ref, map[string]string{"adapter_hash": "other"}) {
		t.Fatal("blockRefMatchesLabels(adapter mismatch) = true")
	}
	if blockRefMatchesLabels(ref, map[string]string{"tokenizer_hash": "other"}) {
		t.Fatal("blockRefMatchesLabels(tokenizer mismatch) = true")
	}

	// cloneBlockCacheLabelsExtra clamps a negative extra to zero rather than
	// passing a negative capacity hint to make.
	cloned := cloneBlockCacheLabelsExtra(map[string]string{"a": "b"}, -4)
	if cloned["a"] != "b" {
		t.Fatalf("cloneBlockCacheLabelsExtra(extra<0) = %+v, want copied entry", cloned)
	}

	// writeStateBlock guards a nil store: a Service whose state store is unset
	// returns an explicit error instead of calling Put on nil.
	if _, err := (&Service{}).writeStateBlock(context.Background(), inference.CacheBlockRef{ID: "x"}, nil); err == nil {
		t.Fatal("writeStateBlock(nil store) error = nil")
	}
	// writeStateBlock substitutes context.Background for a nil context (the
	// documented fast path) — with a recording store the call succeeds.
	okStore := &Service{cfg: Config{StateStore: recordingStateWriter{}}}
	//nolint:staticcheck // SA1012: passing a nil Context is the path under test.
	if _, err := okStore.writeStateBlock(nil, inference.CacheBlockRef{ID: "x"}, []int32{1}); err != nil {
		t.Fatalf("writeStateBlock(nil ctx) error = %v, want nil", err)
	}
}

// TestBlockCacheHelpers_WithStateLabels exercises every optional-field arm of
// withStateLabels: a ChunkRef carrying a codec, a segment, and a frame offset
// emits the corresponding state_* labels, which the in-memory store path leaves
// unset.
func TestBlockCacheHelpers_WithStateLabels(t *testing.T) {
	labelled := withStateLabels(inference.CacheBlockRef{ID: "x"}, state.ChunkRef{
		ChunkID:        7,
		Codec:          "zstd",
		Segment:        "seg-1",
		HasFrameOffset: true,
		FrameOffset:    42,
	})
	if labelled.Labels["cold_store"] != "state" {
		t.Fatalf("cold_store label = %q, want state", labelled.Labels["cold_store"])
	}
	if labelled.Labels["state_chunk_id"] != "7" {
		t.Fatalf("state_chunk_id = %q, want 7", labelled.Labels["state_chunk_id"])
	}
	if labelled.Labels["state_codec"] != "zstd" {
		t.Fatalf("state_codec = %q, want zstd", labelled.Labels["state_codec"])
	}
	if labelled.Labels["state_segment"] != "seg-1" {
		t.Fatalf("state_segment = %q, want seg-1", labelled.Labels["state_segment"])
	}
	if labelled.Labels["state_frame_offset"] != "42" {
		t.Fatalf("state_frame_offset = %q, want 42", labelled.Labels["state_frame_offset"])
	}
}

// TestBlockCacheHelpers_SortPdqsort drives sortCacheBlockRefs past its
// insertion-sort threshold (32) so the pdqsort branch executes. The input is
// reverse-ordered by TokenStart; the result must be ascending.
func TestBlockCacheHelpers_SortPdqsort(t *testing.T) {
	const n = sortCacheBlockRefsInsertionThreshold + 8 // 40 > 32
	refs := make([]inference.CacheBlockRef, n)
	for i := range refs {
		refs[i] = inference.CacheBlockRef{
			ID:         core.Itoa(n - i),
			TokenStart: (n - i) * 2,
		}
	}
	sortCacheBlockRefs(refs)
	for i := 1; i < len(refs); i++ {
		if refs[i-1].TokenStart > refs[i].TokenStart {
			t.Fatalf("sortCacheBlockRefs(n=%d) not ascending at %d: %d > %d", n, i, refs[i-1].TokenStart, refs[i].TokenStart)
		}
	}
}

// ---------------------------------------------------------------------------
// Disk-load error propagation — a DiskPath whose parent is a regular file makes
// the lazy ensureDiskLoadedLocked MkdirAll fail, and every public method surfaces
// that failure on first touch (diskLoaded is false until a load succeeds).
// ---------------------------------------------------------------------------

// unwritableDiskPath returns a DiskPath that cannot be created because a parent
// path component is a regular file, so core.MkdirAll fails.
func unwritableDiskPath(t *testing.T) string {
	t.Helper()
	parent := core.PathJoin(t.TempDir(), "afile")
	if result := core.WriteFile(parent, []byte("x"), 0o600); !result.OK {
		t.Fatalf("WriteFile(parent) error = %s", result.Error())
	}
	return core.PathJoin(parent, "blocks")
}

func TestBlockcache_Service_DiskLoadFailurePropagates(t *testing.T) {
	ctx := context.Background()
	// CacheStats surfaces the ensureDiskLoadedLocked MkdirAll failure.
	if _, err := New(Config{DiskPath: unwritableDiskPath(t)}).CacheStats(ctx); err == nil {
		t.Fatal("CacheStats(unwritable disk) error = nil")
	}
	// CacheEntries surfaces the same failure.
	if _, err := New(Config{DiskPath: unwritableDiskPath(t)}).CacheEntries(ctx, nil); err == nil {
		t.Fatal("CacheEntries(unwritable disk) error = nil")
	}
	// WarmCache surfaces the same failure before any block is recorded.
	if _, err := New(Config{DiskPath: unwritableDiskPath(t)}).WarmCache(ctx, inference.CacheWarmRequest{Tokens: []int32{1, 2}}); err == nil {
		t.Fatal("WarmCache(unwritable disk) error = nil")
	}
	// ClearCache surfaces the same failure.
	if _, err := New(Config{DiskPath: unwritableDiskPath(t)}).ClearCache(ctx, nil); err == nil {
		t.Fatal("ClearCache(unwritable disk) error = nil")
	}
}

// TestBlockcache_Service_DiskRecordUnreadableQuarantined drives the
// readDiskRecord read-failure branch and the quarantine path: a *directory*
// named like a block record is matched by the "*.json" glob but cannot be read
// as a file, so it is quarantined (counted corrupt + evicted) on load.
func TestBlockcache_Service_DiskRecordUnreadableQuarantined(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll(diskPath) error = %s", result.Error())
	}
	// A directory entry that matches *.json: PathGlob returns it, ReadFile on a
	// directory fails, so readDiskRecord reports not-ok and the loader
	// quarantines it.
	if result := core.MkdirAll(core.PathJoin(diskPath, "asdir.json"), 0o700); !result.OK {
		t.Fatalf("MkdirAll(asdir.json) error = %s", result.Error())
	}
	stats, err := New(Config{DiskPath: diskPath}).CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	if stats.Blocks != 0 || stats.Evictions != 1 || stats.Labels["disk_corrupt"] != "1" {
		t.Fatalf("stats = %+v, want unreadable record quarantined", stats)
	}
}

// TestBlockcache_Service_WarmCacheWriteFailure drives the writeDiskBlockLocked
// WriteFile-failure branch: a read-only DiskPath directory already exists (so the
// inner MkdirAll no-ops), but the block record cannot be written into it.
func TestBlockcache_Service_WarmCacheWriteFailure(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll(diskPath) error = %s", result.Error())
	}
	service := New(Config{BlockSize: 2, DiskPath: diskPath})
	// Force the first lazy load to complete on the still-writable directory.
	if _, err := service.CacheStats(context.Background()); err != nil {
		t.Fatalf("CacheStats(warm load) error = %v", err)
	}
	if result := core.Chmod(diskPath, 0o500); !result.OK {
		t.Fatalf("Chmod(read-only) error = %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(diskPath, 0o700) })
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{Tokens: []int32{1, 2}}); err == nil {
		t.Fatal("WarmCache(read-only disk) error = nil")
	}
}

// TestBlockcache_Service_ClearCacheRunsRuntimeHook covers the ClearRuntime hook
// invocation on the clear-all path: clearing with nil labels invokes the
// configured runtime-clear callback.
func TestBlockcache_Service_ClearCacheRunsRuntimeHook(t *testing.T) {
	var cleared bool
	service := New(Config{
		BlockSize:    2,
		ModelHash:    "sha256:model",
		ClearRuntime: func() { cleared = true },
	})
	if _, err := service.ClearCache(context.Background(), nil); err != nil {
		t.Fatalf("ClearCache() error = %v", err)
	}
	if !cleared {
		t.Fatal("ClearRuntime hook was not invoked on clear-all")
	}
}

// TestBlockcache_Service_ClearCacheDiskFailure drives the clearDiskLocked
// RemoveAll-failure path on the clear-all branch. After a normal load, the
// DiskPath's parent directory is made read-only, so the post-load RemoveAll
// inside clearDiskLocked cannot unlink the block directory.
func TestBlockcache_Service_ClearCacheDiskFailure(t *testing.T) {
	parent := core.PathJoin(t.TempDir(), "parent")
	diskPath := core.PathJoin(parent, "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll(diskPath) error = %s", result.Error())
	}
	service := New(Config{BlockSize: 2, DiskPath: diskPath})
	if _, err := service.CacheStats(context.Background()); err != nil {
		t.Fatalf("CacheStats(load) error = %v", err)
	}
	// A read-only parent blocks the unlink of the block directory, so
	// clearDiskLocked's RemoveAll fails.
	if result := core.Chmod(parent, 0o500); !result.OK {
		t.Fatalf("Chmod(read-only parent) error = %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(parent, 0o700) })
	if _, err := service.ClearCache(context.Background(), nil); err == nil {
		t.Fatal("ClearCache(disk RemoveAll failure) error = nil")
	}
}

// TestBlockcache_Service_DiskBytesStatFallback covers the diskBytesLocked
// Stat-then-ReadFile fallback: a zero-byte *.json record reports a Stat size of
// zero, so the byte count is taken from the (empty) ReadFile result instead.
// Also covers the diskEnabled-false early return via a non-disk service.
func TestBlockcache_Service_DiskBytesStatFallback(t *testing.T) {
	// diskBytesLocked on a service with no DiskPath returns 0 without touching
	// the filesystem.
	if got := New(Config{}).diskBytesLocked(); got != 0 {
		t.Fatalf("diskBytesLocked(no disk) = %d, want 0", got)
	}

	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll(diskPath) error = %s", result.Error())
	}
	// A zero-byte record file: Stat reports size 0 (info.Size() > 0 is false),
	// so diskBytesLocked falls back to the ReadFile length (also 0).
	if result := core.WriteFile(core.PathJoin(diskPath, "empty.json"), []byte{}, 0o600); !result.OK {
		t.Fatalf("WriteFile(empty record) error = %s", result.Error())
	}
	service := New(Config{DiskPath: diskPath})
	stats, err := service.CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	// The empty record is unreadable as a record (quarantined), but the byte
	// accounting still walks the glob and exercises the Stat/ReadFile fallback.
	if stats.DiskBytes != 0 {
		t.Fatalf("DiskBytes = %d, want 0 for an empty record", stats.DiskBytes)
	}
}

// TestBlockcache_Service_ClearCacheRemoveBlockFailure drives the
// removeDiskBlockLocked error path on the label-scoped clear branch: after a
// labelled block is persisted, the DiskPath directory is made read-only, so
// unlinking the matched block's record file fails and the error is surfaced.
func TestBlockcache_Service_ClearCacheRemoveBlockFailure(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll(diskPath) error = %s", result.Error())
	}
	service := New(Config{BlockSize: 2, ModelHash: "sha256:model", DiskPath: diskPath})
	if _, err := service.WarmCache(context.Background(), inference.CacheWarmRequest{
		Labels: map[string]string{"tenant": "alpha"},
		Tokens: []int32{1, 2},
	}); err != nil {
		t.Fatalf("WarmCache(alpha) error = %v", err)
	}
	// A read-only DiskPath directory blocks the unlink of the record file it
	// contains, so removeDiskBlockLocked's Remove fails.
	if result := core.Chmod(diskPath, 0o500); !result.OK {
		t.Fatalf("Chmod(read-only diskPath) error = %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(diskPath, 0o700) })
	if _, err := service.ClearCache(context.Background(), map[string]string{"tenant": "alpha"}); err == nil {
		t.Fatal("ClearCache(remove block failure) error = nil")
	}
}

// TestBlockcache_Service_QuarantineRemoveFailure drives the quarantineDiskBlock
// best-effort Remove-failure branch: a corrupt record sits in a read-only
// DiskPath, so the loader can glob and read-fail it but cannot unlink it. The
// load still completes (quarantine is best-effort) and the record is counted
// corrupt + evicted.
func TestBlockcache_Service_QuarantineRemoveFailure(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll(diskPath) error = %s", result.Error())
	}
	if result := core.WriteFile(core.PathJoin(diskPath, "broken.json"), []byte("{broken"), 0o600); !result.OK {
		t.Fatalf("WriteFile(corrupt record) error = %s", result.Error())
	}
	// A read-only DiskPath lets the glob + read run but blocks the unlink, so
	// quarantineDiskBlock's Remove fails (best-effort, non-fatal).
	if result := core.Chmod(diskPath, 0o500); !result.OK {
		t.Fatalf("Chmod(read-only diskPath) error = %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(diskPath, 0o700) })
	stats, err := New(Config{DiskPath: diskPath}).CacheStats(context.Background())
	if err != nil {
		t.Fatalf("CacheStats() error = %v", err)
	}
	if stats.Evictions != 1 || stats.Labels["disk_corrupt"] != "1" {
		t.Fatalf("stats = %+v, want corrupt record counted despite failed unlink", stats)
	}
}
