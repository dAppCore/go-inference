// SPDX-Licence-Identifier: EUPL-1.2

// filestore index-rebuild tests: invalid header and corrupt-record rejection, capacity hint and rebuilt-index shape.
package filestore

import (
	"context"
	"encoding/binary"
	"maps"
	"math"
	"testing"
	"time"

	core "dappco.re/go"
	state "dappco.re/go/inference/model/state"
)

func TestFileStore_Bad_InvalidFile(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "invalid.mvlog")
	if result := core.WriteFile(path, []byte("not a state log"), 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	if _, err := Open(context.Background(), path); err == nil {
		t.Fatal("Open(invalid header) error = nil")
	}
}

func TestFileStore_Bad_CorruptRecords(t *testing.T) {
	cases := []struct {
		name string
		data []byte
	}{
		{
			name: "truncated-record-header",
			data: append(append([]byte(nil), fileMagic...), recordMagic[:2]...),
		},
		{
			name: "invalid-record-header",
			data: append(append([]byte(nil), fileMagic...), make([]byte, recordHeaderLen)...),
		},
		{
			name: "truncated-payload",
			data: append(append(append([]byte(nil), fileMagic...), testHeader(1, 4, 0)...), []byte{1, 2}...),
		},
		{
			name: "invalid-metadata",
			data: append(append(append([]byte(nil), fileMagic...), testHeader(1, 0, 1)...), []byte("{")...),
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := core.PathJoin(t.TempDir(), tc.name+".mvlog")
			if result := core.WriteFile(path, tc.data, 0o600); !result.OK {
				t.Fatalf("WriteFile() error = %s", result.Error())
			}
			if _, err := Open(context.Background(), path); err == nil {
				t.Fatalf("Open(%s) error = nil, want corruption error", tc.name)
			}
		})
	}
}

func TestFileStore_Good_IndexCapacityHintSkipsLargePayloadStores(t *testing.T) {
	if got := indexCapacityHint(int64(len(fileMagic))+1024*indexHintRecordBytes, int64(len(fileMagic))); got != 1024 {
		t.Fatalf("small-record hint = %d, want 1024", got)
	}
	if got := indexCapacityHint(int64(len(fileMagic))+indexHintMaxFileBytes+1, int64(len(fileMagic))); got != 0 {
		t.Fatalf("large-payload hint = %d, want 0", got)
	}
	if got := indexCapacityHint(int64(len(fileMagic)), int64(len(fileMagic))); got != 0 {
		t.Fatalf("empty hint = %d, want 0", got)
	}
}

// testHeader is a test-only wrapper that returns a fresh []byte built
// via encodeRecordHeader's in-place API. Production callers should use
// encodeRecordHeader directly with a stack-allocated [recordHeaderLen]byte.
func testHeader(chunkID, payloadSize, metaSize int) []byte {
	buf := make([]byte, recordHeaderLen)
	encodeRecordHeader(buf, chunkID, payloadSize, metaSize)
	return buf
}

// TestFileStore_Good_RebuildIndexPreservesIndexShape pins the index
// shape across rebuildIndex changes — Wave 8 perf rewrites can alter
// how the meta JSON is parsed, but the resulting index entries (per
// chunk id) must match a Put-built index 1:1 in ref + payload offset.
// The uriIndex must contain exactly the URIs that were Put with a
// non-empty URI, mapped to the same chunk ids.
func TestFileStore_Good_RebuildIndexPreservesIndexShape(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "rebuild-shape.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	// Mix records with URI, without URI, with tag-maps + label-slices,
	// with empty meta — covers every branch rebuildIndex touches.
	cases := []state.PutOptions{
		{URI: "mlx://kv/0", Title: "with-uri", Kind: "bench"},
		{}, // empty meta
		{URI: "mlx://kv/2", Tags: map[string]string{"a": "1", "b": "2"}, Labels: []string{"x", "y"}},
		{Kind: "no-uri", Track: "tr"},
		{URI: "mlx://kv/4", Title: "another", Tags: map[string]string{}},
	}
	payloads := [][]byte{
		[]byte("alpha"),
		[]byte("beta"),
		[]byte("gamma"),
		[]byte("delta"),
		[]byte("epsilon"),
	}
	var putRefs []state.ChunkRef
	for i, opts := range cases {
		ref, err := store.PutBytes(ctx, payloads[i], opts)
		if err != nil {
			t.Fatalf("PutBytes(%d) error = %v", i, err)
		}
		putRefs = append(putRefs, ref)
	}
	// Snapshot the live index built by Put for later comparison.
	store.mu.Lock()
	putIndex := make(map[int]fileIndexEntry, len(store.index))
	maps.Copy(putIndex, store.index)
	putURIIndex := make(map[string]int, len(store.uriIndex))
	maps.Copy(putURIIndex, store.uriIndex)
	putNextID := store.nextID
	putWriteAt := store.writeAt
	store.mu.Unlock()
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	reopened, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open() error = %v", err)
	}
	defer reopened.Close()

	reopened.mu.Lock()
	defer reopened.mu.Unlock()

	if reopened.nextID != putNextID {
		t.Fatalf("rebuilt nextID = %d, want %d", reopened.nextID, putNextID)
	}
	if reopened.writeAt != putWriteAt {
		t.Fatalf("rebuilt writeAt = %d, want %d", reopened.writeAt, putWriteAt)
	}
	if len(reopened.index) != len(putIndex) {
		t.Fatalf("rebuilt index size = %d, want %d", len(reopened.index), len(putIndex))
	}
	for id, want := range putIndex {
		got, ok := reopened.index[id]
		if !ok {
			t.Fatalf("rebuilt index missing chunk id %d", id)
		}
		if got.ref != want.ref {
			t.Fatalf("rebuilt entry[%d].ref = %+v, want %+v", id, got.ref, want.ref)
		}
		if got.payloadAt != want.payloadAt {
			t.Fatalf("rebuilt entry[%d].payloadAt = %d, want %d", id, got.payloadAt, want.payloadAt)
		}
		if got.payloadSize != want.payloadSize {
			t.Fatalf("rebuilt entry[%d].payloadSize = %d, want %d", id, got.payloadSize, want.payloadSize)
		}
	}
	if len(reopened.uriIndex) != len(putURIIndex) {
		t.Fatalf("rebuilt uriIndex size = %d, want %d", len(reopened.uriIndex), len(putURIIndex))
	}
	for uri, wantID := range putURIIndex {
		gotID, ok := reopened.uriIndex[uri]
		if !ok {
			t.Fatalf("rebuilt uriIndex missing %q", uri)
		}
		if gotID != wantID {
			t.Fatalf("rebuilt uriIndex[%q] = %d, want %d", uri, gotID, wantID)
		}
	}
	_ = putRefs
}

// rawHeader builds a 24-byte record header from raw uint64/uint32
// field values, bypassing encodeRecordHeader's int-typed parameters.
// encodeRecordHeader cannot represent a chunk id or payload size above
// math.MaxInt64 (its params are plain int), so overflow-branch tests
// that need a field beyond maxInt() craft the bytes directly here.
func rawHeader(chunkID, payloadSize uint64, metaSize uint32) []byte {
	buf := make([]byte, recordHeaderLen)
	copy(buf[:4], recordMagic[:])
	binary.LittleEndian.PutUint64(buf[4:12], chunkID)
	binary.LittleEndian.PutUint64(buf[12:20], payloadSize)
	binary.LittleEndian.PutUint32(buf[20:24], metaSize)
	return buf
}

// openFileOrFatal opens path with flag/mode via core.OpenFile and
// fails the test on error, returning the live *core.OSFile. Several
// rebuildIndex/detectHeaderLen fault-injection tests need a real,
// already-closed file descriptor to force a deterministic Stat/ReadAt
// error without touching global process state (rlimits) or blocking
// I/O (FIFOs) — closing a file we opened ourselves is the cleanest
// hermetic way to do that.
func openFileOrFatal(t *testing.T, path string, flag int) *core.OSFile {
	t.Helper()
	result := core.OpenFile(path, flag, 0o600)
	if !result.OK {
		t.Fatalf("OpenFile(%q) error = %s", path, result.Error())
	}
	return result.Value.(*core.OSFile)
}

func TestRebuildIndex_Bad_StatError(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "closed-before-rebuild.mvlog")
	if result := core.WriteFile(path, fileMagic, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	file := openFileOrFatal(t, path, core.O_RDONLY)
	if err := file.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	s := &Store{file: file, index: map[int]fileIndexEntry{}, uriIndex: map[string]int{}, nextID: 1}
	if err := s.rebuildIndex(context.Background()); err == nil {
		t.Fatal("rebuildIndex(closed file) error = nil, want stat error")
	}
}

func TestRebuildIndex_Bad_RegionSizeError(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "region-too-large.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if _, err := store.PutBytes(ctx, []byte("payload"), state.PutOptions{}); err != nil {
		t.Fatalf("PutBytes() error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	stat := core.Stat(path)
	if !stat.OK {
		t.Fatalf("Stat() error = %s", stat.Error())
	}
	size := stat.Value.(interface{ Size() int64 }).Size()

	// payloadOffset itself is non-negative (passes openRegionWithSegmentAlias's
	// own guard) but exceeds the real file size, so regionSize's
	// baseAt-vs-fileSize check must reject it from inside rebuildIndex.
	if _, err := OpenRegionWithSegmentAlias(ctx, path, size+1024, 0, ""); err == nil {
		t.Fatal("OpenRegionWithSegmentAlias(baseAt beyond EOF) error = nil")
	}
}

// countdownContext is a minimal context.Context whose Done() channel
// stays open for the first `remaining` polls and is closed on the
// poll that brings the counter to zero. It lets a test reach past an
// earlier checkContext gate (e.g. the one at the top of Open) with a
// live context and then simulate cancellation deep inside a later,
// per-record loop — something context.WithCancel cannot do
// deterministically without a real clock-based race.
type countdownContext struct {
	remaining int
	done      chan struct{}
}

func newCountdownContext(remaining int) *countdownContext {
	return &countdownContext{remaining: remaining, done: make(chan struct{})}
}

func (c *countdownContext) Deadline() (time.Time, bool) { return time.Time{}, false }

func (c *countdownContext) Done() <-chan struct{} {
	if c.remaining > 0 {
		c.remaining--
		if c.remaining == 0 {
			close(c.done)
		}
	}
	return c.done
}

func (c *countdownContext) Err() error {
	select {
	case <-c.done:
		return context.Canceled
	default:
		return nil
	}
}

func (c *countdownContext) Value(key any) any { return nil }

func TestRebuildIndex_Bad_ContextCancelledMidLoop(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "cancel-mid-loop.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	if _, err := store.PutBytes(ctx, []byte("first"), state.PutOptions{}); err != nil {
		t.Fatalf("PutBytes(first) error = %v", err)
	}
	if _, err := store.PutBytes(ctx, []byte("second"), state.PutOptions{}); err != nil {
		t.Fatalf("PutBytes(second) error = %v", err)
	}
	if err := store.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	// remaining=2: the 1st Done() poll is Open's top-of-function
	// checkContext (must still pass, so the file actually opens);
	// the 2nd is rebuildIndex's first per-record loop iteration,
	// which must observe cancellation.
	_, err = Open(newCountdownContext(2), path)
	if !core.Is(err, context.Canceled) {
		t.Fatalf("Open(cancel mid-loop) error = %v, want context.Canceled", err)
	}
}

func TestRebuildIndex_Bad_ChunkIDOverflow(t *testing.T) {
	payload := []byte("x")
	header := rawHeader(uint64(1)<<63, uint64(len(payload)), 2)
	data := append(append(append([]byte(nil), fileMagic...), header...), []byte("{}")...)
	data = append(data, payload...)
	path := core.PathJoin(t.TempDir(), "chunkid-overflow.mvlog")
	if result := core.WriteFile(path, data, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	if _, err := Open(context.Background(), path); err == nil {
		t.Fatal("Open(chunk id overflow) error = nil")
	}
}

func TestRebuildIndex_Bad_PayloadSizeOverflow(t *testing.T) {
	header := rawHeader(1, uint64(1)<<63, 0)
	data := append(append([]byte(nil), fileMagic...), header...)
	path := core.PathJoin(t.TempDir(), "payloadsize-overflow.mvlog")
	if result := core.WriteFile(path, data, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	if _, err := Open(context.Background(), path); err == nil {
		t.Fatal("Open(payload size overflow) error = nil")
	}
}

func TestDetectHeaderLen_Bad_ReadAtError(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "closed-before-detect.mvlog")
	if result := core.WriteFile(path, fileMagic, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	file := openFileOrFatal(t, path, core.O_RDONLY)
	if err := file.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	s := &Store{file: file}
	if _, err := s.detectHeaderLen(int64(len(fileMagic))); err == nil {
		t.Fatal("detectHeaderLen(closed file) error = nil, want ReadAt error")
	}
}

func TestDetectHeaderLen_Good_LegacyExactSize(t *testing.T) {
	// A file that is exactly the legacy header's length (shorter than
	// the current fileMagic) exercises detectHeaderLen's mid-range
	// clamp: maxHeaderLen is trimmed down to the actual file size
	// before the magic prefix comparison runs.
	path := core.PathJoin(t.TempDir(), "legacy-header-only.mvlog")
	if result := core.WriteFile(path, legacyFileMagic, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	store, err := Open(context.Background(), path)
	if err != nil {
		t.Fatalf("Open(legacy header only) error = %v", err)
	}
	defer store.Close()
	if store.ChunkCount() != 0 {
		t.Fatalf("ChunkCount() = %d, want 0", store.ChunkCount())
	}
}

func TestDetectHeaderLen_Bad_NoMagicMatch(t *testing.T) {
	// Long enough to clear minHeaderLen (so it isn't rejected by the
	// earlier "missing header" check) but matching neither fileMagic
	// nor legacyFileMagic — the final fallback error.
	path := core.PathJoin(t.TempDir(), "no-magic-match.mvlog")
	garbage := make([]byte, len(fileMagic)+16)
	for i := range garbage {
		garbage[i] = 'X'
	}
	if result := core.WriteFile(path, garbage, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	if _, err := Open(context.Background(), path); err == nil {
		t.Fatal("Open(no magic match) error = nil")
	}
}

func TestRegionSize_Good_Cases(t *testing.T) {
	cases := []struct {
		name     string
		store    *Store
		fileSize int64
		want     int64
	}{
		{"unbounded-uses-available", &Store{baseAt: 10}, 30, 20},
		{"bounded-within-available", &Store{baseAt: 10, region: 5}, 30, 5},
		{"baseAt-zero", &Store{}, 100, 100},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.store.regionSize(tc.fileSize)
			if err != nil {
				t.Fatalf("regionSize() error = %v", err)
			}
			if got != tc.want {
				t.Fatalf("regionSize() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestRegionSize_Bad_Cases(t *testing.T) {
	cases := []struct {
		name     string
		store    *Store
		fileSize int64
	}{
		{"nil-store", nil, 100},
		{"negative-baseAt", &Store{baseAt: -1}, 100},
		{"negative-region", &Store{region: -1}, 100},
		{"baseAt-beyond-fileSize", &Store{baseAt: 200}, 100},
		{"region-beyond-available", &Store{baseAt: 10, region: 1000}, 100},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.store.regionSize(tc.fileSize); err == nil {
				t.Fatalf("regionSize(%s) error = nil, want errRegionInvalid", tc.name)
			}
		})
	}
}

func TestPhysicalOffset_Good_Cases(t *testing.T) {
	cases := []struct {
		name      string
		store     *Store
		logOffset int64
		want      int64
	}{
		{"zero-base-zero-region", &Store{}, 42, 42},
		{"with-base", &Store{baseAt: 100}, 42, 142},
		{"within-region-bound", &Store{baseAt: 10, region: 50}, 50, 60},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, err := tc.store.physicalOffset(tc.logOffset)
			if err != nil {
				t.Fatalf("physicalOffset() error = %v", err)
			}
			if got != tc.want {
				t.Fatalf("physicalOffset() = %d, want %d", got, tc.want)
			}
		})
	}
}

func TestPhysicalOffset_Bad_Cases(t *testing.T) {
	maxOffset := int64(math.MaxInt64)
	cases := []struct {
		name      string
		store     *Store
		logOffset int64
	}{
		{"nil-store", nil, 0},
		{"negative-offset", &Store{}, -1},
		{"exceeds-region", &Store{region: 10}, 11},
		{"baseAt-overflow-guard", &Store{baseAt: 100}, maxOffset - 10},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.store.physicalOffset(tc.logOffset); err == nil {
				t.Fatalf("physicalOffset(%s) error = nil, want errRegionInvalid", tc.name)
			}
		})
	}
}
