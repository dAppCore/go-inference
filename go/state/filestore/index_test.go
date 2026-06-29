// SPDX-Licence-Identifier: EUPL-1.2

// filestore index-rebuild tests: invalid header and corrupt-record rejection, capacity hint and rebuilt-index shape.
package filestore

import (
	"context"
	"testing"

	core "dappco.re/go"
	state "dappco.re/go/inference/state"
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
	for id, entry := range store.index {
		putIndex[id] = entry
	}
	putURIIndex := make(map[string]int, len(store.uriIndex))
	for uri, id := range store.uriIndex {
		putURIIndex[uri] = id
	}
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
