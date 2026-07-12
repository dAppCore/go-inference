// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"sync"

	core "dappco.re/go"
)

// DedupStore is a content-addressed write-dedup skin over any state.Store: two
// byte payloads with identical content are stored once and share a single
// chunk, so N conversations whose KV blocks overlap a common prefix keep only
// one physical copy of each shared block. Dedup falls out of identity — a
// chunk's address IS its content hash — which is what makes cross-bundle
// sharing safe: no conversation owns a shared chunk, so reclaiming one
// conversation can never pull a block out from under another.
//
// A per-chunk reference count records how many writes resolved to each chunk.
// Release decrements it, and a chunk is physically reclaimed — via an optional
// state.Deleter on the inner store — only when its last reference is released.
// An inner store without a Deleter keeps a zero-referenced chunk resident
// (immortal but safe), never a dangling ref.
//
// Writes: PutBytes is deduped by content; Put (text) passes straight through so
// URI-addressed manifests and wake indexes keep their exact ResolveURI
// semantics — only the ref-addressed binary blocks, where the bytes and the
// savings live, share. Reads pass through to the inner store unchanged, so a
// spilling/reviving inner store (ramspill) pages shared chunks in and out
// transparently and a wake stays byte-identical and unaware.
//
//	inner := state.NewInMemoryStore(nil)
//	store := state.NewDedupStore(inner)
//	// two conversations that sleep a shared prefix now store it once:
//	_, _ = store.PutBytes(ctx, prefixBlock, state.PutOptions{}) // conversation A: written
//	_, _ = store.PutBytes(ctx, prefixBlock, state.PutOptions{}) // conversation B: deduped
//
// Safe for concurrent use.
type DedupStore struct {
	inner   Store
	writer  Writer
	binary  BinaryWriter
	deleter Deleter

	mu     sync.Mutex
	byHash map[string]dedupEntry
	byID   map[int]string
	stats  DedupStats
}

// dedupEntry is one content chunk: the ref every referencing bundle resolves it
// by, and the live reference count that gates reclamation.
type dedupEntry struct {
	ref  ChunkRef
	refs int
}

// DedupStats is a snapshot of a DedupStore's write-sharing accounting — the
// store-savings receipt reads it. Writes counts payloads physically written;
// Dedups counts payloads whose write was avoided (the savings). BytesWritten
// and BytesDeduped are the same split in bytes.
type DedupStats struct {
	UniqueChunks int   // distinct content chunks currently tracked
	Writes       int64 // byte payloads physically written to the inner store
	Dedups       int64 // byte payloads served from an existing chunk (writes avoided)
	BytesWritten int64 // total bytes physically written
	BytesDeduped int64 // total bytes whose physical write dedup avoided
	Released     int64 // references released
	Reclaimed    int64 // chunks physically deleted after their last release
}

// NewDedupStore wraps inner in a content-addressed dedup layer. It probes inner
// for the Writer, BinaryWriter, and Deleter capabilities once: PutBytes dedups
// only when inner is a BinaryWriter, and Release reclaims only when inner is a
// Deleter. A read-only or non-binary inner still yields a usable transparent
// read proxy — the write methods report the missing capability plainly.
//
//	store := state.NewDedupStore(state.NewInMemoryStore(nil))
func NewDedupStore(inner Store) *DedupStore {
	d := &DedupStore{
		inner:  inner,
		byHash: make(map[string]dedupEntry),
		byID:   make(map[int]string),
	}
	if w, ok := inner.(Writer); ok {
		d.writer = w
	}
	if b, ok := inner.(BinaryWriter); ok {
		d.binary = b
	}
	if del, ok := inner.(Deleter); ok {
		d.deleter = del
	}
	return d
}

// Put passes a text write straight through to the inner store without dedup.
// Manifests and indexes are URI-addressed, and deduping a URI-addressed write
// would leave the second write's URI unregistered in the inner store — so text
// stays per-writer and fully ResolveURI-able.
func (d *DedupStore) Put(ctx context.Context, text string, opts PutOptions) (ChunkRef, error) {
	if d == nil || d.writer == nil {
		return ChunkRef{}, core.E("state.DedupStore.Put", "inner store is not a Writer", nil)
	}
	return d.writer.Put(ctx, text, opts)
}

// PutBytes stores data content-addressed: a hash already held returns the
// existing chunk's ref and writes nothing (a dedup hit), otherwise it writes
// once through the inner store and records the hash. The returned ref is the
// canonical chunk for that content — every referencing bundle resolves the same
// id, which is what lets two conversations share one physical block. An inner
// store that is not a BinaryWriter cannot take a stable content write, so this
// reports that plainly rather than dedup a write it cannot make.
func (d *DedupStore) PutBytes(ctx context.Context, data []byte, opts PutOptions) (ChunkRef, error) {
	if d == nil || d.binary == nil {
		return ChunkRef{}, core.E("state.DedupStore.PutBytes", "inner store is not a BinaryWriter", nil)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return ChunkRef{}, ctx.Err()
	default:
	}
	// Hash outside the lock — it is pure and O(payload); the lock then guards
	// only the check-write-record so a concurrent identical write serialises to
	// a single physical chunk with no orphan.
	hash := core.SHA256Hex(data)
	d.mu.Lock()
	defer d.mu.Unlock()
	if entry, ok := d.byHash[hash]; ok {
		entry.refs++
		d.byHash[hash] = entry
		d.stats.Dedups++
		d.stats.BytesDeduped += int64(len(data))
		return entry.ref, nil
	}
	ref, err := d.binary.PutBytes(ctx, data, opts)
	if err != nil {
		return ChunkRef{}, err
	}
	d.byHash[hash] = dedupEntry{ref: ref, refs: 1}
	d.byID[ref.ChunkID] = hash
	d.stats.Writes++
	d.stats.BytesWritten += int64(len(data))
	return ref, nil
}

// Release decrements the reference count of each ref's content chunk. When a
// chunk's last reference is released it is physically reclaimed via the inner
// store's Deleter, or — when the inner store has no Deleter — kept resident and
// still dedup-tracked (safe: an unreferenced chunk is dead, and a live one is
// never reached here). A ref the store never deduped (a manifest written through
// Put, or a foreign id) is skipped. This is how a conversation is reclaimed
// without breaking another: shared blocks keep a reference from every sharer, so
// only the truly-private blocks reach zero.
//
//	// reclaim conversation A: release the block refs its bundle recorded
//	_ = store.Release(ctx, aBlockRefs...)
func (d *DedupStore) Release(ctx context.Context, refs ...ChunkRef) error {
	if d == nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	var reclaim []int
	for _, r := range refs {
		hash, ok := d.byID[r.ChunkID]
		if !ok {
			continue
		}
		entry := d.byHash[hash]
		if entry.refs > 0 {
			entry.refs--
			d.stats.Released++
		}
		if entry.refs > 0 {
			d.byHash[hash] = entry
			continue
		}
		if d.deleter == nil {
			// No physical reclamation available: keep the chunk resident and
			// dedup-tracked. It is unreferenced but immortal — safe, never a
			// dangling ref, and still available to dedup a later identical
			// write.
			d.byHash[hash] = entry
			continue
		}
		delete(d.byHash, hash)
		delete(d.byID, r.ChunkID)
		reclaim = append(reclaim, r.ChunkID)
	}
	for _, id := range reclaim {
		if err := d.deleter.Delete(ctx, id); err != nil {
			return core.E("state.DedupStore.Release", "reclaim chunk", err)
		}
		d.stats.Reclaimed++
	}
	return nil
}

// Stats returns a snapshot of the write-sharing accounting.
func (d *DedupStore) Stats() DedupStats {
	if d == nil {
		return DedupStats{}
	}
	d.mu.Lock()
	defer d.mu.Unlock()
	snapshot := d.stats
	snapshot.UniqueChunks = len(d.byHash)
	return snapshot
}

// Get resolves a chunk's text through the inner store.
func (d *DedupStore) Get(ctx context.Context, chunkID int) (string, error) {
	if d == nil || d.inner == nil {
		return "", &ChunkNotFoundError{ID: chunkID}
	}
	return d.inner.Get(ctx, chunkID)
}

// Resolve resolves a chunk through the inner store — reads are never deduped,
// the shared chunk simply lives in the inner store under one id.
func (d *DedupStore) Resolve(ctx context.Context, chunkID int) (Chunk, error) {
	if d == nil || d.inner == nil {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	return Resolve(ctx, d.inner, chunkID)
}

// ResolveBytes resolves a chunk's bytes through the inner store.
func (d *DedupStore) ResolveBytes(ctx context.Context, chunkID int) (Chunk, error) {
	if d == nil || d.inner == nil {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	return ResolveBytes(ctx, d.inner, chunkID)
}

// ResolveRefBytes resolves a chunk ref's bytes through the inner store,
// preserving the ref's frame-offset/codec/segment overlay for backends (like
// filestore) that resolve by more than the chunk id.
func (d *DedupStore) ResolveRefBytes(ctx context.Context, ref ChunkRef) (Chunk, error) {
	if d == nil || d.inner == nil {
		return Chunk{}, &ChunkNotFoundError{ID: ref.ChunkID}
	}
	return ResolveRefBytes(ctx, d.inner, ref)
}

// ResolveURI resolves a chunk by its Put-time URI through the inner store.
func (d *DedupStore) ResolveURI(ctx context.Context, uri string) (Chunk, error) {
	if d == nil || d.inner == nil {
		return Chunk{}, &URIChunkNotFoundError{URI: uri}
	}
	return ResolveURI(ctx, d.inner, uri)
}

// BorrowBytes borrows a chunk's bytes through the inner store.
func (d *DedupStore) BorrowBytes(ctx context.Context, chunkID int) (BorrowedChunk, error) {
	if d == nil || d.inner == nil {
		return BorrowedChunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	return BorrowBytes(ctx, d.inner, chunkID)
}

// BorrowRefBytes borrows a chunk ref's bytes through the inner store, preserving
// the ref overlay as ResolveRefBytes does.
func (d *DedupStore) BorrowRefBytes(ctx context.Context, ref ChunkRef) (BorrowedChunk, error) {
	if d == nil || d.inner == nil {
		return BorrowedChunk{}, &ChunkNotFoundError{ID: ref.ChunkID}
	}
	return BorrowRefBytes(ctx, d.inner, ref)
}
