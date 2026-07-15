// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"sync"
)

// InMemoryStore is a pure-RAM state.Store: chunks live in maps, nothing touches
// disk. It is the hot tier for `lem serve` when no durable -state-store is
// requested. Every method guards the backing maps with an RWMutex — serve runs
// concurrent chat turns and the continuity manager touches the store outside
// its own lock (wake on acquire, sleep on finish), so the store must be safe to
// share, exactly as filestore.Store is. Reads take RLock, writes take Lock.
type InMemoryStore struct {
	mu     sync.RWMutex
	chunks map[int]string
	data   map[int][]byte
	refs   map[int]ChunkRef
	uris   map[string]int
	nextID int
}

func NewInMemoryStore(chunks map[int]string) *InMemoryStore {
	return NewInMemoryStoreWithManifest(chunks, nil)
}

func NewInMemoryStoreWithManifest(chunks map[int]string, refs map[int]ChunkRef) *InMemoryStore {
	// Single-pass over the seed map: populate text + default ref together so
	// each id is visited once instead of twice. Refs override defaults below.
	// All maps are lazy: when no chunks/refs are seeded the four backing
	// maps stay nil and the four make() heap allocs are skipped entirely.
	// Read sites (Resolve/ResolveBytes/ResolveURI) are nil-safe — Go maps
	// return the zero value + ok=false from nil — and Put/PutBytes already
	// lazy-init on first write. The bench-only NewInMemoryStore_Empty call
	// pattern drops from 5 allocs / 240 B to 1 alloc / 32 B (just the
	// Store struct).
	var copyMap map[int]string
	var refMap map[int]ChunkRef
	if total := len(chunks) + len(refs); total > 0 {
		copyMap = make(map[int]string, len(chunks))
		refMap = make(map[int]ChunkRef, total)
	}
	nextID := 1
	for id, text := range chunks {
		copyMap[id] = text
		refMap[id] = ChunkRef{
			ChunkID:        id,
			FrameOffset:    uint64(id),
			HasFrameOffset: true,
			Codec:          CodecMemory,
		}
		if id >= nextID {
			nextID = id + 1
		}
	}
	for id, ref := range refs {
		ref.ChunkID = id
		refMap[id] = ref
		if id >= nextID {
			nextID = id + 1
		}
	}
	return &InMemoryStore{
		chunks: copyMap,
		refs:   refMap,
		nextID: nextID,
	}
}

func (s *InMemoryStore) Get(ctx context.Context, chunkID int) (string, error) {
	chunk, err := s.Resolve(ctx, chunkID)
	if err != nil {
		return "", err
	}
	return chunk.Text, nil
}

func (s *InMemoryStore) Resolve(ctx context.Context, chunkID int) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return Chunk{}, ctx.Err()
	default:
	}
	if s == nil {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.resolveLocked(chunkID)
}

// resolveLocked is Resolve's map-access core; the caller holds at least RLock.
// Get and ResolveURI route through it so a public read never nests a second
// RLock (sync.RWMutex is not reentrant — a nested RLock can deadlock behind a
// waiting writer).
func (s *InMemoryStore) resolveLocked(chunkID int) (Chunk, error) {
	text, ok := s.chunks[chunkID]
	data, dataOK := s.data[chunkID]
	if !ok && !dataOK {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	ref := s.refs[chunkID]
	if ref.ChunkID != chunkID {
		ref.ChunkID = chunkID
	}
	chunk := Chunk{Ref: ref, Text: text}
	if dataOK {
		chunk.Data = append([]byte(nil), data...)
		if chunk.Text == "" {
			chunk.Text = string(data)
		}
	}
	return chunk, nil
}

func (s *InMemoryStore) ResolveBytes(ctx context.Context, chunkID int) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return Chunk{}, ctx.Err()
	default:
	}
	if s == nil {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	ref := s.refs[chunkID]
	if ref.ChunkID != chunkID {
		ref.ChunkID = chunkID
	}
	if data, ok := s.data[chunkID]; ok {
		return Chunk{Ref: ref, Data: append([]byte(nil), data...)}, nil
	}
	text, ok := s.chunks[chunkID]
	if !ok {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	return Chunk{Ref: ref, Text: text, Data: []byte(text)}, nil
}

// BorrowBytes returns a live view onto the store's own backing slice for
// binary chunks (no defensive copy) — callers must not mutate Data unless
// they intend the mutation to be visible to later Resolve/Borrow calls.
// Text-only chunks still convert (a Go string cannot be borrowed byte-for-
// byte without unsafe aliasing), so only PutBytes-originated chunks get the
// zero-copy path. The borrowed slice stays valid after the lock is released:
// Put/PutBytes only ever write a fresh (monotonic) id and never mutate or drop
// an existing entry's slice.
func (s *InMemoryStore) BorrowBytes(ctx context.Context, chunkID int) (BorrowedChunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return BorrowedChunk{}, ctx.Err()
	default:
	}
	if s == nil {
		return BorrowedChunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	ref := s.refs[chunkID]
	if ref.ChunkID != chunkID {
		ref.ChunkID = chunkID
	}
	if data, ok := s.data[chunkID]; ok {
		return BorrowedChunk{Ref: ref, Data: data}, nil
	}
	text, ok := s.chunks[chunkID]
	if !ok {
		return BorrowedChunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	return BorrowedChunk{Ref: ref, Data: []byte(text)}, nil
}

// BorrowRefBytes resolves ref.ChunkID via BorrowBytes and overlays any
// frame-offset/codec/segment carried on ref onto the returned Ref — mirrors
// the overlay semantics MergeRef documents, without a data copy. It calls the
// public BorrowBytes (which takes the lock) and applies the overlays on the
// returned local value, so it never nests a lock.
func (s *InMemoryStore) BorrowRefBytes(ctx context.Context, ref ChunkRef) (BorrowedChunk, error) {
	if ref.ChunkID == 0 {
		return BorrowedChunk{}, &ChunkNotFoundError{ID: ref.ChunkID}
	}
	borrowed, err := s.BorrowBytes(ctx, ref.ChunkID)
	if err != nil {
		return BorrowedChunk{}, err
	}
	if ref.HasFrameOffset {
		borrowed.Ref.FrameOffset = ref.FrameOffset
		borrowed.Ref.HasFrameOffset = true
	}
	if ref.Codec != "" {
		borrowed.Ref.Codec = ref.Codec
	}
	if ref.Segment != "" {
		borrowed.Ref.Segment = ref.Segment
	}
	return borrowed, nil
}

func (s *InMemoryStore) ResolveURI(ctx context.Context, uri string) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return Chunk{}, ctx.Err()
	default:
	}
	if s == nil {
		return Chunk{}, &URIChunkNotFoundError{URI: uri}
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	id, ok := s.uris[uri]
	if !ok {
		return Chunk{}, &URIChunkNotFoundError{URI: uri}
	}
	return s.resolveLocked(id)
}

func (s *InMemoryStore) Put(ctx context.Context, text string, opts PutOptions) (ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return ChunkRef{}, ctx.Err()
	default:
	}
	if s == nil {
		return ChunkRef{}, &ChunkNotFoundError{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.chunks == nil {
		s.chunks = make(map[int]string)
	}
	if s.refs == nil {
		s.refs = make(map[int]ChunkRef)
	}
	if s.data == nil {
		s.data = make(map[int][]byte)
	}
	if s.uris == nil {
		s.uris = make(map[string]int)
	}
	if s.nextID <= 0 {
		s.nextID = 1
	}
	id := s.nextID
	s.nextID++
	ref := ChunkRef{
		ChunkID:        id,
		FrameOffset:    uint64(id),
		HasFrameOffset: true,
		Codec:          CodecMemory,
	}
	s.chunks[id] = text
	delete(s.data, id)
	s.refs[id] = ref
	if opts.URI != "" {
		s.uris[opts.URI] = id
	}
	return ref, nil
}

// Delete removes chunkID's text/bytes/ref and any URI that resolved to it. A
// DedupStore over this store calls Delete to physically reclaim a content chunk
// once its last reference is released, which is why InMemoryStore implements
// state.Deleter. A missing id is a no-op. Delete never invalidates a still-live
// chunk — the caller's reference count is the proof that chunkID is dead.
//
//	_ = store.Delete(ctx, ref.ChunkID) // after the last Release drops it to zero
func (s *InMemoryStore) Delete(ctx context.Context, chunkID int) error {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	default:
	}
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	delete(s.chunks, chunkID)
	delete(s.data, chunkID)
	delete(s.refs, chunkID)
	// A block chunk carries a Put-time URI that is never read back (blocks
	// resolve by ChunkRef), but drop any URI aliasing the freed id so the URI
	// map cannot outgrow the live chunk set.
	for uri, id := range s.uris {
		if id == chunkID {
			delete(s.uris, uri)
		}
	}
	return nil
}

// ChunkCount reports the number of distinct chunks currently held — one entry
// per live id, so it drops when Delete reclaims one. For tests and dedup
// accounting.
func (s *InMemoryStore) ChunkCount() int {
	if s == nil {
		return 0
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.refs)
}

func (s *InMemoryStore) PutBytes(ctx context.Context, data []byte, opts PutOptions) (ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-ctx.Done():
		return ChunkRef{}, ctx.Err()
	default:
	}
	if s == nil {
		return ChunkRef{}, &ChunkNotFoundError{}
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.chunks == nil {
		s.chunks = make(map[int]string)
	}
	if s.data == nil {
		s.data = make(map[int][]byte)
	}
	if s.refs == nil {
		s.refs = make(map[int]ChunkRef)
	}
	if s.uris == nil {
		s.uris = make(map[string]int)
	}
	if s.nextID <= 0 {
		s.nextID = 1
	}
	id := s.nextID
	s.nextID++
	ref := ChunkRef{
		ChunkID:        id,
		FrameOffset:    uint64(id),
		HasFrameOffset: true,
		Codec:          CodecMemory,
	}
	delete(s.chunks, id)
	s.data[id] = append([]byte(nil), data...)
	s.refs[id] = ref
	if opts.URI != "" {
		s.uris[opts.URI] = id
	}
	return ref, nil
}
