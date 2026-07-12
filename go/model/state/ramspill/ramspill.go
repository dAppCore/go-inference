// SPDX-Licence-Identifier: EUPL-1.2

// Package ramspill is the memory-pressure spill policy for a RAM-resident
// conversation state.Store (#48, tiered-KV follow-up 2). `lem serve`'s
// RAM-default conversation store (state.InMemoryStore) grows unbounded —
// every sleep adds chunks and nothing ever frees them. Store wraps that
// growth in kv/kvtier's byte-budget tiering: once resident payload bytes
// exceed Options.Budget, the coldest (least-recently-touched) chunks page
// out to a durable Cold store — in production a scratch `.kv` filestore.Store
// — and page back in transparently the moment anything resolves them again.
//
// Every existing wake/sleep call already goes through the state.Store
// surface (agent.LoadStateIndex, kv.LoadStateBlockBundle, per-block
// Resolve/Borrow calls all take a state.Store), so Store is a drop-in for
// state.InMemoryStore wherever continuity.Enable wants one:
//
//	cold, _ := filestore.Create(ctx, spillPath)
//	store, _ := ramspill.New(ramspill.Options{Budget: 512 << 20, Cold: cold})
//	_ = continuity.Enable(model, store)
//
// The unit an eviction pass spills is the individual chunk (an index blob, a
// bundle manifest, or one KV block payload) rather than a whole named
// conversation — kv/radix's cross-conversation sharing aside, chunks are
// never shared BETWEEN conversations, and every chunk a single sleep writes
// lands with (near enough) the same recency tick, so an idle conversation's
// whole chunk set ages out — and comes back — together in practice.
package ramspill

import (
	"context"
	"strconv"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference/kv/kvtier"
	state "dappco.re/go/inference/model/state"
)

// reviveAttempts bounds Resolve/Borrow's retry against a concurrent eviction
// that re-spills a chunk in the narrow window between kvtier promoting it and
// this Store reading the now-hot bytes back out (kvtier serialises every
// Put/Access under its own lock, but Store's own bytes-read happens after
// that call returns — see access). A budget too small to hold even one
// resident chunk hits this bound legitimately (kvtier promotes then
// immediately re-demotes the sole occupant every pass) and reports a clear
// error instead of spinning forever.
const reviveAttempts = 3

// ColdStore is the durable backing a Store spills to — the minimal surface
// filestore.Store already satisfies. Kept as an interface (not the concrete
// filestore type) so tests can exercise failure paths without touching disk.
type ColdStore interface {
	state.Writer
	state.BinaryWriter
	state.URIResolver
	state.BinaryResolver
}

// Options configures a Store.
type Options struct {
	// Budget is the byte ceiling for RAM-resident chunk payloads. <= 0 means
	// unlimited — every chunk stays hot forever, byte-identical to the
	// pre-#48 InMemoryStore default.
	Budget int64
	// Cold receives spilled payloads. Required when Budget > 0.
	Cold ColdStore
	// Log receives one line per spill/revival (nil silences it) — mirrors
	// serving's multimodel eviction notices.
	Log core.Writer
}

// entry is one tracked chunk. bytes is nil exactly when spilled is true —
// spilled is the authoritative flag (an empty-but-resident payload also
// serialises to a nil byte slice, so bytes alone can't disambiguate).
// size is captured once at Put and never touched by spill/revive, so a log
// line can report it without needing the (possibly absent) hot bytes.
type entry struct {
	ref     state.ChunkRef
	uri     string
	bytes   []byte
	size    int64
	spilled bool
}

// Store is a byte-budgeted, transparently-spilling state.Store. Construct
// with New. Safe for concurrent use.
type Store struct {
	mu      sync.Mutex
	cold    ColdStore
	log     core.Writer
	tier    *kvtier.Manager
	nextID  int
	entries map[int]*entry
	uris    map[string]int
}

// New builds a Store. Budget <= 0 disables tiering entirely (every chunk
// stays hot; Cold is never touched). Budget > 0 requires a non-nil Cold.
//
//	store, err := ramspill.New(ramspill.Options{Budget: 64 << 20, Cold: cold})
func New(opts Options) (*Store, error) {
	if opts.Budget > 0 && opts.Cold == nil {
		return nil, core.E("ramspill.New", "budget > 0 requires a Cold store", nil)
	}
	s := &Store{
		cold:    opts.Cold,
		log:     opts.Log,
		nextID:  1,
		entries: make(map[int]*entry),
		uris:    make(map[string]int),
	}
	// CPU: 0 folds kvtier's GPU->CPU->Disk cascade into a single rebalance
	// pass: this Store only has two real locations (hot RAM and Cold), so
	// "GPU" and "CPU" both mean hot here — see Move.
	s.tier = kvtier.New(kvtier.Budget{GPU: opts.Budget}, s)
	return s, nil
}

// Move implements kvtier.Store: the real byte transfer behind the tiering
// policy's plan. This Store models only two physical locations (hot RAM,
// Cold), so the classification is by Disk endpoint, not by the exact
// GPU/CPU label — kvtier's rollback path can reverse an already-applied
// spill via a Disk->CPU hop just as often as Access promotes via Disk->GPU,
// and both mean the same thing here: bring the bytes back. A hop between GPU
// and CPU never touches Disk on either end, so it is pure accounting.
func (s *Store) Move(ctx context.Context, blockID string, from, to kvtier.Tier) error {
	id, err := strconv.Atoi(blockID)
	if err != nil {
		return core.E("ramspill.Store.Move", "block id "+blockID, err)
	}
	switch {
	case to == kvtier.TierDisk:
		return s.spill(ctx, id)
	case from == kvtier.TierDisk:
		return s.revive(ctx, id)
	default:
		return nil
	}
}

// spill writes id's current bytes to Cold and frees the RAM copy. A no-op
// when id is unknown or already spilled.
func (s *Store) spill(ctx context.Context, id int) error {
	s.mu.Lock()
	e, ok := s.entries[id]
	if !ok || e.spilled {
		s.mu.Unlock()
		return nil
	}
	payload := e.bytes
	uri := e.uri
	s.mu.Unlock()

	if s.cold == nil {
		return core.E("ramspill.Store.spill", "chunk "+strconv.Itoa(id)+" has no Cold store configured", nil)
	}
	if _, err := s.cold.PutBytes(ctx, payload, state.PutOptions{URI: spillKey(id), Title: "ramspill chunk"}); err != nil {
		return core.E("ramspill.Store.spill", "write chunk "+strconv.Itoa(id)+" to cold store", err)
	}

	s.mu.Lock()
	if e, ok := s.entries[id]; ok {
		e.bytes = nil
		e.spilled = true
	}
	s.mu.Unlock()
	core.Print(s.log, "state: spilled conversation chunk %d (%s), ~%d bytes freed", id, uri, len(payload))
	return nil
}

// revive reads id's bytes back from Cold into RAM. A no-op when id is
// unknown or already resident.
func (s *Store) revive(ctx context.Context, id int) error {
	s.mu.Lock()
	e, ok := s.entries[id]
	if !ok {
		s.mu.Unlock()
		return core.Wrap(kvtier.ErrUnknownBlock, "ramspill.Store.revive", "chunk "+strconv.Itoa(id))
	}
	if !e.spilled {
		s.mu.Unlock()
		return nil
	}
	s.mu.Unlock()

	if s.cold == nil {
		return core.E("ramspill.Store.revive", "chunk "+strconv.Itoa(id)+" has no Cold store configured", nil)
	}
	chunk, err := s.cold.ResolveURI(ctx, spillKey(id))
	if err != nil {
		return core.E("ramspill.Store.revive", "read chunk "+strconv.Itoa(id)+" from cold store", err)
	}
	data := chunk.Data
	if len(data) == 0 && chunk.Text != "" {
		data = []byte(chunk.Text)
	}

	s.mu.Lock()
	if e, ok := s.entries[id]; ok {
		e.bytes = data
		e.spilled = false
	}
	s.mu.Unlock()
	core.Print(s.log, "state: revived conversation chunk %d, ~%d bytes restored", id, len(data))
	return nil
}

// spillKey is the URI a chunk's payload is addressed by inside Cold —
// namespaced so it can never collide with a caller-supplied PutOptions.URI.
func spillKey(id int) string {
	return "ramspill://chunk/" + strconv.Itoa(id)
}

// Get returns chunkID's text, reviving it from Cold first if it is spilled.
func (s *Store) Get(ctx context.Context, chunkID int) (string, error) {
	chunk, err := s.Resolve(ctx, chunkID)
	if err != nil {
		return "", err
	}
	return chunk.Text, nil
}

// Resolve returns chunkID's chunk, reviving it from Cold first if it is
// spilled, and marks it most-recently-used either way.
func (s *Store) Resolve(ctx context.Context, chunkID int) (state.Chunk, error) {
	data, ref, err := s.access(ctx, chunkID)
	if err != nil {
		return state.Chunk{}, err
	}
	return state.Chunk{Ref: ref, Text: string(data), Data: data}, nil
}

// ResolveBytes is Resolve; every chunk here is already byte-native.
func (s *Store) ResolveBytes(ctx context.Context, chunkID int) (state.Chunk, error) {
	return s.Resolve(ctx, chunkID)
}

// BorrowBytes returns a defensive copy of chunkID's bytes. Unlike
// state.InMemoryStore's zero-copy borrow, Store always copies: a borrowed
// slice can otherwise straddle a concurrent spill of the same id from a
// different goroutine's eviction pass, which would hand the caller a slice
// whose backing bytes this Store no longer accounts for as resident.
func (s *Store) BorrowBytes(ctx context.Context, chunkID int) (state.BorrowedChunk, error) {
	data, ref, err := s.access(ctx, chunkID)
	if err != nil {
		return state.BorrowedChunk{}, err
	}
	return state.BorrowedChunk{Ref: ref, Data: data}, nil
}

// ResolveURI resolves a chunk by its Put-time URI, reviving it from Cold
// first if it is spilled.
func (s *Store) ResolveURI(ctx context.Context, uri string) (state.Chunk, error) {
	s.mu.Lock()
	id, ok := s.uris[uri]
	s.mu.Unlock()
	if !ok {
		return state.Chunk{}, &state.URIChunkNotFoundError{URI: uri}
	}
	return s.Resolve(ctx, id)
}

// access marks chunkID most-recently-used (reviving it first if spilled) and
// returns a copy of its current bytes. See reviveAttempts for why this
// retries a bounded number of times rather than once.
func (s *Store) access(ctx context.Context, chunkID int) ([]byte, state.ChunkRef, error) {
	if s == nil {
		return nil, state.ChunkRef{}, &state.ChunkNotFoundError{ID: chunkID}
	}
	select {
	case <-ctx.Done():
		return nil, state.ChunkRef{}, ctx.Err()
	default:
	}
	s.mu.Lock()
	_, known := s.entries[chunkID]
	s.mu.Unlock()
	if !known {
		return nil, state.ChunkRef{}, &state.ChunkNotFoundError{ID: chunkID}
	}

	blockID := strconv.Itoa(chunkID)
	for attempt := 0; attempt < reviveAttempts; attempt++ {
		if err := s.tier.Access(ctx, blockID); err != nil && !core.Is(err, kvtier.ErrUnknownBlock) {
			return nil, state.ChunkRef{}, core.E("ramspill.Store", "revive chunk "+blockID, err)
		}
		s.mu.Lock()
		e := s.entries[chunkID]
		spilled := e.spilled
		data := append([]byte(nil), e.bytes...)
		ref := e.ref
		s.mu.Unlock()
		if !spilled {
			return data, ref, nil
		}
		// Lost a race with a concurrent eviction that re-spilled chunkID
		// right after we promoted it — retry.
	}
	return nil, state.ChunkRef{}, core.E("ramspill.Store", "chunk "+blockID+" would not stay resident (budget smaller than one chunk?)", nil)
}

// Put stores text under a fresh chunk id, tracked for tiering.
func (s *Store) Put(ctx context.Context, text string, opts state.PutOptions) (state.ChunkRef, error) {
	return s.put(ctx, []byte(text), opts)
}

// PutBytes stores data under a fresh chunk id, tracked for tiering.
func (s *Store) PutBytes(ctx context.Context, data []byte, opts state.PutOptions) (state.ChunkRef, error) {
	return s.put(ctx, append([]byte(nil), data...), opts)
}

func (s *Store) put(ctx context.Context, data []byte, opts state.PutOptions) (state.ChunkRef, error) {
	if s == nil {
		return state.ChunkRef{}, core.E("ramspill.Store.Put", "store is nil", nil)
	}
	select {
	case <-ctx.Done():
		return state.ChunkRef{}, ctx.Err()
	default:
	}

	s.mu.Lock()
	id := s.nextID
	s.nextID++
	ref := state.ChunkRef{ChunkID: id, FrameOffset: uint64(id), HasFrameOffset: true, Codec: state.CodecMemory}
	size := int64(len(data))
	s.entries[id] = &entry{ref: ref, uri: opts.URI, bytes: data, size: size}
	if opts.URI != "" {
		s.uris[opts.URI] = id
	}
	s.mu.Unlock()

	// tier.Put may synchronously call back into Move (spilling THIS or an
	// older chunk) — it must never run while s.mu is held, or that callback
	// deadlocks retaking the same lock.
	blockID := strconv.Itoa(id)
	if err := s.tier.Put(ctx, kvtier.Block{ID: blockID, SizeBytes: size}); err != nil {
		// A chunk bigger than the whole budget (or a Cold I/O failure while
		// demoting an older one) never breaks the write that's already
		// landed hot — it just stays outside the tiering policy's future
		// accounting, exactly like Budget <= 0. Serving degrades, it never
		// breaks.
		core.Error("ramspill: chunk not budget-tracked; staying resident", "chunk", id, "error", err)
	}
	return ref, nil
}

// Resident reports whether chunkID currently holds its payload in RAM.
// Unknown ids report false. Does not affect recency.
func (s *Store) Resident(chunkID int) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	e, ok := s.entries[chunkID]
	return ok && !e.spilled
}

// ChunkIDForURI returns the chunk id a Put-time URI resolved to.
func (s *Store) ChunkIDForURI(uri string) (int, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	id, ok := s.uris[uri]
	return id, ok
}

// ChunkCount reports the number of chunks tracked (resident + spilled).
func (s *Store) ChunkCount() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.entries)
}
