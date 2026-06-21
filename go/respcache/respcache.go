// SPDX-Licence-Identifier: EUPL-1.2

// Package respcache is the exact-match response cache for the serving
// surface (RFC.md §6.11, "Response cache"). It returns a stored completion
// with NO inference at all, keyed on the canonicalised request — messages plus
// model plus sampling params (RFC.md §6.1). It is distinct from prompt/KV
// (prefix) caching, which still runs the model: this short-circuits the run
// entirely for a repeated identical prompt (evals, idempotent tool calls).
//
// Key(req) derives a stable, field-order-independent key; Cache wraps a
// pluggable Store with optional per-entry TTL; the default Store is an
// in-memory, goroutine-safe map. A request can opt out of the cache for one
// call via Request.Bypass.
//
//	c := respcache.New(nil) // in-memory store
//	if hit, ok := c.Get(req); ok {
//		return hit // no inference
//	}
//	out := runInference(req)
//	c.Set(req, out, time.Hour)
package respcache

import (
	"sort"
	"sync"
	"time"

	core "dappco.re/go"
)

// Message is one canonicalised chat message. Only the fields that affect the
// completion form the key — role and content (RFC.md §6.1, messages). The JSON
// tags fix the field order so two messages with the same values serialise
// identically regardless of how the caller built them.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// Request is the cache view of a chat request: the subset of RFC.md §6.1 that
// determines the output. Two requests with these fields equal are the same
// generation and share a key. Bypass is NOT part of the key — it is a per-call
// switch (RFC.md §6.11, "bypassable per request"), not a property of the
// request's identity.
type Request struct {
	Model       string    `json:"model"`
	Messages    []Message `json:"messages"`
	Temperature float64   `json:"temperature"`
	TopP        float64   `json:"top_p"`
	MaxTokens   int       `json:"max_tokens"`
	Seed        int       `json:"seed"`
	Stop        []string  `json:"stop"`
	Bypass      bool      `json:"-"` // skip the cache for this call; not keyed
}

// Completion is the stored model output returned on a cache hit — what the
// caller would otherwise have run inference to produce.
type Completion struct {
	Text         string `json:"text"`
	Model        string `json:"model"`
	FinishReason string `json:"finish_reason,omitempty"`
}

// Entry is what a Store holds: the completion plus an optional absolute expiry.
// A zero Expiry means the entry never expires.
type Entry struct {
	Completion Completion
	Expiry     time.Time // zero = no expiry
}

// Store is the pluggable backing for a Cache. Implementations must be
// goroutine-safe. Get reports ok=false for a missing key; expiry is enforced by
// the Cache, not the Store, so a Store is a plain key→Entry medium.
//
//	type RedisStore struct{ ... }
//	func (r *RedisStore) Get(key string) (respcache.Entry, bool) { ... }
//	func (r *RedisStore) Set(key string, e respcache.Entry)      { ... }
type Store interface {
	Get(key string) (entry Entry, ok bool)
	Set(key string, entry Entry)
}

// Cache is an exact-match response cache over a Store. Construct it with New.
// Safe for concurrent use when its Store is (the default MemoryStore is).
type Cache struct {
	store Store
	now   func() time.Time // injectable clock for TTL tests; defaults to time.Now
}

// New builds a Cache over store. Pass nil to use the in-memory default.
//
//	c := respcache.New(nil)                 // in-memory
//	c := respcache.New(respcache.NewMemoryStore())
func New(store Store) *Cache {
	if store == nil {
		store = NewMemoryStore()
	}
	return &Cache{store: store, now: time.Now}
}

// Get returns the stored completion for req, or ok=false on a miss, on an
// expired entry, or when req.Bypass is set. No inference is performed — a hit
// IS the answer (RFC.md §6.11). An expired entry is treated as a miss.
//
//	if out, ok := c.Get(req); ok { return out }
func (c *Cache) Get(req Request) (Completion, bool) {
	if req.Bypass {
		return Completion{}, false
	}
	e, ok := c.store.Get(Key(req))
	if !ok {
		return Completion{}, false
	}
	if !e.Expiry.IsZero() && !c.now().Before(e.Expiry) {
		return Completion{}, false
	}
	return e.Completion, true
}

// Set stores out under req's key. A non-zero ttl sets an absolute expiry from
// now; ttl <= 0 stores with no expiry. A Set with req.Bypass set is a no-op —
// a bypassed call neither reads nor writes the cache. Re-Setting the same key
// overwrites the prior entry.
//
//	c.Set(req, out, time.Hour) // expires in an hour
//	c.Set(req, out, 0)         // never expires
func (c *Cache) Set(req Request, out Completion, ttl time.Duration) {
	if req.Bypass {
		return
	}
	e := Entry{Completion: out}
	if ttl > 0 {
		e.Expiry = c.now().Add(ttl)
	}
	c.store.Set(Key(req), e)
}

// Key derives a deterministic, field-order-independent cache key from req. The
// same request shape always yields the same key; any change to the model,
// messages, or a sampling param yields a different key (so a different
// generation never collides). Bypass is excluded — it is a per-call switch, not
// part of the request's identity.
//
// Canonicalisation: the request is copied into a fixed-field struct (stable
// JSON field order via core.JSONMarshalString) with the stop list sorted, so a
// caller passing the same stop strings in a different order — or a nil vs
// empty stop slice — maps to one key. The canonical JSON is hashed with
// core.SHA3_256Hex for a fixed-width, collision-resistant key.
//
//	k := respcache.Key(req)
func Key(req Request) string {
	// Copy the stop list before sorting so we never mutate the caller's slice.
	// nil and empty both normalise to nil, so they share a key.
	var stop []string
	if len(req.Stop) > 0 {
		stop = make([]string, len(req.Stop))
		copy(stop, req.Stop)
		sort.Strings(stop)
	}

	canonical := Request{
		Model:       req.Model,
		Messages:    req.Messages,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		Seed:        req.Seed,
		Stop:        stop,
	}
	return core.SHA3_256Hex(core.AsBytes(core.JSONMarshalString(canonical)))
}

// MemoryStore is the default Store — an in-memory, goroutine-safe map. Suitable
// for a single-process host; swap in a shared Store (Redis, go-store KV) for a
// fleet. Expiry is enforced by the Cache, so this never prunes on its own.
type MemoryStore struct {
	mu      sync.RWMutex
	entries map[string]Entry
}

// NewMemoryStore builds an empty in-memory Store.
//
//	c := respcache.New(respcache.NewMemoryStore())
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{entries: make(map[string]Entry)}
}

// Get returns the entry for key, or ok=false when absent.
func (m *MemoryStore) Get(key string) (Entry, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	e, ok := m.entries[key]
	return e, ok
}

// Set stores entry under key, overwriting any prior entry.
func (m *MemoryStore) Set(key string, entry Entry) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.entries[key] = entry
}
