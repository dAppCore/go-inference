// SPDX-Licence-Identifier: EUPL-1.2

package registry

import (
	"sort"
	"sync"

	core "dappco.re/go"
)

// Store is the pluggable persistence behind a Registry. The default is
// MemStore; a go-store / DuckDB implementation slots in unchanged (out of scope
// here). Keys are canonical entry ids.
//
//	var s registry.Store = registry.NewMemStore()
//	s.Put(entry)
//	r := s.Get("gemma-4-4b-it") // r.Value is Entry when r.OK
type Store interface {
	// Put inserts or replaces the entry keyed by its id.
	//
	//	s.Put(registry.Entry{ID: "gemma-4-4b-it"})
	Put(e Entry) core.Result

	// Get returns the entry for id, or a failed Result when absent.
	//
	//	r := s.Get("gemma-4-4b-it")
	Get(id string) core.Result

	// List returns every stored entry, sorted by id.
	//
	//	for _, e := range s.List() { ... }
	List() []Entry

	// Delete removes the entry for id, or a failed Result when absent.
	//
	//	s.Delete("gemma-4-4b-it")
	Delete(id string) core.Result
}

// MemStore is an in-memory, goroutine-safe Store — the default backing for a
// Registry and the store used in tests.
//
//	s := registry.NewMemStore()
type MemStore struct {
	mu      sync.RWMutex
	entries map[string]Entry
}

// NewMemStore returns an empty in-memory Store.
//
//	r := registry.NewWithStore(registry.NewMemStore())
func NewMemStore() *MemStore {
	return &MemStore{entries: map[string]Entry{}}
}

// Put inserts or replaces e by its id.
//
//	s.Put(registry.Entry{ID: "gemma-4-4b-it"})
func (s *MemStore) Put(e Entry) core.Result {
	if e.ID == "" {
		return core.Fail(core.E("registry.MemStore.Put", "entry id is empty", nil))
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.entries[e.ID] = e
	return core.Ok(e)
}

// Get returns the entry for id.
//
//	r := s.Get("gemma-4-4b-it")
func (s *MemStore) Get(id string) core.Result {
	s.mu.RLock()
	defer s.mu.RUnlock()
	e, ok := s.entries[id]
	if !ok {
		return core.Fail(core.E("registry.MemStore.Get", core.Sprintf("no entry with id %q", id), nil))
	}
	return core.Ok(e)
}

// List returns every entry sorted by id.
//
//	all := s.List()
func (s *MemStore) List() []Entry {
	s.mu.RLock()
	out := make([]Entry, 0, len(s.entries))
	for _, e := range s.entries {
		out = append(out, e)
	}
	s.mu.RUnlock()
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// Delete removes the entry for id.
//
//	s.Delete("gemma-4-4b-it")
func (s *MemStore) Delete(id string) core.Result {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, ok := s.entries[id]; !ok {
		return core.Fail(core.E("registry.MemStore.Delete", core.Sprintf("no entry with id %q", id), nil))
	}
	delete(s.entries, id)
	return core.Ok(id)
}
