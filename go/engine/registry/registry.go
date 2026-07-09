// SPDX-Licence-Identifier: EUPL-1.2

package registry

import (
	"cmp"
	"slices"
	"sync"

	core "dappco.re/go"
)

// Registry is the model catalogue. It wraps a Store with an alias index and the
// resolution / filtering / device-fit queries the registry consumers use.
//
//	r := registry.New()
//	r.Put(registry.Entry{ID: "gemma-4-4b-it", Aliases: []string{"lemma"}})
//	e := r.Resolve("lemma").Value.(registry.Entry)
type Registry struct {
	store Store

	mu      sync.RWMutex
	aliases map[string]string // normalised alias/id → canonical id
}

// New returns a Registry backed by an in-memory Store.
//
//	r := registry.New()
func New() *Registry {
	return NewWithStore(NewMemStore())
}

// NewWithStore returns a Registry over a caller-supplied Store, rebuilding the
// alias index from whatever the store already holds.
//
//	r := registry.NewWithStore(registry.NewMemStore())
func NewWithStore(s Store) *Registry {
	r := &Registry{store: s, aliases: map[string]string{}}
	for _, e := range s.List() {
		r.indexEntry(e)
	}
	return r
}

// normalise lower-cases and trims a name so resolution is case- and
// whitespace-insensitive.
//
//	normalise("  LEMMA ") == "lemma"
func normalise(name string) string {
	return core.Lower(core.Trim(name))
}

// indexEntry adds an entry's id and aliases to the alias index. Callers hold
// r.mu. It assumes conflicts were already rejected by Put.
func (r *Registry) indexEntry(e Entry) {
	r.aliases[normalise(e.ID)] = e.ID
	for _, a := range e.Aliases {
		if n := normalise(a); n != "" {
			r.aliases[n] = e.ID
		}
	}
}

// deindexEntry removes an entry's id and aliases from the alias index. Callers
// hold r.mu.
func (r *Registry) deindexEntry(e Entry) {
	delete(r.aliases, normalise(e.ID))
	for _, a := range e.Aliases {
		delete(r.aliases, normalise(a))
	}
}

// Put inserts or replaces an entry. The id and every alias must be unique
// across the catalogue (a name may only point at one entry); re-putting the
// same id updates it in place. An empty id, or an alias that collides with a
// different entry's id or alias, is rejected.
//
//	r.Put(registry.Entry{ID: "gemma-4-4b-it", Aliases: []string{"lemma"}})
func (r *Registry) Put(e Entry) core.Result {
	if e.ID == "" {
		return core.Fail(core.E("registry.Put", "entry id is empty", nil))
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	// Every name this entry claims (its id + aliases) must be free, or already
	// owned by this same id (the update-in-place case). Check the id first, then
	// each alias, without allocating a combined slice.
	if n := normalise(e.ID); n != "" {
		if owner, taken := r.aliases[n]; taken && owner != e.ID {
			return core.Fail(core.E("registry.Put",
				core.Sprintf("name %q already maps to entry %q", e.ID, owner), nil))
		}
	}
	for _, name := range e.Aliases {
		n := normalise(name)
		if n == "" {
			continue
		}
		if owner, taken := r.aliases[n]; taken && owner != e.ID {
			return core.Fail(core.E("registry.Put",
				core.Sprintf("name %q already maps to entry %q", name, owner), nil))
		}
	}

	// Drop the previous index for this id so removed aliases do not linger.
	if prev := r.store.Get(e.ID); prev.OK {
		r.deindexEntry(prev.Value.(Entry))
	}

	res := r.store.Put(e)
	if !res.OK {
		// Re-index the previous entry if the store rejected the new one.
		if prev := r.store.Get(e.ID); prev.OK {
			r.indexEntry(prev.Value.(Entry))
		}
		return res
	}
	r.indexEntry(e)
	return res
}

// Get returns the entry stored under the exact canonical id (no alias
// resolution). Use Resolve for id-or-alias lookup.
//
//	r.Get("gemma-4-4b-it")
func (r *Registry) Get(id string) core.Result {
	return r.store.Get(id)
}

// Resolve maps an id or alias (case- and whitespace-insensitive) to its Entry.
//
//	r.Resolve("lemma")          // → the gemma-4-4b-it entry
//	r.Resolve("gemma-4-4b-it")  // → same entry by canonical id
func (r *Registry) Resolve(idOrAlias string) core.Result {
	n := normalise(idOrAlias)
	if n == "" {
		return core.Fail(core.E("registry.Resolve", "empty id or alias", nil))
	}
	r.mu.RLock()
	id, ok := r.aliases[n]
	r.mu.RUnlock()
	if !ok {
		return core.Fail(core.E("registry.Resolve", core.Sprintf("unknown model %q", idOrAlias), nil))
	}
	return r.store.Get(id)
}

// List returns every entry, sorted by id.
//
//	for _, e := range r.List() { ... }
func (r *Registry) List() []Entry {
	return r.store.List()
}

// Filter returns the entries matching f, sorted by id.
//
//	r.Filter(registry.Filter{Tools: true, ReadyOnly: true})
func (r *Registry) Filter(f Filter) []Entry {
	all := r.store.List()
	out := make([]Entry, 0, len(all))
	for _, e := range all {
		if f.matches(e) {
			out = append(out, e)
		}
	}
	return out
}

// Delete removes an entry by its canonical id, freeing its id and aliases for
// reuse.
//
//	r.Delete("gemma-4-4b-it")
func (r *Registry) Delete(id string) core.Result {
	r.mu.Lock()
	defer r.mu.Unlock()

	got := r.store.Get(id)
	if !got.OK {
		return core.Fail(core.E("registry.Delete", core.Sprintf("no entry with id %q", id), nil))
	}
	res := r.store.Delete(id)
	if !res.OK {
		return res
	}
	r.deindexEntry(got.Value.(Entry))
	return res
}

// FitsDevice returns the entries whose memory footprint fits within budgetBytes,
// largest-footprint-first (the biggest model a device can hold ranks first).
// Entries with an unknown (zero) footprint never fit. This is what the
// residency policy consumes to place models on a device.
//
//	fits := r.FitsDevice(96 << 30) // models that fit a 96 GiB device
func (r *Registry) FitsDevice(budgetBytes uint64) []Entry {
	return r.FitsDeviceWith(budgetBytes, Filter{})
}

// FitsDeviceWith is FitsDevice narrowed by a capability / status Filter — e.g.
// "the ready, vision-capable models that fit this budget".
//
//	r.FitsDeviceWith(96<<30, registry.Filter{Vision: true, ReadyOnly: true})
func (r *Registry) FitsDeviceWith(budgetBytes uint64, f Filter) []Entry {
	all := r.store.List()
	out := make([]Entry, 0, len(all))
	for _, e := range all {
		if e.MemoryBytes == 0 || e.MemoryBytes > budgetBytes {
			continue
		}
		if !f.matches(e) {
			continue
		}
		out = append(out, e)
	}
	slices.SortFunc(out, func(a, b Entry) int {
		if a.MemoryBytes != b.MemoryBytes {
			if a.MemoryBytes > b.MemoryBytes {
				return -1
			}
			return 1
		}
		return cmp.Compare(a.ID, b.ID)
	})
	return out
}
