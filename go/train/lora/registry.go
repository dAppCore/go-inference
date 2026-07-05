// SPDX-Licence-Identifier: EUPL-1.2

// Registry: the ref-counted catalogue of known adapters (with entry storage).

package lora

import (
	"slices"
	"sync"

	core "dappco.re/go"
)

// entry is one registered adapter: its ref, the cached deterministic id, its
// current ref-count (outstanding Use leases), and whether it is currently
// resident on the base model. id is AdapterRef.ID() — a SHA-256 + hex of the
// immutable Name+Path — computed once at Register so the hot name→id paths
// (Acquire, IsResident, Pin) never re-hash.
type entry struct {
	ref      AdapterRef
	id       string
	refs     int
	resident bool
}

// Registry is the catalogue of known adapters with ref-counted leases. Register
// adds an adapter (keyed by Name), Acquire/Release fence in-flight use so the Pool
// never evicts an adapter mid-request, and Unregister removes a free adapter. It
// is safe for concurrent use.
//
//	reg := lora.NewRegistry()
//	reg.Register(lora.AdapterRef{Name: "a", Path: "/x"})
//	id, _ := reg.Acquire("a"); defer reg.Release(id)
type Registry struct {
	mu     sync.Mutex
	byName map[string]*entry
	byID   map[string]*entry
}

// NewRegistry builds an empty adapter registry.
//
//	reg := lora.NewRegistry()
func NewRegistry() *Registry {
	return &Registry{
		byName: make(map[string]*entry),
		byID:   make(map[string]*entry),
	}
}

// Register records a new adapter under its Name. A missing Name, or a Name that
// is already registered, returns a typed core error.
//
//	if err := reg.Register(ref); err != nil { … }
func (r *Registry) Register(ref AdapterRef) error {
	if ref.Name == "" {
		return core.E("ai", "lora: adapter name is required", nil)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.byName[ref.Name]; ok {
		return core.E("ai", "lora: adapter already registered: "+ref.Name, nil)
	}
	id := ref.ID()
	e := &entry{ref: ref, id: id}
	r.byName[ref.Name] = e
	r.byID[id] = e
	return nil
}

// Unregister removes a free adapter by Name. An unknown name, or an adapter with
// outstanding refs (an in-flight request), returns a typed core error so the Pool
// can never lose an adapter from under a live request.
func (r *Registry) Unregister(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.byName[name]
	if !ok {
		return core.E("ai", "lora: unknown adapter: "+name, nil)
	}
	if e.refs > 0 {
		return core.E("ai", "lora: adapter in use, cannot unregister: "+name, nil)
	}
	delete(r.byName, name)
	delete(r.byID, e.id)
	return nil
}

// Get returns the adapter ref registered under name, or a typed error if unknown.
//
//	ref, err := reg.Get("a")
func (r *Registry) Get(name string) (AdapterRef, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.byName[name]
	if !ok {
		return AdapterRef{}, core.E("ai", "lora: unknown adapter: "+name, nil)
	}
	return e.ref, nil
}

// List returns every registered adapter ref, sorted by Name for deterministic
// output.
//
//	for _, ref := range reg.List() { … }
func (r *Registry) List() []AdapterRef {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make([]AdapterRef, 0, len(r.byName))
	for _, e := range r.byName {
		out = append(out, e.ref)
	}
	slices.SortFunc(out, func(a, b AdapterRef) int {
		switch {
		case a.Name < b.Name:
			return -1
		case a.Name > b.Name:
			return 1
		default:
			return 0
		}
	})
	return out
}

// Acquire bumps the ref-count for the adapter named and returns its id. While the
// count is non-zero the adapter is in use and the Pool will not evict it. An
// unknown name returns a typed error. Balance every Acquire with a Release.
//
//	id, err := reg.Acquire("a"); defer reg.Release(id)
func (r *Registry) Acquire(name string) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.byName[name]
	if !ok {
		return "", core.E("ai", "lora: unknown adapter: "+name, nil)
	}
	e.refs++
	return e.id, nil
}

// Release drops one ref for the adapter id. It clamps at zero, so an over-release
// or a release of an unknown id is a harmless no-op.
//
//	reg.Release(id)
func (r *Registry) Release(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.byID[id]
	if !ok {
		return
	}
	if e.refs > 0 {
		e.refs--
	}
}

// RefCount reports the number of outstanding leases on the adapter id. An unknown
// id reports 0.
func (r *Registry) RefCount(id string) int {
	r.mu.Lock()
	defer r.mu.Unlock()
	if e, ok := r.byID[id]; ok {
		return e.refs
	}
	return 0
}

// InUse reports whether the adapter id has any outstanding lease (and is thus
// ineligible for eviction).
func (r *Registry) InUse(id string) bool {
	return r.RefCount(id) > 0
}

// idByName resolves name to its cached deterministic id, with ok=false for an
// unknown name. It lets the Pool's residency/pin paths get an id without
// recomputing AdapterRef.ID() (a SHA-256 + hex) on every call.
//
//	id, ok := reg.idByName("a")
func (r *Registry) idByName(name string) (string, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	e, ok := r.byName[name]
	if !ok {
		return "", false
	}
	return e.id, true
}
