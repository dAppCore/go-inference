// SPDX-Licence-Identifier: EUPL-1.2

package prompt

import (
	"sort"
	"sync"

	core "dappco.re/go"
)

// Store keeps versioned prompt templates addressable by id — the persistence
// seam behind the stored prompts (RFC §6.10). Put stores a template
// (auto-assigning the next version when Version is zero); Get resolves one
// explicit version; Latest resolves the highest version; List returns every
// version of an id. Unknown id or version is a typed error.
//
//	s := prompt.NewMemoryStore()
//	stored, _ := s.Put(prompt.Template{ID: "greet", Body: "hi {{name}}"})
//	latest, _ := s.Latest("greet")
type Store interface {
	Put(t Template) (Template, error)
	Get(id string, version int) (Template, error)
	Latest(id string) (Template, error)
	List(id string) ([]Template, error)
}

// MemoryStore is a goroutine-safe in-memory Store. The zero value is not
// usable — construct it with NewMemoryStore.
type MemoryStore struct {
	mu       sync.RWMutex
	versions map[string]map[int]Template
}

// NewMemoryStore returns an empty, ready-to-use in-memory Store.
//
//	s := prompt.NewMemoryStore()
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{versions: make(map[string]map[int]Template)}
}

// Put stores t and returns it as stored. An empty ID is rejected — the id is
// the storage key. When t.Version is zero the next version (one above the
// highest stored for the id, or 1 for a fresh id) is assigned; a non-zero
// version is honoured and overwrites that version in place.
//
//	stored, _ := s.Put(prompt.Template{ID: "greet", Body: "hi"})  // version 1
func (s *MemoryStore) Put(t Template) (Template, error) {
	if t.ID == "" {
		return Template{}, core.E("prompt", "template id is required", nil)
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	byVer := s.versions[t.ID]
	if byVer == nil {
		byVer = make(map[int]Template)
		s.versions[t.ID] = byVer
	}

	if t.Version == 0 {
		t.Version = nextVersion(byVer)
	}
	byVer[t.Version] = cloneTemplate(t)
	return cloneTemplate(t), nil
}

// Get returns the template for id at the given version, or a typed error when
// the id or the version is unknown.
//
//	got, _ := s.Get("greet", 1)
func (s *MemoryStore) Get(id string, version int) (Template, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	byVer, ok := s.versions[id]
	if !ok {
		return Template{}, core.E("prompt", core.Concat("unknown template id ", id), nil)
	}
	t, ok := byVer[version]
	if !ok {
		return Template{}, core.E("prompt", core.Concat("unknown version ", core.Itoa(version), " for template ", id), nil)
	}
	return cloneTemplate(t), nil
}

// Latest returns the highest-versioned template for id, or a typed error when
// the id is unknown.
//
//	latest, _ := s.Latest("greet")
func (s *MemoryStore) Latest(id string) (Template, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	byVer, ok := s.versions[id]
	if !ok || len(byVer) == 0 {
		return Template{}, core.E("prompt", core.Concat("unknown template id ", id), nil)
	}
	highest := 0
	for v := range byVer {
		if v > highest {
			highest = v
		}
	}
	return cloneTemplate(byVer[highest]), nil
}

// List returns every version of id in ascending version order, or a typed
// error when the id is unknown.
//
//	all, _ := s.List("greet")
func (s *MemoryStore) List(id string) ([]Template, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	byVer, ok := s.versions[id]
	if !ok || len(byVer) == 0 {
		return nil, core.E("prompt", core.Concat("unknown template id ", id), nil)
	}
	vers := make([]int, 0, len(byVer))
	for v := range byVer {
		vers = append(vers, v)
	}
	sort.Ints(vers)
	out := make([]Template, 0, len(vers))
	for _, v := range vers {
		out = append(out, cloneTemplate(byVer[v]))
	}
	return out, nil
}

// nextVersion returns one above the highest version present, or 1 when empty.
func nextVersion(byVer map[int]Template) int {
	highest := 0
	for v := range byVer {
		if v > highest {
			highest = v
		}
	}
	return highest + 1
}

// cloneTemplate returns a deep copy of t so stored entries and returned values
// never alias the same InputVars slice — a caller mutating a returned template
// must not corrupt the store, and vice versa.
func cloneTemplate(t Template) Template {
	t.InputVars = append([]string(nil), t.InputVars...)
	return t
}
