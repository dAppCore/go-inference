// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"sync"

	core "dappco.re/go"
)

// ErrNotFound is the typed error a Store returns when a session id is unknown.
// Callers compare against it to tell "no such session" from an I/O failure.
//
//	if _, err := store.Get(id); err == session.ErrNotFound { … }
var ErrNotFound = core.E("session", "session not found", nil)

// Store persists sessions by id. The in-memory implementation is the default;
// a durable backend (go-store KV) plugs in behind the same three methods so the
// registry survives a restart without changing the Manager.
//
//	var s session.Store = session.NewMemoryStore()
type Store interface {
	// Get returns the session for id, or ErrNotFound if none exists.
	Get(id string) (Session, error)
	// Put stores (creates or replaces) the session under sess.ID.
	Put(sess Session) error
	// Delete removes the session for id; deleting a missing id is not an error.
	Delete(id string) error
}

// MemoryStore is a goroutine-safe in-memory Store — the default registry backing
// for a single process (RFC §6.10). Sessions are held as values, copied in and
// out, so a caller can never reach the stored map through a returned Session.
type MemoryStore struct {
	mu       sync.RWMutex
	sessions map[string]Session
}

// NewMemoryStore builds an empty in-memory Store.
//
//	m := session.NewManager(session.NewMemoryStore())
func NewMemoryStore() *MemoryStore {
	return &MemoryStore{sessions: make(map[string]Session)}
}

// Get returns a copy of the stored session, or ErrNotFound.
func (m *MemoryStore) Get(id string) (Session, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	sess, ok := m.sessions[id]
	if !ok {
		return Session{}, ErrNotFound
	}
	return sess.clone(), nil
}

// Put stores a copy of the session under its id.
func (m *MemoryStore) Put(sess Session) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sessions[sess.ID] = sess.clone()
	return nil
}

// Delete removes the session for id (a no-op if absent).
func (m *MemoryStore) Delete(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.sessions, id)
	return nil
}
