// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"sync"

	core "dappco.re/go"
	chat "dappco.re/go/inference/serving/chat"
)

// Manager is the conversation registry (RFC §6.10): it opens sessions, appends
// turns, and resolves a `previous_response_id` back to the session + context a
// caller continues from — so the next request never resends the transcript.
//
// id generation and the clock are injected (WithIDGen / WithClock) so tests are
// deterministic; the defaults mint a random id and read core.Now.
//
//	m := session.NewManager(session.NewMemoryStore())
//	s := m.Open("lemma")
//	resp, _ := m.Append(s.ID, chat.Message{Role: chat.User, Content: []chat.ContentBlock{chat.Text("hi")}})
//	prior, _ := m.Continue(resp)   // s with its turns, ready to continue
type Manager struct {
	store Store
	idGen func() string
	clock func() core.Time

	mu        sync.Mutex          // guards the response→position map below
	responses map[string]position // responseID → where in which session it points
}

// position pins a responseID to a session and the turn count at the moment it
// was minted, so Continue can hand back the transcript as it stood then.
type position struct {
	sessionID string
	turnCount int
}

// Option configures a Manager.
type Option func(*Manager)

// WithIDGen injects the id generator used for both session and response ids —
// inject a deterministic sequence in tests.
//
//	session.NewManager(store, session.WithIDGen(seq("sess-1", "resp-1")))
func WithIDGen(gen func() string) Option {
	return func(m *Manager) {
		if gen != nil {
			m.idGen = gen
		}
	}
}

// WithClock injects the time source for created/updated stamps.
//
//	session.NewManager(store, session.WithClock(func() core.Time { return at }))
func WithClock(clock func() core.Time) Option {
	return func(m *Manager) {
		if clock != nil {
			m.clock = clock
		}
	}
}

// NewManager builds a registry over the given Store, applying defaults (random
// ids, core.Now clock) before any Option overrides.
//
//	m := session.NewManager(session.NewMemoryStore())
func NewManager(store Store, opts ...Option) *Manager {
	m := &Manager{
		store:     store,
		idGen:     defaultIDGen,
		clock:     core.Now,
		responses: make(map[string]position),
	}
	for _, opt := range opts {
		opt(m)
	}
	return m
}

// defaultIDGen mints a random id when none is injected.
func defaultIDGen() string {
	return core.RandomString(24).Must().(string)
}

// Open starts a fresh session for model and returns it (a copy). The session has
// a new id, no turns, and no KV handle yet.
//
//	s := m.Open("lemma")
func (m *Manager) Open(model string) Session {
	now := m.clock()
	sess := Session{
		ID:      m.idGen(),
		Model:   model,
		Created: now,
		Updated: now,
	}
	_ = m.store.Put(sess)
	return sess.clone()
}

// Append adds turn to the session and mints a responseID that points at the
// session's new position — this is the `previous_response_id` the caller sends
// next time. An empty or unknown sessionID is a typed error.
//
//	resp, err := m.Append(s.ID, chat.Message{Role: chat.User, Content: []chat.ContentBlock{chat.Text("hello")}})
func (m *Manager) Append(sessionID string, turn chat.Message) (string, error) {
	if sessionID == "" {
		return "", core.E("session", "append: empty session id", nil)
	}
	sess, err := m.store.Get(sessionID)
	if err != nil {
		return "", core.E("session", "append: "+sessionID, err)
	}

	// store.Get handed back an exactly-sized clone (cap == len), so a plain
	// append would geometrically over-allocate just to add one turn. The final
	// length is known (one more), so build the slice at exact size; Put re-clones
	// it to exact size anyway, so the intermediate capacity is never observed.
	n := len(sess.Turns)
	turns := make([]chat.Message, n+1)
	copy(turns, sess.Turns)
	turns[n] = turn
	sess.Turns = turns
	sess.Updated = m.clock()
	if err := m.store.Put(sess); err != nil {
		return "", core.E("session", "append: put "+sessionID, err)
	}

	respID := m.idGen()
	m.mu.Lock()
	m.responses[respID] = position{sessionID: sessionID, turnCount: len(sess.Turns)}
	m.mu.Unlock()
	return respID, nil
}

// Continue resolves a previousResponseID back to its session and the context as
// it stood when that response was minted — the caller continues from here with
// 0% transcript replay. An empty or unknown id is a typed error.
//
//	prior, err := m.Continue(previousResponseID)
func (m *Manager) Continue(previousResponseID string) (Session, error) {
	if previousResponseID == "" {
		return Session{}, core.E("session", "continue: empty response id", nil)
	}
	m.mu.Lock()
	pos, ok := m.responses[previousResponseID]
	m.mu.Unlock()
	if !ok {
		return Session{}, core.E("session", "continue: unknown response id "+previousResponseID, ErrNotFound)
	}

	sess, err := m.store.Get(pos.sessionID)
	if err != nil {
		return Session{}, core.E("session", "continue: "+pos.sessionID, err)
	}

	// Hand back the transcript as it stood at this response's position — a later
	// response id sees more turns, an earlier one fewer. store.Get already
	// returned an isolated copy that only this call holds (the same copy Get
	// hands straight back), so the truncating reslice touches private backing and
	// no further clone is needed before returning it.
	if pos.turnCount < len(sess.Turns) {
		sess.Turns = sess.Turns[:pos.turnCount]
	}
	return sess, nil
}

// Get returns the current session for id (a copy), or a typed error.
//
//	s, err := m.Get(sessionID)
func (m *Manager) Get(sessionID string) (Session, error) {
	if sessionID == "" {
		return Session{}, core.E("session", "get: empty session id", nil)
	}
	sess, err := m.store.Get(sessionID)
	if err != nil {
		return Session{}, core.E("session", "get: "+sessionID, err)
	}
	return sess, nil
}

// SetStateHandle attaches the opaque go-mlx KV reference to the session, so a
// later request re-attaches the same Wake/Sleep blocks instead of re-prefilling
// (RFC §6.10). An empty or unknown sessionID is a typed error.
//
//	err := m.SetStateHandle(s.ID, "mlx-kv://node-a/slab/42")
func (m *Manager) SetStateHandle(sessionID, handle string) error {
	if sessionID == "" {
		return core.E("session", "set state handle: empty session id", nil)
	}
	sess, err := m.store.Get(sessionID)
	if err != nil {
		return core.E("session", "set state handle: "+sessionID, err)
	}
	sess.StateHandle = handle
	sess.Updated = m.clock()
	if err := m.store.Put(sess); err != nil {
		return core.E("session", "set state handle: put "+sessionID, err)
	}
	return nil
}

// Delete removes a session and forgets every responseID that pointed at it, so a
// stale `previous_response_id` can't resolve to a deleted conversation.
//
//	err := m.Delete(sessionID)
func (m *Manager) Delete(sessionID string) error {
	if sessionID == "" {
		return core.E("session", "delete: empty session id", nil)
	}
	m.mu.Lock()
	for respID, pos := range m.responses {
		if pos.sessionID == sessionID {
			delete(m.responses, respID)
		}
	}
	m.mu.Unlock()
	if err := m.store.Delete(sessionID); err != nil {
		return core.E("session", "delete: "+sessionID, err)
	}
	return nil
}
