// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	core "dappco.re/go"
)

// flakyStore is a Store test double whose Get / Put / Delete can each be made to
// fail on demand, so the Manager's I/O-error branches (which a healthy
// MemoryStore never triggers) can be exercised through the public API. When a
// fail flag is unset it delegates to an embedded MemoryStore, so happy-path
// behaviour is identical to the real backing.
type flakyStore struct {
	inner      *MemoryStore
	failGet    bool
	failPut    bool
	failDelete bool
}

func newFlakyStore() *flakyStore {
	return &flakyStore{inner: NewMemoryStore()}
}

func (f *flakyStore) Get(id string) (Session, error) {
	if f.failGet {
		return Session{}, core.E("sessiontest", "get exploded", nil)
	}
	return f.inner.Get(id)
}

func (f *flakyStore) Put(sess Session) error {
	if f.failPut {
		return core.E("sessiontest", "put exploded", nil)
	}
	return f.inner.Put(sess)
}

func (f *flakyStore) Delete(id string) error {
	if f.failDelete {
		return core.E("sessiontest", "delete exploded", nil)
	}
	return f.inner.Delete(id)
}

// TestSession_DefaultIDGen_Good — with no WithIDGen override the Manager mints a
// non-empty random id for each opened session, and two opens never collide.
func TestSession_DefaultIDGen_Good(t *core.T) {
	m := NewManager(NewMemoryStore()) // defaults: random id, core.Now clock

	a := m.Open("lemma")
	b := m.Open("lemma")
	core.AssertNotEmpty(t, a.ID, "the default generator mints a non-empty id")
	core.AssertNotEmpty(t, b.ID, "the default generator mints a non-empty id")
	core.AssertNotEqual(t, a.ID, b.ID, "two sessions get distinct random ids")
}

// TestSession_Get_Good — Get returns the live session (with its turns) for a
// known id, as a copy distinct from any later mutation.
func TestSession_Get_Good(t *core.T) {
	m := NewManager(NewMemoryStore(), WithIDGen(seqIDs("sess-1", "resp-1")))
	sess := m.Open("lemma")
	_, err := m.Append(sess.ID, userTurn("hi"))
	core.AssertNoError(t, err)

	got, err := m.Get(sess.ID)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "sess-1", got.ID)
	core.AssertLen(t, got.Turns, 1, "Get returns the current turns")
}

// TestSession_Get_Ugly — an empty id is a typed error rather than a store hit,
// and an unknown id surfaces the store's ErrNotFound.
func TestSession_Get_Ugly(t *core.T) {
	m := NewManager(NewMemoryStore())

	_, err := m.Get("")
	core.AssertError(t, err, "empty session id")

	_, err = m.Get("sess-missing")
	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrNotFound)
}

// TestSession_Append_StorePutFails_Bad — when the backing store fails to persist
// the appended turn, Append surfaces a typed error and never mints a dangling
// responseID for a turn that wasn't stored.
func TestSession_Append_StorePutFails_Bad(t *core.T) {
	store := newFlakyStore()
	m := NewManager(store, WithIDGen(seqIDs("sess-1", "resp-1")))
	sess := m.Open("lemma") // first Put succeeds (store healthy)

	store.failPut = true // now persistence of the appended turn fails
	resp, err := m.Append(sess.ID, userTurn("won't persist"))
	core.AssertError(t, err, "append: put")
	core.AssertEqual(t, "", resp, "a failed append mints no responseID")
}

// TestSession_Get_StoreFails_Bad — a store read failure for a known-shaped id is
// surfaced as a typed error (distinct from the empty-id guard).
func TestSession_Get_StoreFails_Bad(t *core.T) {
	store := newFlakyStore()
	m := NewManager(store, WithIDGen(seqIDs("sess-1")))
	sess := m.Open("lemma")

	store.failGet = true
	_, err := m.Get(sess.ID)
	core.AssertError(t, err, "get:")
}

// TestSession_Continue_StoreGetFails_Bad — a responseID resolves to a position,
// but the session it points at has vanished from the store (deleted out from
// under the registry). Continue surfaces the store error rather than handing
// back a stale empty session.
func TestSession_Continue_StoreGetFails_Bad(t *core.T) {
	store := NewMemoryStore()
	m := NewManager(store, WithIDGen(seqIDs("sess-1", "resp-1")))
	sess := m.Open("lemma")
	resp, err := m.Append(sess.ID, userTurn("hi"))
	core.AssertNoError(t, err)

	// Drop the session directly from the store, bypassing Manager.Delete (which
	// would also forget the responseID) so the position outlives its session.
	core.AssertNoError(t, store.Delete(sess.ID))

	_, err = m.Continue(resp)
	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrNotFound, "a position pointing at a gone session errors")
}

// TestSession_SetStateHandle_Ugly — an empty id is a typed error before any
// store access.
func TestSession_SetStateHandle_Ugly(t *core.T) {
	m := NewManager(NewMemoryStore())
	err := m.SetStateHandle("", "mlx-kv://x")
	core.AssertError(t, err, "empty session id")
}

// TestSession_SetStateHandle_StorePutFails_Bad — when persisting the handle
// fails, SetStateHandle surfaces a typed error rather than silently dropping it.
func TestSession_SetStateHandle_StorePutFails_Bad(t *core.T) {
	store := newFlakyStore()
	m := NewManager(store, WithIDGen(seqIDs("sess-1")))
	sess := m.Open("lemma")

	store.failPut = true
	err := m.SetStateHandle(sess.ID, "mlx-kv://node/slab/1")
	core.AssertError(t, err, "set state handle: put")
}

// TestSession_Delete_Good — Delete removes the session AND forgets every
// responseID that pointed at it, so a later Get fails and a stale
// previous_response_id can no longer resolve to the gone conversation.
func TestSession_Delete_Good(t *core.T) {
	m := NewManager(NewMemoryStore(), WithIDGen(seqIDs("sess-1", "resp-1", "resp-2")))
	sess := m.Open("lemma")
	r1, err := m.Append(sess.ID, userTurn("one"))
	core.AssertNoError(t, err)
	r2, err := m.Append(sess.ID, assistantTurn("two"))
	core.AssertNoError(t, err)

	core.AssertNoError(t, m.Delete(sess.ID))

	// The session is gone.
	_, err = m.Get(sess.ID)
	core.AssertErrorIs(t, err, ErrNotFound, "a deleted session is no longer gettable")

	// Both response ids that pointed at it are forgotten — they no longer resolve.
	_, err = m.Continue(r1)
	core.AssertErrorIs(t, err, ErrNotFound, "a stale response id can't resolve a deleted session")
	_, err = m.Continue(r2)
	core.AssertErrorIs(t, err, ErrNotFound, "every pointing response id is forgotten")
}

// TestSession_Delete_Ugly — an empty id is a typed error, and deleting an
// unknown id is a no-op success (the store treats a missing id as already gone).
func TestSession_Delete_Ugly(t *core.T) {
	m := NewManager(NewMemoryStore())

	err := m.Delete("")
	core.AssertError(t, err, "empty session id")

	// Deleting a never-opened id is not an error (MemoryStore.Delete is a no-op).
	core.AssertNoError(t, m.Delete("sess-never-existed"))
}

// TestSession_Delete_StoreFails_Bad — when the store's Delete fails, Manager
// surfaces a typed error. The responseID map is still cleared first (best-effort
// forgetting), but the operation reports the persistence failure.
func TestSession_Delete_StoreFails_Bad(t *core.T) {
	store := newFlakyStore()
	m := NewManager(store, WithIDGen(seqIDs("sess-1", "resp-1")))
	sess := m.Open("lemma")
	_, err := m.Append(sess.ID, userTurn("hi"))
	core.AssertNoError(t, err)

	store.failDelete = true
	err = m.Delete(sess.ID)
	core.AssertError(t, err, "delete:")
}

// TestSession_MemoryStore_Delete_Good — the MemoryStore Delete removes a stored
// session so a subsequent Get returns ErrNotFound, and deleting an absent id is
// a no-op rather than an error.
func TestSession_MemoryStore_Delete_Good(t *core.T) {
	store := NewMemoryStore()
	core.AssertNoError(t, store.Put(Session{ID: "s1", Model: "lemma"}))

	got, err := store.Get("s1")
	core.AssertNoError(t, err)
	core.AssertEqual(t, "s1", got.ID)

	core.AssertNoError(t, store.Delete("s1"))
	_, err = store.Get("s1")
	core.AssertErrorIs(t, err, ErrNotFound, "a deleted session is gone from the store")

	// Deleting a missing id is a no-op, not an error.
	core.AssertNoError(t, store.Delete("s1"), "deleting an absent id is a no-op")
}
