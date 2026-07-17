// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"reflect"
	"testing"
	"time"

	core "dappco.re/go"
)

func TestSessionManager_Good(t *testing.T) {
	paths := testReactivePaths(t)
	repository := openSessionRepository(t, paths.Database)
	state := openSessionState(t, paths)
	ids := sequenceIDs("session-one", "session-two")
	clock := sequenceClock(time.Date(2026, time.July, 17, 15, 0, 0, 0, time.UTC))

	opened := newSessionManager(repository, state, ids, clock)
	if !opened.OK {
		t.Fatalf("newSessionManager empty: %v", opened.Value)
	}
	manager, ok := opened.Value.(*sessionManager)
	if !ok {
		t.Fatalf("newSessionManager value = %T, want *sessionManager", opened.Value)
	}
	if manager.Active() != nil || len(manager.Recent()) != 0 {
		t.Fatalf("empty manager active/recent = %#v / %#v", manager.Active(), manager.Recent())
	}

	firstResult := manager.Create()
	if !firstResult.OK {
		t.Fatalf("create first session: %v", firstResult.Value)
	}
	first := firstResult.Value.(*chatSession)
	if first.Record.ID != "session-one" || manager.Active().Record.ID != first.Record.ID {
		t.Fatalf("first session = %#v, active %#v", first, manager.Active())
	}
	if result := manager.SetDraft(first.Record.ID, "first draft"); !result.OK {
		t.Fatalf("set first draft: %v", result.Value)
	}
	if result := manager.SetViewport(first.Record.ID, 17, false); !result.OK {
		t.Fatalf("set first viewport: %v", result.Value)
	}

	secondResult := manager.Create()
	if !secondResult.OK {
		t.Fatalf("create second session: %v", secondResult.Value)
	}
	second := secondResult.Value.(*chatSession)
	if result := manager.SetDraft(second.Record.ID, "second draft"); !result.OK {
		t.Fatalf("set second draft: %v", result.Value)
	}
	if result := manager.SetViewport(second.Record.ID, 29, true); !result.OK {
		t.Fatalf("set second viewport: %v", result.Value)
	}
	if result := manager.Previous(); !result.OK || manager.Active().Record.ID != first.Record.ID {
		t.Fatalf("Previous = %#v, active %v, want %q", result.Value, manager.Active(), first.Record.ID)
	}
	if result := manager.Next(); !result.OK || manager.Active().Record.ID != second.Record.ID {
		t.Fatalf("Next = %#v, active %v, want %q", result.Value, manager.Active(), second.Record.ID)
	}

	if result := state.Close(); !result.OK {
		t.Fatalf("close first state: %v", result.Value)
	}
	if result := repository.Close(); !result.OK {
		t.Fatalf("close first repository: %v", result.Value)
	}

	repository = openSessionRepository(t, paths.Database)
	state = openSessionState(t, paths)
	defer closeSessionFixture(t, repository, state)
	restoredResult := newSessionManager(repository, state, sequenceIDs("unused"), clock)
	if !restoredResult.OK {
		t.Fatalf("restore session manager: %v", restoredResult.Value)
	}
	restored := restoredResult.Value.(*sessionManager)
	if restored.Active() == nil || restored.Active().Record.ID != second.Record.ID {
		t.Fatalf("restored active = %#v, want %q", restored.Active(), second.Record.ID)
	}
	recent := restored.Recent()
	if len(recent) != 2 || recent[0].Record.ID != second.Record.ID || recent[1].Record.ID != first.Record.ID {
		t.Fatalf("restored recent order = %#v, want [%s %s]", recent, second.Record.ID, first.Record.ID)
	}
	firstRestored := restored.sessions[first.Record.ID]
	secondRestored := restored.sessions[second.Record.ID]
	if firstRestored.Draft != "first draft" || firstRestored.ViewportOffset != 17 || firstRestored.Follow {
		t.Fatalf("restored first UI state = %#v", firstRestored)
	}
	if secondRestored.Draft != "second draft" || secondRestored.ViewportOffset != 29 || !secondRestored.Follow {
		t.Fatalf("restored second UI state = %#v", secondRestored)
	}
	if firstRestored.Turns != nil {
		t.Fatalf("inactive turns eagerly loaded = %#v, want nil", firstRestored.Turns)
	}
	if secondRestored.Turns == nil {
		t.Fatal("active turns were not loaded")
	}
}

func TestSessionManager_Bad(t *testing.T) {
	manager := openTestSessionManager(t, sequenceIDs("session-known"))
	if result := manager.Create(); !result.OK {
		t.Fatalf("create known session: %v", result.Value)
	}
	activeBefore := manager.Active().Record.ID
	orderBefore := append([]string(nil), manager.order...)

	result := manager.Switch("session-missing")
	if result.OK {
		t.Fatalf("Switch unknown session = %#v, want failure", result.Value)
	}
	if manager.Active().Record.ID != activeBefore || !reflect.DeepEqual(manager.order, orderBefore) {
		t.Fatalf("unknown switch mutated active/order to %q / %#v", manager.Active().Record.ID, manager.order)
	}
}

func TestSessionManager_Ugly(t *testing.T) {
	manager := openTestSessionManager(t, sequenceIDs("session-a", "session-b", "session-c"))
	for range 3 {
		if result := manager.Create(); !result.OK {
			t.Fatalf("create session: %v", result.Value)
		}
	}
	if result := manager.Switch("session-a"); !result.OK {
		t.Fatalf("switch to session-a: %v", result.Value)
	}
	if result := manager.Complete("session-b"); !result.OK {
		t.Fatalf("complete hidden session-b: %v", result.Value)
	}
	if result := manager.Complete("session-c"); !result.OK {
		t.Fatalf("complete hidden session-c: %v", result.Value)
	}
	if !manager.sessions["session-b"].Attention || !manager.sessions["session-c"].Attention {
		t.Fatalf("hidden attention = b:%v c:%v, want both", manager.sessions["session-b"].Attention, manager.sessions["session-c"].Attention)
	}

	if result := manager.Switch("session-b"); !result.OK {
		t.Fatalf("switch to completed session-b: %v", result.Value)
	}
	if manager.sessions["session-b"].Attention {
		t.Fatal("active session-b retained attention marker")
	}
	if !manager.sessions["session-c"].Attention {
		t.Fatal("switching session-b cleared session-c attention")
	}
}

func TestSessionManager_Title_Good(t *testing.T) {
	manager := openTestSessionManager(t, sequenceIDs("session-title", "session-pre-renamed"))
	created := manager.Create()
	if !created.OK {
		t.Fatalf("create title session: %v", created.Value)
	}
	session := created.Value.(*chatSession)
	firstText := "  Explain\n\twhy a durable multi-session terminal workspace should preserve context across restarts and model changes  "
	first := testTurnRecord("turn-title-first", session.Record.ID, 1, "user", firstText, time.Now().UTC())
	if result := manager.AddTurn(first); !result.OK {
		t.Fatalf("add first user turn: %v", result.Value)
	}
	derived := session.Record.Title
	if derived == "New session" || core.ContainsAny(derived, "\n\t") || core.RuneCount(derived) > sessionTitleMaxRunes {
		t.Fatalf("derived title = %q (%d runes)", derived, core.RuneCount(derived))
	}

	if result := manager.Rename(session.Record.ID, "Hand edited title"); !result.OK {
		t.Fatalf("rename session: %v", result.Value)
	}
	second := testTurnRecord("turn-title-second", session.Record.ID, 2, "user", "This later prompt must not rename it", time.Now().UTC())
	if result := manager.AddTurn(second); !result.OK {
		t.Fatalf("add later user turn: %v", result.Value)
	}
	if session.Record.Title != "Hand edited title" {
		t.Fatalf("later prompt renamed session to %q", session.Record.Title)
	}

	preRenamedResult := manager.Create()
	if !preRenamedResult.OK {
		t.Fatalf("create pre-renamed session: %v", preRenamedResult.Value)
	}
	preRenamed := preRenamedResult.Value.(*chatSession)
	if result := manager.Rename(preRenamed.Record.ID, "Named before prompting"); !result.OK {
		t.Fatalf("pre-rename session: %v", result.Value)
	}
	turn := testTurnRecord("turn-pre-renamed", preRenamed.Record.ID, 1, "user", "First prompt", time.Now().UTC())
	if result := manager.AddTurn(turn); !result.OK {
		t.Fatalf("add pre-renamed first turn: %v", result.Value)
	}
	if preRenamed.Record.Title != "Named before prompting" {
		t.Fatalf("first prompt replaced manual title with %q", preRenamed.Record.Title)
	}

	stored := manager.repository.Session(session.Record.ID)
	if !stored.OK || stored.Value.(sessionRecord).Title != "Hand edited title" {
		t.Fatalf("persisted edited title = %#v", stored.Value)
	}
}

func openTestSessionManager(t *testing.T, ids func() string) *sessionManager {
	t.Helper()
	paths := testReactivePaths(t)
	repository := openSessionRepository(t, paths.Database)
	state := openSessionState(t, paths)
	t.Cleanup(func() { closeSessionFixture(t, repository, state) })
	result := newSessionManager(
		repository,
		state,
		ids,
		sequenceClock(time.Date(2026, time.July, 17, 16, 0, 0, 0, time.UTC)),
	)
	if !result.OK {
		t.Fatalf("newSessionManager: %v", result.Value)
	}
	return result.Value.(*sessionManager)
}

func openSessionRepository(t *testing.T, path string) workspaceRepository {
	t.Helper()
	result := openDuckRepository(path)
	if !result.OK {
		t.Fatalf("open session repository: %v", result.Value)
	}
	repository, ok := result.Value.(workspaceRepository)
	if !ok {
		t.Fatalf("openDuckRepository value = %T, want workspaceRepository", result.Value)
	}
	return repository
}

func openSessionState(t *testing.T, paths appPaths) reactiveState {
	t.Helper()
	result := openReactiveState(paths)
	if !result.OK {
		t.Fatalf("open session state: %v", result.Value)
	}
	state, ok := result.Value.(reactiveState)
	if !ok {
		t.Fatalf("openReactiveState value = %T, want reactiveState", result.Value)
	}
	return state
}

func closeSessionFixture(t *testing.T, repository workspaceRepository, state reactiveState) {
	t.Helper()
	if result := state.Close(); !result.OK {
		t.Errorf("close session state: %v", result.Value)
	}
	if result := repository.Close(); !result.OK {
		t.Errorf("close session repository: %v", result.Value)
	}
}

func sequenceIDs(ids ...string) func() string {
	index := 0
	return func() string {
		if index >= len(ids) {
			return ""
		}
		id := ids[index]
		index++
		return id
	}
}

func sequenceClock(start time.Time) func() time.Time {
	current := start.Add(-time.Second)
	return func() time.Time {
		current = current.Add(time.Second)
		return current
	}
}
