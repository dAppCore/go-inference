// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"time"

	core "dappco.re/go"
)

const (
	newSessionTitle      = "New session"
	sessionTitleMaxRunes = 48
	stateActiveSession   = "active_session"
)

type chatSession struct {
	Record         sessionRecord
	Turns          []turnRecord
	Draft          string
	ViewportOffset int
	Follow         bool
	Attention      bool
	ToolHops       int
	ActiveJobID    string
}

type sessionManager struct {
	repository workspaceRepository
	state      reactiveState
	ids        func() string
	now        func() time.Time
	order      []string
	activeID   string
	sessions   map[string]*chatSession
}

type persistedViewport struct {
	Offset int  `json:"offset"`
	Follow bool `json:"follow"`
}

func newSessionManager(
	repository workspaceRepository,
	state reactiveState,
	ids func() string,
	now func() time.Time,
) core.Result {
	if repository == nil {
		return core.Fail(core.E("tui.newSessionManager", "repository is required", nil))
	}
	if state == nil {
		return core.Fail(core.E("tui.newSessionManager", "reactive state is required", nil))
	}
	if ids == nil {
		ids = newRecordID
	}
	if now == nil {
		now = time.Now
	}

	listed := repository.ListSessions(false)
	if !listed.OK {
		return core.Fail(core.E("tui.newSessionManager", "list sessions", resultError(listed)))
	}
	records, ok := listed.Value.([]sessionRecord)
	if !ok {
		return core.Fail(core.E("tui.newSessionManager", "invalid session list result", nil))
	}
	manager := &sessionManager{
		repository: repository,
		state:      state,
		ids:        ids,
		now:        now,
		order:      make([]string, 0, len(records)),
		sessions:   make(map[string]*chatSession, len(records)),
	}
	for _, record := range records {
		session := &chatSession{Record: record, Follow: true}
		manager.restoreSessionState(session)
		manager.sessions[record.ID] = session
		manager.order = append(manager.order, record.ID)
	}

	if activeID, result := state.Get(reactiveGroupWorkspace, stateActiveSession); result.OK {
		if _, exists := manager.sessions[activeID]; exists {
			manager.activeID = activeID
		}
	}
	if manager.activeID == "" && len(manager.order) > 0 {
		manager.activeID = manager.order[0]
	}
	if manager.activeID != "" {
		manager.moveToFront(manager.activeID)
		if result := manager.loadSessionTurns(manager.sessions[manager.activeID]); !result.OK {
			return result
		}
	}
	return core.Ok(manager)
}

func (manager *sessionManager) Active() *chatSession {
	if manager == nil || manager.activeID == "" {
		return nil
	}
	return manager.sessions[manager.activeID]
}

func (manager *sessionManager) Recent() []*chatSession {
	if manager == nil {
		return nil
	}
	recent := make([]*chatSession, 0, len(manager.order))
	for _, id := range manager.order {
		if session, ok := manager.sessions[id]; ok {
			recent = append(recent, session)
		}
	}
	return recent
}

func (manager *sessionManager) Session(id string) core.Result {
	if manager == nil {
		return core.Fail(core.E("tui.sessionManager.Session", "session manager is unavailable", nil))
	}
	session, ok := manager.sessions[id]
	if !ok {
		return core.Fail(core.E("tui.sessionManager.Session", core.Concat("unknown session: ", id), nil))
	}
	if result := manager.loadSessionTurns(session); !result.OK {
		return result
	}
	return core.Ok(session)
}

func (manager *sessionManager) Create() core.Result {
	if manager == nil || manager.repository == nil {
		return core.Fail(core.E("tui.sessionManager.Create", "session manager is unavailable", nil))
	}
	id := core.Trim(manager.ids())
	if id == "" {
		return core.Fail(core.E("tui.sessionManager.Create", "session ID generator returned an empty value", nil))
	}
	if _, exists := manager.sessions[id]; exists {
		return core.Fail(core.E("tui.sessionManager.Create", core.Concat("duplicate session ID: ", id), nil))
	}
	createdAt := manager.now().UTC()
	record := sessionRecord{
		ID:             id,
		Title:          newSessionTitle,
		Status:         "idle",
		Mode:           "chat",
		GenerationJSON: "{}",
		ToolsJSON:      "[]",
		CreatedAt:      createdAt,
		UpdatedAt:      createdAt,
		LastOpenedAt:   createdAt,
		ArchivedAt:     unsetRecordTime(),
	}
	if result := manager.repository.SaveSession(record); !result.OK {
		return result
	}
	session := &chatSession{
		Record: record,
		Turns:  []turnRecord{},
		Follow: true,
	}
	manager.sessions[id] = session
	manager.moveToFront(id)
	manager.activeID = id
	manager.persistActive()
	return core.Ok(session)
}

func (manager *sessionManager) Switch(id string) core.Result {
	if manager == nil {
		return core.Fail(core.E("tui.sessionManager.Switch", "session manager is unavailable", nil))
	}
	session, ok := manager.sessions[id]
	if !ok {
		return core.Fail(core.E("tui.sessionManager.Switch", core.Concat("unknown session: ", id), nil))
	}
	if result := manager.loadSessionTurns(session); !result.OK {
		return result
	}
	updated := session.Record
	updated.LastOpenedAt = manager.now().UTC()
	updated.UpdatedAt = updated.LastOpenedAt
	if result := manager.repository.SaveSession(updated); !result.OK {
		return result
	}
	session.Record = updated
	session.Attention = false
	manager.activeID = id
	manager.moveToFront(id)
	manager.persistActive()
	return core.Ok(session)
}

func (manager *sessionManager) Previous() core.Result {
	return manager.switchRelative(1, "Previous")
}

func (manager *sessionManager) Next() core.Result {
	return manager.switchRelative(-1, "Next")
}

func (manager *sessionManager) SetDraft(sessionID, draft string) core.Result {
	session, result := manager.knownSession("SetDraft", sessionID)
	if !result.OK {
		return result
	}
	session.Draft = draft
	return manager.state.Set(reactiveGroupDrafts, sessionID, draft)
}

func (manager *sessionManager) SetViewport(sessionID string, offset int, follow bool) core.Result {
	session, result := manager.knownSession("SetViewport", sessionID)
	if !result.OK {
		return result
	}
	if offset < 0 {
		offset = 0
	}
	session.ViewportOffset = offset
	session.Follow = follow
	persisted := core.JSONMarshalString(persistedViewport{Offset: offset, Follow: follow})
	return manager.state.Set(reactiveGroupViewport, sessionID, persisted)
}

func (manager *sessionManager) Complete(sessionID string) core.Result {
	session, result := manager.knownSession("Complete", sessionID)
	if !result.OK {
		return result
	}
	updated := session.Record
	updated.Status = "idle"
	updated.UpdatedAt = manager.now().UTC()
	if result := manager.repository.SaveSession(updated); !result.OK {
		return result
	}
	session.Record = updated
	session.ActiveJobID = ""
	session.ToolHops = 0
	session.Attention = sessionID != manager.activeID
	return core.Ok(session)
}

func (manager *sessionManager) AddTurn(record turnRecord) core.Result {
	session, result := manager.knownSession("AddTurn", record.SessionID)
	if !result.OK {
		return result
	}
	if loadResult := manager.loadSessionTurns(session); !loadResult.OK {
		return loadResult
	}
	hadUserTurn := sessionHasUserTurn(session.Turns)
	if result := manager.repository.SaveTurn(record); !result.OK {
		return result
	}
	session.Turns = upsertOrderedTurn(session.Turns, record)

	if record.Role == "user" && !hadUserTurn && session.Record.Title == newSessionTitle {
		title := compactSessionTitle(record.Visible)
		if title != "" {
			updated := session.Record
			updated.Title = title
			updated.UpdatedAt = manager.now().UTC()
			if result := manager.repository.SaveSession(updated); !result.OK {
				return result
			}
			session.Record = updated
		}
	}
	return core.Ok(session)
}

func (manager *sessionManager) Rename(sessionID, title string) core.Result {
	session, result := manager.knownSession("Rename", sessionID)
	if !result.OK {
		return result
	}
	title = compactSessionTitle(title)
	if title == "" {
		return core.Fail(core.E("tui.sessionManager.Rename", "session title is required", nil))
	}
	updated := session.Record
	updated.Title = title
	updated.UpdatedAt = manager.now().UTC()
	if result := manager.repository.SaveSession(updated); !result.OK {
		return result
	}
	session.Record = updated
	return core.Ok(session)
}

func (manager *sessionManager) Archive(sessionID string) core.Result {
	session, result := manager.knownSession("Archive", sessionID)
	if !result.OK {
		return result
	}
	updated := session.Record
	updated.Archived = true
	updated.ArchivedAt = manager.now().UTC()
	updated.UpdatedAt = updated.ArchivedAt
	if result := manager.repository.SaveSession(updated); !result.OK {
		return result
	}
	delete(manager.sessions, sessionID)
	manager.removeFromOrder(sessionID)
	if manager.activeID == sessionID {
		manager.activeID = ""
		if len(manager.order) > 0 {
			manager.activeID = manager.order[0]
			if loadResult := manager.loadSessionTurns(manager.sessions[manager.activeID]); !loadResult.OK {
				return loadResult
			}
		}
		manager.persistActive()
	}
	return core.Ok(updated)
}

func (manager *sessionManager) Reopen(sessionID string) core.Result {
	if manager == nil {
		return core.Fail(core.E("tui.sessionManager.Reopen", "session manager is unavailable", nil))
	}
	if _, exists := manager.sessions[sessionID]; exists {
		return manager.Switch(sessionID)
	}
	loaded := manager.repository.Session(sessionID)
	if !loaded.OK {
		return loaded
	}
	record, ok := loaded.Value.(sessionRecord)
	if !ok {
		return core.Fail(core.E("tui.sessionManager.Reopen", "invalid session result", nil))
	}
	record.Archived = false
	record.ArchivedAt = unsetRecordTime()
	record.UpdatedAt = manager.now().UTC()
	record.LastOpenedAt = record.UpdatedAt
	if result := manager.repository.SaveSession(record); !result.OK {
		return result
	}
	session := &chatSession{Record: record, Follow: true}
	manager.restoreSessionState(session)
	manager.sessions[sessionID] = session
	manager.moveToFront(sessionID)
	manager.activeID = sessionID
	if result := manager.loadSessionTurns(session); !result.OK {
		return result
	}
	manager.persistActive()
	return core.Ok(session)
}

func (manager *sessionManager) switchRelative(delta int, operation string) core.Result {
	if manager == nil || manager.activeID == "" || len(manager.order) == 0 {
		return core.Fail(core.E(core.Concat("tui.sessionManager.", operation), "no active session", nil))
	}
	if len(manager.order) == 1 {
		return core.Ok(manager.Active())
	}
	index := 0
	for candidate, id := range manager.order {
		if id == manager.activeID {
			index = candidate
			break
		}
	}
	target := (index + delta) % len(manager.order)
	if target < 0 {
		target += len(manager.order)
	}
	return manager.Switch(manager.order[target])
}

func (manager *sessionManager) knownSession(operation, id string) (*chatSession, core.Result) {
	if manager == nil {
		return nil, core.Fail(core.E(core.Concat("tui.sessionManager.", operation), "session manager is unavailable", nil))
	}
	session, ok := manager.sessions[id]
	if !ok {
		return nil, core.Fail(core.E(core.Concat("tui.sessionManager.", operation), core.Concat("unknown session: ", id), nil))
	}
	return session, core.Ok(nil)
}

func (manager *sessionManager) loadSessionTurns(session *chatSession) core.Result {
	if session == nil {
		return core.Fail(core.E("tui.sessionManager.loadSessionTurns", "session is required", nil))
	}
	if session.Turns != nil {
		return core.Ok(nil)
	}
	loaded := manager.repository.Turns(session.Record.ID)
	if !loaded.OK {
		return loaded
	}
	turns, ok := loaded.Value.([]turnRecord)
	if !ok {
		return core.Fail(core.E("tui.sessionManager.loadSessionTurns", "invalid turns result", nil))
	}
	if turns == nil {
		turns = []turnRecord{}
	}
	session.Turns = turns
	return core.Ok(nil)
}

func (manager *sessionManager) restoreSessionState(session *chatSession) {
	if session == nil {
		return
	}
	if draft, result := manager.state.Get(reactiveGroupDrafts, session.Record.ID); result.OK {
		session.Draft = draft
	}
	if raw, result := manager.state.Get(reactiveGroupViewport, session.Record.ID); result.OK {
		var persisted persistedViewport
		if decoded := core.JSONUnmarshalString(raw, &persisted); decoded.OK {
			if persisted.Offset > 0 {
				session.ViewportOffset = persisted.Offset
			}
			session.Follow = persisted.Follow
		}
	}
}

func (manager *sessionManager) persistActive() {
	if manager == nil || manager.state == nil {
		return
	}
	if result := manager.state.Set(reactiveGroupWorkspace, stateActiveSession, manager.activeID); !result.OK {
		core.Warn("tui.sessions.persist_active", "session", manager.activeID, "error", result.Value)
	}
}

func (manager *sessionManager) moveToFront(id string) {
	manager.removeFromOrder(id)
	manager.order = append([]string{id}, manager.order...)
}

func (manager *sessionManager) removeFromOrder(id string) {
	for index, candidate := range manager.order {
		if candidate != id {
			continue
		}
		copy(manager.order[index:], manager.order[index+1:])
		manager.order = manager.order[:len(manager.order)-1]
		return
	}
}

func sessionHasUserTurn(turns []turnRecord) bool {
	for _, turn := range turns {
		if turn.Role == "user" {
			return true
		}
	}
	return false
}

func upsertOrderedTurn(turns []turnRecord, record turnRecord) []turnRecord {
	for index := range turns {
		if turns[index].ID == record.ID {
			turns[index] = record
			return turns
		}
	}
	insertAt := len(turns)
	for index, turn := range turns {
		if turn.Sequence > record.Sequence {
			insertAt = index
			break
		}
	}
	turns = append(turns, turnRecord{})
	copy(turns[insertAt+1:], turns[insertAt:])
	turns[insertAt] = record
	return turns
}

func compactSessionTitle(text string) string {
	title := core.Join(" ", core.Fields(text)...)
	if title == "" {
		return ""
	}
	runes := []rune(title)
	if len(runes) <= sessionTitleMaxRunes {
		return title
	}
	return core.Concat(string(runes[:sessionTitleMaxRunes-1]), "…")
}
