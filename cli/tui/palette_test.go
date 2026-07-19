// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

func TestCommandPalette_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	palette := newCommandPalette(styles)
	matches := palette.Filter("models panel")
	if len(matches) == 0 || matches[0].ID != commandPanelModels {
		t.Fatalf("models filter = %#v", matches)
	}
	a := newApp("", 0, 64)
	a.activePanel = panelWork
	if result := palette.Invoke(commandPanelModels, &a); !result.OK {
		t.Fatalf("Invoke models: %v", result.Value)
	}
	if a.activePanel != panelModels {
		t.Fatalf("active panel = %d, want Models", a.activePanel)
	}
	model, _ := a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a = model.(app)
	beforeHeight := a.view.Height
	if result := palette.Invoke(commandToggleInspector, &a); !result.OK {
		t.Fatalf("Invoke inspector: %v", result.Value)
	}
	if !a.inspectorOpen || a.view.Height == beforeHeight || a.view.Height != a.transcriptHeight() {
		t.Fatalf("palette inspector layout = open %v height %d (before %d)", a.inspectorOpen, a.view.Height, beforeHeight)
	}
}

func TestAgentCommandPalette_LifecycleActionMatrix(t *testing.T) {
	capabilities := []agentCapability{
		{Feature: agentFeatureCancel, Available: true},
		{Feature: agentFeatureChangesReview, Available: true},
	}
	statuses := []string{"queued", "preparing", "running", "completed", "waiting", "cancelling", "cancelled", "failed", "interrupted", "accepted", "rejected"}
	for _, status := range statuses {
		t.Run(status, func(t *testing.T) {
			selected := workItemRecord{ID: "work-1", Status: status}
			state := agentWorkSnapshot{NativeRunID: "run-1"}
			commands := agentWorkspaceCommandsForContext(capabilities, &selected, state)
			available := make(map[agentFeature]bool, len(commands))
			for _, command := range commands {
				for _, feature := range []agentFeature{agentFeatureCancel, agentFeatureChangesReview} {
					if command.ID == agentCommandID(feature) {
						available[feature] = command.Available
					}
				}
			}
			wantCancel := status == "queued" || status == "running"
			wantReview := status == "completed"
			if available[agentFeatureCancel] != wantCancel || available[agentFeatureChangesReview] != wantReview {
				t.Fatalf("status %q availability = cancel %v review %v, want %v/%v", status, available[agentFeatureCancel], available[agentFeatureChangesReview], wantCancel, wantReview)
			}
		})
	}
}

func TestCommandPalette_Bad(t *testing.T) {
	palette := newCommandPalette(newUIStyles(midnightTheme()))
	a := newApp("", 0, 64)
	a.activePanel = panelService
	before := a.activePanel
	result := palette.Invoke(commandID("missing.command"), &a)
	if result.OK {
		t.Fatal("unknown command invocation succeeded")
	}
	if a.activePanel != before || a.inspectorOpen {
		t.Fatalf("unknown command mutated app: panel=%d inspector=%v", a.activePanel, a.inspectorOpen)
	}
}

func TestCommandPalette_AgentDispatchNeedsSelectedWork(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	palette := newCommandPalette(styles)
	palette.SetAgentContext([]agentCapability{{Feature: agentFeatureDispatch, Available: true}}, nil)
	a := newApp("", 0, 64)
	command := palette.byID[agentCommandID(agentFeatureDispatch)]
	if command.Available || !strings.Contains(command.Reason, "selected Work") {
		t.Fatalf("dispatch without Work = %#v", command)
	}
	if result := palette.Invoke(agentCommandID(agentFeatureDispatch), &a); result.OK {
		t.Fatal("dispatch without Work was invokable")
	}
	selected := workItemRecord{Status: workStatusActive}
	palette.SetAgentContext([]agentCapability{{Feature: agentFeatureDispatch, Available: true}}, &selected)
	if command = palette.byID[agentCommandID(agentFeatureDispatch)]; !command.Available {
		t.Fatalf("dispatch with selected Work = %#v", command)
	}
}

func TestCommandPalette_AgentWorkStatus(t *testing.T) {
	capabilities := []agentCapability{
		{Feature: agentFeatureDispatch, Available: true}, {Feature: agentFeatureCancel, Available: true},
		{Feature: agentFeatureAnswer, Available: true}, {Feature: agentFeatureRetry, Available: true},
		{Feature: agentFeatureResume, Available: true}, {Feature: agentFeatureChangesReview, Available: true},
		{Feature: agentFeatureAccept, Available: true}, {Feature: agentFeatureReject, Available: true},
	}
	cases := []struct {
		status string
		ready  []agentFeature
	}{
		{workStatusActive, []agentFeature{agentFeatureDispatch}}, {"queued", []agentFeature{agentFeatureCancel}}, {"running", []agentFeature{agentFeatureCancel}},
		{workStatusWaiting, []agentFeature{agentFeatureAnswer, agentFeatureResume}}, {workStatusFailed, []agentFeature{agentFeatureRetry}}, {"interrupted", []agentFeature{agentFeatureResume}},
		{workStatusCompleted, []agentFeature{}},
	}
	for _, test := range cases {
		t.Run(test.status, func(t *testing.T) {
			palette := newCommandPalette(newUIStyles(midnightTheme()))
			selected := workItemRecord{Status: test.status}
			palette.SetAgentContext(capabilities, &selected)
			for _, feature := range []agentFeature{agentFeatureDispatch, agentFeatureCancel, agentFeatureAnswer, agentFeatureRetry, agentFeatureResume, agentFeatureChangesReview} {
				command := palette.byID[agentCommandID(feature)]
				want := false
				for _, ready := range test.ready {
					want = want || feature == ready
				}
				if got := command.Available; got != want {
					t.Fatalf("%s for %s available=%v reason=%q", feature, test.status, got, command.Reason)
				}
			}
			if command := palette.byID[agentCommandID(agentFeatureChangesReview)]; command.Available || !strings.Contains(command.Reason, "Task 14") {
				t.Fatalf("change review state = %#v", command)
			}
			for _, feature := range []agentFeature{agentFeatureAccept, agentFeatureReject} {
				if command := palette.byID[agentCommandID(feature)]; command.Available || !strings.Contains(command.Reason, "review-ready") {
					t.Fatalf("%s state = %#v", feature, command)
				}
			}
		})
	}
}

func TestCommandPaletteAnsweredWaitingDisablesAnswerAndEnablesResume(t *testing.T) {
	capabilities := []agentCapability{{Feature: agentFeatureAnswer, Available: true}, {Feature: agentFeatureResume, Available: true}}
	selected := workItemRecord{ID: "work-1", Status: workStatusWaiting}
	state := agentWorkSnapshot{
		ExternalID: "work-1", NativeRunID: "waiting-1", QuestionID: "question-1",
		AnswerID: "answer-1", ResumeRunID: "resume-1",
	}
	palette := newCommandPalette(newUIStyles(midnightTheme()))
	palette.SetAgentContext(capabilities, &selected, state)
	if palette.byID[agentCommandID(agentFeatureAnswer)].Available {
		t.Fatal("Answer remains available after its durable response")
	}
	if !palette.byID[agentCommandID(agentFeatureResume)].Available {
		t.Fatal("Resume is unavailable after its durable response")
	}
}

func TestCommandPaletteRecoveryAbandonRequiresPendingReceipt(t *testing.T) {
	capabilities := []agentCapability{{Feature: agentFeatureRecoveryAbandon, Available: true}}
	selected := workItemRecord{ID: "work-1", Status: workStatusFailed}
	palette := newCommandPalette(newUIStyles(midnightTheme()))
	palette.SetAgentContext(capabilities, &selected, agentWorkSnapshot{NativeRunID: "attempt-9"})
	command := palette.byID[agentCommandID(agentFeatureRecoveryAbandon)]
	if command.Available || !strings.Contains(command.Reason, "retained recovery") {
		t.Fatalf("recovery action without receipt = %#v", command)
	}

	state := agentWorkSnapshot{Recovery: agentPendingRecovery{EventID: "recovery-event-9", Receipt: agentRecoveryReceipt{
		Kind: "run", WorkID: selected.ID, RunID: "attempt-9", RunNumber: 9, WorkspaceRunID: "lineage-root",
	}}}
	palette.SetAgentContext(capabilities, &selected, state)
	command = palette.byID[agentCommandID(agentFeatureRecoveryAbandon)]
	if !command.Available {
		t.Fatalf("recovery action with receipt = %#v", command)
	}
}

func TestCommandPaletteReviewChangesDisabledByPendingCleanupRecovery(t *testing.T) {
	capabilities := []agentCapability{{Feature: agentFeatureChangesReview, Available: true}}
	selected := workItemRecord{ID: "work-1", Status: workStatusCompleted}
	for _, state := range []agentWorkSnapshot{
		{NativeRunID: "run-1", RecoveryCount: 1},
		{NativeRunID: "run-1", Recovery: agentPendingRecovery{EventID: "retained-cleanup-1"}},
	} {
		palette := newCommandPalette(newUIStyles(midnightTheme()))
		palette.SetAgentContext(capabilities, &selected, state)
		command := palette.byID[agentCommandID(agentFeatureChangesReview)]
		if command.Available || !strings.Contains(command.Reason, "retained cleanup recovery") {
			t.Fatalf("Review Changes with retained cleanup = %#v", command)
		}
	}
}

func TestCommandPalette_AgentNativeRunAndReviewState(t *testing.T) {
	caps := []agentCapability{{Feature: agentFeatureCancel, Available: true}, {Feature: agentFeatureChangesReview, Available: true}, {Feature: agentFeatureAccept, Available: true}, {Feature: agentFeatureReject, Available: true}}
	selected := workItemRecord{ID: "local", Status: "completed"}
	state := agentWorkSnapshot{NativeRunID: "run-7"}
	p := newCommandPalette(newUIStyles(midnightTheme()))
	p.SetAgentContext(caps, &selected, state)
	if p.byID[agentCommandID(agentFeatureCancel)].Available || !p.byID[agentCommandID(agentFeatureChangesReview)].Available {
		t.Fatalf("run actions unavailable: %#v", p.byID)
	}
	if p.byID[agentCommandID(agentFeatureAccept)].Available || p.byID[agentCommandID(agentFeatureReject)].Available {
		t.Fatalf("accept/reject available before reviewed receipt")
	}
	state.ReviewID, state.ReviewStatus, state.Review = "review-1", "prepared", agentReview{Feature: agentFeatureChangesReview, Payload: "opaque", AcceptanceAllowed: true}
	p.SetAgentContext(caps, &selected, state)
	if !p.byID[agentCommandID(agentFeatureAccept)].Available || !p.byID[agentCommandID(agentFeatureReject)].Available {
		t.Fatalf("review actions unavailable: %#v", p.byID)
	}
	state.ReviewStatus = "validation_failed"
	p.SetAgentContext(caps, &selected, state)
	if p.byID[agentCommandID(agentFeatureAccept)].Available || !p.byID[agentCommandID(agentFeatureReject)].Available {
		t.Fatalf("failed review actions = %#v", p.byID)
	}
}

func TestCommandPalette_AgentQueueState(t *testing.T) {
	caps := []agentCapability{{Feature: agentFeatureQueueStart, Available: true}, {Feature: agentFeatureQueueStop, Available: true}}
	for _, test := range []struct {
		status      string
		start, stop bool
	}{{"frozen", true, false}, {"accepting", false, true}, {"draining", false, false}} {
		p := newCommandPalette(newUIStyles(midnightTheme()))
		p.SetAgentContext(caps, nil, agentWorkSnapshot{QueueStatus: test.status})
		if p.byID[agentCommandID(agentFeatureQueueStart)].Available != test.start || p.byID[agentCommandID(agentFeatureQueueStop)].Available != test.stop {
			t.Fatalf("queue %s = %#v", test.status, p.byID)
		}
	}
}

func TestCommandPaletteLocalRefresh_Bad(t *testing.T) {
	palette := newCommandPalette(newUIStyles(midnightTheme()))
	a := newApp("", 0, 64)
	for _, id := range []commandID{commandRefreshRuntimes, commandRefreshKnowledge} {
		command := palette.byID[id]
		if command.Available || !strings.Contains(command.Reason, "restart") {
			t.Fatalf("refresh command %q = %#v", id, command)
		}
		if result := palette.Invoke(id, &a); result.OK {
			t.Fatalf("unwired refresh command %q reported success", id)
		}
	}
	if command := palette.byID[commandExportJSON]; command.Title != "Export JSON" || strings.Contains(command.Description, "JSON Lines") {
		t.Fatalf("JSON export command = %#v", command)
	}
}

func TestSessionSwitcher_Good(t *testing.T) {
	manager := openTestSessionManager(t, sequenceIDs("session-one", "session-two"))
	first := manager.Create().Value.(*chatSession)
	second := manager.Create().Value.(*chatSession)
	first.Record.Status = "idle"
	first.Record.PreferredModel = "gemma-4"
	second.Record.Status = "generating"
	second.Record.PreferredModel = "qwen-3"
	second.ActiveJobID = "job-hidden"
	second.Attention = true
	if result := manager.repository.SaveSession(first.Record); !result.OK {
		t.Fatalf("save first metadata: %v", result.Value)
	}
	if result := manager.repository.SaveSession(second.Record); !result.OK {
		t.Fatalf("save second metadata: %v", result.Value)
	}
	if result := manager.Switch(first.Record.ID); !result.OK {
		t.Fatalf("switch first: %v", result.Value)
	}
	second.Attention = true

	switcherResult := newSessionSwitcher(manager, newUIStyles(midnightTheme()), 72, 14)
	if !switcherResult.OK {
		t.Fatalf("newSessionSwitcher: %v", switcherResult.Value)
	}
	switcher := switcherResult.Value.(*sessionSwitcher)
	items := switcher.Items()
	if len(items) != 2 || items[0].SessionID != first.Record.ID || items[1].SessionID != second.Record.ID {
		t.Fatalf("recent switcher order = %#v", items)
	}
	if !strings.Contains(items[1].Title, "!") || !strings.Contains(items[1].Description, "generating") || !strings.Contains(items[1].Description, "qwen-3") {
		t.Fatalf("hidden session metadata = %#v", items[1])
	}

	switcher.list.Select(0)
	switcher.Update(tea.KeyMsg{Type: tea.KeyDown})
	if result := switcher.ActivateSelected(); !result.OK {
		t.Fatalf("activate selected: %v", result.Value)
	}
	if manager.Active().Record.ID != second.Record.ID {
		t.Fatalf("active session = %q, want %q", manager.Active().Record.ID, second.Record.ID)
	}
	if second.ActiveJobID != "job-hidden" {
		t.Fatalf("switch cancelled hidden job: %q", second.ActiveJobID)
	}
}

func TestHistorySearch_Good(t *testing.T) {
	manager := openTestSessionManager(t, sequenceIDs("session-match", "session-other"))
	matched := manager.Create().Value.(*chatSession)
	first := testTurnRecord("turn-before", matched.Record.ID, 1, "user", "ordinary opening", time.Now().UTC())
	second := testTurnRecord("turn-match", matched.Record.ID, 2, "assistant", "the durable needle is here", time.Now().UTC())
	if result := manager.AddTurn(first); !result.OK {
		t.Fatalf("add first turn: %v", result.Value)
	}
	if result := manager.AddTurn(second); !result.OK {
		t.Fatalf("add matching turn: %v", result.Value)
	}
	other := manager.Create().Value.(*chatSession)
	if manager.Active().Record.ID != other.Record.ID {
		t.Fatal("search fixture did not leave another session active")
	}

	searchResult := newHistorySearch(manager.repository, manager, newUIStyles(midnightTheme()), 72, 14)
	if !searchResult.OK {
		t.Fatalf("newHistorySearch: %v", searchResult.Value)
	}
	search := searchResult.Value.(*historySearch)
	if result := search.Search("durable needle"); !result.OK {
		t.Fatalf("Search: %v", result.Value)
	}
	if len(search.Hits()) != 1 || search.Hits()[0].Session.ID != matched.Record.ID {
		t.Fatalf("search hits = %#v", search.Hits())
	}
	if result := search.ActivateSelected(); !result.OK {
		t.Fatalf("activate search hit: %v", result.Value)
	}
	if manager.Active().Record.ID != matched.Record.ID {
		t.Fatalf("active after hit = %q", manager.Active().Record.ID)
	}
	if manager.Active().ViewportOffset != 1 || manager.Active().Follow {
		t.Fatalf("matched viewport = offset %d follow %v, want 1/false", manager.Active().ViewportOffset, manager.Active().Follow)
	}
	if search.MatchTurnID() != second.ID {
		t.Fatalf("match turn = %q, want %q", search.MatchTurnID(), second.ID)
	}
}

func TestHistorySearchUsesRenderedTurnOffset_Ugly(t *testing.T) {
	a := newApp("", 0, 64)
	a.view.Width = 24
	a.turns = []turn{
		{id: "turn-wrapped", role: "assistant", text: "This is a deliberately long markdown paragraph that wraps across several terminal lines before the next match."},
		{id: "turn-match", role: "assistant", text: "durable needle"},
	}
	offset := a.transcriptTurnOffset("turn-match")
	if offset <= 1 {
		t.Fatalf("rendered match offset = %d, want wrapped line offset", offset)
	}
}

func TestOverlayRouting_Ugly(t *testing.T) {
	a := newApp("", 0, 64)
	m, _ := a.Update(tea.WindowSizeMsg{Width: 100, Height: 24})
	a = m.(app)
	a.activePanel = panelService
	a.generating = true

	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyCtrlK})
	a = m.(app)
	if a.activeOverlay != overlayCommands {
		t.Fatalf("ctrl+k overlay = %d, want commands", a.activeOverlay)
	}
	a.palette.list.SetFilterText("models panel")
	beforeAddress := a.svc.addrIdx
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyDown})
	a = m.(app)
	if a.activePanel != panelService || a.svc.addrIdx != beforeAddress {
		t.Fatal("overlay arrow leaked into the Service panel")
	}
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if a.activePanel != panelModels || a.svc.running || a.activeOverlay != overlayNone {
		t.Fatalf("overlay Enter: panel=%d service=%v overlay=%d", a.activePanel, a.svc.running, a.activeOverlay)
	}

	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyCtrlK})
	a = m.(app)
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEsc})
	a = m.(app)
	if a.activeOverlay != overlayNone || !a.generating {
		t.Fatalf("overlay Escape: overlay=%d generating=%v", a.activeOverlay, a.generating)
	}
}
