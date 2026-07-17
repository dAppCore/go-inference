// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"context"
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/list"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/x/ansi"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/parser"
)

func TestAppBootstrap_Good(t *testing.T) {
	resources := openAppTestWorkspace(t)
	resources.Warnings = append(resources.Warnings, "reactive state: degraded fixture")
	a := newWorkspaceApp("", 0, 64, func() core.Result { return core.Ok(resources) })
	if a.activePanel != panelChat || a.boot.phase != bootLoading {
		t.Fatalf("initial workspace app = panel %d boot %#v", a.activePanel, a.boot)
	}
	message := workspaceBootstrap(a.workspaceLoader)()
	m, command := a.Update(message)
	a = m.(app)
	if a.boot.phase != bootReady || a.resources != resources || a.sessions == nil || a.sessions.Active() == nil {
		t.Fatalf("ready app = boot %#v resources=%p sessions=%#v", a.boot, a.resources, a.sessions)
	}
	if a.repository == nil || a.preferences == nil || a.work == nil || a.knowledge == nil {
		t.Fatalf("composition incomplete: repository=%v preferences=%v work=%v knowledge=%v", a.repository != nil, a.preferences != nil, a.work != nil, a.knowledge != nil)
	}
	if command == nil || !strings.Contains(a.statusLine(), "degraded fixture") {
		t.Fatalf("ready command/warning = %v / %q", command != nil, a.statusLine())
	}
	if result := a.shutdown(); !result.OK {
		t.Fatalf("shutdown: %v", result.Value)
	}
}

func TestAppBootstrap_Bad(t *testing.T) {
	attempts := 0
	var resources *workspaceResources
	loader := func() core.Result {
		attempts++
		if attempts == 1 {
			return core.Fail(core.E("test.bootstrap", "/tmp/lem.duckdb is unavailable", nil))
		}
		resources = openAppTestWorkspace(t)
		return core.Ok(resources)
	}
	a := newWorkspaceApp("", 0, 64, loader)
	m, _ := a.Update(workspaceBootstrap(loader)())
	a = m.(app)
	if a.boot.phase != bootFailed || !strings.Contains(a.View(), "/tmp/lem.duckdb") || !strings.Contains(a.View(), "Retry") {
		t.Fatalf("blocking storage view:\n%s", a.View())
	}
	m, command := a.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'r'}})
	a = m.(app)
	if a.boot.phase != bootLoading || command == nil {
		t.Fatalf("retry state = %#v command=%v", a.boot, command != nil)
	}
	m, _ = a.Update(command())
	a = m.(app)
	if a.boot.phase != bootReady || attempts != 2 || a.resources != resources {
		t.Fatalf("retry result = boot %#v attempts=%d", a.boot, attempts)
	}
	_ = a.shutdown()
}

func TestAppBootstrap_Ugly(t *testing.T) {
	a := newWorkspaceApp("", 0, 64, func() core.Result {
		return core.Fail(core.E("test.bootstrap", "storage offline", nil))
	})
	m, _ := a.Update(workspaceBootstrap(a.workspaceLoader)())
	a = m.(app)
	m, command := a.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'q'}})
	a = m.(app)
	if command == nil {
		t.Fatal("quit from blocking storage screen returned no command")
	}
	if _, ok := command().(tea.QuitMsg); !ok {
		t.Fatalf("quit command produced %T", command())
	}
}

func TestAppLifecycleLoaders_Ugly(t *testing.T) {
	t.Run("workspace", func(t *testing.T) {
		resources := openAppTestWorkspace(t)
		started := make(chan struct{})
		release := make(chan struct{})
		a := newWorkspaceApp("", 0, 64, func() core.Result {
			close(started)
			<-release
			return core.Ok(resources)
		})
		command := a.lifecycle.command(workspaceBootstrap(a.workspaceLoader))
		message := make(chan tea.Msg, 1)
		go func() { message <- command() }()
		<-started
		shutdown := make(chan core.Result, 1)
		go func() { shutdown <- a.shutdown() }()
		<-a.lifecycle.stopCh
		close(release)
		if result := <-shutdown; !result.OK {
			t.Fatalf("shutdown workspace loader: %s", result.Error())
		}
		if _, ok := (<-message).(lifecycleStoppedMsg); !ok {
			t.Fatal("late workspace load was delivered after shutdown")
		}
		if resources.Repository != nil || resources.State != nil {
			t.Fatal("late workspace resources were not closed")
		}
	})

	t.Run("model", func(t *testing.T) {
		base := newFakeTextModel(nil)
		started := make(chan struct{})
		release := make(chan struct{})
		a := newApp("", 0, 64)
		command := a.lifecycle.command(func() tea.Msg {
			close(started)
			<-release
			return loadedMsg{model: base, name: "late"}
		})
		message := make(chan tea.Msg, 1)
		go func() { message <- command() }()
		<-started
		shutdown := make(chan core.Result, 1)
		go func() { shutdown <- a.shutdown() }()
		<-a.lifecycle.stopCh
		close(release)
		if result := <-shutdown; !result.OK {
			t.Fatalf("shutdown model loader: %s", result.Error())
		}
		if _, ok := (<-message).(lifecycleStoppedMsg); !ok {
			t.Fatal("late model load was delivered after shutdown")
		}
		if base.closes.Load() != 1 {
			t.Fatalf("late model close count = %d", base.closes.Load())
		}
	})

	t.Run("completed but undelivered model", func(t *testing.T) {
		base := newFakeTextModel(nil)
		a := newApp("", 0, 64)
		message := a.lifecycle.command(func() tea.Msg {
			return loadedMsg{model: base, name: "pending"}
		})()
		if result := a.shutdown(); !result.OK {
			t.Fatalf("shutdown pending model: %s", result.Error())
		}
		if base.closes.Load() != 1 {
			t.Fatalf("pending model close count = %d", base.closes.Load())
		}
		model, _ := a.Update(message)
		a = model.(app)
		if base.closes.Load() != 1 {
			t.Fatalf("pending model closed twice: %d", base.closes.Load())
		}
	})
}

func TestAppSessionGeneration_Good(t *testing.T) {
	resources := openAppTestWorkspace(t)
	a := newApp("", 0, 64)
	if result := a.connectWorkspace(resources); !result.OK {
		t.Fatalf("connect workspace: %v", result.Value)
	}
	base := newFakeTextModel(map[string][]string{"alpha": {"answer-a"}, "beta": {"answer-b"}})
	alphaGate := base.block("alpha")
	m, _ := a.Update(loadedMsg{model: base, name: "fixture"})
	a = m.(app)
	sessionA := a.sessions.Active().Record.ID
	a.input.SetValue("alpha")
	m, commandA := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if commandA == nil || a.sessionJobs[sessionA] == nil {
		t.Fatal("session A did not enqueue generation")
	}
	if started := waitFakeStarted(t, base.started); started != "alpha" {
		t.Fatalf("first prompt = %q", started)
	}

	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyCtrlN})
	a = m.(app)
	sessionB := a.sessions.Active().Record.ID
	if sessionB == sessionA {
		t.Fatal("Ctrl+N did not create session B")
	}
	a.input.SetValue("beta")
	m, commandB := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if commandB == nil || a.sessionJobs[sessionB] == nil || a.sessionJobs[sessionA] == nil {
		t.Fatalf("independent jobs = A:%v B:%v", a.sessionJobs[sessionA] != nil, a.sessionJobs[sessionB] != nil)
	}
	assertNoFakeStarted(t, base.started)
	close(alphaGate)
	driveAppGeneration(t, &a, sessionA, commandA)
	if started := waitFakeStarted(t, base.started); started != "beta" {
		t.Fatalf("second prompt = %q", started)
	}
	if !a.sessions.sessions[sessionA].Attention || a.sessions.Active().Record.ID != sessionB {
		t.Fatalf("hidden completion = active %q attention %v", a.sessions.Active().Record.ID, a.sessions.sessions[sessionA].Attention)
	}
	driveAppGeneration(t, &a, sessionB, commandB)

	for sessionID, want := range map[string]string{sessionA: "answer-a", sessionB: "answer-b"} {
		turnResult := resources.Repository.Turns(sessionID)
		if !turnResult.OK {
			t.Fatalf("turns %s: %v", sessionID, turnResult.Value)
		}
		turns := turnResult.Value.([]turnRecord)
		if len(turns) != 2 || turns[0].Role != "user" || turns[1].Role != "assistant" || turns[1].Visible != want {
			t.Fatalf("session %s turns = %#v", sessionID, turns)
		}
		jobs := resources.Repository.Jobs(sessionID).Value.([]generationJobRecord)
		if len(jobs) != 1 || jobs[0].Status != "completed" {
			t.Fatalf("session %s jobs = %#v", sessionID, jobs)
		}
	}
	if base.maxActive.Load() != 1 {
		t.Fatalf("shared lane max concurrency = %d", base.maxActive.Load())
	}
	_ = a.shutdown()
}

func TestAppSessionPersistenceFailure_Bad(t *testing.T) {
	tests := []struct {
		name        string
		failTurn    bool
		failJobCall int
	}{
		{name: "final turn", failTurn: true},
		{name: "final job", failJobCall: 3},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			resources := openAppTestWorkspace(t)
			faults := &failingWorkspaceRepository{workspaceRepository: resources.Repository, failJobCall: test.failJobCall}
			resources.Repository = faults
			a := newApp("", 0, 64)
			if result := a.connectWorkspace(resources); !result.OK {
				t.Fatalf("connect workspace: %v", result.Value)
			}
			base := newFakeTextModel(map[string][]string{"persist": {"answer"}})
			model, _ := a.Update(loadedMsg{model: base, name: "fixture"})
			a = model.(app)
			sessionID := a.sessions.Active().Record.ID
			a.input.SetValue("persist")
			model, command := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
			a = model.(app)
			faults.failTurn = test.failTurn
			driveAppGeneration(t, &a, sessionID, command)

			if status := a.sessions.sessions[sessionID].Record.Status; status != "failed" {
				t.Fatalf("session status = %q, want failed", status)
			}
			if !strings.Contains(a.errText, "injected persistence failure") {
				t.Fatalf("error text = %q", a.errText)
			}
			if events := faults.workspaceRepository.Events(sessionID); !events.OK || len(events.Value.([]eventRecord)) != 1 || events.Value.([]eventRecord)[0].Kind != "generation.failed" {
				t.Fatalf("failure events = %#v (%s)", events.Value, events.Error())
			}
			_ = a.shutdown()
		})
	}
}

func TestAppManagedStreamBatchesPersistence_Good(t *testing.T) {
	resources := openAppTestWorkspace(t)
	tracked := &failingWorkspaceRepository{workspaceRepository: resources.Repository}
	resources.Repository = tracked
	a := newApp("", 0, 256)
	if result := a.connectWorkspace(resources); !result.OK {
		t.Fatalf("connect workspace: %v", result.Value)
	}
	tokens := make([]string, 128)
	for index := range tokens {
		tokens[index] = "x"
	}
	base := newFakeTextModel(map[string][]string{"burst": tokens})
	model, _ := a.Update(loadedMsg{model: base, name: "fixture"})
	a = model.(app)
	sessionID := a.sessions.Active().Record.ID
	a.input.SetValue("burst")
	model, command := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = model.(app)
	driveAppGeneration(t, &a, sessionID, command)

	if tracked.turnCalls > 8 {
		t.Fatalf("128-token stream wrote turns %d times, want bounded batches", tracked.turnCalls)
	}
	turnResult := tracked.workspaceRepository.Turns(sessionID)
	turns := turnResult.Value.([]turnRecord)
	if !turnResult.OK || len(turns) != 2 || len(turns[1].Visible) != 128 {
		t.Fatalf("batched transcript = %#v (%s)", turnResult.Value, turnResult.Error())
	}
	_ = a.shutdown()
}

func TestAppSessionToolLoop_Good(t *testing.T) {
	resources := openAppTestWorkspace(t)
	a := newApp("", 0, 64)
	if result := a.connectWorkspace(resources); !result.OK {
		t.Fatalf("connect workspace: %v", result.Value)
	}
	a.tools.setEnabled(true)
	toolCall := parser.RenderGemmaToolCall("word_count", `{"text":"one two"}`)
	toolResponse := parser.RenderGemmaToolResponse("2 words")
	base := newFakeTextModel(map[string][]string{
		"please count": {toolCall},
		toolResponse:   {"The count is two."},
	})
	m, _ := a.Update(loadedMsg{model: base, name: "tool-fixture"})
	a = m.(app)
	sessionID := a.sessions.Active().Record.ID
	a.input.SetValue("please count")
	m, command := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	driveAppGeneration(t, &a, sessionID, command)

	turnResult := resources.Repository.Turns(sessionID)
	if !turnResult.OK {
		t.Fatalf("load tool turns: %v", turnResult.Value)
	}
	turns := turnResult.Value.([]turnRecord)
	if len(turns) != 4 {
		t.Fatalf("tool loop turns = %#v", turns)
	}
	if turns[0].Role != "user" || turns[1].Role != "assistant" || turns[2].Role != "tool" || turns[3].Role != "assistant" {
		t.Fatalf("tool loop roles = %#v", turns)
	}
	if turns[2].ToolName != "word_count" || turns[3].Visible != "The count is two." {
		t.Fatalf("tool loop result = %#v", turns)
	}
	events := resources.Repository.Events(sessionID).Value.([]eventRecord)
	if len(events) != 1 || events[0].Kind != "tool.call" || events[0].Status != "completed" {
		t.Fatalf("tool events = %#v", events)
	}
	jobs := resources.Repository.Jobs(sessionID).Value.([]generationJobRecord)
	if len(jobs) != 2 || jobs[0].Status != "completed" || jobs[1].Status != "completed" {
		t.Fatalf("tool jobs = %#v", jobs)
	}
	transcript := ansi.Strip(a.renderTranscript())
	for _, want := range []string{"word_count", "2 words", "The count is two."} {
		if !strings.Contains(transcript, want) {
			t.Fatalf("durable tool transcript missing %q:\n%s", want, transcript)
		}
	}
	_ = a.shutdown()
}

func TestAppSharedServiceLane_Good(t *testing.T) {
	resources := openAppTestWorkspace(t)
	a := newApp("", 0, 64)
	if result := a.connectWorkspace(resources); !result.OK {
		t.Fatalf("connect workspace: %v", result.Value)
	}
	base := newFakeTextModel(map[string][]string{
		"chat": {"chat-answer"}, "api": {"api-answer"}, "later": {"later-answer"},
	})
	chatGate := base.block("chat")
	m, _ := a.Update(loadedMsg{model: base, name: "shared"})
	a = m.(app)
	a.svc.custom = "127.0.0.1:0"
	if command := a.svc.start(a.model); command == nil || !a.svc.running {
		t.Fatal("service did not start on the shared lane")
	}
	a.input.SetValue("chat")
	m, chatCommand := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if started := waitFakeStarted(t, base.started); started != "chat" {
		t.Fatalf("first shared request = %q", started)
	}
	apiDone := make(chan string, 1)
	go consumeFakeChat(a.model, "api", apiDone)
	assertNoFakeStarted(t, base.started)
	close(chatGate)
	driveAppGeneration(t, &a, a.sessionID, chatCommand)
	if started := waitFakeStarted(t, base.started); started != "api" {
		t.Fatalf("queued API request = %q", started)
	}
	if answer := waitFakeDone(t, apiDone); answer != "api-answer" {
		t.Fatalf("API answer = %q", answer)
	}
	a.svc.teardown("test stop")
	select {
	case event := <-a.svc.events:
		a.svc.finish(event.err)
	case <-time.After(2 * time.Second):
		t.Fatal("service listener did not stop")
	}
	if base.closes.Load() != 0 {
		t.Fatalf("service stop closed base model %d times", base.closes.Load())
	}
	a.input.SetValue("later")
	m, laterCommand := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if laterCommand == nil {
		t.Fatal("later chat did not start after service stop")
	}
	if started := waitFakeStarted(t, base.started); started != "later" {
		t.Fatalf("later request = %q", started)
	}
	driveAppGeneration(t, &a, a.sessionID, laterCommand)
	if base.maxActive.Load() != 1 {
		t.Fatalf("shared service lane concurrency = %d", base.maxActive.Load())
	}
	_ = a.shutdown()
}

func TestAppModelSwap_Bad(t *testing.T) {
	resources := openAppTestWorkspace(t)
	a := newApp("", 0, 64)
	if result := a.connectWorkspace(resources); !result.OK {
		t.Fatalf("connect workspace: %v", result.Value)
	}
	base := newFakeTextModel(map[string][]string{"busy": {"done"}})
	gate := base.block("busy")
	m, _ := a.Update(loadedMsg{model: base, name: "old"})
	a = m.(app)
	a.input.SetValue("busy")
	m, generationCommand := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	waitFakeStarted(t, base.started)
	loads := 0
	a.modelLoader = func(string, int) tea.Cmd {
		loads++
		return func() tea.Msg { return loadErrMsg{err: errFor("must not load")} }
	}
	a.activePanel = panelModels
	a.picker.SetItems([]list.Item{modelItem{path: "/models/new", name: "new", modelType: "fake"}})
	m, command := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if command != nil || loads != 0 || base.closes.Load() != 0 || !strings.Contains(a.errText, "jobs are active") {
		t.Fatalf("blocked swap = command=%v loads=%d closes=%d err=%q", command != nil, loads, base.closes.Load(), a.errText)
	}
	close(gate)
	driveAppGeneration(t, &a, a.sessionID, generationCommand)
	_ = a.shutdown()
}

func TestAppModelSwap_Good(t *testing.T) {
	oldBase := newFakeTextModel(nil)
	newBase := newFakeTextModel(nil)
	a := newApp("", 0, 64)
	m, _ := a.Update(loadedMsg{model: oldBase, name: "old"})
	a = m.(app)
	a.svc.custom = "127.0.0.1:0"
	if command := a.svc.start(a.model); command == nil {
		t.Fatal("service did not start before swap")
	}
	loads := 0
	a.modelLoader = func(path string, _ int) tea.Cmd {
		loads++
		return func() tea.Msg {
			if strings.Contains(path, "broken") {
				return loadErrMsg{err: errFor("broken checkpoint")}
			}
			return loadedMsg{model: newBase, name: "new"}
		}
	}
	a.activePanel = panelModels
	a.picker.SetItems([]list.Item{modelItem{path: "/models/new", name: "new", modelType: "fake"}})
	m, command := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if command != nil || !a.svc.stopping || a.pendingModel != "/models/new" || oldBase.closes.Load() != 0 {
		t.Fatalf("draining swap = command=%v stopping=%v pending=%q closes=%d", command != nil, a.svc.stopping, a.pendingModel, oldBase.closes.Load())
	}
	var serviceEvent serviceEvent
	select {
	case serviceEvent = <-a.svc.events:
	case <-time.After(2 * time.Second):
		t.Fatal("service did not drain for model swap")
	}
	m, command = a.Update(serviceMsg{ev: serviceEvent})
	a = m.(app)
	if command == nil || oldBase.closes.Load() != 1 || a.model != nil {
		t.Fatalf("post-drain unload = command=%v closes=%d model=%v", command != nil, oldBase.closes.Load(), a.model != nil)
	}
	m, _ = a.Update(command())
	a = m.(app)
	if a.model == nil || a.modelName != "new" || loads != 1 || newBase.closes.Load() != 0 {
		t.Fatalf("new model = name=%q loads=%d closes=%d", a.modelName, loads, newBase.closes.Load())
	}

	a.activePanel = panelModels
	a.picker.SetItems([]list.Item{modelItem{path: "/models/broken", name: "broken", modelType: "fake"}})
	m, command = a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if command == nil || newBase.closes.Load() != 1 {
		t.Fatalf("failed-swap unload = command=%v closes=%d", command != nil, newBase.closes.Load())
	}
	m, _ = a.Update(command())
	a = m.(app)
	if a.model != nil || a.lane != nil || a.loading != "" || !strings.Contains(a.errText, "broken checkpoint") {
		t.Fatalf("failed load state = model=%v lane=%v loading=%q err=%q", a.model != nil, a.lane != nil, a.loading, a.errText)
	}
	_ = a.shutdown()
}

func TestAppQuit_Ugly(t *testing.T) {
	order := make([]string, 0, 4)
	files := testWorkspaceFiles(t)
	opened := openWorkspaceWith(files, workspaceOpeners{
		Repository: func(path string) core.Result {
			result := openDuckRepository(path)
			if !result.OK {
				return result
			}
			return core.Ok(workspaceRepository(&trackingWorkspaceRepository{
				workspaceRepository: result.Value.(workspaceRepository), closeOrder: &order,
			}))
		},
		State: func(paths appPaths) core.Result {
			result := openReactiveState(paths)
			if !result.OK {
				return result
			}
			return core.Ok(reactiveState(&trackingReactiveState{
				reactiveState: result.Value.(reactiveState), closeOrder: &order,
			}))
		},
	})
	if !opened.OK {
		t.Fatalf("open tracked workspace: %v", opened.Value)
	}
	resources := opened.Value.(*workspaceResources)
	a := newApp("", 0, 64)
	a.agent = &orderedAgentProvider{order: &order}
	if result := a.connectWorkspace(resources); !result.OK {
		t.Fatalf("connect workspace: %v", result.Value)
	}
	base := &orderedFakeTextModel{
		fakeTextModel: newFakeTextModel(map[string][]string{"running": {"one"}, "queued": {"two"}}),
		order:         &order,
	}
	base.block("running")
	m, _ := a.Update(loadedMsg{model: base, name: "ordered"})
	a = m.(app)
	a.svc.custom = "127.0.0.1:0"
	if command := a.svc.start(a.model); command == nil {
		t.Fatal("service did not start")
	}
	a.input.SetValue("running")
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	waitFakeStarted(t, base.started)
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyCtrlN})
	a = m.(app)
	a.input.SetValue("queued")
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if a.jobs.ActiveCount() != 2 {
		t.Fatalf("queued job count = %d", a.jobs.ActiveCount())
	}

	type quitResult struct {
		app     app
		command tea.Cmd
	}
	finished := make(chan quitResult, 1)
	go func() {
		model, command := a.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
		finished <- quitResult{app: model.(app), command: command}
	}()
	select {
	case result := <-finished:
		a = result.app
		if result.command == nil {
			t.Fatal("quit returned no Bubble Tea command")
		}
	case <-time.After(2 * time.Second):
		t.Fatal("quit blocked with queued jobs")
	}
	if got := strings.Join(order, ","); got != "agent,model,state,repository" {
		t.Fatalf("shutdown close order = %q", got)
	}
	if base.closes.Load() != 1 || a.svc.running || a.jobs.ActiveCount() != 0 {
		t.Fatalf("shutdown state = closes=%d service=%v jobs=%d", base.closes.Load(), a.svc.running, a.jobs.ActiveCount())
	}
	select {
	case <-a.svc.events:
	case <-time.After(2 * time.Second):
		t.Fatal("service goroutine did not finish after quit")
	}
}

func TestAppQuitPersistsPartialGeneration_Ugly(t *testing.T) {
	files := testWorkspaceFiles(t)
	opened := openWorkspaceWith(files, workspaceOpeners{})
	if !opened.OK {
		t.Fatalf("open workspace: %v", opened.Value)
	}
	a := newApp("", 0, 64)
	if result := a.connectWorkspace(opened.Value.(*workspaceResources)); !result.OK {
		t.Fatalf("connect workspace: %v", result.Value)
	}
	base := newFakeTextModel(map[string][]string{"quit during reply": {"durable partial", "discarded"}})
	base.blockAfterFirst("quit during reply")
	model, _ := a.Update(loadedMsg{model: base, name: "fixture"})
	a = model.(app)
	sessionID := a.sessions.Active().Record.ID
	a.input.SetValue("quit during reply")
	model, _ = a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = model.(app)
	select {
	case <-base.firstYielded:
	case <-time.After(2 * time.Second):
		t.Fatal("model did not yield partial output")
	}

	model, command := a.Update(tea.KeyMsg{Type: tea.KeyCtrlC})
	a = model.(app)
	if command == nil {
		t.Fatal("quit returned no Bubble Tea command")
	}

	reopened := openWorkspaceWith(files, workspaceOpeners{})
	if !reopened.OK {
		t.Fatalf("reopen workspace: %v", reopened.Value)
	}
	defer reopened.Value.(*workspaceResources).Close()
	repository := reopened.Value.(*workspaceResources).Repository
	turnResult := repository.Turns(sessionID)
	turns, ok := turnResult.Value.([]turnRecord)
	if !turnResult.OK || !ok || len(turns) != 2 || turns[1].Visible != "durable partial" {
		t.Fatalf("reopened turns = %#v (%s)", turnResult.Value, turnResult.Error())
	}
	jobResult := repository.Jobs(sessionID)
	jobs, ok := jobResult.Value.([]generationJobRecord)
	if !jobResult.OK || !ok || len(jobs) != 1 || jobs[0].Status != "cancelled" {
		t.Fatalf("reopened jobs = %#v (%s)", jobResult.Value, jobResult.Error())
	}
	sessionResult := repository.Session(sessionID)
	if !sessionResult.OK || sessionResult.Value.(sessionRecord).Status != "cancelled" {
		t.Fatalf("reopened session = %#v (%s)", sessionResult.Value, sessionResult.Error())
	}
	eventResult := repository.Events(sessionID)
	events, ok := eventResult.Value.([]eventRecord)
	if !eventResult.OK || !ok || len(events) != 1 || events[0].Kind != "generation.cancelled" {
		t.Fatalf("reopened events = %#v (%s)", eventResult.Value, eventResult.Error())
	}
}

func TestAppInterruptedSessionIsVisible_Ugly(t *testing.T) {
	files := testWorkspaceFiles(t)
	opened := openWorkspaceWith(files, workspaceOpeners{})
	if !opened.OK {
		t.Fatalf("open workspace: %v", opened.Value)
	}
	resources := opened.Value.(*workspaceResources)
	now := time.Date(2026, time.July, 17, 20, 0, 0, 0, time.UTC)
	session := testSessionRecord("session-interrupted", "Interrupted durable work", now)
	session.Status = "generating"
	if result := resources.Repository.SaveSession(session); !result.OK {
		t.Fatalf("save session: %s", result.Error())
	}
	answer := testTurnRecord("answer-interrupted", session.ID, 1, "assistant", "partial answer survives", now)
	if result := resources.Repository.SaveTurn(answer); !result.OK {
		t.Fatalf("save answer: %s", result.Error())
	}
	job := testJobRecord("job-interrupted", session.ID, "generating", now, unsetRecordTime())
	job.AnswerTurnID = answer.ID
	if result := resources.Repository.SaveJob(job); !result.OK {
		t.Fatalf("save job: %s", result.Error())
	}
	if result := resources.Close(); !result.OK {
		t.Fatalf("close crashed workspace: %s", result.Error())
	}

	reopened := openWorkspaceWith(files, workspaceOpeners{Now: func() time.Time { return now.Add(time.Minute) }})
	if !reopened.OK {
		t.Fatalf("reopen workspace: %v", reopened.Value)
	}
	a := newApp("", 0, 64)
	if result := a.connectWorkspace(reopened.Value.(*workspaceResources)); !result.OK {
		t.Fatalf("connect recovered workspace: %s", result.Error())
	}
	transcript := ansi.Strip(a.renderTranscript())
	if !strings.Contains(transcript, "Generation interrupted") || !strings.Contains(transcript, "partial answer survives") {
		t.Fatalf("recovered transcript:\n%s", transcript)
	}
	if strip := a.sessionStrip(); !strings.Contains(strip, "! Interrupted durable work") {
		t.Fatalf("recovered strip = %q", strip)
	}
	_ = a.shutdown()
}

type orderedAgentProvider struct {
	order *[]string
}

type failingWorkspaceRepository struct {
	workspaceRepository
	failTurn    bool
	failJobCall int
	jobCalls    int
	turnCalls   int
}

func (repository *failingWorkspaceRepository) SaveTurn(record turnRecord) core.Result {
	repository.turnCalls++
	if repository.failTurn {
		return core.Fail(core.E("test.repository.SaveTurn", "injected persistence failure", nil))
	}
	return repository.workspaceRepository.SaveTurn(record)
}

func (repository *failingWorkspaceRepository) SaveJob(record generationJobRecord) core.Result {
	repository.jobCalls++
	if repository.failJobCall > 0 && repository.jobCalls >= repository.failJobCall {
		return core.Fail(core.E("test.repository.SaveJob", "injected persistence failure", nil))
	}
	return repository.workspaceRepository.SaveJob(record)
}

func (*orderedAgentProvider) Capabilities() []agentCapability {
	return agentFeatureCatalog(defaultAgentUnavailableReason)
}

func (*orderedAgentProvider) Snapshot(context.Context) core.Result {
	return core.Ok(agentSnapshot{Work: []agentWorkSnapshot{}, Events: []agentEventSnapshot{}})
}

func (*orderedAgentProvider) Run(context.Context, agentRequest) core.Result {
	return core.Ok(nil)
}

func (provider *orderedAgentProvider) Close() core.Result {
	*provider.order = append(*provider.order, "agent")
	return core.Ok(nil)
}

type orderedFakeTextModel struct {
	*fakeTextModel
	order *[]string
}

func (model *orderedFakeTextModel) Close() core.Result {
	*model.order = append(*model.order, "model")
	return model.fakeTextModel.Close()
}

// TestAppUpdateTransitions drives the pure state machine without a terminal:
// sizing readies the viewport, discovery fills the picker, enter on a picked
// item moves to loading, a loadErr falls back to the picker, and ctrl+t
// toggles the thinking opt-out in chat.
func TestAppUpdateTransitions(t *testing.T) {
	a := newApp("", 0, 512)
	m, _ := a.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	a = m.(app)
	if !a.ready {
		t.Fatal("window size did not ready the app")
	}
	if a.activePanel != panelChat {
		t.Fatalf("activePanel = %d, want Chat on a pickerless start", a.activePanel)
	}
	// a load failure lands back on Models with the error surfaced
	m, _ = a.Update(loadErrMsg{err: errFor("no backends")})
	a = m.(app)
	if a.activePanel != panelModels || a.errText == "" {
		t.Fatalf("loadErr: panel=%d err=%q, want Models + message", a.activePanel, a.errText)
	}
	// tab cycles through every pane and wraps
	for i := 0; i < int(panelCount); i++ {
		m, _ = a.Update(tea.KeyMsg{Type: tea.KeyTab})
		a = m.(app)
	}
	if a.activePanel != panelModels {
		t.Fatalf("panel cycle did not wrap: %d", a.activePanel)
	}
	// ctrl+t flips the thinking override to an explicit state
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyCtrlT})
	a = m.(app)
	if a.cfg.thinking() == nil {
		t.Fatal("ctrl+t left thinking on the model default")
	}
	if v := a.View(); !strings.Contains(v, "thinking") {
		t.Fatalf("status line missing thinking state: %q", v)
	}
	// inspector is a global surface, independent of the active primary panel.
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyCtrlO})
	a = m.(app)
	if !a.inspectorOpen || !strings.Contains(a.View(), "INSPECTOR") {
		t.Fatal("ctrl+o did not open the inspector")
	}
	// service: enter with no model declines with a note, never starts
	a.inspectorOpen = false
	a.activePanel = panelService
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if a.svc.running || a.svc.note == "" {
		t.Fatalf("service start without a model: running=%v note=%q", a.svc.running, a.svc.note)
	}
	// address presets cycle while stopped and render in the tab
	before := a.svc.addrIdx
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyRight})
	a = m.(app)
	if a.svc.addrIdx == before {
		t.Fatal("service right-adjust did not change the address preset")
	}
	if v := a.View(); !strings.Contains(v, a.svc.addr()) {
		t.Fatalf("service tab does not render the listen address %q", a.svc.addr())
	}
}

func TestAppComposerNewline_Good(t *testing.T) {
	a := newApp("", 0, 64)
	a.input.SetValue("first line")

	model, _ := a.Update(tea.KeyMsg{Type: tea.KeyEnter, Alt: true})
	a = model.(app)

	if a.input.Value() != "first line\n" {
		t.Fatalf("Alt+Enter composer value = %q", a.input.Value())
	}
}

func TestAppSessionStrip_Good(t *testing.T) {
	manager := openTestSessionManager(t, sequenceIDs("session-one", "session-two", "session-three"))
	first := manager.Create().Value.(*chatSession)
	second := manager.Create().Value.(*chatSession)
	third := manager.Create().Value.(*chatSession)
	if result := manager.Rename(first.Record.ID, "Renamed durable session"); !result.OK {
		t.Fatalf("rename first: %s", result.Error())
	}
	if result := manager.Rename(second.Record.ID, "Background answer"); !result.OK {
		t.Fatalf("rename second: %s", result.Error())
	}
	if result := manager.Rename(third.Record.ID, "Interrupted work"); !result.OK {
		t.Fatalf("rename third: %s", result.Error())
	}
	if result := manager.Switch(first.Record.ID); !result.OK {
		t.Fatalf("switch first: %s", result.Error())
	}
	second.Record.Status = "generating"
	second.ActiveJobID = "job-two"
	second.Attention = true
	third.Record.Status = "interrupted"
	third.Attention = true
	manager.order = []string{first.Record.ID, second.Record.ID, third.Record.ID}

	a := newApp("", 0, 64)
	a.sessions = manager
	a.sessionID = first.Record.ID
	a.width = 120
	a.recentSessionLimit = 3
	strip := a.sessionStrip()
	for _, want := range []string{"● Renamed durable", "◉ Background answer", "◆ Interrupted work"} {
		if !strings.Contains(strip, want) {
			t.Fatalf("session strip missing %q: %s", want, strip)
		}
	}

	a.recentSessionLimit = 2
	strip = a.sessionStrip()
	if !strings.Contains(strip, "+1") || strings.Contains(strip, "Interrupted work") {
		t.Fatalf("limited session strip = %q", strip)
	}
}

func TestTranscriptPreservesTurnModel_Good(t *testing.T) {
	a := newApp("", 0, 64)
	a.modelName = "current-model"
	a.turns = []turn{{id: "answer-old", role: "assistant", model: "original-model", text: "historical answer"}}
	a.view.Width = 72
	transcript := ansi.Strip(a.renderTranscript())
	if !strings.Contains(transcript, "original-model") || strings.Contains(transcript, "current-model") {
		t.Fatalf("historical model label:\n%s", transcript)
	}
}

// TestAppLiveChatDrive (LTHN_PROBE_MODEL-gated) is the headless end-to-end
// receipt: load a real checkpoint, send a prompt through the real Update loop,
// consume the stream to done, and assert an answer landed with metrics — the
// whole TUI path minus the terminal.
func TestAppLiveChatDrive(t *testing.T) {
	dir := os.Getenv("LTHN_PROBE_MODEL")
	if dir == "" || os.Getenv("MLX_METALLIB_PATH") == "" {
		t.Skip("needs LTHN_PROBE_MODEL + MLX_METALLIB_PATH")
	}
	a := newApp(dir, 0, 128)
	if result := a.connectWorkspace(openAppTestWorkspace(t)); !result.OK {
		t.Fatalf("connect live workspace: %v", result.Value)
	}
	defer func() { _ = a.shutdown() }()
	m, _ := a.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	a = m.(app)

	msg := loadModel(dir, 0)()
	loaded, ok := msg.(loadedMsg)
	if !ok {
		t.Fatalf("loadModel returned %#v", msg)
	}
	m, _ = a.Update(loaded)
	a = m.(app)
	if a.activePanel != panelChat || a.model == nil {
		t.Fatalf("panel = %d model=%v, want chat + loaded", a.activePanel, a.model != nil)
	}
	a.cfg.thinkIdx = 2 // thinking off — deterministic short answers for the drive
	a.input.SetValue("Say hello in exactly two words.")
	m, cmd := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if !a.generating || cmd == nil {
		t.Fatal("enter did not start a generation")
	}
	for i := 0; i < 4096 && cmd != nil; i++ {
		m, cmd = a.Update(cmd())
		a = m.(app)
		if !a.generating {
			break
		}
	}
	if a.generating {
		t.Fatal("generation never completed")
	}
	last := a.turns[len(a.turns)-1]
	if last.role != "assistant" || strings.TrimSpace(last.text) == "" {
		t.Fatalf("assistant turn empty: %+v", last)
	}
	if a.lastTokS <= 0 {
		t.Fatalf("decode tok/s not recorded: %v", a.lastTokS)
	}
	t.Logf("live drive: %q at %.1f tok/s", strings.TrimSpace(last.text), a.lastTokS)
}

// TestAppLiveServiceAPI (LTHN_PROBE_MODEL-gated) is the Service tab's
// end-to-end receipt: load a real checkpoint through the Update loop, start
// the API from the Service tab, drive a real OpenAI chat completion at it
// over HTTP (through the shared serial lane), then stop it cleanly.
func TestAppLiveServiceAPI(t *testing.T) {
	dir := os.Getenv("LTHN_PROBE_MODEL")
	if dir == "" || os.Getenv("MLX_METALLIB_PATH") == "" {
		t.Skip("needs LTHN_PROBE_MODEL + MLX_METALLIB_PATH")
	}
	a := newApp(dir, 0, 128)
	if result := a.connectWorkspace(openAppTestWorkspace(t)); !result.OK {
		t.Fatalf("connect live workspace: %v", result.Value)
	}
	defer func() { _ = a.shutdown() }()
	m, _ := a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a = m.(app)

	msg := loadModel(dir, 0)()
	loaded, ok := msg.(loadedMsg)
	if !ok {
		t.Fatalf("loadModel returned %#v", msg)
	}
	m, _ = a.Update(loaded)
	a = m.(app)
	const addr = "127.0.0.1:36917"
	a.svc.custom = addr
	a.activePanel = panelService
	m, cmd := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if !a.svc.running || cmd == nil {
		t.Fatalf("service did not start: running=%v note=%q", a.svc.running, a.svc.note)
	}
	if v := a.View(); !strings.Contains(v, addr) {
		t.Fatal("service tab does not render the live address")
	}

	// wait for the listener, then a real OpenAI request through the lane
	client := &http.Client{Timeout: 120 * time.Second}
	body := `{"model":"lem","max_tokens":48,"chat_template_kwargs":{"enable_thinking":false},` +
		`"messages":[{"role":"user","content":"Reply with the single word OK."}]}`
	var resp *http.Response
	var err error
	for i := 0; i < 100; i++ {
		resp, err = client.Post("http://"+addr+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err == nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("API never answered: %v", err)
	}
	defer resp.Body.Close()
	payload, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat completion status %d: %s", resp.StatusCode, payload)
	}
	if !strings.Contains(string(payload), `"content"`) {
		t.Fatalf("no content in completion: %s", payload)
	}
	if a.svc.requests.Load() == 0 {
		t.Fatal("request counter did not move")
	}
	t.Logf("live API: %d req · %s", a.svc.requests.Load(), payload)

	// stop from the tab and drain Serve's return through the update loop
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if !a.svc.stopping {
		t.Fatal("enter while running did not begin the stop")
	}
	m, _ = a.Update(waitService(a.svc.events)())
	a = m.(app)
	if a.svc.running {
		t.Fatal("service did not finish cleanly")
	}
}

func TestAppServiceUsesLoadedLaneWithoutOwningIt(t *testing.T) {
	base := newFakeTextModel(map[string][]string{"hello": {"world"}})
	a := newApp("", 0, 32)
	m, _ := a.Update(loadedMsg{model: base, name: "fake"})
	a = m.(app)
	if a.lane == nil || a.model != a.lane.Model() {
		t.Fatal("loaded model was not wrapped in the application lane")
	}

	a.svc.custom = "127.0.0.1:0"
	cmd := a.svc.start(a.model)
	if cmd == nil || !a.svc.running {
		t.Fatalf("service did not start: running=%v note=%q", a.svc.running, a.svc.note)
	}
	a.svc.teardown("test stop")
	select {
	case event := <-a.svc.events:
		a.svc.finish(event.err)
	case <-time.After(2 * time.Second):
		t.Fatal("service listener did not stop")
	}
	if got := base.closes.Load(); got != 0 {
		t.Fatalf("service stop closed the loaded model %d times", got)
	}
	if r := a.lane.Close(); !r.OK {
		t.Fatalf("lane Close error = %s", r.Error())
	}
	if got := base.closes.Load(); got != 1 {
		t.Fatalf("application lane closed base %d times, want 1", got)
	}
}

func TestTranscriptFollow_Good(t *testing.T) {
	a := transcriptTestApp(t)
	a.follow = true
	a.refreshTranscriptOutput()
	if !a.view.AtBottom() || !a.follow || a.newOutput {
		t.Fatalf("follow refresh: bottom=%v follow=%v marker=%v", a.view.AtBottom(), a.follow, a.newOutput)
	}
	a.turns[len(a.turns)-1].text += "\nnew streamed line"
	a.refreshTranscriptOutput()
	if !a.view.AtBottom() || a.newOutput {
		t.Fatalf("streamed follow: bottom=%v marker=%v", a.view.AtBottom(), a.newOutput)
	}
}

func TestTranscriptFollow_Bad(t *testing.T) {
	a := transcriptTestApp(t)
	a.generating = true // keep the last assistant in streaming/plain render mode
	a.follow = true
	a.refreshTranscriptOutput()
	if !a.view.AtBottom() || a.view.YOffset == 0 {
		t.Fatalf("fixture did not overflow viewport: bottom=%v offset=%d", a.view.AtBottom(), a.view.YOffset)
	}

	m, _ := a.Update(tea.KeyMsg{Type: tea.KeyUp})
	a = m.(app)
	if a.follow {
		t.Fatal("scrolling upward left transcript follow enabled")
	}
	offset := a.view.YOffset
	a.turns[len(a.turns)-1].text += "\noutput while reading older text"
	a.refreshTranscriptOutput()
	if a.view.YOffset != offset || !a.newOutput {
		t.Fatalf("scrolled refresh: offset=%d want=%d marker=%v", a.view.YOffset, offset, a.newOutput)
	}
	if !strings.Contains(a.View(), "new output") {
		t.Fatal("scrolled transcript did not render its new-output marker")
	}

	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEnd})
	a = m.(app)
	if !a.follow || !a.view.AtBottom() || a.newOutput {
		t.Fatalf("End: follow=%v bottom=%v marker=%v", a.follow, a.view.AtBottom(), a.newOutput)
	}
}

func TestTranscriptFollow_Ugly(t *testing.T) {
	manager := openTestSessionManager(t, sequenceIDs("session-active", "session-hidden"))
	if result := manager.Create(); !result.OK {
		t.Fatalf("create active session: %v", result.Value)
	}
	if result := manager.Create(); !result.OK {
		t.Fatalf("create hidden session: %v", result.Value)
	}
	if result := manager.Switch("session-active"); !result.OK {
		t.Fatalf("switch active session: %v", result.Value)
	}
	if result := manager.SetViewport("session-active", 13, false); !result.OK {
		t.Fatalf("set active viewport: %v", result.Value)
	}
	if result := manager.Complete("session-hidden"); !result.OK {
		t.Fatalf("complete hidden session: %v", result.Value)
	}
	active := manager.Active()
	if active.Record.ID != "session-active" || active.ViewportOffset != 13 || active.Follow {
		t.Fatalf("hidden completion moved active viewport: %#v", active)
	}
	if !manager.sessions["session-hidden"].Attention {
		t.Fatal("hidden completion did not mark the session for attention")
	}
}

func transcriptTestApp(t *testing.T) app {
	t.Helper()
	a := newApp("", 0, 64)
	m, _ := a.Update(tea.WindowSizeMsg{Width: 72, Height: 18})
	a = m.(app)
	a.activePanel = panelChat
	a.turns = make([]turn, 0, 24)
	for i := 0; i < 12; i++ {
		a.turns = append(a.turns,
			turn{id: benchmarkTurnID(i * 2), role: "user", text: "Question with enough words to occupy a transcript line"},
			turn{id: benchmarkTurnID(i*2 + 1), role: "assistant", text: "Answer with enough words to occupy another transcript line"},
		)
	}
	return a
}

// errFor builds a plain error for transition tests.
func errFor(text string) error { return &driveErr{text} }

type driveErr struct{ s string }

func (e *driveErr) Error() string { return e.s }

func openAppTestWorkspace(t *testing.T) *workspaceResources {
	t.Helper()
	result := openWorkspaceWith(testWorkspaceFiles(t), workspaceOpeners{})
	if !result.OK {
		t.Fatalf("open app test workspace: %v", result.Value)
	}
	return result.Value.(*workspaceResources)
}

func driveAppGeneration(t *testing.T, target *app, sessionID string, command tea.Cmd) {
	t.Helper()
	for step := 0; step < 1024 && command != nil; step++ {
		model, next := target.Update(command())
		*target = model.(app)
		command = next
		if target.sessionJobs[sessionID] == nil {
			return
		}
	}
	t.Fatalf("session %s generation did not complete", sessionID)
}
