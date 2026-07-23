// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"context"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/orchestrator"
	"dappco.re/go/inference/agent/provider"
	"dappco.re/go/inference/agent/work"
	"dappco.re/go/inference/agent/workspace"
	"dappco.re/go/inference/dataset"
	"dappco.re/go/inference/decode/parser"
	tea "dappco.re/go/render/display/tui"
	"dappco.re/go/render/display/tui/list"
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
	if a.boot.phase != bootFailed || !strings.Contains(a.View().Content, "/tmp/lem.duckdb") || !strings.Contains(a.View().Content, "Retry") {
		t.Fatalf("blocking storage view:\n%s", a.View().Content)
	}
	m, command := a.Update(testTextPress('r'))
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
	m, command := a.Update(testTextPress('q'))
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
	m, commandA := a.Update(testKeyPress(tea.KeyEnter))
	a = m.(app)
	if commandA == nil || a.sessionJobs[sessionA] == nil {
		t.Fatal("session A did not enqueue generation")
	}
	if started := waitFakeStarted(t, base.started); started != "alpha" {
		t.Fatalf("first prompt = %q", started)
	}

	m, _ = a.Update(testModifiedKeyPress('n', tea.ModCtrl))
	a = m.(app)
	sessionB := a.sessions.Active().Record.ID
	if sessionB == sessionA {
		t.Fatal("Ctrl+N did not create session B")
	}
	a.input.SetValue("beta")
	m, commandB := a.Update(testKeyPress(tea.KeyEnter))
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
			model, command := a.Update(testKeyPress(tea.KeyEnter))
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
	model, command := a.Update(testKeyPress(tea.KeyEnter))
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
	m, command := a.Update(testKeyPress(tea.KeyEnter))
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
	m, chatCommand := a.Update(testKeyPress(tea.KeyEnter))
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
	m, laterCommand := a.Update(testKeyPress(tea.KeyEnter))
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
	m, generationCommand := a.Update(testKeyPress(tea.KeyEnter))
	a = m.(app)
	waitFakeStarted(t, base.started)
	loads := 0
	a.modelLoader = func(string, int) tea.Cmd {
		loads++
		return func() tea.Msg { return loadErrMsg{err: errFor("must not load")} }
	}
	a.activePanel = panelModels
	a.picker.SetItems([]list.Item{modelItem{path: "/models/new", name: "new", modelType: "fake"}})
	m, command := a.Update(testKeyPress(tea.KeyEnter))
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
	m, command := a.Update(testKeyPress(tea.KeyEnter))
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
	m, command = a.Update(testKeyPress(tea.KeyEnter))
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
	resources.Agent = &orderedAgentProvider{order: &order}
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
	m, _ = a.Update(testKeyPress(tea.KeyEnter))
	a = m.(app)
	waitFakeStarted(t, base.started)
	m, _ = a.Update(testModifiedKeyPress('n', tea.ModCtrl))
	a = m.(app)
	a.input.SetValue("queued")
	m, _ = a.Update(testKeyPress(tea.KeyEnter))
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
		model, command := a.Update(testModifiedKeyPress('c', tea.ModCtrl))
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

func TestAppShutdownPreservesAllCloseErrors_Ugly(t *testing.T) {
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
				closeFailure: "repository close failed",
			}))
		},
		State: func(paths appPaths) core.Result {
			result := openReactiveState(paths)
			if !result.OK {
				return result
			}
			return core.Ok(reactiveState(&trackingReactiveState{
				reactiveState: result.Value.(reactiveState), closeOrder: &order,
				closeFailure: "state close failed",
			}))
		},
	})
	if !opened.OK {
		t.Fatalf("open tracked workspace: %s", opened.Error())
	}
	resources := opened.Value.(*workspaceResources)
	resources.Agent = &orderedAgentProvider{order: &order, closeFailure: "agent close failed"}
	a := newApp("", 0, 64)
	if result := a.connectWorkspace(resources); !result.OK {
		t.Fatalf("connect workspace: %s", result.Error())
	}
	model := &orderedFakeTextModel{
		fakeTextModel: newFakeTextModel(map[string][]string{}), order: &order,
		closeFailure: "model close failed",
	}
	updated, _ := a.Update(loadedMsg{model: model, name: "ordered-failure"})
	a = updated.(app)

	result := a.shutdown()
	if result.OK {
		t.Fatal("shutdown with retained agent ownership succeeded")
	}
	want := core.E(
		"tui.app.shutdown",
		"test.agent.Close: agent close failed; test.model.Close: model close failed",
		nil,
	).Error()
	if result.Error() != want {
		t.Fatalf("shutdown error = %q, want %q", result.Error(), want)
	}
	if got := strings.Join(order, ","); got != "agent,model" {
		t.Fatalf("shutdown close order = %q", got)
	}
	if resources.State == nil || resources.Repository == nil {
		t.Fatal("shutdown destroyed retained-agent retry dependencies")
	}
	if model.closes.Load() != 1 {
		t.Fatalf("underlying model Close calls = %d, want 1", model.closes.Load())
	}
}

func TestAppShutdownRetriesAgentOwnershipBeforeClosingDependencies(t *testing.T) {
	repository := openTestDuckRepository(t)
	state := &retryCloseReactiveState{}
	agent := &retryCloseAgent{repository: repository}
	resources := &workspaceResources{Agent: agent, State: state, Repository: repository}
	a := newApp("", 0, 64)
	a.resources = resources
	a.agent = agent

	first := a.shutdown()
	core.AssertFalse(t, first.OK)
	core.AssertEqual(t, 1, agent.calls)
	core.AssertFalse(t, state.closed)
	core.AssertTrue(t, resources.State != nil)
	core.AssertTrue(t, resources.Repository != nil)
	core.AssertTrue(t, repository.ListSessions(false).OK)

	second := a.shutdown()
	core.AssertTrue(t, second.OK, second.Error())
	core.AssertEqual(t, 2, agent.calls)
	core.AssertTrue(t, state.closed)
	core.AssertTrue(t, resources.State == nil)
	core.AssertTrue(t, resources.Repository == nil)
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
	model, _ = a.Update(testKeyPress(tea.KeyEnter))
	a = model.(app)
	select {
	case <-base.firstYielded:
	case <-time.After(2 * time.Second):
		t.Fatal("model did not yield partial output")
	}

	model, command := a.Update(testModifiedKeyPress('c', tea.ModCtrl))
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
	order        *[]string
	closeFailure string
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

func (*orderedAgentProvider) Review(context.Context, agentReviewRequest) core.Result {
	return core.Ok(agentReview{})
}

func (*orderedAgentProvider) Run(context.Context, agentRequest) core.Result {
	return core.Ok(nil)
}

func (provider *orderedAgentProvider) Close() core.Result {
	*provider.order = append(*provider.order, "agent")
	if provider.closeFailure != "" {
		return core.Fail(core.E("test.agent.Close", provider.closeFailure, nil))
	}
	return core.Ok(nil)
}

type orderedFakeTextModel struct {
	*fakeTextModel
	order        *[]string
	closeFailure string
}

func (model *orderedFakeTextModel) Close() core.Result {
	*model.order = append(*model.order, "model")
	result := model.fakeTextModel.Close()
	if !result.OK {
		return result
	}
	if model.closeFailure != "" {
		return core.Fail(core.E("test.model.Close", model.closeFailure, nil))
	}
	return core.Ok(nil)
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
		m, _ = a.Update(testKeyPress(tea.KeyTab))
		a = m.(app)
	}
	if a.activePanel != panelModels {
		t.Fatalf("panel cycle did not wrap: %d", a.activePanel)
	}
	// ctrl+t flips the thinking override to an explicit state
	m, _ = a.Update(testModifiedKeyPress('t', tea.ModCtrl))
	a = m.(app)
	if a.cfg.thinking() == nil {
		t.Fatal("ctrl+t left thinking on the model default")
	}
	if v := a.View().Content; !strings.Contains(v, "thinking") {
		t.Fatalf("status line missing thinking state: %q", v)
	}
	// inspector is a global surface, independent of the active primary panel.
	m, _ = a.Update(testModifiedKeyPress('o', tea.ModCtrl))
	a = m.(app)
	if !a.inspectorOpen || !strings.Contains(a.View().Content, "INSPECTOR") {
		t.Fatal("ctrl+o did not open the inspector")
	}
	// service: enter with no model declines with a note, never starts
	a.inspectorOpen = false
	a.activePanel = panelService
	m, _ = a.Update(testKeyPress(tea.KeyEnter))
	a = m.(app)
	if a.svc.running || a.svc.note == "" {
		t.Fatalf("service start without a model: running=%v note=%q", a.svc.running, a.svc.note)
	}
	// address presets cycle while stopped and render in the tab
	before := a.svc.addrIdx
	m, _ = a.Update(testKeyPress(tea.KeyRight))
	a = m.(app)
	if a.svc.addrIdx == before {
		t.Fatal("service right-adjust did not change the address preset")
	}
	if v := a.View().Content; !strings.Contains(v, a.svc.addr()) {
		t.Fatalf("service tab does not render the listen address %q", a.svc.addr())
	}
}

func TestAppComposerNewline_Good(t *testing.T) {
	a := newApp("", 0, 64)
	a.input.SetValue("first line")

	model, _ := a.Update(testModifiedKeyPress(tea.KeyEnter, tea.ModAlt))
	a = model.(app)

	if a.input.Value() != "first line\n" {
		t.Fatalf("Alt+Enter composer value = %q", a.input.Value())
	}
}

func TestAppPanelView_ChatEmptyStateMatchesCanonicalCopy(t *testing.T) {
	a := newApp("", 0, 64)
	model, _ := a.Update(tea.WindowSizeMsg{Width: 120, Height: 30})
	a = model.(app)

	view := ansi.Strip(a.panelView())
	compact := strings.Join(strings.Fields(view), " ")
	for _, want := range []string{
		"○ no model loaded — open Models and choose one. The prompt below stays put either way.",
		"○ no model — pick one in Models to send. You can still type.",
		"›",
		"select a model to start chatting…",
	} {
		if !strings.Contains(compact, want) {
			t.Fatalf("empty Chat panel missing %q:\n%s", want, view)
		}
	}

	model, _ = a.Update(testTextPress('x'))
	a = model.(app)
	if got := a.input.Value(); got != "x" {
		t.Fatalf("no-model composer value = %q, want typing to remain available", got)
	}
}

func TestAppPanelView_WorkAndDataEmptyStatesMatchCanonicalCopy(t *testing.T) {
	a := newApp("", 0, 64)

	a.activePanel = panelWork
	workView := ansi.Strip(a.panelView())
	for _, want := range []string{
		"WORK  ACTIVE 0 · WAITING 0 · DONE 0",
		"○ No work yet",
		"A connected provider or workspace action will create work here. Open the inspector and dispatch one.",
	} {
		if !strings.Contains(workView, want) {
			t.Fatalf("empty Work panel missing %q:\n%s", want, workView)
		}
	}

	a.activePanel = panelData
	dataView := ansi.Strip(a.panelView())
	for _, want := range []string{
		"DATA  0 items · sort date",
		"○ No items match this filter",
		"Import or capture data with lem data import or lem serve --capture.",
	} {
		if !strings.Contains(dataView, want) {
			t.Fatalf("empty Data panel missing %q:\n%s", want, dataView)
		}
	}
}

func TestAppFooterLine_Good(t *testing.T) {
	a := newApp("", 0, 64)
	footer := ansi.Strip(a.footerLine())
	want := "tab panels · ctrl+k commands · ctrl+o inspector · f1 help"
	if !strings.Contains(footer, want) || strings.Contains(footer, "settings") {
		t.Fatalf("footer = %q, want canonical key strip %q", footer, want)
	}

	a.width, a.height = 140, 30
	footer = ansi.Strip(a.footerLine())
	if !strings.HasSuffix(footer, want) {
		t.Fatalf("sized footer must preserve its right-hand key strip: %q", footer)
	}
	if got, limit := lipgloss.Width(footer), measureFrame(a.width, a.height, a.inspectorOpen).innerWidth; got > limit {
		t.Fatalf("footer width = %d, exceeds frame inner width %d: %q", got, limit, footer)
	}
	if frame := ansi.Strip(a.View().Content); !strings.Contains(frame, want) {
		t.Fatalf("rendered wide frame clipped the canonical key strip:\n%s", frame)
	}
}

func TestAppSessionStrip_Good(t *testing.T) {
	unmanaged := newApp("", 0, 64)
	if strip := unmanaged.sessionStrip(); strip != "session 1" {
		t.Fatalf("initial session chip = %q, want %q", strip, "session 1")
	}

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
	a.view.SetWidth(72)
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
	m, cmd := a.Update(testKeyPress(tea.KeyEnter))
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
	m, cmd := a.Update(testKeyPress(tea.KeyEnter))
	a = m.(app)
	if !a.svc.running || cmd == nil {
		t.Fatalf("service did not start: running=%v note=%q", a.svc.running, a.svc.note)
	}
	if v := a.View().Content; !strings.Contains(v, addr) {
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
	m, _ = a.Update(testKeyPress(tea.KeyEnter))
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
	if !a.view.AtBottom() || a.view.YOffset() == 0 {
		t.Fatalf("fixture did not overflow viewport: bottom=%v offset=%d", a.view.AtBottom(), a.view.YOffset())
	}

	m, _ := a.Update(testKeyPress(tea.KeyUp))
	a = m.(app)
	if a.follow {
		t.Fatal("scrolling upward left transcript follow enabled")
	}
	offset := a.view.YOffset()
	a.turns[len(a.turns)-1].text += "\noutput while reading older text"
	a.refreshTranscriptOutput()
	if a.view.YOffset() != offset || !a.newOutput {
		t.Fatalf("scrolled refresh: offset=%d want=%d marker=%v", a.view.YOffset(), offset, a.newOutput)
	}
	if !strings.Contains(a.View().Content, "new output") {
		t.Fatal("scrolled transcript did not render its new-output marker")
	}

	m, _ = a.Update(testKeyPress(tea.KeyEnd))
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

func TestApp_AgentReviewLaunchesOnlyAfterBothConfirmations(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	provider := &launchReviewProvider{
		caps:    []agentCapability{{Feature: agentFeatureDispatch, Available: true}},
		reviews: []agentReview{{Feature: agentFeatureDispatch, Title: "Review project registration", Body: "Source: /tmp/repo", ConfirmRequired: true}},
	}
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	if result := a.work.CreateWork("Launch", "Run the approved task", "/tmp/repo"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	if result := a.palette.Invoke(agentCommandID(agentFeatureDispatch), &a); !result.OK {
		t.Fatalf("dispatch palette invoke: %v", result.Value)
	}
	startAgentProjectReview(t, &a, "codex", "gpt-5")
	if a.activeOverlay != overlayProjectReview || len(provider.runs) != 0 {
		t.Fatalf("project review state = overlay %d runs %d", a.activeOverlay, len(provider.runs))
	}
	model, command := a.Update(testKeyPress(tea.KeyEnter))
	a = model.(app)
	if len(provider.runs) != 0 || command == nil {
		t.Fatalf("registration confirmation = runs=%d command=%v", len(provider.runs), command != nil)
	}
	model, _ = a.Update(command())
	a = model.(app)
	if a.activeOverlay != overlayLaunchReview || len(provider.runs) != 1 {
		t.Fatalf("launch review state = overlay %d runs %d", a.activeOverlay, len(provider.runs))
	}
	model, command = a.Update(testKeyPress(tea.KeyEnter))
	a = model.(app)
	if command == nil || len(provider.runs) != 1 {
		t.Fatalf("launch confirmation = runs=%d command=%v", len(provider.runs), command != nil)
	}
	model, _ = a.Update(command())
	a = model.(app)
	selected, ok := a.work.Selected()
	if len(provider.runs) != 2 || a.activeOverlay != overlayNone || !ok || selected.Status != "queued" || provider.runs[0].Provider != "codex" || provider.runs[0].Model != "gpt-5" || provider.runs[1].Provider != "codex" || provider.runs[1].Model != "gpt-5" {
		t.Fatalf("final dispatch = runs=%#v overlay=%d", provider.runs, a.activeOverlay)
	}
	a.refreshAgentPalette()
	if command := a.palette.byID[agentCommandID(agentFeatureDispatch)]; command.Available {
		t.Fatalf("dispatch remained available after queued receipt: %#v", command)
	}
}

func TestApp_AgentReviewRejectsOverlappingAndStaleOperations(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	provider := &launchReviewProvider{caps: []agentCapability{{Feature: agentFeatureDispatch, Available: true}}, reviews: []agentReview{{Feature: agentFeatureDispatch, Title: "Project", ConfirmRequired: true}}}
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	if result := a.work.CreateWork("Launch", "Run the task", "/tmp/repo"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	if result := a.queueAgentAction(agentFeatureDispatch); !result.OK {
		t.Fatalf("first action: %v", result.Value)
	}
	a.launchReview.providerInput.SetValue("codex")
	a.launchReview.modelInput.SetValue("gpt-5")
	if result := a.confirmAgentSelection(); !result.OK {
		t.Fatalf("confirm selection: %v", result.Value)
	}
	operationID := a.agentOperationID
	if result := a.queueAgentAction(agentFeatureDispatch); result.OK {
		t.Fatal("overlapping agent action was accepted")
	}
	beforeRequest := a.agentRequest
	beforeOverlay := a.activeOverlay
	model, _ := a.Update(agentActionMsg{operationID: operationID + 1, feature: agentFeatureDispatch, stage: agentReviewLaunch, request: agentRequest{Provider: "stale", Model: "stale"}, result: core.Ok(agentReview{Title: "stale"})})
	a = model.(app)
	if a.agentRequest != beforeRequest || a.agentOperationID != operationID || a.activeOverlay != beforeOverlay || len(provider.runs) != 0 {
		t.Fatalf("stale result mutated app: request=%#v operation=%d overlay=%d runs=%d", a.agentRequest, a.agentOperationID, a.activeOverlay, len(provider.runs))
	}
}

func TestApp_AgentReviewReceiptUsesRequestedLocalWorkIDAndResets(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	provider := &launchReviewProvider{caps: []agentCapability{{Feature: agentFeatureDispatch, Available: true}, {Feature: agentFeatureQueueStart, Available: true}}}
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	a.work.ids = sequenceIDs("requested", "other")
	first := a.work.CreateWork("Requested", "Task one", "/tmp/one")
	second := a.work.CreateWork("Other", "Task two", "/tmp/two")
	if !first.OK || !second.OK {
		t.Fatalf("CreateWork = %#v / %#v", first, second)
	}
	requested := first.Value.(workItemRecord)
	other := second.Value.(workItemRecord)
	a.agentOperationID, a.agentOperationNext, a.agentInFlight = 4, 5, true
	a.agentRequest = agentRequest{Feature: agentFeatureDispatch, WorkID: requested.ID, Review: agentReview{Payload: agentProjectRegistration{Provider: "codex"}}}
	a.agentReview = a.agentRequest.Review
	a.agentStage = agentReviewLaunch
	model, _ := a.Update(agentActionMsg{operationID: 4, feature: agentFeatureDispatch, stage: agentReviewLaunch, request: a.agentRequest, result: core.Ok(agentActionReceipt{Feature: agentFeatureDispatch, WorkID: other.ID, Status: "queued"})})
	a = model.(app)
	items := a.work.Items()
	statuses := map[string]string{}
	for _, item := range items {
		statuses[item.ID] = item.Status
	}
	if statuses[requested.ID] != "queued" || statuses[other.ID] != workStatusActive || !strings.Contains(a.errText, "receipt WorkID") {
		t.Fatalf("receipt reconciliation = statuses=%#v error=%q", statuses, a.errText)
	}
	if a.agentOperationID != 0 || a.agentRequest.Feature != "" || a.agentReview.Payload != nil || a.agentStage != agentReviewNone || a.agentCommand != nil || a.agentInFlight {
		t.Fatalf("terminal success retained transaction: id=%d request=%#v review=%#v stage=%d command=%v inFlight=%v", a.agentOperationID, a.agentRequest, a.agentReview, a.agentStage, a.agentCommand != nil, a.agentInFlight)
	}
	a.work.selectWork(requested.ID)
	a.refreshAgentPalette()
	if command := a.palette.byID[agentCommandID(agentFeatureDispatch)]; command.Available {
		t.Fatalf("dispatch remained available for queued requested Work: %#v", command)
	}
	if result := a.queueAgentAction(agentFeatureQueueStart); !result.OK || a.agentOperationID != 6 {
		t.Fatalf("fresh operation = result=%#v id=%d", result, a.agentOperationID)
	}

	a.agentInFlight = true
	a.agentReview = agentReview{Payload: agentProjectRegistration{Provider: "stale"}}
	model, _ = a.Update(agentActionMsg{operationID: a.agentOperationID, feature: agentFeatureQueueStart, request: a.agentRequest, result: core.Fail(core.E("test.agent", "provider failed", nil))})
	a = model.(app)
	if !strings.Contains(a.errText, "provider failed") || a.agentOperationID != 0 || a.agentRequest.Feature != "" || a.agentReview.Payload != nil || a.agentStage != agentReviewNone || a.agentCommand != nil || a.agentInFlight {
		t.Fatalf("terminal error retained transaction: err=%q id=%d request=%#v review=%#v stage=%d command=%v inFlight=%v", a.errText, a.agentOperationID, a.agentRequest, a.agentReview, a.agentStage, a.agentCommand != nil, a.agentInFlight)
	}
}

func TestApp_AgentActionTargetsNativeRunAndRequiresReviewReceipt(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	provider := &launchReviewProvider{caps: []agentCapability{{Feature: agentFeatureCancel, Available: true}, {Feature: agentFeatureAnswer, Available: true}, {Feature: agentFeatureChangesReview, Available: true}, {Feature: agentFeatureAccept, Available: true}, {Feature: agentFeatureReject, Available: true}}, reviews: []agentReview{{Feature: agentFeatureChangesReview, Title: "Review", ConfirmRequired: true, Payload: "opaque-review"}}}
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	a.work.ids = sequenceIDs("local")
	if result := a.work.CreateWork("Local", "task", "/tmp/repo"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	if result := a.work.updateWork("snapshot", "local", func(record *workItemRecord) { record.Status = "running" }); !result.OK {
		t.Fatalf("update: %v", result.Value)
	}
	a.work.agentWork["local"] = agentWorkSnapshot{NativeRunID: "run-immutable"}
	if result := a.queueAgentAction(agentFeatureCancel); !result.OK {
		t.Fatalf("cancel: %v", result.Value)
	}
	if a.agentRequest.WorkID != "local" || a.agentRequest.RunID != "run-immutable" {
		t.Fatalf("cancel request = %#v", a.agentRequest)
	}
	if result := a.queueAgentAction(agentFeatureAccept); result.OK {
		t.Fatal("accepted without exact change review receipt")
	}
}

func TestApp_AgentRefreshRunsWhileAnotherPanelIsActive(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, newUnavailableAgentProvider(defaultAgentUnavailableReason)); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	a.work.ids = sequenceIDs("local")
	if result := a.work.CreateWork("Local", "task", "/tmp/repo"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	if result := a.work.updateWork("running", "local", func(record *workItemRecord) { record.Status = "running" }); !result.OK {
		t.Fatalf("update: %v", result.Value)
	}
	a.work.agentWork["local"] = agentWorkSnapshot{NativeRunID: "run-1", Status: "running"}
	a.activePanel = panelChat
	if command := a.armAgentRefresh(); command == nil || !a.agentRefreshArmed {
		t.Fatalf("refresh = %v armed=%v", command != nil, a.agentRefreshArmed)
	}
}

func TestApp_AgentAnswerStoresExactResumeIdentity(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	provider := &launchReviewProvider{caps: []agentCapability{{Feature: agentFeatureAnswer, Available: true}, {Feature: agentFeatureResume, Available: true}}}
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	a.work.ids = sequenceIDs("local")
	if result := a.work.CreateWork("Local", "task", "/tmp/repo"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	if result := a.work.updateWork("waiting", "local", func(record *workItemRecord) { record.Status = "waiting"; record.Question = "Which target?" }); !result.OK {
		t.Fatalf("update: %v", result.Value)
	}
	a.work.agentWork["local"] = agentWorkSnapshot{NativeRunID: "run-parent", QuestionID: "question-1"}
	if result := a.queueAgentAction(agentFeatureAnswer); !result.OK {
		t.Fatalf("answer: %v", result.Value)
	}
	a.agentInFlight = true
	model, _ := a.Update(agentActionMsg{operationID: a.agentOperationID, feature: agentFeatureAnswer, request: a.agentRequest, result: core.Ok(agentActionReceipt{Feature: agentFeatureAnswer, WorkID: "local", RunID: "run-reserved", Detail: "answer-1", Status: "answered"})})
	a = model.(app)
	if result := a.work.ApplyAgentSnapshot(agentSnapshot{Work: []agentWorkSnapshot{{ExternalID: "local", NativeRunID: "run-parent", Status: "waiting"}}}); !result.OK {
		t.Fatalf("snapshot: %v", result.Value)
	}
	state := a.work.AgentState(a.work.Items()[0])
	if state.NativeRunID != "run-parent" || state.AnswerID != "answer-1" || state.ResumeRunID != "run-reserved" {
		t.Fatalf("answer state = %#v", state)
	}
	if result := a.queueAgentAction(agentFeatureResume); !result.OK || a.agentRequest.RunID != "run-parent" || a.agentRequest.Input != "answer-1" {
		t.Fatalf("resume request = %#v / %#v", result, a.agentRequest)
	}
}

func TestApp_WorkspaceReadyCorrelatesSnapshotAndArmsOneRefresh(t *testing.T) {
	resources := openAppTestWorkspace(t)
	at := time.Date(2026, time.July, 18, 13, 0, 0, 0, time.UTC)
	if result := resources.Repository.SaveWorkItem(testWorkRecord("work-1", "Existing work", workStatusActive, at)); !result.OK {
		t.Fatalf("SaveWorkItem: %v", result.Value)
	}
	provider := &correctiveAgentProvider{
		snapshot: func(context.Context, int) core.Result {
			return core.Ok(agentSnapshot{Work: []agentWorkSnapshot{{
				ExternalID: "work-1", NativeRunID: "run-existing", Status: "running",
			}}})
		},
	}
	resources.Agent = provider
	a := newWorkspaceApp("", 0, 64, nil)
	a.runtimeDetector, a.knowledgeScan = nil, nil
	a.activePanel = panelChat

	model, command := a.Update(workspaceReadyMsg{resources: resources})
	a = model.(app)
	if !a.agentSnapshotInFlight || a.agentSnapshotCurrent != 1 || provider.SnapshotCalls() != 0 {
		t.Fatalf("workspace-ready snapshot correlation = inFlight %v current %d calls %d", a.agentSnapshotInFlight, a.agentSnapshotCurrent, provider.SnapshotCalls())
	}
	message := command()
	batch, ok := message.(tea.BatchMsg)
	if !ok {
		t.Fatalf("workspace-ready command = %T, want tea.BatchMsg", message)
	}
	var refresh tea.Cmd
	for _, child := range batch {
		if child == nil {
			continue
		}
		model, next := a.Update(child())
		a = model.(app)
		if next != nil {
			refresh = next
		}
	}
	items := a.work.Items()
	if provider.SnapshotCalls() != 1 || len(items) != 1 || items[0].Status != "running" || a.work.AgentState(items[0]).NativeRunID != "run-existing" {
		t.Fatalf("startup snapshot = calls %d items %#v state %#v", provider.SnapshotCalls(), items, a.work.AgentState(items[0]))
	}
	if a.activePanel != panelChat || refresh == nil || !a.agentRefreshArmed || a.armAgentRefresh() != nil {
		t.Fatalf("startup refresh = panel %d command %v armed %v", a.activePanel, refresh != nil, a.agentRefreshArmed)
	}
	if result := a.shutdown(); !result.OK {
		t.Fatalf("shutdown: %s", result.Error())
	}
}

func TestApp_ManualAgentRefreshCoalescesAndRejectsStaleSnapshots(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	started := make(chan struct{}, 1)
	release := make(chan struct{})
	provider := &correctiveAgentProvider{snapshot: func(ctx context.Context, call int) core.Result {
		if call == 1 {
			started <- struct{}{}
			select {
			case <-release:
			case <-ctx.Done():
				return core.Fail(core.E("test.snapshot", "cancelled", ctx.Err()))
			}
			return core.Ok(agentSnapshot{Work: []agentWorkSnapshot{{ExternalID: "local", NativeRunID: "run-old", Status: "running"}}})
		}
		return core.Ok(agentSnapshot{Work: []agentWorkSnapshot{{ExternalID: "local", NativeRunID: "run-new", Status: "completed"}}})
	}}
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	a.work.ids = sequenceIDs("local")
	if result := a.work.CreateWork("Local", "refresh it", "/src/local"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	if result := a.palette.Invoke(commandRefreshWork, &a); !result.OK {
		t.Fatalf("first Refresh Work: %v", result.Value)
	}
	first := a.takeAgentCommand()
	if first == nil || provider.SnapshotCalls() != 0 {
		t.Fatalf("manual refresh called Snapshot synchronously: command %v calls %d", first != nil, provider.SnapshotCalls())
	}
	message := make(chan tea.Msg, 1)
	go func() { message <- first() }()
	select {
	case <-started:
	case <-time.After(time.Second):
		t.Fatal("first snapshot did not start")
	}
	if result := a.palette.Invoke(commandRefreshWork, &a); !result.OK {
		t.Fatalf("coalesced Refresh Work: %v", result.Value)
	}
	if !a.agentSnapshotPending || a.takeAgentCommand() != nil || provider.SnapshotCalls() != 1 {
		t.Fatalf("coalesced refresh = pending %v calls %d", a.agentSnapshotPending, provider.SnapshotCalls())
	}
	close(release)
	model, second := a.Update(<-message)
	a = model.(app)
	if second == nil || a.agentSnapshotCurrent != 2 || a.work.Items()[0].Status != "running" {
		t.Fatalf("first completion = second %v current %d items %#v", second != nil, a.agentSnapshotCurrent, a.work.Items())
	}
	model, _ = a.Update(agentSnapshotMsg{requestID: 1, result: core.Ok(agentSnapshot{Work: []agentWorkSnapshot{{ExternalID: "local", NativeRunID: "run-stale", Status: "failed"}}})})
	a = model.(app)
	if state := a.work.AgentState(a.work.Items()[0]); state.NativeRunID != "run-old" || a.work.Items()[0].Status != "running" {
		t.Fatalf("stale snapshot replaced state: %#v / %#v", state, a.work.Items())
	}
	model, _ = a.Update(second())
	a = model.(app)
	if state := a.work.AgentState(a.work.Items()[0]); provider.SnapshotCalls() != 2 || state.NativeRunID != "run-new" || a.work.Items()[0].Status != "completed" {
		t.Fatalf("newest snapshot = calls %d state %#v items %#v", provider.SnapshotCalls(), state, a.work.Items())
	}
	model, _ = a.Update(agentSnapshotMsg{requestID: 1, result: core.Ok(agentSnapshot{Work: []agentWorkSnapshot{{ExternalID: "local", NativeRunID: "run-stale-after-new", Status: "failed"}}})})
	a = model.(app)
	if state := a.work.AgentState(a.work.Items()[0]); state.NativeRunID != "run-new" || a.work.Items()[0].Status != "completed" {
		t.Fatalf("stale snapshot replaced newest applied state: %#v / %#v", state, a.work.Items())
	}
}

func TestApp_AgentSnapshotFailurePreservesViewAndRefreshesLiveWork(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	provider := &correctiveAgentProvider{}
	provider.snapshot = func(_ context.Context, call int) core.Result {
		if call == 1 {
			return core.Fail(core.E("test.snapshot", "provider temporarily unavailable", nil))
		}
		return core.Ok(agentSnapshot{Work: []agentWorkSnapshot{{ExternalID: "local", NativeRunID: "run-1", Status: "completed"}}})
	}
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	a.work.ids = sequenceIDs("local")
	if result := a.work.CreateWork("Local", "keep last good", "/src/local"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	if result := a.work.ApplyAgentSnapshot(agentSnapshot{Work: []agentWorkSnapshot{{ExternalID: "local", NativeRunID: "run-1", Status: "running"}}}); !result.OK {
		t.Fatalf("seed snapshot: %v", result.Value)
	}
	command := a.requestAgentSnapshot()
	model, refresh := a.Update(command())
	a = model.(app)
	if refresh == nil || !a.agentRefreshArmed || a.work.Items()[0].Status != "running" || !strings.Contains(a.errText, "temporarily unavailable") {
		t.Fatalf("failed snapshot = refresh %v armed %v items %#v error %q", refresh != nil, a.agentRefreshArmed, a.work.Items(), a.errText)
	}
	model, command = a.Update(agentRefreshMsg{})
	a = model.(app)
	if command == nil {
		t.Fatal("armed refresh did not issue a new snapshot request")
	}
	model, _ = a.Update(command())
	a = model.(app)
	if provider.SnapshotCalls() != 2 || a.work.Items()[0].Status != "completed" {
		t.Fatalf("refreshed snapshot = calls %d items %#v", provider.SnapshotCalls(), a.work.Items())
	}
}

func TestApp_AgentRefreshTimerStopsWithLifecycle(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	provider := &correctiveAgentProvider{}
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	a.work.ids = sequenceIDs("local")
	if result := a.work.CreateWork("Local", "stop refresh", "/src/local"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	if result := a.work.ApplyAgentSnapshot(agentSnapshot{Work: []agentWorkSnapshot{{ExternalID: "local", NativeRunID: "run-1", Status: "running"}}}); !result.OK {
		t.Fatalf("seed snapshot: %v", result.Value)
	}
	command := a.armAgentRefresh()
	if command == nil {
		t.Fatal("live work did not arm a stoppable refresh timer")
	}
	message := make(chan tea.Msg, 1)
	go func() { message <- command() }()
	time.Sleep(10 * time.Millisecond)
	started := time.Now()
	a.lifecycle.stop()
	select {
	case got := <-message:
		if _, ok := got.(lifecycleStoppedMsg); !ok {
			t.Fatalf("cancelled timer message = %T, want lifecycleStoppedMsg", got)
		}
	case <-time.After(250 * time.Millisecond):
		t.Fatal("lifecycle cancellation did not stop the refresh timer promptly")
	}
	a.lifecycle.wait()
	if elapsed := time.Since(started); elapsed >= 250*time.Millisecond || provider.SnapshotCalls() != 0 {
		t.Fatalf("stopped lifecycle = elapsed %s snapshots %d", elapsed, provider.SnapshotCalls())
	}
}

func TestApp_AgentResumeUsesNativeResumeAndInterruptedRetry(t *testing.T) {
	t.Run("waiting", func(t *testing.T) {
		engine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities()}
		a := testNativeAgentApp(t, requireAgentAdapter(t, engine))
		if result := a.work.updateWork("waiting", "work-1", func(record *workItemRecord) { record.Status = "waiting" }); !result.OK {
			t.Fatalf("set waiting: %v", result.Value)
		}
		a.work.agentWork["work-1"] = agentWorkSnapshot{NativeRunID: "run-parent", AnswerID: "answer-exact", Agent: "codex", Runtime: "gpt-5"}
		if result := a.queueAgentAction(agentFeatureResume); !result.OK {
			t.Fatalf("Resume: %v", result.Value)
		}
		driveCorrectiveCommand(t, &a, a.takeAgentCommand())
		if a.activeOverlay != overlayLaunchReview || countAgentCall(engine.calls, "resume") != 0 {
			t.Fatalf("resume review = overlay %d calls %#v", a.activeOverlay, engine.calls)
		}
		model, command := a.Update(testKeyPress(tea.KeyEnter))
		a = model.(app)
		driveCorrectiveCommand(t, &a, command)
		want := work.ResumeRequest{
			Work:        nativeWorkItem(agentWorkRequest{ID: "work-1", ExternalID: "local:work-1", Title: "Ship", Task: "Implement the slice", Repository: "/src/project"}, "work-1", ""),
			ParentRunID: "run-parent", AnswerID: "answer-exact", Provider: "codex", Model: "gpt-5",
		}
		if core.JSONMarshalString(engine.resumeRequest) != core.JSONMarshalString(want) || engine.retryParent != "" {
			t.Fatalf("native waiting Resume = %#v retry parent %q, want %#v", engine.resumeRequest, engine.retryParent, want)
		}
	})

	t.Run("interrupted", func(t *testing.T) {
		engine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities()}
		a := testNativeAgentApp(t, requireAgentAdapter(t, engine))
		if result := a.work.updateWork("interrupted", "work-1", func(record *workItemRecord) { record.Status = "interrupted" }); !result.OK {
			t.Fatalf("set interrupted: %v", result.Value)
		}
		a.work.agentWork["work-1"] = agentWorkSnapshot{NativeRunID: "run-interrupted"}
		if result := a.queueAgentAction(agentFeatureResume); !result.OK {
			t.Fatalf("Resume interrupted: %v", result.Value)
		}
		driveCorrectiveCommand(t, &a, a.takeAgentCommand())
		if a.activeOverlay != overlayLaunchReview || countAgentCall(engine.calls, "retry") != 0 {
			t.Fatalf("interrupted retry review = overlay %d calls %#v", a.activeOverlay, engine.calls)
		}
		model, command := a.Update(testKeyPress(tea.KeyEnter))
		a = model.(app)
		driveCorrectiveCommand(t, &a, command)
		if engine.retryParent != "run-interrupted" || engine.resumeRequest.ParentRunID != "" || engine.retryItem.ID != "work-1" {
			t.Fatalf("interrupted Resume = retry item %#v parent %q resume %#v", engine.retryItem, engine.retryParent, engine.resumeRequest)
		}
	})
}

func TestApp_AgentRunReplacementClearsRunScopedContinuation(t *testing.T) {
	t.Run("answered parent replaced by interrupted child", func(t *testing.T) {
		engine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities()}
		a := testNativeAgentApp(t, requireAgentAdapter(t, engine))
		if result := a.work.updateWork("waiting", "work-1", func(record *workItemRecord) { record.Status = "waiting" }); !result.OK {
			t.Fatalf("set waiting: %v", result.Value)
		}
		a.work.agentWork["work-1"] = agentWorkSnapshot{NativeRunID: "run-parent", QuestionID: "question-parent", Question: "Which target?", Status: "waiting"}
		a.work.syncList()
		engine.snapshot = work.Snapshot{
			Runs: []work.Run{
				{ID: "run-parent", WorkID: "work-1", Status: work.RunWaiting},
				{ID: "run-child", WorkID: "work-1", Status: work.RunInterrupted},
			},
			Questions: []work.Question{{ID: "question-parent", RunID: "run-parent", Text: "Which target?", CreatedAt: time.Now().UTC()}},
		}
		if result := a.queueAgentAction(agentFeatureAnswer); !result.OK {
			t.Fatalf("Answer: %v", result.Value)
		}
		a.answerOverlay.input.SetValue("Use child target")
		model, answerCommand := a.Update(testKeyPress(tea.KeyEnter))
		a = model.(app)
		model, snapshotCommand := a.Update(answerCommand())
		a = model.(app)
		if snapshotCommand == nil {
			t.Fatal("answered parent did not schedule a snapshot")
		}
		if state := a.work.AgentState(a.work.Items()[0]); state.AnswerID != "answer-1" || state.ResumeRunID != "run-2" {
			t.Fatalf("parent answer identity = %#v", state)
		}
		model, _ = a.Update(snapshotCommand())
		a = model.(app)
		state := a.work.AgentState(a.work.Items()[0])
		if state.NativeRunID != "run-child" || state.AnswerID != "" || state.ResumeRunID != "" || state.QuestionID != "" || state.Question != "" || a.work.Items()[0].Status != "interrupted" {
			t.Fatalf("child replacement retained parent state: %#v item %#v", state, a.work.Items()[0])
		}
		if result := a.queueAgentAction(agentFeatureResume); !result.OK {
			t.Fatalf("Resume interrupted child: %v", result.Value)
		}
		driveCorrectiveCommand(t, &a, a.takeAgentCommand())
		if engine.retryParent != "run-child" || engine.resumeRequest.ParentRunID != "" {
			t.Fatalf("interrupted child Resume = retry %q resume %#v", engine.retryParent, engine.resumeRequest)
		}
	})

	t.Run("late parent answer cannot attach to selected child", func(t *testing.T) {
		engine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities(), snapshot: work.Snapshot{
			Runs: []work.Run{
				{ID: "run-parent", WorkID: "work-1", Status: work.RunWaiting},
				{ID: "run-child", WorkID: "work-1", Status: work.RunInterrupted},
			},
		}}
		a := testNativeAgentApp(t, requireAgentAdapter(t, engine))
		if result := a.work.updateWork("waiting", "work-1", func(record *workItemRecord) { record.Status = "waiting" }); !result.OK {
			t.Fatalf("set waiting: %v", result.Value)
		}
		a.work.agentWork["work-1"] = agentWorkSnapshot{NativeRunID: "run-parent", QuestionID: "question-parent", Question: "Which target?", Status: "waiting"}
		a.work.syncList()
		if result := a.queueAgentAction(agentFeatureAnswer); !result.OK {
			t.Fatalf("Answer: %v", result.Value)
		}
		a.answerOverlay.input.SetValue("Use child target")
		model, answerCommand := a.Update(testKeyPress(tea.KeyEnter))
		a = model.(app)
		lateAnswer := answerCommand()
		snapshotCommand := a.requestAgentSnapshot()
		model, _ = a.Update(snapshotCommand())
		a = model.(app)
		model, _ = a.Update(lateAnswer)
		a = model.(app)
		state := a.work.AgentState(a.work.Items()[0])
		if state.NativeRunID != "run-child" || state.AnswerID != "" || state.ResumeRunID != "" {
			t.Fatalf("late parent answer attached to child: %#v", state)
		}
	})
}

func TestApp_AgentRunReplacementDropsOldPreparedReview(t *testing.T) {
	review := correctiveChangeReview(true)
	review.RunID = "run-reviewed-old"
	engine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities(), snapshot: work.Snapshot{
		Runs: []work.Run{
			{ID: review.RunID, WorkID: review.WorkID, Status: work.RunCompleted},
			{ID: "run-new-attempt", WorkID: review.WorkID, Status: work.RunRunning},
		},
		Acceptances: []work.Acceptance{{ID: "review-old", WorkID: review.WorkID, RunID: review.RunID, Status: "prepared", ValidationJSON: core.JSONMarshalString(review)}},
	}}
	a := testNativeAgentApp(t, requireAgentAdapter(t, engine))
	driveCorrectiveCommand(t, &a, a.requestAgentSnapshot())
	state := a.work.AgentState(a.work.Items()[0])
	if state.NativeRunID != "run-new-attempt" || state.ReviewID != "" || state.ReviewStatus != "" || state.Review.Payload != nil {
		t.Fatalf("new attempt retained old review: %#v", state)
	}
	a.refreshAgentPalette()
	if command := a.palette.byID[agentCommandID(agentFeatureAccept)]; command.Available {
		t.Fatalf("Accept available for old prepared review: %#v", command)
	}
}

func TestAppRecoveryAbandonUsesReviewConfirmationAndRefreshesLedger(t *testing.T) {
	at := time.Date(2026, time.July, 19, 14, 0, 0, 0, time.UTC)
	recovery := agentPendingRecovery{EventID: "recovery-event-11", Receipt: agentRecoveryReceipt{
		Kind: "run", ProjectID: "project-1", WorkID: "work-1", RunID: "attempt-11",
		RunNumber: 11, WorkspaceRunID: "lineage-root", Branch: "lem/work/work-1/run-11",
		Worktree: "/private/workspaces/project-1/runs/lineage-root/worktree",
	}}
	engine := &fixtureNativeAgentEngine{
		capabilities: append(nativeFixtureCapabilities(), work.Capability{Name: "recovery.abandon", Available: true}),
		snapshot: work.Snapshot{
			Runs: []work.Run{{ID: recovery.Receipt.RunID, WorkID: recovery.Receipt.WorkID, ProjectID: recovery.Receipt.ProjectID, Number: recovery.Receipt.RunNumber, Status: work.RunFailed}},
			Events: []work.Event{{
				ID: recovery.EventID, RunID: recovery.Receipt.RunID, WorkID: recovery.Receipt.WorkID,
				Kind: "workspace_cleanup_retained", Detail: recovery.Receipt.Worktree,
				DetailJSON: core.JSONMarshalString(recovery.Receipt), CreatedAt: at,
			}},
		},
	}
	a := testNativeAgentApp(t, requireAgentAdapter(t, engine))
	driveCorrectiveCommand(t, &a, a.requestAgentSnapshot())
	if result := a.queueAgentAction(agentFeatureRecoveryAbandon); !result.OK {
		t.Fatalf("queue recovery abandon: %s", result.Error())
	}
	driveCorrectiveCommand(t, &a, a.takeAgentCommand())
	if a.activeOverlay != overlayLaunchReview || engine.abandonRecoveryCalls != 0 {
		t.Fatalf("recovery review = overlay %d calls %d err %q", a.activeOverlay, engine.abandonRecoveryCalls, a.errText)
	}

	model, confirmation := a.Update(testKeyPress(tea.KeyEnter))
	a = model.(app)
	next := driveCorrectiveCommand(t, &a, confirmation)
	if engine.abandonRecoveryCalls != 1 || engine.abandonRecoveryRunID != recovery.Receipt.RunID || engine.abandonRecoveryEventID != recovery.EventID {
		t.Fatalf("recovery action = calls %d run %q event %q", engine.abandonRecoveryCalls, engine.abandonRecoveryRunID, engine.abandonRecoveryEventID)
	}
	engine.snapshot.Events = append(engine.snapshot.Events, work.Event{
		ID: "recovery-success-11", RunID: recovery.Receipt.RunID, WorkID: recovery.Receipt.WorkID,
		Kind: "cleanup_recovery_succeeded", Detail: recovery.Receipt.Worktree,
		DetailJSON: core.JSONMarshalString(agentRecoveryOutcome{RecoveryEventID: recovery.EventID, Receipt: recovery.Receipt}), CreatedAt: at.Add(time.Second),
	})
	driveCorrectiveCommand(t, &a, next)
	state := a.work.AgentState(a.work.Items()[0])
	if state.RecoveryCount != 0 || state.Recovery.EventID != "" {
		t.Fatalf("recovery remained after success refresh: %#v", state)
	}
	a.refreshAgentPalette()
	if command := a.palette.byID[agentCommandID(agentFeatureRecoveryAbandon)]; command.Available {
		t.Fatalf("recovery action remained available after success: %#v", command)
	}
}

func TestApp_AgentReviewPreparedSnapshotAndRejectPreserveLocalWork(t *testing.T) {
	review := correctiveChangeReview(true)
	engine := &fixtureNativeAgentEngine{
		capabilities: nativeFixtureCapabilities(), changeReview: review,
		snapshot: work.Snapshot{
			Runs:        []work.Run{{ID: review.RunID, WorkID: review.WorkID, Status: work.RunCompleted}},
			Acceptances: []work.Acceptance{{ID: "review-prepared", WorkID: review.WorkID, RunID: review.RunID, Status: "prepared", ValidationJSON: core.JSONMarshalString(review)}},
		},
	}
	a := testNativeAgentApp(t, requireAgentAdapter(t, engine))
	if result := a.work.updateWork("completed", review.WorkID, func(record *workItemRecord) { record.Status = "completed" }); !result.OK {
		t.Fatalf("set completed: %v", result.Value)
	}
	a.work.agentWork[review.WorkID] = agentWorkSnapshot{NativeRunID: review.RunID, Status: "completed"}
	localEvent := eventRecord{ID: "local-review-note", SessionID: workEventSessionID(a.work.Items()[0]), WorkItemID: review.WorkID, Kind: "local.note", Status: "recorded", Title: "keep me", PayloadJSON: "{}", CreatedAt: time.Now().UTC()}
	if result := a.work.repository.SaveEvent(localEvent); !result.OK {
		t.Fatalf("SaveEvent: %v", result.Value)
	}
	if result := a.work.RefreshLocal(); !result.OK {
		t.Fatalf("RefreshLocal: %v", result.Value)
	}
	if result := a.queueAgentAction(agentFeatureChangesReview); !result.OK {
		t.Fatalf("Review Changes: %v", result.Value)
	}
	model, snapshotCommand := a.Update(a.takeAgentCommand()())
	a = model.(app)
	if snapshotCommand == nil || a.activeOverlay != overlayChangeReview || engine.reviewChangesRunID != review.RunID || core.JSONMarshalString(a.agentReview.Payload) != core.JSONMarshalString(review) {
		t.Fatalf("fresh review = snapshot %v overlay %d run %q payload %#v", snapshotCommand != nil, a.activeOverlay, engine.reviewChangesRunID, a.agentReview.Payload)
	}
	model, _ = a.Update(snapshotCommand())
	a = model.(app)
	state := a.work.AgentState(a.work.Items()[0])
	if state.ReviewID != "review-prepared" || state.ReviewStatus != "prepared" || core.JSONMarshalString(state.Review.Payload) != core.JSONMarshalString(review) {
		t.Fatalf("durable prepared review state = %#v", state)
	}
	model, _ = a.Update(testKeyPress(tea.KeyEsc))
	a = model.(app)
	if command := a.palette.byID[agentCommandID(agentFeatureReject)]; !command.Available {
		t.Fatalf("Reject unavailable after review escape: %#v", command)
	}
	if result := a.palette.Invoke(agentCommandID(agentFeatureReject), &a); !result.OK {
		t.Fatalf("Reject invoke: %v", result.Value)
	}
	driveCorrectiveCommand(t, &a, a.takeAgentCommand())
	if engine.rejectRunID != review.RunID {
		t.Fatalf("Reject native run = %q, want %q", engine.rejectRunID, review.RunID)
	}
	storedItems := a.work.repository.ListWorkItems(false).Value.([]workItemRecord)
	storedEvents := a.work.repository.Events(workEventSessionID(storedItems[0])).Value.([]eventRecord)
	if len(storedItems) != 1 || storedItems[0].ID != review.WorkID || storedItems[0].Title != "Ship" || len(storedEvents) != 1 || storedEvents[0].ID != localEvent.ID {
		t.Fatalf("Reject mutated local state: items %#v events %#v", storedItems, storedEvents)
	}
}

func TestApp_AgentPreparedAcceptRequiresReviewAndFinalConfirmation(t *testing.T) {
	for _, test := range []struct {
		name            string
		validated       bool
		acknowledgement bool
	}{
		{name: "validated", validated: true},
		{name: "no validation", acknowledgement: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			review := correctiveChangeReview(test.validated)
			engine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities()}
			a := testNativeAgentApp(t, requireAgentAdapter(t, engine))
			a.work.agentWork[review.WorkID] = agentWorkSnapshot{
				NativeRunID: review.RunID, ReviewID: "review-prepared", ReviewStatus: "prepared",
				Review: mapChangeReview(review, true),
			}
			a.refreshAgentPalette()
			if result := a.palette.Invoke(agentCommandID(agentFeatureAccept), &a); !result.OK {
				t.Fatalf("Accept invoke: %v", result.Value)
			}
			if a.activeOverlay != overlayChangeReview || engine.acceptRequest.Confirmed {
				t.Fatalf("prepared Accept = overlay %d request %#v", a.activeOverlay, engine.acceptRequest)
			}
			if test.acknowledgement {
				model, command := a.Update(testKeyPress(tea.KeyEnter))
				a = model.(app)
				if command != nil || a.changeOverlay.final || engine.acceptRequest.Confirmed {
					t.Fatalf("unacknowledged enter = command %v final %v accept %#v", command != nil, a.changeOverlay.final, engine.acceptRequest)
				}
				model, _ = a.Update(testTextPress('a'))
				a = model.(app)
				if !a.changeOverlay.acknowledged {
					t.Fatal("a did not acknowledge missing validation")
				}
			}
			model, command := a.Update(testKeyPress(tea.KeyEnter))
			a = model.(app)
			if command != nil || !a.changeOverlay.final || engine.acceptRequest.Confirmed {
				t.Fatalf("review confirmation = command %v final %v accept %#v", command != nil, a.changeOverlay.final, engine.acceptRequest)
			}
			model, command = a.Update(testKeyPress(tea.KeyEnter))
			a = model.(app)
			if command == nil || engine.acceptRequest.Confirmed {
				t.Fatalf("final confirmation scheduling = command %v accept %#v", command != nil, engine.acceptRequest)
			}
			driveCorrectiveCommand(t, &a, command)
			want := workspace.AcceptRequest{Review: review, Confirmed: true}
			if core.JSONMarshalString(engine.acceptRequest) != core.JSONMarshalString(want) {
				t.Fatalf("Accept request = %#v, want %#v", engine.acceptRequest, want)
			}
		})
	}
}

func TestApp_AgentBlockedReviewStaysScrollableAndRejectable(t *testing.T) {
	for _, test := range []struct {
		name   string
		review workspace.ChangeReview
	}{
		{name: "conflict", review: func() workspace.ChangeReview {
			review := correctiveChangeReview(true)
			review.Conflicts = []string{"a.go content conflict"}
			return review
		}()},
		{name: "failed validation", review: func() workspace.ChangeReview {
			review := correctiveChangeReview(true)
			review.Validation[0].Passed = false
			review.Validation[0].ExitCode = 1
			review.Validation[0].Output = "FAIL package"
			return review
		}()},
	} {
		t.Run(test.name, func(t *testing.T) {
			lines := make([]string, 80)
			for index := range lines {
				lines[index] = core.Sprintf("diff line %03d", index)
			}
			test.review.Diff = core.Join("\n", lines...)
			engine := &fixtureNativeAgentEngine{
				capabilities: nativeFixtureCapabilities(), changeReview: test.review,
				snapshot: work.Snapshot{
					Runs:        []work.Run{{ID: test.review.RunID, WorkID: test.review.WorkID, Status: work.RunCompleted}},
					Acceptances: []work.Acceptance{{ID: "review-blocked", WorkID: test.review.WorkID, RunID: test.review.RunID, Status: "prepared", ValidationJSON: core.JSONMarshalString(test.review)}},
				},
			}
			a := testNativeAgentApp(t, requireAgentAdapter(t, engine))
			if result := a.work.updateWork("completed", test.review.WorkID, func(record *workItemRecord) { record.Status = "completed" }); !result.OK {
				t.Fatalf("set completed: %v", result.Value)
			}
			a.work.agentWork[test.review.WorkID] = agentWorkSnapshot{NativeRunID: test.review.RunID, Status: "completed"}
			if result := a.queueAgentAction(agentFeatureChangesReview); !result.OK {
				t.Fatalf("Review Changes: %v", result.Value)
			}
			model, snapshotCommand := a.Update(a.takeAgentCommand()())
			a = model.(app)
			model, _ = a.Update(snapshotCommand())
			a = model.(app)
			a.changeOverlay.View(48, 12, a.styles)
			model, _ = a.Update(testKeyPress(tea.KeyPgDown))
			a = model.(app)
			if a.changeOverlay.viewport.YOffset() == 0 || a.changeOverlay.viewport.TotalLineCount() < len(lines) {
				t.Fatalf("blocked review viewport = offset %d lines %d", a.changeOverlay.viewport.YOffset(), a.changeOverlay.viewport.TotalLineCount())
			}
			model, command := a.Update(testKeyPress(tea.KeyEnter))
			a = model.(app)
			if command != nil || a.activeOverlay != overlayChangeReview || strings.Contains(core.JSONMarshalString(engine.calls), "accept") || !strings.Contains(a.errText, "prevent acceptance") {
				t.Fatalf("blocked Accept = command %v overlay %d calls %#v error %q", command != nil, a.activeOverlay, engine.calls, a.errText)
			}
			model, _ = a.Update(testKeyPress(tea.KeyEsc))
			a = model.(app)
			if result := a.palette.Invoke(agentCommandID(agentFeatureReject), &a); !result.OK {
				t.Fatalf("Reject blocked review: %v", result.Value)
			}
			driveCorrectiveCommand(t, &a, a.takeAgentCommand())
			if engine.rejectRunID != test.review.RunID || strings.Contains(core.JSONMarshalString(engine.calls), "accept") {
				t.Fatalf("blocked review mutation = reject %q calls %#v", engine.rejectRunID, engine.calls)
			}
		})
	}
}

func TestApp_AgentQueueStateGatesAndExecutesNativeControls(t *testing.T) {
	engine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities()}
	a := testNativeAgentApp(t, requireAgentAdapter(t, engine))
	a.work.queueStatus = "frozen"
	a.refreshAgentPalette()
	if result := a.palette.Invoke(agentCommandID(agentFeatureQueueStart), &a); !result.OK {
		t.Fatalf("Start frozen queue: %v", result.Value)
	}
	driveCorrectiveCommand(t, &a, a.takeAgentCommand())
	if countAgentCall(engine.calls, "queue.start") != 1 {
		t.Fatalf("queue start calls = %#v", engine.calls)
	}
	a.work.queueStatus = "accepting"
	a.refreshAgentPalette()
	if result := a.palette.Invoke(agentCommandID(agentFeatureQueueStop), &a); !result.OK {
		t.Fatalf("Stop accepting queue: %v", result.Value)
	}
	driveCorrectiveCommand(t, &a, a.takeAgentCommand())
	if countAgentCall(engine.calls, "queue.stop") != 1 {
		t.Fatalf("queue stop calls = %#v", engine.calls)
	}
	a.work.queueStatus = "draining"
	a.refreshAgentPalette()
	for _, feature := range []agentFeature{agentFeatureQueueStart, agentFeatureQueueStop} {
		before := len(engine.calls)
		if result := a.palette.Invoke(agentCommandID(feature), &a); result.OK {
			t.Fatalf("%s admitted while draining", feature)
		}
		if len(engine.calls) != before {
			t.Fatalf("%s mutated provider while draining: %#v", feature, engine.calls)
		}
	}
}

func TestApp_AgentReviewEscapeAbortsTransaction(t *testing.T) {
	for _, overlay := range []overlayKind{overlayAgentSelection, overlayProjectReview, overlayGitEnableReview, overlayLaunchReview} {
		t.Run(core.Sprintf("overlay-%d", overlay), func(t *testing.T) {
			a := newApp("", 0, 64)
			a.agentOperationID, a.agentOperationNext = 7, 7
			a.agentStage = agentReviewProject
			a.agentRequest = agentRequest{Feature: agentFeatureDispatch, Provider: "codex"}
			a.agentReview = agentReview{Title: "review"}
			a.activeOverlay = overlay
			a.launchReview = newAgentSelectionOverlay("codex", "gpt-5")
			model, _ := a.Update(testKeyPress(tea.KeyEsc))
			a = model.(app)
			if a.activeOverlay != overlayNone || a.agentStage != agentReviewNone || a.agentOperationID != 0 || a.agentRequest.Feature != "" || a.agentReview.Title != "" {
				t.Fatalf("escape state = overlay=%d stage=%d id=%d request=%#v review=%#v", a.activeOverlay, a.agentStage, a.agentOperationID, a.agentRequest, a.agentReview)
			}
		})
	}
}

func TestApp_AgentReviewNativeAdapterTransactions(t *testing.T) {
	project := testNativeProjectReview("work-1", false)
	dispatch := testNativeDispatchReview(project)
	engine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities(), projectReview: project, registeredProject: dispatch.Project, dispatchReview: dispatch}
	adapter := requireAgentAdapter(t, engine)
	a := testNativeAgentApp(t, adapter)

	if result := a.queueAgentAction(agentFeatureDispatch); !result.OK {
		t.Fatalf("dispatch: %v", result.Value)
	}
	startAgentProjectReview(t, &a, "codex", "gpt-5")
	if !strings.Contains(a.agentReview.Body, "Included files: 2") {
		t.Fatalf("ad-hoc included-file review:\n%s", a.agentReview.Body)
	}
	registration, ok := a.agentReview.Payload.(agentProjectRegistration)
	if !ok || core.JSONMarshalString(registration.Review) != core.JSONMarshalString(project) || registration.Provider != "codex" || registration.Model != "gpt-5" {
		t.Fatalf("project payload = %#v", a.agentReview.Payload)
	}
	model, command := a.Update(testKeyPress(tea.KeyEnter))
	a = model.(app)
	if command == nil || engine.registerCalls != 0 || engine.dispatchCalls != 0 || a.agentRequest.EnableGit {
		t.Fatalf("clean project confirmation = command=%v register=%d dispatch=%d enableGit=%v", command != nil, engine.registerCalls, engine.dispatchCalls, a.agentRequest.EnableGit)
	}
	model, _ = a.Update(command())
	a = model.(app)
	launch, ok := a.agentReview.Payload.(orchestrator.DispatchReview)
	if !ok || core.JSONMarshalString(launch) != core.JSONMarshalString(dispatch) || a.activeOverlay != overlayLaunchReview || engine.registerCalls != 1 || engine.dispatchCalls != 0 {
		t.Fatalf("launch payload = %#v register=%d dispatch=%d", a.agentReview.Payload, engine.registerCalls, engine.dispatchCalls)
	}
	for _, want := range []string{"Provider: codex", "Model: gpt-5", "Command: codex exec --api-key [REDACTED] --model gpt-5", "Source: /src/project", "Branch: main", "Revision: abc123", "Private repository: work-1", "Worktree: /private/runs/pending-run/worktree", "Queue: ready for admission"} {
		if !strings.Contains(a.agentReview.Body, want) {
			t.Fatalf("launch body missing %q:\n%s", want, a.agentReview.Body)
		}
	}
	for _, width := range []int{48, 120} {
		a.width, a.height = width, 22
		view := a.View().Content
		wants := []string{"Command:", "codex", "exec", "--api-key", "[REDACTED]", "--model", "gpt-5", "native host access"}
		if width >= 100 {
			wants = append(wants, "Command: codex exec --api-key [REDACTED] --model gpt-5")
		}
		for _, want := range wants {
			if !strings.Contains(view, want) {
				t.Fatalf("width %d launch view missing %q:\n%s", width, want, view)
			}
		}
		for line, text := range strings.Split(view, "\n") {
			if got := lipgloss.Width(text); got > width {
				t.Fatalf("width %d line %d overflows at %d", width, line, got)
			}
		}
	}
	model, command = a.Update(testKeyPress(tea.KeyEnter))
	a = model.(app)
	if command == nil || engine.dispatchCalls != 0 {
		t.Fatalf("launch confirmation = command=%v dispatch=%d", command != nil, engine.dispatchCalls)
	}
	model, _ = a.Update(command())
	a = model.(app)
	if engine.dispatchCalls != 1 || a.work.Items()[0].Status != "queued" {
		t.Fatalf("final dispatch = calls=%d work=%#v", engine.dispatchCalls, a.work.Items())
	}
}

func TestApp_AgentReviewNativeAdapterGitConfirmationAndFailure(t *testing.T) {
	project := testNativeProjectReview("work-1", true)
	dispatch := testNativeDispatchReview(project)
	engine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities(), projectReview: project, registeredProject: dispatch.Project, dispatchReview: dispatch}
	adapter := requireAgentAdapter(t, engine)
	a := testNativeAgentApp(t, adapter)
	if result := a.queueAgentAction(agentFeatureDispatch); !result.OK {
		t.Fatalf("dispatch: %v", result.Value)
	}
	startAgentProjectReview(t, &a, "codex", "gpt-5")
	model, command := a.Update(testKeyPress(tea.KeyEnter))
	a = model.(app)
	if command != nil || a.activeOverlay != overlayGitEnableReview || engine.registerCalls != 0 {
		t.Fatalf("ad-hoc project enter = command=%v overlay=%d register=%d", command != nil, a.activeOverlay, engine.registerCalls)
	}
	model, _ = a.Update(testKeyPress(tea.KeyEsc))
	a = model.(app)
	if engine.registerCalls != 0 || a.agentStage != agentReviewNone {
		t.Fatalf("Git cancel = register=%d stage=%d", engine.registerCalls, a.agentStage)
	}

	if result := a.queueAgentAction(agentFeatureDispatch); !result.OK {
		t.Fatalf("fresh dispatch: %v", result.Value)
	}
	startAgentProjectReview(t, &a, "codex", "gpt-5")
	model, _ = a.Update(testKeyPress(tea.KeyEnter))
	a = model.(app)
	model, command = a.Update(testKeyPress(tea.KeyEnter))
	a = model.(app)
	if command == nil || !a.agentRequest.EnableGit || engine.registerCalls != 0 {
		t.Fatalf("Git confirmation = command=%v enable=%v register=%d", command != nil, a.agentRequest.EnableGit, engine.registerCalls)
	}
	model, _ = a.Update(command())
	a = model.(app)
	if engine.registerCalls != 1 || a.activeOverlay != overlayLaunchReview {
		t.Fatalf("Git registration = register=%d overlay=%d", engine.registerCalls, a.activeOverlay)
	}

	failure := core.Fail(core.E("test.register", "included file hash changed", nil))
	failedEngine := &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities(), projectReview: project, registeredProject: dispatch.Project, dispatchReview: dispatch, registerProject: func(context.Context, orchestrator.ProjectReview, bool) core.Result { return failure }}
	failedApp := testNativeAgentApp(t, requireAgentAdapter(t, failedEngine))
	if result := failedApp.queueAgentAction(agentFeatureDispatch); !result.OK {
		t.Fatalf("failed dispatch: %v", result.Value)
	}
	startAgentProjectReview(t, &failedApp, "codex", "gpt-5")
	model, _ = failedApp.Update(testKeyPress(tea.KeyEnter))
	failedApp = model.(app)
	model, command = failedApp.Update(testKeyPress(tea.KeyEnter))
	failedApp = model.(app)
	model, _ = failedApp.Update(command())
	failedApp = model.(app)
	if !strings.Contains(failedApp.errText, "included file hash changed") || failedEngine.reviewDispatchCalls != 0 || failedEngine.dispatchCalls != 0 {
		t.Fatalf("changed hash failure = err=%q reviewDispatch=%d dispatch=%d", failedApp.errText, failedEngine.reviewDispatchCalls, failedEngine.dispatchCalls)
	}
}

func TestApp_AgentReviewNativeAdapterProjectFailuresDoNotRegister(t *testing.T) {
	for _, reason := range []string{"source repository is dirty", "source repository is detached"} {
		t.Run(reason, func(t *testing.T) {
			engine := &failingProjectNativeEngine{fixtureNativeAgentEngine: &fixtureNativeAgentEngine{capabilities: nativeFixtureCapabilities()}, failure: core.Fail(core.E("test.project", reason, nil))}
			a := testNativeAgentApp(t, requireAgentAdapter(t, engine))
			if result := a.queueAgentAction(agentFeatureDispatch); !result.OK {
				t.Fatalf("dispatch: %v", result.Value)
			}
			startAgentProjectReview(t, &a, "codex", "gpt-5")
			if !strings.Contains(a.errText, reason) || engine.registerCalls != 0 || engine.dispatchCalls != 0 {
				t.Fatalf("project failure = err=%q register=%d dispatch=%d", a.errText, engine.registerCalls, engine.dispatchCalls)
			}
		})
	}
}

func TestWorkEditor_CtrlSSavesAndEscapeCancels(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	provider := &launchReviewProvider{caps: []agentCapability{{Feature: agentFeatureDispatch, Available: true}}}
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	if result := a.openWorkEditor(workItemRecord{}); !result.OK {
		t.Fatalf("open editor: %v", result.Value)
	}
	a.workEditor.title.SetValue("Keyboard save")
	a.workEditor.task.SetValue("Complete task")
	a.workEditor.repository.SetValue("/tmp/repository")
	model, _ := a.Update(testModifiedKeyPress('s', tea.ModCtrl))
	a = model.(app)
	if a.activeOverlay != overlayNone || len(a.work.Items()) != 1 || len(provider.reviewRequests) != 0 || len(provider.runs) != 0 {
		t.Fatalf("ctrl+s state = overlay=%d work=%#v", a.activeOverlay, a.work.Items())
	}
	if result := a.openWorkEditor(workItemRecord{}); !result.OK {
		t.Fatalf("open cancellation editor: %v", result.Value)
	}
	a.workEditor.title.SetValue("Cancelled")
	model, _ = a.Update(testKeyPress(tea.KeyEsc))
	a = model.(app)
	if a.activeOverlay != overlayNone || len(a.work.Items()) != 1 {
		t.Fatalf("escape state = overlay=%d work=%#v", a.activeOverlay, a.work.Items())
	}
}

// ---- Data panel wiring (Task 8) ----

func assertDataReviewPending(t *testing.T, store dataset.Store, itemID string) {
	t.Helper()
	review := core.MustCast[dataset.Review](store.ReviewLatest(itemID))
	if review.Status != dataset.StatusPending {
		t.Fatalf("item %s status = %q, want still pending (no write yet)", itemID, review.Status)
	}
}

func TestApp_AttachDataWiresPanelAndPalette(t *testing.T) {
	store := newTestDataPanelStore(t)
	a := newApp("", 0, 64)
	if result := a.attachData(store); !result.OK {
		t.Fatalf("attachData: %v", result.Value)
	}
	if a.data == nil {
		t.Fatal("attachData left a.data nil")
	}
	if _, exists := a.palette.byID[dataCommandID(dataActionApprove, false)]; !exists {
		t.Fatal("attachData did not seed the palette's data.* commands")
	}
}

func TestApp_AttachDataNilStoreLeavesCanonicalEmptyPanel(t *testing.T) {
	a := newApp("", 0, 64)
	if result := a.attachData(nil); !result.OK {
		t.Fatalf("attachData(nil): %v", result.Value)
	}
	if a.data != nil {
		t.Fatalf("attachData(nil) left a.data non-nil: %#v", a.data)
	}
	a.activePanel = panelData
	view := ansi.Strip(a.panelView())
	compact := strings.Join(strings.Fields(view), " ")
	if !strings.Contains(compact, "DATA 0 items · sort date") ||
		!strings.Contains(compact, "○ No items match this filter") ||
		!strings.Contains(compact, "Import or capture data with lem data import or lem serve --capture.") {
		t.Fatalf("panelView with no data store:\n%s", view)
	}
}

func TestApp_RoutePanelDataDispatchesToDataPanel(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	seedDataItem(t, store, ds.ID, "first", "first", at)
	seedDataItem(t, store, ds.ID, "second", "second", at.Add(time.Second))

	a := newApp("", 0, 64)
	if result := a.attachData(store); !result.OK {
		t.Fatalf("attachData: %v", result.Value)
	}
	a.activePanel = panelData
	before, ok := a.data.Selected()
	if !ok {
		t.Fatal("no initial selection")
	}
	model, _ := a.Update(keyMsg("j"))
	a = model.(app)
	after, ok := a.data.Selected()
	if !ok || after.Item.ID == before.Item.ID {
		t.Fatalf("j navigation did not move the selection: before=%s after=%s ok=%v", before.Item.ID, after.Item.ID, ok)
	}
}

func TestApp_PanelViewDispatchesToDataPanel(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	seedDataItem(t, store, ds.ID, "prompt-marker", "response-marker", at)

	a := newApp("", 0, 64)
	if result := a.attachData(store); !result.OK {
		t.Fatalf("attachData: %v", result.Value)
	}
	a.activePanel, a.width, a.height = panelData, 120, 40
	view := a.panelView()
	if !strings.Contains(view, "DATA") {
		t.Fatalf("panelView for panelData missing the panel's own header:\n%s", view)
	}
}

func TestApp_DataPanelHotkeys_ApproveReject(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	item := seedDataItem(t, store, ds.ID, "p", "r", at)

	a := newApp("", 0, 64)
	if result := a.attachData(store); !result.OK {
		t.Fatalf("attachData: %v", result.Value)
	}
	a.activePanel = panelData

	model, _ := a.Update(keyMsg("a"))
	a = model.(app)
	if a.errText != "" {
		t.Fatalf("approve hotkey errText = %q", a.errText)
	}
	if review := core.MustCast[dataset.Review](store.ReviewLatest(item.ID)); review.Status != dataset.StatusApproved {
		t.Fatalf("status after 'a' = %q", review.Status)
	}

	model, _ = a.Update(keyMsg("r"))
	a = model.(app)
	if review := core.MustCast[dataset.Review](store.ReviewLatest(item.ID)); review.Status != dataset.StatusRejected {
		t.Fatalf("status after 'r' = %q", review.Status)
	}
}

func TestApp_DataPanelHotkeys_TagOverlayFlow(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	item := seedDataItem(t, store, ds.ID, "p", "r", at)

	a := newApp("", 0, 64)
	if result := a.attachData(store); !result.OK {
		t.Fatalf("attachData: %v", result.Value)
	}
	a.activePanel = panelData

	model, _ := a.Update(keyMsg("t"))
	a = model.(app)
	if a.activeOverlay != overlayDataNote || a.dataNote == nil {
		t.Fatalf("t did not open the note overlay: overlay=%d note=%v", a.activeOverlay, a.dataNote)
	}
	for _, r := range "hand-picked" {
		model, _ = a.Update(keyMsg(string(r)))
		a = model.(app)
	}
	model, _ = a.Update(keyMsg("enter"))
	a = model.(app)
	if a.activeOverlay != overlayNone || a.dataNote != nil {
		t.Fatalf("enter did not close the note overlay: overlay=%d note=%v", a.activeOverlay, a.dataNote)
	}
	if review := core.MustCast[dataset.Review](store.ReviewLatest(item.ID)); review.Note != "tag: hand-picked" {
		t.Fatalf("tag review = %#v", review)
	}
}

func TestApp_DataPanelHotkeys_QuarantineClearRequiresNote(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	item := seedDataItem(t, store, ds.ID, "p", "r", at)
	if result := store.ReviewAppend(dataset.Review{ItemID: item.ID, Status: dataset.StatusQuarantined, Reviewer: dataset.ReviewerAutoWelfare, CreatedAt: at}); !result.OK {
		t.Fatalf("seed quarantine: %v", result.Value)
	}

	a := newApp("", 0, 64)
	if result := a.attachData(store); !result.OK {
		t.Fatalf("attachData: %v", result.Value)
	}
	a.activePanel = panelData

	model, _ := a.Update(keyMsg("c"))
	a = model.(app)
	if a.activeOverlay != overlayDataNote {
		t.Fatalf("c did not open the note overlay: overlay=%d", a.activeOverlay)
	}
	// An empty Enter must not submit — no write.
	model, _ = a.Update(keyMsg("enter"))
	a = model.(app)
	if a.activeOverlay != overlayDataNote {
		t.Fatalf("empty enter closed the overlay without a note: overlay=%d", a.activeOverlay)
	}
	if review := core.MustCast[dataset.Review](store.ReviewLatest(item.ID)); review.Status != dataset.StatusQuarantined {
		t.Fatalf("status changed despite an empty note: %q", review.Status)
	}
	for _, r := range "false positive" {
		model, _ = a.Update(keyMsg(string(r)))
		a = model.(app)
	}
	model, _ = a.Update(keyMsg("enter"))
	a = model.(app)
	if a.activeOverlay != overlayNone {
		t.Fatalf("non-empty enter did not close the overlay: overlay=%d", a.activeOverlay)
	}
	if review := core.MustCast[dataset.Review](store.ReviewLatest(item.ID)); review.Status != dataset.StatusApproved || review.Note != "false positive" {
		t.Fatalf("cleared review = %#v", review)
	}
}

func TestApp_DataPanelHotkeys_EditAsDerivedFlow(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	original := seedDataItem(t, store, ds.ID, "original", "original response", at)

	a := newApp("", 0, 64)
	if result := a.attachData(store); !result.OK {
		t.Fatalf("attachData: %v", result.Value)
	}
	a.activePanel = panelData

	model, _ := a.Update(keyMsg("e"))
	a = model.(app)
	if a.activeOverlay != overlayDataEditor || a.dataEditor == nil {
		t.Fatalf("e did not open the editor: overlay=%d editor=%v", a.activeOverlay, a.dataEditor)
	}
	a.dataEditor.focus = 1
	a.dataEditor.applyFocus()
	a.dataEditor.response.SetValue("edited response")

	model, _ = a.Update(testModifiedKeyPress('s', tea.ModCtrl))
	a = model.(app)
	if a.activeOverlay != overlayNone || a.dataEditor != nil {
		t.Fatalf("ctrl+s did not close the editor: overlay=%d editor=%v", a.activeOverlay, a.dataEditor)
	}
	if reread := core.MustCast[dataset.Item](store.Item(original.ID)); !reread.Archived {
		t.Fatal("ctrl+s did not archive the original")
	}
}

// TestApp_DataPanelBulkConfirmGate is the explicit "no confirm, no writes"
// proof the task requires, driven end-to-end through real key messages:
// opening the bulk overlay writes nothing; Escape cancels and still
// writes nothing; a single Enter only arms (still nothing written); only
// the SECOND Enter applies the action.
func TestApp_DataPanelBulkConfirmGate(t *testing.T) {
	store := newTestDataPanelStore(t)
	at := time.Now()
	ds := seedDataDataset(t, store, "vents", at)
	itemA := seedDataItem(t, store, ds.ID, "a", "a", at)
	itemB := seedDataItem(t, store, ds.ID, "b", "b", at.Add(time.Second))

	a := newApp("", 0, 64)
	if result := a.attachData(store); !result.OK {
		t.Fatalf("attachData: %v", result.Value)
	}
	a.activePanel = panelData

	model, _ := a.Update(keyMsg("A"))
	a = model.(app)
	if a.activeOverlay != overlayDataBulk || a.dataBulk == nil {
		t.Fatalf("A did not open the bulk overlay: overlay=%d bulk=%v", a.activeOverlay, a.dataBulk)
	}
	assertDataReviewPending(t, store, itemA.ID)
	assertDataReviewPending(t, store, itemB.ID)

	model, _ = a.Update(keyMsg("esc"))
	a = model.(app)
	if a.activeOverlay != overlayNone || a.dataBulk != nil {
		t.Fatalf("esc did not clear the bulk overlay: overlay=%d bulk=%v", a.activeOverlay, a.dataBulk)
	}
	assertDataReviewPending(t, store, itemA.ID)
	assertDataReviewPending(t, store, itemB.ID)

	model, _ = a.Update(keyMsg("A"))
	a = model.(app)
	model, _ = a.Update(keyMsg("enter"))
	a = model.(app)
	if a.activeOverlay != overlayDataBulk || a.dataBulk == nil || !a.dataBulk.armed {
		t.Fatalf("first enter did not arm: overlay=%d bulk=%#v", a.activeOverlay, a.dataBulk)
	}
	assertDataReviewPending(t, store, itemA.ID)
	assertDataReviewPending(t, store, itemB.ID)

	model, _ = a.Update(keyMsg("enter"))
	a = model.(app)
	if a.activeOverlay != overlayNone || a.dataBulk != nil {
		t.Fatalf("second enter did not close the overlay: overlay=%d bulk=%v", a.activeOverlay, a.dataBulk)
	}
	for _, id := range []string{itemA.ID, itemB.ID} {
		if review := core.MustCast[dataset.Review](store.ReviewLatest(id)); review.Status != dataset.StatusApproved {
			t.Fatalf("item %s status = %q after confirm, want approved", id, review.Status)
		}
	}
}

func testNativeAgentApp(t *testing.T, provider agentProvider) app {
	t.Helper()
	repository := openTestDuckRepository(t)
	t.Cleanup(func() { closeTestDuckRepository(t, repository) })
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	a.work.ids = sequenceIDs("work-1")
	if result := a.work.CreateWork("Ship", "Implement the slice", "/src/project"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	return a
}

func testNativeProjectReview(workID string, enableGit bool) orchestrator.ProjectReview {
	return orchestrator.ProjectReview{Work: work.Item{ID: workID, Title: "Ship", Task: "Implement the slice", Repository: "/src/project"}, Source: workspace.SourceReview{Path: "/src/project", Root: "/src/project", Branch: "main", Revision: "abc123", IncludedHash: "hash", Included: []string{"go.mod", "main.go"}}, RepositoryName: workID, RequiresGitEnable: enableGit}
}

func testNativeDispatchReview(project orchestrator.ProjectReview) orchestrator.DispatchReview {
	return orchestrator.DispatchReview{Request: work.DispatchRequest{Work: project.Work, Provider: "codex", Model: "gpt-5", ConfirmedSourceRevision: project.Source.Revision}, Project: work.Project{ID: project.Work.ID, SourcePath: project.Source.Path, SourceRevision: project.Source.Revision, RepositoryName: project.RepositoryName}, Source: project.Source, Command: provider.Command{Provider: "codex", Executable: "codex", Args: []string{"exec", "--api-key", "[REDACTED]", "--model", "gpt-5"}, Receipt: "codex exec --api-key [REDACTED] --model gpt-5"}, WorktreePath: "/private/runs/pending-run/worktree", Warning: "native host access"}
}

type failingProjectNativeEngine struct {
	*fixtureNativeAgentEngine
	failure core.Result
}

func (engine *failingProjectNativeEngine) ReviewProject(context.Context, work.Item) core.Result {
	return engine.failure
}

func TestApp_AgentReviewUsesStageInsteadOfReviewTitle(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	provider := &launchReviewProvider{caps: []agentCapability{{Feature: agentFeatureDispatch, Available: true}}, reviews: []agentReview{{Feature: agentFeatureDispatch, Title: "Localised provider wording", Body: "Source: /tmp/repo", ConfirmRequired: true}}}
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	if result := a.work.CreateWork("Launch", "Run the task", "/tmp/repo"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	if result := a.palette.Invoke(agentCommandID(agentFeatureDispatch), &a); !result.OK {
		t.Fatalf("dispatch: %v", result.Value)
	}
	startAgentProjectReview(t, &a, "codex", "gpt-5")
	if a.activeOverlay != overlayProjectReview {
		t.Fatalf("project review overlay = %d, want %d", a.activeOverlay, overlayProjectReview)
	}
}

func TestApp_AgentReviewSelectsProviderAndModelBeforeProjectReview(t *testing.T) {
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	provider := &launchReviewProvider{caps: []agentCapability{{Feature: agentFeatureDispatch, Available: true}}, reviews: []agentReview{{Feature: agentFeatureDispatch, Title: "Project", ConfirmRequired: true}}}
	a := newApp("", 0, 64)
	if result := a.attachWork(repository, provider); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	if result := a.work.CreateWork("Launch", "Run the task", "/tmp/repo"); !result.OK {
		t.Fatalf("CreateWork: %v", result.Value)
	}
	if result := a.palette.Invoke(agentCommandID(agentFeatureDispatch), &a); !result.OK {
		t.Fatalf("dispatch: %v", result.Value)
	}
	if a.activeOverlay != overlayAgentSelection || a.agentCommand != nil {
		t.Fatalf("selection state = overlay %d command=%v", a.activeOverlay, a.agentCommand != nil)
	}
	a.launchReview.providerInput.SetValue("codex")
	a.launchReview.modelInput.SetValue("gpt-5")
	model, command := a.Update(testKeyPress(tea.KeyEnter))
	a = model.(app)
	if command == nil {
		t.Fatal("provider selection did not schedule a project review")
	}
	model, _ = a.Update(command())
	a = model.(app)
	if len(provider.reviewRequests) != 1 || provider.reviewRequests[0].Provider != "codex" || provider.reviewRequests[0].Model != "gpt-5" || a.activeOverlay != overlayProjectReview {
		t.Fatalf("project review = requests=%#v overlay=%d", provider.reviewRequests, a.activeOverlay)
	}
}

func TestApp_AgentReviewFailureDoesNotRun(t *testing.T) {
	for _, reason := range []string{"source repository is dirty", "source repository is detached", "included file hash changed"} {
		t.Run(reason, func(t *testing.T) {
			repository := openTestDuckRepository(t)
			defer closeTestDuckRepository(t, repository)
			provider := &failingLaunchReviewProvider{reason: reason}
			a := newApp("", 0, 64)
			if result := a.attachWork(repository, provider); !result.OK {
				t.Fatalf("attachWork: %v", result.Value)
			}
			if result := a.work.CreateWork("Launch", "Run the task", "/tmp/repo"); !result.OK {
				t.Fatalf("CreateWork: %v", result.Value)
			}
			if result := a.palette.Invoke(agentCommandID(agentFeatureDispatch), &a); !result.OK {
				t.Fatalf("dispatch: %v", result.Value)
			}
			startAgentProjectReview(t, &a, "codex", "gpt-5")
			if !strings.Contains(a.errText, reason) || provider.runs != 0 || a.activeOverlay != overlayNone {
				t.Fatalf("failure=%q err=%q runs=%d overlay=%d", reason, a.errText, provider.runs, a.activeOverlay)
			}
		})
	}
}

func startAgentProjectReview(t *testing.T, target *app, provider, model string) {
	t.Helper()
	if target.activeOverlay != overlayAgentSelection || target.launchReview == nil {
		t.Fatalf("agent selection = overlay %d review=%v", target.activeOverlay, target.launchReview != nil)
	}
	target.launchReview.providerInput.SetValue(provider)
	target.launchReview.modelInput.SetValue(model)
	next, command := target.Update(testKeyPress(tea.KeyEnter))
	*target = next.(app)
	if command == nil {
		t.Fatal("agent selection did not schedule project review")
	}
	next, _ = target.Update(command())
	*target = next.(app)
}

type failingLaunchReviewProvider struct {
	reason string
	runs   int
}

func (*failingLaunchReviewProvider) Capabilities() []agentCapability {
	return []agentCapability{{Feature: agentFeatureDispatch, Available: true}}
}
func (*failingLaunchReviewProvider) Snapshot(context.Context) core.Result {
	return core.Ok(agentSnapshot{})
}
func (provider *failingLaunchReviewProvider) Review(context.Context, agentReviewRequest) core.Result {
	return core.Fail(core.E("test.review", provider.reason, nil))
}
func (provider *failingLaunchReviewProvider) Run(context.Context, agentRequest) core.Result {
	provider.runs++
	return core.Ok(agentActionReceipt{})
}
func (*failingLaunchReviewProvider) Close() core.Result { return core.Ok(nil) }

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

type correctiveAgentProvider struct {
	mu        sync.Mutex
	caps      []agentCapability
	snapshot  func(context.Context, int) core.Result
	review    func(context.Context, agentReviewRequest) core.Result
	run       func(context.Context, agentRequest) core.Result
	snapshots int
	reviews   int
	runs      int
	closes    int
}

func (provider *correctiveAgentProvider) Capabilities() []agentCapability {
	provider.mu.Lock()
	defer provider.mu.Unlock()
	if provider.caps != nil {
		return append([]agentCapability(nil), provider.caps...)
	}
	capabilities := agentFeatureCatalog("")
	for index := range capabilities {
		capabilities[index].Available = true
	}
	return capabilities
}

func (provider *correctiveAgentProvider) Snapshot(ctx context.Context) core.Result {
	provider.mu.Lock()
	provider.snapshots++
	call := provider.snapshots
	snapshot := provider.snapshot
	provider.mu.Unlock()
	if snapshot == nil {
		return core.Ok(agentSnapshot{})
	}
	return snapshot(ctx, call)
}

func (provider *correctiveAgentProvider) Review(ctx context.Context, request agentReviewRequest) core.Result {
	provider.mu.Lock()
	provider.reviews++
	review := provider.review
	provider.mu.Unlock()
	if review == nil {
		return core.Ok(agentReview{})
	}
	return review(ctx, request)
}

func (provider *correctiveAgentProvider) Run(ctx context.Context, request agentRequest) core.Result {
	provider.mu.Lock()
	provider.runs++
	run := provider.run
	provider.mu.Unlock()
	if run == nil {
		return core.Ok(agentActionReceipt{Feature: request.Feature, WorkID: request.WorkID, RunID: request.RunID})
	}
	return run(ctx, request)
}

func (provider *correctiveAgentProvider) Close() core.Result {
	provider.mu.Lock()
	provider.closes++
	provider.mu.Unlock()
	return core.Ok(nil)
}

func (provider *correctiveAgentProvider) SnapshotCalls() int {
	provider.mu.Lock()
	defer provider.mu.Unlock()
	return provider.snapshots
}

func (provider *correctiveAgentProvider) CallCounts() (int, int, int, int) {
	provider.mu.Lock()
	defer provider.mu.Unlock()
	return provider.snapshots, provider.reviews, provider.runs, provider.closes
}

func correctiveChangeReview(validated bool) workspace.ChangeReview {
	review := workspace.ChangeReview{
		WorkID: "work-1", RunID: "run-reviewed", SourceBranch: "main", SourceRevision: "source-123",
		AgentBase: "source-123", AgentTip: "agent-456", IntegrationBranch: "lem/integration/run-reviewed",
		IntegrationPath: "/private/reviews/run-reviewed", ResultRevision: "result-789",
		Diff: "diff --git a/main.go b/main.go\n+reviewed change", CommitLog: "agent-456 implement reviewed change",
	}
	if validated {
		review.Validation = []workspace.ValidationResult{{
			Command: workspace.Command{Dir: "/src/project", Executable: "go", Args: []string{"test", "./..."}},
			Output:  "ok all packages", Receipt: "validation-receipt", Passed: true,
		}}
	}
	return review
}

func driveCorrectiveCommand(t *testing.T, target *app, command tea.Cmd) tea.Cmd {
	t.Helper()
	if command == nil {
		t.Fatal("expected lifecycle-owned command")
	}
	model, next := target.Update(command())
	*target = model.(app)
	return next
}

func countAgentCall(calls []string, want string) int {
	count := 0
	for _, call := range calls {
		if call == want {
			count++
		}
	}
	return count
}

func TestSelectPanel_Good(t *testing.T) {
	a := newApp("", 0, 64)
	if cmd := a.selectPanel(panelWork); cmd != nil {
		t.Fatal("selecting a non-Models panel must not schedule a command")
	}
	if a.activePanel != panelWork {
		t.Fatalf("activePanel = %v, want work", a.activePanel)
	}
}

func TestSelectPanel_Ugly(t *testing.T) {
	a := newApp("", 0, 64)
	cmd := a.selectPanel(panelModels)
	if a.activePanel != panelModels {
		t.Fatalf("activePanel = %v, want models", a.activePanel)
	}
	if cmd == nil {
		t.Fatal("a first Models visit with an empty picker must schedule discovery")
	}
}

// tabClick builds the left-press message for the centre of one tab's hit
// span — the native tabs-slot box plus the cell walk panelBarHit itself
// uses — offset from strip-local to screen cells by the frame insets.
func tabClick(t *testing.T, a app, panel panelID) tea.MouseClickMsg {
	t.Helper()
	metrics := measureFrame(a.width, a.height, a.inspectorOpen)
	_, boxes := renderPanelBarBoxes(a.activePanel, metrics.innerWidth, metrics.kind, a.styles)
	start, end, ok := panelTabSpan(boxes, a.activePanel, metrics.kind, panel)
	if !ok {
		t.Fatalf("no hit span for panel %v", panel)
	}
	return tea.MouseClickMsg{
		X:      frameInsetCols + (start+end)/2,
		Y:      frameInsetRows + boxes[tabsSlotID].Row,
		Button: tea.MouseLeft,
	}
}

func TestOnMouse_Good(t *testing.T) {
	a := newApp("", 0, 64)
	model, _ := a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a = model.(app)
	model, cmd := a.Update(tabClick(t, a, panelWork))
	a = model.(app)
	if a.activePanel != panelWork {
		t.Fatalf("activePanel = %v, want work", a.activePanel)
	}
	if cmd != nil {
		t.Fatal("switching to Work must not schedule a command")
	}
	model, cmd = a.Update(tabClick(t, a, panelModels))
	a = model.(app)
	if a.activePanel != panelModels {
		t.Fatalf("activePanel = %v, want models", a.activePanel)
	}
	if cmd == nil {
		t.Fatal("a first Models visit through a click must schedule discovery")
	}
}

func TestOnMouse_Bad(t *testing.T) {
	a := newApp("", 0, 64)
	model, _ := a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a = model.(app)
	press := func(x, y int, button tea.MouseButton) {
		t.Helper()
		var message tea.Msg = tea.MouseClickMsg{X: x, Y: y, Button: button}
		if button == tea.MouseWheelUp || button == tea.MouseWheelDown {
			message = tea.MouseWheelMsg{X: x, Y: y, Button: button}
		}
		model, _ = a.Update(message)
		a = model.(app)
		if a.activePanel != panelChat {
			t.Fatalf("press at (%d, %d) switched panel to %v", x, y, a.activePanel)
		}
	}
	press(frameInsetCols+6, 0, tea.MouseLeft)                 // the border row above the strip
	press(frameInsetCols, frameInsetRows, tea.MouseLeft)      // the brand cell resolves to the strip, not a tab
	press(frameInsetCols+8, frameInsetRows, tea.MouseWheelUp) // wheel messages keep their transcript route
}

func TestOnMouse_Ugly(t *testing.T) {
	// Before any WindowSizeMsg the frame has no geometry: no panic, no switch.
	a := newApp("", 0, 64)
	model, _ := a.Update(tea.MouseClickMsg{X: 8, Y: 1, Button: tea.MouseLeft})
	a = model.(app)
	if a.activePanel != panelChat {
		t.Fatalf("sizeless click switched panel to %v", a.activePanel)
	}

	// Under an overlay the bar takes no clicks — the keyboard gate mirrored.
	b := newApp("", 0, 64)
	model, _ = b.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	b = model.(app)
	click := tabClick(t, b, panelWork)
	b.activeOverlay = overlayHelp
	model, _ = b.Update(click)
	b = model.(app)
	if b.activePanel != panelChat {
		t.Fatalf("overlay click switched panel to %v", b.activePanel)
	}
}
