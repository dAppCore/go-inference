package main

import (
	core "dappco.re/go"
	"github.com/wailsapp/wails/v3/pkg/application"
)

// --- AX-7 canonical triplets ---

func TestAgentRunner_NewAgentRunner_Good(t *core.T) {
	runner := NewAgentRunner("api", "influx", "db", "m3", "model", "work")
	name := runner.ServiceName()
	running := runner.IsRunning()

	core.AssertEqual(t, "AgentRunner", name)
	core.AssertFalse(t, running)
}

func TestAgentRunner_NewAgentRunner_Bad(t *core.T) {
	runner := NewAgentRunner("", "", "", "", "", "")
	task := runner.CurrentTask()
	running := runner.IsRunning()

	core.AssertEqual(t, "", task)
	core.AssertFalse(t, running)
}

func TestAgentRunner_NewAgentRunner_Ugly(t *core.T) {
	runner := NewAgentRunner("api", "influx", "db", "m3", "model", "work")
	runner.running = true
	running := runner.IsRunning()

	core.AssertTrue(t, running)
	core.AssertEqual(t, "AgentRunner", runner.ServiceName())
}

func TestAgentRunner_AgentRunner_ServiceName_Good(t *core.T) {
	runner := &AgentRunner{}
	got := runner.ServiceName()
	want := "AgentRunner"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestAgentRunner_AgentRunner_ServiceName_Bad(t *core.T) {
	var runner *AgentRunner
	got := runner.ServiceName()
	want := "AgentRunner"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestAgentRunner_AgentRunner_ServiceName_Ugly(t *core.T) {
	runner := NewAgentRunner("", "", "", "", "", "")
	first := runner.ServiceName()
	second := runner.ServiceName()

	core.AssertEqual(t, first, second)
	core.AssertEqual(t, "AgentRunner", first)
}

func TestAgentRunner_AgentRunner_ServiceStartup_Good(t *core.T) {
	runner := &AgentRunner{}
	err := runner.ServiceStartup(core.Background(), application.ServiceOptions{})
	got := runner.ServiceName()

	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, "AgentRunner", got)
}

func TestAgentRunner_AgentRunner_ServiceStartup_Bad(t *core.T) {
	runner := &AgentRunner{}
	ctx, cancel := core.WithCancel(core.Background())
	cancel()

	err := runner.ServiceStartup(ctx, application.ServiceOptions{})
	core.AssertTrue(t, err.OK)
	core.AssertFalse(t, runner.IsRunning())
}

func TestAgentRunner_AgentRunner_ServiceStartup_Ugly(t *core.T) {
	var runner AgentRunner
	err := runner.ServiceStartup(core.Background(), application.ServiceOptions{})
	task := runner.CurrentTask()

	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, "", task)
}

func TestAgentRunner_AgentRunner_IsRunning_Good(t *core.T) {
	runner := &AgentRunner{running: true}
	got := runner.IsRunning()
	want := true

	core.AssertEqual(t, want, got)
	core.AssertTrue(t, got)
}

func TestAgentRunner_AgentRunner_IsRunning_Bad(t *core.T) {
	runner := &AgentRunner{}
	got := runner.IsRunning()
	want := false

	core.AssertEqual(t, want, got)
	core.AssertFalse(t, got)
}

func TestAgentRunner_AgentRunner_IsRunning_Ugly(t *core.T) {
	runner := &AgentRunner{running: true}
	runner.Stop()
	got := runner.IsRunning()

	core.AssertFalse(t, got)
	core.AssertEqual(t, "", runner.CurrentTask())
}

func TestAgentRunner_AgentRunner_CurrentTask_Good(t *core.T) {
	runner := &AgentRunner{task: "scoring"}
	got := runner.CurrentTask()
	want := "scoring"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestAgentRunner_AgentRunner_CurrentTask_Bad(t *core.T) {
	runner := &AgentRunner{}
	got := runner.CurrentTask()
	want := ""

	core.AssertEqual(t, want, got)
	core.AssertEmpty(t, got)
}

func TestAgentRunner_AgentRunner_CurrentTask_Ugly(t *core.T) {
	runner := &AgentRunner{running: true, task: "stopping"}
	runner.Stop()
	got := runner.CurrentTask()

	core.AssertEqual(t, "", got)
	core.AssertFalse(t, runner.IsRunning())
}

func TestAgentRunner_AgentRunner_Start_Good(t *core.T) {
	runner := &AgentRunner{running: true}
	err := runner.Start()
	running := runner.IsRunning()

	core.AssertTrue(t, err.OK)
	core.AssertTrue(t, running)
}

func TestAgentRunner_AgentRunner_Start_Bad(t *core.T) {
	runner := &AgentRunner{running: true, task: "already running"}
	err := runner.Start()
	task := runner.CurrentTask()

	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, "already running", task)
}

func TestAgentRunner_AgentRunner_Start_Ugly(t *core.T) {
	runner := NewAgentRunner("", "", "", "", "", "")
	runner.running = true
	err := runner.Start()

	core.AssertTrue(t, err.OK)
	core.AssertTrue(t, runner.IsRunning())
}

func TestAgentRunner_AgentRunner_Stop_Good(t *core.T) {
	_, cancel := core.WithCancel(core.Background())
	runner := &AgentRunner{running: true, task: "scoring", cancel: cancel}
	runner.Stop()

	core.AssertFalse(t, runner.IsRunning())
	core.AssertEqual(t, "", runner.CurrentTask())
}

func TestAgentRunner_AgentRunner_Stop_Bad(t *core.T) {
	runner := &AgentRunner{}
	runner.Stop()
	got := runner.IsRunning()

	core.AssertFalse(t, got)
	core.AssertEqual(t, "", runner.CurrentTask())
}

func TestAgentRunner_AgentRunner_Stop_Ugly(t *core.T) {
	runner := &AgentRunner{running: true, task: "queued"}
	runner.Stop()
	got := runner.CurrentTask()

	core.AssertEqual(t, "", got)
	core.AssertFalse(t, runner.IsRunning())
}
