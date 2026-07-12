package main

import (
	core "dappco.re/go"
	"github.com/wailsapp/wails/v3/pkg/application"
)

// --- AX-7 canonical triplets ---

func TestTray_NewTrayService_Good(t *core.T) {
	service := NewTrayService(nil)
	name := service.ServiceName()
	snapshot := service.GetSnapshot()

	core.AssertEqual(t, "TrayService", name)
	core.AssertEqual(t, TraySnapshot{}, snapshot)
}

func TestTray_NewTrayService_Bad(t *core.T) {
	service := NewTrayService(nil)
	err := service.StartStack()
	got := core.ErrorMessage(err)

	core.AssertError(t, err)
	core.AssertContains(t, got, "docker service")
}

func TestTray_NewTrayService_Ugly(t *core.T) {
	service := NewTrayService(nil)
	service.SetServices(&DashboardService{}, &DockerService{services: map[string]ContainerStatus{}}, nil, &AgentRunner{})
	snapshot := service.GetSnapshot()

	core.AssertEqual(t, "TrayService", service.ServiceName())
	core.AssertFalse(t, snapshot.StackRunning)
}

func TestTray_TrayService_SetServices_Good(t *core.T) {
	tray := NewTrayService(nil)
	dashboard := &DashboardService{}
	docker := NewDockerService("/tmp/deploy")
	contained := NewContainerService("svc", "")

	tray.SetServices(dashboard, docker, contained, &AgentRunner{})
	core.AssertNotNil(t, tray.dashboard)
	core.AssertNotNil(t, tray.docker)
	core.AssertNotNil(t, tray.container)
}

func TestTray_TrayService_SetServices_Bad(t *core.T) {
	tray := NewTrayService(nil)
	tray.SetServices(nil, nil, nil, nil)
	snapshot := tray.GetSnapshot()

	core.AssertNil(t, tray.dashboard)
	core.AssertEqual(t, TraySnapshot{}, snapshot)
}

func TestTray_TrayService_SetServices_Ugly(t *core.T) {
	tray := NewTrayService(nil)
	tray.SetServices(&DashboardService{dbPath: "db"}, NewDockerService("/tmp/deploy"), nil, &AgentRunner{task: "queued"})
	snapshot := tray.GetSnapshot()

	core.AssertEqual(t, "queued", snapshot.AgentTask)
	core.AssertEqual(t, "db", tray.dashboard.dbPath)
}

func TestTray_TrayService_ServiceName_Good(t *core.T) {
	tray := &TrayService{}
	got := tray.ServiceName()
	want := "TrayService"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestTray_TrayService_ServiceName_Bad(t *core.T) {
	var tray *TrayService
	got := tray.ServiceName()
	want := "TrayService"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestTray_TrayService_ServiceName_Ugly(t *core.T) {
	tray := NewTrayService(nil)
	first := tray.ServiceName()
	second := tray.ServiceName()

	core.AssertEqual(t, first, second)
	core.AssertEqual(t, "TrayService", first)
}

func TestTray_TrayService_ServiceStartup_Good(t *core.T) {
	tray := &TrayService{}
	err := tray.ServiceStartup(core.Background(), application.ServiceOptions{})
	got := tray.ServiceName()

	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, "TrayService", got)
}

func TestTray_TrayService_ServiceStartup_Bad(t *core.T) {
	tray := &TrayService{}
	ctx, cancel := core.WithCancel(core.Background())
	cancel()

	err := tray.ServiceStartup(ctx, application.ServiceOptions{})
	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, TraySnapshot{}, tray.GetSnapshot())
}

func TestTray_TrayService_ServiceStartup_Ugly(t *core.T) {
	var tray TrayService
	err := tray.ServiceStartup(core.Background(), application.ServiceOptions{})
	snapshot := tray.GetSnapshot()

	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, TraySnapshot{}, snapshot)
}

func TestTray_TrayService_ServiceShutdown_Good(t *core.T) {
	tray := &TrayService{}
	err := tray.ServiceShutdown()
	got := tray.ServiceName()

	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, "TrayService", got)
}

func TestTray_TrayService_ServiceShutdown_Bad(t *core.T) {
	var tray TrayService
	err := tray.ServiceShutdown()
	snapshot := tray.GetSnapshot()

	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, TraySnapshot{}, snapshot)
}

func TestTray_TrayService_ServiceShutdown_Ugly(t *core.T) {
	tray := NewTrayService(nil)
	err := tray.ServiceShutdown()
	got := tray.GetSnapshot()

	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, TraySnapshot{}, got)
}

func TestTray_TrayService_GetSnapshot_Good(t *core.T) {
	tray := NewTrayService(nil)
	tray.SetServices(&DashboardService{modelInventory: []ModelInfo{{Name: "m"}}}, NewDockerService("/tmp/deploy"), nil, &AgentRunner{task: "queued"})
	snapshot := tray.GetSnapshot()

	core.AssertLen(t, snapshot.Models, 1)
	core.AssertEqual(t, "queued", snapshot.AgentTask)
}

func TestTray_TrayService_GetSnapshot_Bad(t *core.T) {
	tray := NewTrayService(nil)
	snapshot := tray.GetSnapshot()
	got := snapshot.DockerServices

	core.AssertEqual(t, 0, got)
	core.AssertFalse(t, snapshot.StackRunning)
}

func TestTray_TrayService_GetSnapshot_Ugly(t *core.T) {
	docker := NewDockerService("/tmp/deploy")
	docker.services["db"] = ContainerStatus{Running: true}
	tray := NewTrayService(nil)
	tray.SetServices(nil, docker, nil, &AgentRunner{running: true})

	snapshot := tray.GetSnapshot()
	core.AssertTrue(t, snapshot.StackRunning)
	core.AssertTrue(t, snapshot.AgentRunning)
}

func TestTray_TrayService_StartStack_Good(t *core.T) {
	tray := NewTrayService(nil)
	err := tray.StartStack()
	got := core.ErrorMessage(err)

	core.AssertError(t, err)
	core.AssertContains(t, got, "docker service")
}

func TestTray_TrayService_StartStack_Bad(t *core.T) {
	t.Setenv("PATH", "")
	tray := NewTrayService(nil)
	tray.SetServices(nil, NewDockerService(t.TempDir()), nil, nil)

	err := tray.StartStack()
	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestTray_TrayService_StartStack_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	tray := NewTrayService(nil)
	tray.SetServices(nil, &DockerService{}, nil, nil)

	err := tray.StartStack()
	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestTray_TrayService_StopStack_Good(t *core.T) {
	tray := NewTrayService(nil)
	err := tray.StopStack()
	got := core.ErrorMessage(err)

	core.AssertError(t, err)
	core.AssertContains(t, got, "docker service")
}

func TestTray_TrayService_StopStack_Bad(t *core.T) {
	t.Setenv("PATH", "")
	tray := NewTrayService(nil)
	tray.SetServices(nil, NewDockerService(t.TempDir()), nil, nil)

	err := tray.StopStack()
	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestTray_TrayService_StopStack_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	tray := NewTrayService(nil)
	tray.SetServices(nil, &DockerService{}, nil, nil)

	err := tray.StopStack()
	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestTray_TrayService_StartAgent_Good(t *core.T) {
	tray := NewTrayService(nil)
	err := tray.StartAgent()
	got := core.ErrorMessage(err)

	core.AssertError(t, err)
	core.AssertContains(t, got, "agent service")
}

func TestTray_TrayService_StartAgent_Bad(t *core.T) {
	tray := NewTrayService(nil)
	tray.SetServices(nil, nil, nil, &AgentRunner{running: true})
	err := tray.StartAgent()

	core.AssertTrue(t, err.OK)
	core.AssertTrue(t, tray.agent.IsRunning())
}

func TestTray_TrayService_StartAgent_Ugly(t *core.T) {
	tray := NewTrayService(nil)
	tray.SetServices(nil, nil, nil, &AgentRunner{running: true, task: "queued"})
	err := tray.StartAgent()

	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, "queued", tray.agent.CurrentTask())
}

func TestTray_TrayService_StopAgent_Good(t *core.T) {
	tray := NewTrayService(nil)
	tray.SetServices(nil, nil, nil, &AgentRunner{running: true, task: "queued"})
	tray.StopAgent()

	core.AssertFalse(t, tray.agent.IsRunning())
	core.AssertEqual(t, "", tray.agent.CurrentTask())
}

func TestTray_TrayService_StopAgent_Bad(t *core.T) {
	tray := NewTrayService(nil)
	tray.StopAgent()
	snapshot := tray.GetSnapshot()

	core.AssertEqual(t, TraySnapshot{}, snapshot)
	core.AssertNil(t, tray.agent)
}

func TestTray_TrayService_StopAgent_Ugly(t *core.T) {
	tray := NewTrayService(nil)
	tray.SetServices(nil, nil, nil, &AgentRunner{})
	tray.StopAgent()

	core.AssertFalse(t, tray.agent.IsRunning())
	core.AssertEqual(t, "", tray.agent.CurrentTask())
}

// --- serve menu label rendering ---

func TestTray_serveStatusLabel_Good(t *core.T) {
	got := serveStatusLabel(ServeSnapshot{Up: true, ModelName: "gemma-4-e2b-it-4bit"})
	core.AssertEqual(t, "Serve: gemma-4-e2b-it-4bit (running)", got)
}

func TestTray_serveStatusLabel_Bad(t *core.T) {
	got := serveStatusLabel(ServeSnapshot{})
	core.AssertEqual(t, "Serve: stopped", got)
}

func TestTray_serveStatusLabel_Ugly(t *core.T) {
	got := serveStatusLabel(ServeSnapshot{Managed: true})
	core.AssertEqual(t, "Serve: starting…", got)
}

func TestTray_serveStatusLabel_ModelLess(t *core.T) {
	got := serveStatusLabel(ServeSnapshot{Up: true})
	core.AssertEqual(t, "Serve: running (no model)", got)
}

func TestTray_serveToggleLabel_Good(t *core.T) {
	got := serveToggleLabel(ServeSnapshot{})
	core.AssertEqual(t, "Start Serve", got)
}

func TestTray_serveToggleLabel_Bad(t *core.T) {
	got := serveToggleLabel(ServeSnapshot{Up: true})
	core.AssertEqual(t, "Stop Serve", got)
}

func TestTray_serveToggleLabel_Ugly(t *core.T) {
	got := serveToggleLabel(ServeSnapshot{Managed: true})
	core.AssertEqual(t, "Stop Serve", got)
}
