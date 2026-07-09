package main

import (
	core "dappco.re/go"
	"github.com/wailsapp/wails/v3/pkg/application"
)

// --- AX-7 canonical triplets ---

func TestDocker_NewDockerService_Good(t *core.T) {
	service := NewDockerService("/tmp/deploy")
	name := service.ServiceName()
	status := service.GetStatus()

	core.AssertEqual(t, "DockerService", name)
	core.AssertEqual(t, "/tmp/deploy", status.ComposeDir)
}

func TestDocker_NewDockerService_Bad(t *core.T) {
	service := NewDockerService("")
	status := service.GetStatus()
	running := service.IsRunning()

	core.AssertFalse(t, running)
	core.AssertNotEqual(t, "", status.ComposeDir)
}

func TestDocker_NewDockerService_Ugly(t *core.T) {
	service := NewDockerService("/tmp/deploy")
	service.services["db"] = ContainerStatus{Running: true}
	status := service.GetStatus()

	core.AssertTrue(t, status.Running)
	core.AssertLen(t, status.Services, 1)
}

func TestDocker_DockerService_ServiceName_Good(t *core.T) {
	service := &DockerService{}
	got := service.ServiceName()
	want := "DockerService"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestDocker_DockerService_ServiceName_Bad(t *core.T) {
	var service *DockerService
	got := service.ServiceName()
	want := "DockerService"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestDocker_DockerService_ServiceName_Ugly(t *core.T) {
	service := NewDockerService("/tmp/deploy")
	first := service.ServiceName()
	second := service.ServiceName()

	core.AssertEqual(t, first, second)
	core.AssertEqual(t, "DockerService", first)
}

func TestDocker_DockerService_ServiceStartup_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService(t.TempDir())
	ctx, cancel := core.WithCancel(core.Background())
	cancel()

	err := service.ServiceStartup(ctx, application.ServiceOptions{})
	core.AssertNoError(t, err)
	core.AssertEqual(t, "DockerService", service.ServiceName())
}

func TestDocker_DockerService_ServiceStartup_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService("")
	ctx, cancel := core.WithCancel(core.Background())
	cancel()

	err := service.ServiceStartup(ctx, application.ServiceOptions{})
	core.AssertNoError(t, err)
	core.AssertFalse(t, service.IsRunning())
}

func TestDocker_DockerService_ServiceStartup_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService(t.TempDir())
	err := service.ServiceStartup(core.Background(), application.ServiceOptions{})
	status := service.GetStatus()

	core.AssertNoError(t, err)
	core.AssertFalse(t, status.Running)
}

func TestDocker_DockerService_Start_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService(t.TempDir())
	err := service.Start()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_Start_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService("")
	err := service.Start()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_Start_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	service := &DockerService{}
	err := service.Start()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_Stop_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService(t.TempDir())
	err := service.Stop()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_Stop_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService("")
	err := service.Stop()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_Stop_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	service := &DockerService{}
	err := service.Stop()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_Restart_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService(t.TempDir())
	err := service.Restart()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_Restart_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService("")
	err := service.Restart()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_Restart_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	service := &DockerService{}
	err := service.Restart()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_StartService_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService(t.TempDir())
	err := service.StartService("db")

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_StartService_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService("")
	err := service.StartService("")

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_StartService_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	service := &DockerService{}
	err := service.StartService("db")

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_StopService_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService(t.TempDir())
	err := service.StopService("db")

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_StopService_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService("")
	err := service.StopService("")

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_StopService_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	service := &DockerService{}
	err := service.StopService("db")

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_RestartService_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService(t.TempDir())
	err := service.RestartService("db")

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_RestartService_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService("")
	err := service.RestartService("")

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_RestartService_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	service := &DockerService{}
	err := service.RestartService("db")

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_Logs_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService(t.TempDir())
	logs, err := service.Logs("db", 10)

	core.AssertEqual(t, "", logs)
	core.AssertError(t, err)
}

func TestDocker_DockerService_Logs_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService("")
	logs, err := service.Logs("", 0)

	core.AssertEqual(t, "", logs)
	core.AssertError(t, err)
}

func TestDocker_DockerService_Logs_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	service := &DockerService{}
	logs, err := service.Logs("db", -1)

	core.AssertEqual(t, "", logs)
	core.AssertError(t, err)
}

func TestDocker_DockerService_GetStatus_Good(t *core.T) {
	service := NewDockerService("/tmp/deploy")
	service.services["db"] = ContainerStatus{Running: true}
	status := service.GetStatus()

	core.AssertTrue(t, status.Running)
	core.AssertLen(t, status.Services, 1)
}

func TestDocker_DockerService_GetStatus_Bad(t *core.T) {
	service := NewDockerService("/tmp/deploy")
	status := service.GetStatus()
	got := status.Running

	core.AssertFalse(t, got)
	core.AssertEmpty(t, status.Services)
}

func TestDocker_DockerService_GetStatus_Ugly(t *core.T) {
	service := NewDockerService("")
	service.services["db"] = ContainerStatus{Running: false}
	status := service.GetStatus()

	core.AssertFalse(t, status.Running)
	core.AssertLen(t, status.Services, 1)
}

func TestDocker_DockerService_IsRunning_Good(t *core.T) {
	service := NewDockerService("/tmp/deploy")
	service.services["db"] = ContainerStatus{Running: true}
	got := service.IsRunning()

	core.AssertTrue(t, got)
	core.AssertEqual(t, true, got)
}

func TestDocker_DockerService_IsRunning_Bad(t *core.T) {
	service := NewDockerService("/tmp/deploy")
	service.services["db"] = ContainerStatus{Running: false}
	got := service.IsRunning()

	core.AssertFalse(t, got)
	core.AssertEqual(t, false, got)
}

func TestDocker_DockerService_IsRunning_Ugly(t *core.T) {
	service := NewDockerService("/tmp/deploy")
	service.services["db"] = ContainerStatus{Running: false}
	service.services["api"] = ContainerStatus{Running: true}

	core.AssertTrue(t, service.IsRunning())
	core.AssertLen(t, service.services, 2)
}

func TestDocker_DockerService_Pull_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService(t.TempDir())
	err := service.Pull()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_Pull_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewDockerService("")
	err := service.Pull()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}

func TestDocker_DockerService_Pull_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	service := &DockerService{}
	err := service.Pull()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "docker")
}
