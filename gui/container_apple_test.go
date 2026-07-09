package main

import (
	core "dappco.re/go"
	"dappco.re/go/container"
	"github.com/wailsapp/wails/v3/pkg/application"
)

// --- AX-7 canonical triplets ---
//
// The Apple container CLI is mocked exactly as docker_test mocks `docker`:
// emptying PATH makes go-container's proc.LookPath("container") fail, so the
// provider reports unavailable and every shelling method returns a failed
// Result. No real `container` binary is required to exercise the wiring.

func TestContainer_NewContainerService_Good(t *core.T) {
	service := NewContainerService("lem-contained", "docker.io/library/alpine:latest")
	name := service.ServiceName()
	status := service.GetStatus()

	core.AssertEqual(t, "ContainerService", name)
	core.AssertEqual(t, "", status.Name)
}

func TestContainer_NewContainerService_Bad(t *core.T) {
	service := NewContainerService("", "")
	running := service.IsRunning()

	core.AssertFalse(t, running)
	core.AssertEqual(t, "lem-contained", service.name)
	core.AssertEqual(t, defaultContainedImage, service.image)
}

func TestContainer_NewContainerService_Ugly(t *core.T) {
	service := NewContainerService("svc", "docker.io/library/busybox:latest")
	service.container = ContainerStatus{Running: true, Name: "svc"}
	status := service.GetStatus()

	core.AssertTrue(t, status.Running)
	core.AssertEqual(t, "svc", status.Name)
}

func TestContainer_ContainerService_ServiceName_Good(t *core.T) {
	service := &ContainerService{}
	got := service.ServiceName()
	want := "ContainerService"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestContainer_ContainerService_ServiceName_Bad(t *core.T) {
	var service *ContainerService
	got := service.ServiceName()
	want := "ContainerService"

	core.AssertEqual(t, want, got)
	core.AssertNotEqual(t, "", got)
}

func TestContainer_ContainerService_ServiceName_Ugly(t *core.T) {
	service := NewContainerService("svc", "")
	first := service.ServiceName()
	second := service.ServiceName()

	core.AssertEqual(t, first, second)
	core.AssertEqual(t, "ContainerService", first)
}

func TestContainer_ContainerService_ServiceStartup_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("svc", "")
	ctx, cancel := core.WithCancel(core.Background())
	cancel()

	err := service.ServiceStartup(ctx, application.ServiceOptions{})
	core.AssertTrue(t, err.OK)
	core.AssertEqual(t, "ContainerService", service.ServiceName())
}

func TestContainer_ContainerService_ServiceStartup_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("", "")
	ctx, cancel := core.WithCancel(core.Background())
	cancel()

	err := service.ServiceStartup(ctx, application.ServiceOptions{})
	core.AssertTrue(t, err.OK)
	core.AssertFalse(t, service.IsRunning())
}

func TestContainer_ContainerService_ServiceStartup_Ugly(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("svc", "")
	err := service.ServiceStartup(core.Background(), application.ServiceOptions{})
	status := service.GetStatus()

	core.AssertTrue(t, err.OK)
	core.AssertFalse(t, status.Running)
}

func TestContainer_ContainerService_Available_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("svc", "")

	core.AssertFalse(t, service.Available())
}

func TestContainer_ContainerService_Available_Bad(t *core.T) {
	var service *ContainerService

	core.AssertFalse(t, service.Available())
}

func TestContainer_ContainerService_Available_Ugly(t *core.T) {
	service := &ContainerService{}

	core.AssertFalse(t, service.Available())
}

func TestContainer_ContainerService_Start_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("svc", "")
	err := service.Start()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "apple container")
}

func TestContainer_ContainerService_Start_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("", "")
	err := service.Start()

	core.AssertError(t, err)
	core.AssertFalse(t, service.IsRunning())
}

func TestContainer_ContainerService_Start_Ugly(t *core.T) {
	service := &ContainerService{}
	err := service.Start()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "provider not available")
}

func TestContainer_ContainerService_Stop_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("svc", "")
	err := service.Stop()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "container")
}

func TestContainer_ContainerService_Stop_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("", "")
	err := service.Stop()

	core.AssertError(t, err)
	core.AssertFalse(t, service.IsRunning())
}

func TestContainer_ContainerService_Stop_Ugly(t *core.T) {
	service := &ContainerService{}
	err := service.Stop()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "provider not available")
}

func TestContainer_ContainerService_Restart_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("svc", "")
	err := service.Restart()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "container")
}

func TestContainer_ContainerService_Restart_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("", "")
	err := service.Restart()

	core.AssertError(t, err)
	core.AssertFalse(t, service.IsRunning())
}

func TestContainer_ContainerService_Restart_Ugly(t *core.T) {
	service := &ContainerService{}
	err := service.Restart()

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "provider not available")
}

func TestContainer_ContainerService_Logs_Good(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("svc", "")
	err := service.Logs(10)

	core.AssertError(t, err)
}

func TestContainer_ContainerService_Logs_Bad(t *core.T) {
	t.Setenv("PATH", "")
	service := NewContainerService("", "")
	err := service.Logs(0)

	core.AssertError(t, err)
}

func TestContainer_ContainerService_Logs_Ugly(t *core.T) {
	service := &ContainerService{}
	err := service.Logs(-1)

	core.AssertError(t, err)
	core.AssertContains(t, core.ErrorMessage(err), "provider not available")
}

func TestContainer_ContainerService_GetStatus_Good(t *core.T) {
	service := NewContainerService("svc", "")
	service.container = ContainerStatus{Running: true, Name: "svc"}
	status := service.GetStatus()

	core.AssertTrue(t, status.Running)
	core.AssertEqual(t, "svc", status.Name)
}

func TestContainer_ContainerService_GetStatus_Bad(t *core.T) {
	service := NewContainerService("svc", "")
	status := service.GetStatus()

	core.AssertFalse(t, status.Running)
	core.AssertEqual(t, "", status.Name)
}

func TestContainer_ContainerService_GetStatus_Ugly(t *core.T) {
	service := NewContainerService("", "")
	service.container = ContainerStatus{Running: false}
	status := service.GetStatus()

	core.AssertFalse(t, status.Running)
	core.AssertEqual(t, "", status.Status)
}

func TestContainer_ContainerService_IsRunning_Good(t *core.T) {
	service := NewContainerService("svc", "")
	service.container = ContainerStatus{Running: true}
	got := service.IsRunning()

	core.AssertTrue(t, got)
	core.AssertEqual(t, true, got)
}

func TestContainer_ContainerService_IsRunning_Bad(t *core.T) {
	service := NewContainerService("svc", "")
	service.container = ContainerStatus{Running: false}
	got := service.IsRunning()

	core.AssertFalse(t, got)
	core.AssertEqual(t, false, got)
}

func TestContainer_ContainerService_IsRunning_Ugly(t *core.T) {
	service := NewContainerService("svc", "")
	got := service.IsRunning()

	core.AssertFalse(t, got)
	core.AssertEqual(t, false, got)
}

// --- status-parse coverage: containerStatusFrom maps a go-container Container
// onto the ContainerStatus shape the frontend already consumes. ---

func TestContainer_containerStatusFrom_Good(t *core.T) {
	ctr := &container.Container{
		Name:   "svc",
		Image:  "docker.io/library/alpine:latest",
		Status: container.StatusRunning,
		Ports:  map[int]int{8080: 80},
	}
	status := containerStatusFrom(ctr)

	core.AssertEqual(t, "svc", status.Name)
	core.AssertTrue(t, status.Running)
}

func TestContainer_containerStatusFrom_Bad(t *core.T) {
	ctr := &container.Container{
		Name:   "svc",
		Status: container.StatusStopped,
	}
	status := containerStatusFrom(ctr)

	core.AssertFalse(t, status.Running)
	core.AssertEqual(t, "stopped", status.Status)
}

func TestContainer_containerStatusFrom_Ugly(t *core.T) {
	ctr := &container.Container{
		Name:   "svc",
		Image:  "img",
		Status: container.StatusRunning,
		Ports:  map[int]int{9100: 9100},
	}
	status := containerStatusFrom(ctr)

	core.AssertEqual(t, "9100:9100", status.Ports)
	core.AssertEqual(t, "img", status.Image)
}
