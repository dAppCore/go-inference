package main

import (
	"context"

	core "dappco.re/go"
	"github.com/wailsapp/wails/v3/pkg/application"
)

func ExampleNewTrayService() {
	tray := NewTrayService(nil)

	core.Println(tray.ServiceName())
	// Output:
	// TrayService
}

func ExampleTrayService_SetServices() {
	tray := NewTrayService(nil)
	dashboard := NewDashboardService("", "", "")
	docker := newExampleDockerService()
	contained := NewContainerService("svc", "")
	agent := NewAgentRunner("", "", "", "", "", "")
	tray.SetServices(dashboard, docker, contained, agent)

	core.Println(tray.dashboard == dashboard)
	core.Println(tray.docker == docker)
	core.Println(tray.container == contained)
	core.Println(tray.agent == agent)
	// Output:
	// true
	// true
	// true
	// true
}

func ExampleTrayService_ServiceName() {
	tray := NewTrayService(nil)

	core.Println(tray.ServiceName())
	// Output:
	// TrayService
}

func ExampleTrayService_ServiceStartup() {
	tray := NewTrayService(nil)
	err := tray.ServiceStartup(context.Background(), application.ServiceOptions{})

	core.Println(err.OK)
	// Output:
	// true
}

func ExampleTrayService_ServiceShutdown() {
	tray := NewTrayService(nil)
	err := tray.ServiceShutdown()

	core.Println(err.OK)
	// Output:
	// true
}

func ExampleTrayService_GetSnapshot() {
	tray := NewTrayService(nil)
	snapshot := tray.GetSnapshot()

	core.Println(snapshot.StackRunning)
	core.Println(snapshot.DockerServices)
	// Output:
	// false
	// 0
}

func ExampleTrayService_StartStack() {
	tray := NewTrayService(nil)
	err := tray.StartStack()

	core.Println(!err.OK)
	// Output:
	// true
}

func ExampleTrayService_StopStack() {
	tray := NewTrayService(nil)
	err := tray.StopStack()

	core.Println(!err.OK)
	// Output:
	// true
}

func ExampleTrayService_StartAgent() {
	tray := NewTrayService(nil)
	err := tray.StartAgent()

	core.Println(!err.OK)
	// Output:
	// true
}

func ExampleTrayService_StopAgent() {
	tray := NewTrayService(nil)
	tray.StopAgent()

	core.Println(tray.GetSnapshot().AgentRunning)
	// Output:
	// false
}
