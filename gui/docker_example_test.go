package main

import (
	"context"
	"time"

	. "dappco.re/go"
	"github.com/wailsapp/wails/v3/pkg/application"
)

func withDockerExamplePath(fn func()) {
	previous := Getenv("PATH")
	Setenv("PATH", "")
	defer Setenv("PATH", previous)
	fn()
}

func newExampleDockerService() *DockerService {
	return NewDockerService(PathJoin(TempDir(), "lem-missing-compose"))
}

func ExampleNewDockerService() {
	service := newExampleDockerService()

	Println(service.ServiceName())
	// Output:
	// DockerService
}

func ExampleDockerService_ServiceName() {
	service := newExampleDockerService()

	Println(service.ServiceName())
	// Output:
	// DockerService
}

func ExampleDockerService_ServiceStartup() {
	withDockerExamplePath(func() {
		ctx, cancel := context.WithCancel(context.Background())
		service := newExampleDockerService()
		err := service.ServiceStartup(ctx, application.ServiceOptions{})
		cancel()
		time.Sleep(10 * time.Millisecond)

		Println(err.OK)
	})
	// Output:
	// true
}

func ExampleDockerService_Start() {
	withDockerExamplePath(func() {
		service := newExampleDockerService()
		err := service.Start()

		Println(!err.OK)
	})
	// Output:
	// true
}

func ExampleDockerService_Stop() {
	withDockerExamplePath(func() {
		service := newExampleDockerService()
		err := service.Stop()

		Println(!err.OK)
	})
	// Output:
	// true
}

func ExampleDockerService_Restart() {
	withDockerExamplePath(func() {
		service := newExampleDockerService()
		err := service.Restart()

		Println(!err.OK)
	})
	// Output:
	// true
}

func ExampleDockerService_StartService() {
	withDockerExamplePath(func() {
		service := newExampleDockerService()
		err := service.StartService("influxdb")

		Println(!err.OK)
	})
	// Output:
	// true
}

func ExampleDockerService_StopService() {
	withDockerExamplePath(func() {
		service := newExampleDockerService()
		err := service.StopService("influxdb")

		Println(!err.OK)
	})
	// Output:
	// true
}

func ExampleDockerService_RestartService() {
	withDockerExamplePath(func() {
		service := newExampleDockerService()
		err := service.RestartService("influxdb")

		Println(!err.OK)
	})
	// Output:
	// true
}

func ExampleDockerService_Logs() {
	withDockerExamplePath(func() {
		service := newExampleDockerService()
		_, err := service.Logs("influxdb", 5)

		Println(!err.OK)
	})
	// Output:
	// true
}

func ExampleDockerService_GetStatus() {
	service := newExampleDockerService()
	status := service.GetStatus()

	Println(status.Running)
	Println(len(status.Services))
	// Output:
	// false
	// 0
}

func ExampleDockerService_IsRunning() {
	service := newExampleDockerService()

	Println(service.IsRunning())
	// Output:
	// false
}

func ExampleDockerService_Pull() {
	withDockerExamplePath(func() {
		service := newExampleDockerService()
		err := service.Pull()

		Println(!err.OK)
	})
	// Output:
	// true
}
