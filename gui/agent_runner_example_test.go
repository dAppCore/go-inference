package main

import (
	"context"

	core "dappco.re/go"
	"github.com/wailsapp/wails/v3/pkg/application"
)

func ExampleNewAgentRunner() {
	runner := NewAgentRunner("http://api", "http://influx", "lem", "127.0.0.1:9000", "base", "/tmp")

	core.Println(runner.ServiceName())
	// Output:
	// AgentRunner
}

func ExampleAgentRunner_ServiceName() {
	runner := NewAgentRunner("", "", "", "", "", "")

	core.Println(runner.ServiceName())
	// Output:
	// AgentRunner
}

func ExampleAgentRunner_ServiceStartup() {
	runner := NewAgentRunner("", "", "", "", "", "")
	err := runner.ServiceStartup(context.Background(), application.ServiceOptions{})

	core.Println(err.OK)
	// Output:
	// true
}

func ExampleAgentRunner_IsRunning() {
	runner := NewAgentRunner("", "", "", "", "", "")

	core.Println(runner.IsRunning())
	// Output:
	// false
}

func ExampleAgentRunner_CurrentTask() {
	runner := NewAgentRunner("", "", "", "", "", "")

	core.Println(runner.CurrentTask() == "")
	// Output:
	// true
}

func ExampleAgentRunner_Start() {
	runner := NewAgentRunner("", "", "", "", "", "")
	runner.running = true
	err := runner.Start()

	core.Println(err.OK)
	core.Println(runner.IsRunning())
	// Output:
	// true
	// true
}

func ExampleAgentRunner_Stop() {
	runner := NewAgentRunner("", "", "", "", "", "")
	runner.running = true
	runner.task = "scoring"
	runner.Stop()

	core.Println(runner.IsRunning())
	core.Println(runner.CurrentTask() == "")
	// Output:
	// false
	// true
}
