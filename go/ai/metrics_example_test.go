package ai

import (
	"time"

	. "dappco.re/go"
)

func withMetricsExampleHome(fn func()) {
	previousCoreHome := Getenv("CORE_HOME")
	previousHome := Getenv("HOME")
	previousDirHome := Getenv("DIR_HOME")
	tempHomeResult := MkdirTemp("", "ai-metrics-example-*")
	if !tempHomeResult.OK {
		Println(false)
		return
	}
	tempHome := tempHomeResult.Value.(string)
	defer RemoveAll(tempHome)
	defer Setenv("DIR_HOME", previousDirHome)
	defer Setenv("HOME", previousHome)
	defer Setenv("CORE_HOME", previousCoreHome)

	Setenv("CORE_HOME", "")
	Setenv("DIR_HOME", "")
	Setenv("HOME", tempHome)
	fn()
}

func ExampleRecord() {
	withMetricsExampleHome(func() {
		result := Record(Event{Type: "security.scan", Repo: "core/the inference stack"})

		Println(result.OK)
	})
	// Output:
	// true
}

func ExampleReadEvents() {
	withMetricsExampleHome(func() {
		now := time.Date(2026, 4, 29, 12, 0, 0, 0, time.UTC)
		result := Record(Event{Type: "security.scan", Timestamp: now})
		readResult := ReadEvents(now.Add(-time.Hour))
		events := readResult.Value.([]Event)

		Println(result.OK)
		Println(readResult.OK)
		Println(len(events))
	})
	// Output:
	// true
	// true
	// 1
}

func ExampleSummary() {
	summary := Summary([]Event{{Type: "scan", Repo: "core/the inference stack", AgentID: "agent-1"}})
	byType := summary["by_type"].(map[string]int)
	recent := summary["recent"].([]Event)

	Println(byType["scan"])
	Println(recent[0].Repo)
	// Output:
	// 1
	// core/the inference stack
}
