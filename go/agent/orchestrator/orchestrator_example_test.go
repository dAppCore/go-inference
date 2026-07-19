// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
)

func ExampleNew() {
	result := New(Options{})
	core.Println(result.OK)
	// Output: false
}

func ExampleOptions_HardenedRuntimeContract() {
	result := (Options{}).HardenedRuntimeContract(context.Background())
	core.Println(result.OK)
	// Output: true
}

func ExampleOrchestrator_Capabilities() {
	var orchestrator *Orchestrator
	capabilities := orchestrator.Capabilities()
	core.Println(len(capabilities), capabilities[0].Available)
	// Output: 11 false
}

func ExampleOrchestrator_Snapshot() {
	var orchestrator *Orchestrator
	result := orchestrator.Snapshot(context.Background(), "work-1")
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_AbandonRecovery() {
	var orchestrator *Orchestrator
	result := orchestrator.AbandonRecovery(context.Background(), "run-1", "recovery-event-1")
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_ReviewProject() {
	var orchestrator *Orchestrator
	result := orchestrator.ReviewProject(context.Background(), work.Item{ID: "work-1", Repository: "/source"})
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_RegisterProject() {
	var orchestrator *Orchestrator
	result := orchestrator.RegisterProject(context.Background(), ProjectReview{}, true)
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_Close() {
	var orchestrator *Orchestrator
	result := orchestrator.Close()
	core.Println(result.OK)
	// Output: false
}
