// SPDX-License-Identifier: EUPL-1.2

package queue

import (
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
)

func exampleQueueController(status work.QueueStatus) *Controller {
	result := NewController(defaultPolicy(), work.QueueState{ID: "default", Status: status}, nil)
	if !result.OK {
		return nil
	}
	return result.Value.(*Controller)
}

func ExampleNewController() {
	result := NewController(defaultPolicy(), work.QueueState{}, nil)
	core.Println(result.OK)
	// Output: true
}

func ExampleController_Start() {
	controller := exampleQueueController(work.QueueFrozen)
	result := controller.Start(time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC))
	state := result.Value.(work.QueueState)
	core.Println(state.Status)
	// Output: accepting
}

func ExampleController_Stop() {
	controller := exampleQueueController(work.QueueAccepting)
	result := controller.Stop(2, time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC))
	state := result.Value.(work.QueueState)
	core.Println(state.Status)
	// Output: draining
}

func ExampleController_Decide() {
	controller := exampleQueueController(work.QueueAccepting)
	now := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	result := controller.Decide(
		Candidate{RunID: "run-1", Provider: "codex", QueuedAt: now},
		Runtime{Now: now},
	)
	decision := result.Value.(Decision)
	core.Println(decision.Allowed)
	// Output: true
}

func ExampleController_RecordStart() {
	controller := exampleQueueController(work.QueueAccepting)
	result := controller.RecordStart("codex", "run-1", time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC))
	state := result.Value.(work.ProviderState)
	core.Println(state.LastRunID, state.WindowAdmissions)
	// Output: run-1 1
}

func ExampleController_RecordBackoff() {
	controller := exampleQueueController(work.QueueAccepting)
	at := time.Date(2026, 7, 18, 12, 0, 0, 0, time.UTC)
	result := controller.RecordBackoff("codex", "quota", at.Add(time.Hour), at)
	state := result.Value.(work.ProviderState)
	core.Println(state.Provider, state.BackoffReason)
	// Output: codex quota
}

func ExampleController_Restore() {
	controller := exampleQueueController(work.QueueAccepting)
	result := controller.Restore(work.QueueState{ID: "default", Status: work.QueueFrozen}, nil)
	core.Println(result.OK)
	// Output: true
}
