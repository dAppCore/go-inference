// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
)

func ExampleOrchestrator_ReviewDispatch() {
	var orchestrator *Orchestrator
	result := orchestrator.ReviewDispatch(context.Background(), work.DispatchRequest{})
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_Dispatch() {
	var orchestrator *Orchestrator
	result := orchestrator.Dispatch(context.Background(), DispatchReview{})
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_Cancel() {
	var orchestrator *Orchestrator
	result := orchestrator.Cancel(context.Background(), "run-1")
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_Answer() {
	var orchestrator *Orchestrator
	result := orchestrator.Answer(context.Background(), "run-1", "Keep the Adapter API")
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_Resume() {
	var orchestrator *Orchestrator
	result := orchestrator.Resume(context.Background(), work.ResumeRequest{})
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_Retry() {
	var orchestrator *Orchestrator
	result := orchestrator.Retry(context.Background(), work.Item{}, "run-1")
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_ReviewRetry() {
	var orchestrator *Orchestrator
	result := orchestrator.ReviewRetry(context.Background(), work.Item{}, "run-1")
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_ConfirmRetry() {
	var orchestrator *Orchestrator
	result := orchestrator.ConfirmRetry(context.Background(), ChildReview{})
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_ReviewResume() {
	var orchestrator *Orchestrator
	result := orchestrator.ReviewResume(context.Background(), work.ResumeRequest{})
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_ConfirmResume() {
	var orchestrator *Orchestrator
	result := orchestrator.ConfirmResume(context.Background(), ChildReview{})
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_StartQueue() {
	var orchestrator *Orchestrator
	result := orchestrator.StartQueue(context.Background())
	core.Println(result.OK)
	// Output: false
}

func ExampleOrchestrator_StopQueue() {
	var orchestrator *Orchestrator
	result := orchestrator.StopQueue(context.Background())
	core.Println(result.OK)
	// Output: false
}
