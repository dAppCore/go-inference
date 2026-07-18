// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/workspace"
)

func ExampleOrchestrator_ReviewChanges() {
	var orchestrator *Orchestrator
	core.Println(orchestrator.ReviewChanges(context.Background(), "run").OK)
	// Output: false
}

func ExampleOrchestrator_Accept() {
	var orchestrator *Orchestrator
	core.Println(orchestrator.Accept(context.Background(), workspace.AcceptRequest{Confirmed: true}).OK)
	// Output: false
}

func ExampleOrchestrator_Reject() {
	var orchestrator *Orchestrator
	core.Println(orchestrator.Reject(context.Background(), "run").OK)
	// Output: false
}
