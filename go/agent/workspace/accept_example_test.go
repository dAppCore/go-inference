// SPDX-License-Identifier: EUPL-1.2

package workspace

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
)

func ExampleChangeReview() {
	review := ChangeReview{WorkID: "work-1", RunID: "run-1", ResultRevision: "abc123"}
	core.Println(review.WorkID, review.RunID, review.ResultRevision)
	// Output: work-1 run-1 abc123
}

func ExampleValidationResult() {
	validation := ValidationResult{Command: Command{Executable: "go", Args: []string{"test", "./..."}}, Passed: true}
	core.Println(validation.Command.Executable, validation.Passed)
	// Output: go true
}

func ExampleAcceptRequest() {
	request := AcceptRequest{Review: ChangeReview{RunID: "run-1"}, Confirmed: true}
	core.Println(request.Review.RunID, request.Confirmed)
	// Output: run-1 true
}

func ExampleManager_ReviewChanges() {
	var manager *Manager
	result := manager.ReviewChanges(context.Background(), work.Project{}, work.Run{}, nil)
	core.Println(result.OK)
	// Output: false
}

func ExampleManager_RetainReview() {
	var manager *Manager
	result := manager.RetainReview(ChangeReview{})
	core.Println(result.OK)
	// Output: false
}

func ExampleManager_AbandonReview() {
	var manager *Manager
	result := manager.AbandonReview(context.Background(), ChangeReview{})
	core.Println(result.OK)
	// Output: false
}

func ExampleManager_Apply() {
	var manager *Manager
	result := manager.Apply(context.Background(), AcceptRequest{Confirmed: true})
	core.Println(result.OK)
	// Output: false
}

func ExampleManager_Reject() {
	var manager *Manager
	result := manager.Reject(context.Background(), ChangeReview{})
	core.Println(result.OK)
	// Output: false
}
