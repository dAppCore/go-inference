// SPDX-License-Identifier: EUPL-1.2

package work

import (
	core "dappco.re/go"
)

func ExampleTransition() {
	result := Transition(RunPreparing, RunRunning)
	if !result.OK {
		core.Println(result.Error())
		return
	}
	core.Println(result.Value)
	// Output: running
}

func ExampleValidateDispatch() {
	result := ValidateDispatch(DispatchRequest{
		Work: Item{
			ID:         "work-1",
			Title:      "Improve the TUI",
			Task:       "Add a durable agent workspace.",
			Repository: "/code/project",
		},
		Provider:                "codex",
		ConfirmedSourceRevision: "abc123",
	})
	if !result.OK {
		core.Println(result.Error())
		return
	}
	request := result.Value.(DispatchRequest)
	core.Println(request.Work.ID, request.Provider, request.ConfirmedSourceRevision)
	// Output: work-1 codex abc123
}
