// SPDX-License-Identifier: EUPL-1.2

package work

import (
	"testing"
)

func TestWork_Transition_Good(t *testing.T) {
	tests := []struct {
		name string
		from RunStatus
		to   RunStatus
	}{
		{name: "prepare queued work", from: RunQueued, to: RunPreparing},
		{name: "cancel queued work", from: RunQueued, to: RunCancelled},
		{name: "interrupt queued work", from: RunQueued, to: RunInterrupted},
		{name: "start prepared work", from: RunPreparing, to: RunRunning},
		{name: "fail prepared work", from: RunPreparing, to: RunFailed},
		{name: "interrupt prepared work", from: RunPreparing, to: RunInterrupted},
		{name: "wait for an answer", from: RunRunning, to: RunWaiting},
		{name: "begin cancellation", from: RunRunning, to: RunCancelling},
		{name: "fail running work", from: RunRunning, to: RunFailed},
		{name: "complete running work", from: RunRunning, to: RunCompleted},
		{name: "interrupt a running process", from: RunRunning, to: RunInterrupted},
		{name: "finish cancellation", from: RunCancelling, to: RunCancelled},
		{name: "fail cancellation", from: RunCancelling, to: RunFailed},
		{name: "interrupt cancellation", from: RunCancelling, to: RunInterrupted},
		{name: "accept completed work", from: RunCompleted, to: RunAccepted},
		{name: "reject completed work", from: RunCompleted, to: RunRejected},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := Transition(test.from, test.to)
			if !result.OK {
				t.Fatalf("Transition(%q, %q) failed: %s", test.from, test.to, result.Error())
			}

			status, ok := result.Value.(RunStatus)
			if !ok {
				t.Fatalf("Transition(%q, %q) returned %T, want RunStatus", test.from, test.to, result.Value)
			}
			if status != test.to {
				t.Fatalf("Transition(%q, %q) = %q, want %q", test.from, test.to, status, test.to)
			}
		})
	}
}

func TestWork_Transition_Bad(t *testing.T) {
	tests := []struct {
		from RunStatus
		to   RunStatus
	}{
		{from: RunCompleted, to: RunRunning},
		{from: RunWaiting, to: RunQueued},
		{from: RunFailed, to: RunQueued},
		{from: RunCancelled, to: RunRunning},
		{from: RunInterrupted, to: RunQueued},
		{from: RunAccepted, to: RunRejected},
		{from: RunRejected, to: RunAccepted},
		{from: RunRunning, to: RunRunning},
	}

	for _, test := range tests {
		result := Transition(test.from, test.to)
		if result.OK {
			t.Fatalf("Transition(%q, %q) unexpectedly succeeded", test.from, test.to)
		}
	}
}

func TestWork_Transition_Ugly(t *testing.T) {
	tests := []struct {
		name string
		from RunStatus
		to   RunStatus
	}{
		{name: "empty source", from: "", to: RunRunning},
		{name: "empty target", from: RunRunning, to: ""},
		{name: "unknown source", from: RunStatus("lost"), to: RunRunning},
		{name: "unknown target", from: RunRunning, to: RunStatus("lost")},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := Transition(test.from, test.to)
			if result.OK {
				t.Fatalf("Transition(%q, %q) unexpectedly succeeded", test.from, test.to)
			}
		})
	}
}

func TestWork_ValidateDispatch_Good(t *testing.T) {
	request := DispatchRequest{
		Work: Item{
			ID:         " work-1 ",
			ExternalID: " issue-42 ",
			Title:      " Improve the TUI ",
			Task:       " Add a durable agent workspace. ",
			Repository: " /tmp/project with spaces ",
		},
		Provider:                " codex ",
		Model:                   " gpt-5.6 ",
		ConfirmedSourceRevision: " abc123 ",
		UnsafeFlags:             []string{"--search", "allow network"},
	}

	result := ValidateDispatch(request)
	if !result.OK {
		t.Fatalf("ValidateDispatch failed: %s", result.Error())
	}
	normalized, ok := result.Value.(DispatchRequest)
	if !ok {
		t.Fatalf("ValidateDispatch returned %T, want DispatchRequest", result.Value)
	}

	if normalized.Work.ID != "work-1" || normalized.Work.ExternalID != "issue-42" {
		t.Fatalf("ValidateDispatch work IDs = %q/%q", normalized.Work.ID, normalized.Work.ExternalID)
	}
	if normalized.Work.Title != "Improve the TUI" || normalized.Work.Task != "Add a durable agent workspace." {
		t.Fatalf("ValidateDispatch work text = %q/%q", normalized.Work.Title, normalized.Work.Task)
	}
	if normalized.Work.Repository != "/tmp/project with spaces" {
		t.Fatalf("ValidateDispatch repository = %q", normalized.Work.Repository)
	}
	if normalized.Provider != "codex" || normalized.Model != "gpt-5.6" {
		t.Fatalf("ValidateDispatch provider/model = %q/%q", normalized.Provider, normalized.Model)
	}
	if normalized.ConfirmedSourceRevision != "abc123" {
		t.Fatalf("ValidateDispatch source revision = %q", normalized.ConfirmedSourceRevision)
	}

	request.UnsafeFlags[0] = "mutated"
	if normalized.UnsafeFlags[0] != "--search" {
		t.Fatalf("ValidateDispatch retained caller slice: %q", normalized.UnsafeFlags[0])
	}
}

func TestWork_ValidateDispatch_Bad(t *testing.T) {
	valid := DispatchRequest{
		Work: Item{
			ID:         "work-1",
			Title:      "Improve the TUI",
			Task:       "Add a durable agent workspace.",
			Repository: "/tmp/project",
		},
		Provider:                "codex",
		ConfirmedSourceRevision: "abc123",
	}

	tests := []struct {
		name    string
		request DispatchRequest
	}{
		{name: "missing work ID", request: func() DispatchRequest { request := valid; request.Work.ID = ""; return request }()},
		{name: "missing title", request: func() DispatchRequest { request := valid; request.Work.Title = ""; return request }()},
		{name: "missing task", request: func() DispatchRequest { request := valid; request.Work.Task = ""; return request }()},
		{name: "missing repository", request: func() DispatchRequest { request := valid; request.Work.Repository = ""; return request }()},
		{name: "missing provider", request: func() DispatchRequest { request := valid; request.Provider = ""; return request }()},
		{name: "missing source revision", request: func() DispatchRequest { request := valid; request.ConfirmedSourceRevision = ""; return request }()},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := ValidateDispatch(test.request)
			if result.OK {
				t.Fatalf("ValidateDispatch unexpectedly accepted %s", test.name)
			}
		})
	}
}

func TestWork_ValidateDispatch_Ugly(t *testing.T) {
	tests := []DispatchRequest{
		{},
		{
			Work: Item{
				ID:         " \t ",
				Title:      "\n",
				Task:       "  ",
				Repository: "\t",
			},
			Provider:                " ",
			ConfirmedSourceRevision: "\n",
		},
	}

	for index, request := range tests {
		result := ValidateDispatch(request)
		if result.OK {
			t.Fatalf("ValidateDispatch unexpectedly accepted ugly request %d", index)
		}
	}
}
