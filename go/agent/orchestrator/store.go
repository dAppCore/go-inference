// SPDX-License-Identifier: EUPL-1.2

package orchestrator

import (
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/work"
)

// Commit is one atomic durable write across related agent records.
type Commit struct {
	Project        *work.Project
	Run            *work.Run
	ExpectedStatus *work.RunStatus
	CreateRun      bool
	Event          *work.Event
	Logs           []work.LogChunk
	Question       *work.Question
	Answer         *work.Answer
	Acceptance     *work.Acceptance
	Queue          *work.QueueState
	Provider       *work.ProviderState
}

// Store is the transactional persistence boundary implemented by the CLI.
type Store interface {
	Recover(time.Time) core.Result
	Commit(Commit) core.Result
	Project(string) core.Result
	ProjectBySource(string) core.Result
	Run(string) core.Result
	NextRunNumber(string) core.Result
	Continuation(string) core.Result
	Snapshot(string) core.Result
}

func commitStore(store Store, commit Commit) core.Result {
	if store == nil {
		return core.Fail(core.NewError("agent orchestrator store is required"))
	}
	validated := validateCommit(commit)
	if !validated.OK {
		return validated
	}
	return store.Commit(commit)
}

func recoverStore(store Store, at time.Time) core.Result {
	if store == nil {
		return core.Fail(core.NewError("agent orchestrator store is required"))
	}
	if at.IsZero() {
		return core.Fail(core.NewError("agent orchestrator recovery time is required"))
	}
	return store.Recover(at)
}

func validateCommit(commit Commit) core.Result {
	if commit.Project == nil && commit.Run == nil && commit.Event == nil && len(commit.Logs) == 0 && commit.Question == nil && commit.Answer == nil && commit.Acceptance == nil && commit.Queue == nil && commit.Provider == nil {
		return core.Fail(core.NewError("agent orchestrator commit requires at least one durable record"))
	}
	if commit.CreateRun {
		if commit.Run == nil || commit.ExpectedStatus != nil {
			return core.Fail(core.NewError("agent orchestrator run creation requires a run and no expected status"))
		}
	} else if commit.Run != nil && commit.ExpectedStatus == nil {
		return core.Fail(core.NewError("agent orchestrator run update requires an expected status"))
	}
	if commit.Run == nil && commit.ExpectedStatus != nil {
		return core.Fail(core.NewError("agent orchestrator expected status requires a run update"))
	}
	if commit.Project != nil {
		if core.Trim(commit.Project.ID) == "" || core.Trim(commit.Project.SourcePath) == "" || core.Trim(commit.Project.RepositoryRoot) == "" || core.Trim(commit.Project.RepositoryName) == "" || core.Trim(commit.Project.ClonePath) == "" {
			return core.Fail(core.NewError("agent orchestrator project commit requires durable source and repository identity"))
		}
		if commit.Project.CreatedAt.IsZero() || commit.Project.UpdatedAt.IsZero() {
			return core.Fail(core.NewError("agent orchestrator project commit requires timestamps"))
		}
	}
	if commit.Run != nil {
		if core.Trim(commit.Run.ID) == "" || core.Trim(commit.Run.WorkID) == "" || core.Trim(commit.Run.ProjectID) == "" || !knownRunStatus(commit.Run.Status) {
			return core.Fail(core.NewError("agent orchestrator run commit requires IDs and a known status"))
		}
		if commit.Run.Number <= 0 || commit.Run.Attempt <= 0 || commit.Run.UpdatedAt.IsZero() {
			return core.Fail(core.NewError("agent orchestrator run commit requires positive numbering and update time"))
		}
		if commit.ExpectedStatus != nil && !knownRunStatus(*commit.ExpectedStatus) {
			return core.Fail(core.NewError("agent orchestrator run commit expected status is unknown"))
		}
	}
	if commit.Event != nil {
		if core.Trim(commit.Event.ID) == "" || core.Trim(commit.Event.RunID) == "" || core.Trim(commit.Event.WorkID) == "" || core.Trim(commit.Event.Kind) == "" || commit.Event.CreatedAt.IsZero() {
			return core.Fail(core.NewError("agent orchestrator event commit requires IDs, kind, and time"))
		}
	}
	previousSequence := int64(0)
	for _, chunk := range commit.Logs {
		if core.Trim(chunk.RunID) == "" || chunk.Sequence <= previousSequence || core.Trim(chunk.Stream) == "" || chunk.Text == "" || chunk.CreatedAt.IsZero() {
			return core.Fail(core.NewError("agent orchestrator log commit requires ordered positive sequences and content"))
		}
		previousSequence = chunk.Sequence
	}
	if commit.Question != nil {
		if core.Trim(commit.Question.ID) == "" || core.Trim(commit.Question.RunID) == "" || core.Trim(commit.Question.Text) == "" || commit.Question.CreatedAt.IsZero() {
			return core.Fail(core.NewError("agent orchestrator question commit requires IDs, text, and time"))
		}
	}
	if commit.Answer != nil {
		if core.Trim(commit.Answer.ID) == "" || core.Trim(commit.Answer.QuestionID) == "" || core.Trim(commit.Answer.ResumeRunID) == "" || core.Trim(commit.Answer.Text) == "" || commit.Answer.CreatedAt.IsZero() {
			return core.Fail(core.NewError("agent orchestrator answer commit requires IDs, text, and time"))
		}
	}
	if commit.Acceptance != nil {
		if core.Trim(commit.Acceptance.ID) == "" || core.Trim(commit.Acceptance.WorkID) == "" || core.Trim(commit.Acceptance.RunID) == "" || core.Trim(commit.Acceptance.Status) == "" || commit.Acceptance.CreatedAt.IsZero() || commit.Acceptance.UpdatedAt.IsZero() {
			return core.Fail(core.NewError("agent orchestrator acceptance commit requires IDs, status, and times"))
		}
	}
	if commit.Queue != nil {
		if commit.Queue.ID != "default" || !knownQueueStatus(commit.Queue.Status) || commit.Queue.UpdatedAt.IsZero() {
			return core.Fail(core.NewError("agent orchestrator queue commit requires default ID, known status, and time"))
		}
	}
	if commit.Provider != nil {
		if core.Trim(commit.Provider.Provider) == "" || commit.Provider.WindowAdmissions < 0 || commit.Provider.UpdatedAt.IsZero() {
			return core.Fail(core.NewError("agent orchestrator provider commit requires identity, non-negative admissions, and time"))
		}
	}
	if commit.Run != nil {
		if commit.Event != nil && (commit.Event.RunID != commit.Run.ID || commit.Event.WorkID != commit.Run.WorkID) {
			return core.Fail(core.NewError("agent orchestrator event does not belong to the committed run"))
		}
		for _, chunk := range commit.Logs {
			if chunk.RunID != commit.Run.ID {
				return core.Fail(core.NewError("agent orchestrator log does not belong to the committed run"))
			}
		}
		if commit.Question != nil && commit.Question.RunID != commit.Run.ID {
			return core.Fail(core.NewError("agent orchestrator question does not belong to the committed run"))
		}
		if commit.Answer != nil && commit.Answer.ResumeRunID != commit.Run.ID {
			return core.Fail(core.NewError("agent orchestrator answer does not belong to the committed resume run"))
		}
		if commit.Acceptance != nil && (commit.Acceptance.RunID != commit.Run.ID || commit.Acceptance.WorkID != commit.Run.WorkID) {
			return core.Fail(core.NewError("agent orchestrator acceptance does not belong to the committed run"))
		}
	}
	return core.Ok(commit)
}

func knownRunStatus(status work.RunStatus) bool {
	switch status {
	case work.RunQueued, work.RunPreparing, work.RunRunning, work.RunWaiting,
		work.RunCancelling, work.RunCancelled, work.RunFailed, work.RunCompleted,
		work.RunInterrupted, work.RunAccepted, work.RunRejected:
		return true
	default:
		return false
	}
}

func knownQueueStatus(status work.QueueStatus) bool {
	switch status {
	case work.QueueFrozen, work.QueueAccepting, work.QueueDraining:
		return true
	default:
		return false
	}
}
