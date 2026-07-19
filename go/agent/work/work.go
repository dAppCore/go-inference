// SPDX-License-Identifier: EUPL-1.2

// Package work defines durable values shared by agent orchestration layers.
package work

import (
	"time"

	core "dappco.re/go"
)

// RunStatus is the durable lifecycle state of one immutable process attempt.
type RunStatus string

const (
	// RunQueued is durably recorded and waiting for queue admission.
	RunQueued RunStatus = "queued"
	// RunPreparing is creating or recovering its isolated workspace.
	RunPreparing RunStatus = "preparing"
	// RunRunning owns a live native provider process.
	RunRunning RunStatus = "running"
	// RunWaiting exited with a valid question for the user.
	RunWaiting RunStatus = "waiting"
	// RunCancelling is terminating a live process group.
	RunCancelling RunStatus = "cancelling"
	// RunCancelled was cancelled by the user before completion.
	RunCancelled RunStatus = "cancelled"
	// RunFailed ended unsuccessfully.
	RunFailed RunStatus = "failed"
	// RunCompleted has captured provider work ready for review.
	RunCompleted RunStatus = "completed"
	// RunInterrupted was stopped by recovery or application shutdown.
	RunInterrupted RunStatus = "interrupted"
	// RunAccepted was applied to the reviewed source checkout.
	RunAccepted RunStatus = "accepted"
	// RunRejected was reviewed and declined.
	RunRejected RunStatus = "rejected"
)

// QueueStatus is the durable admission state shared across restarts.
type QueueStatus string

const (
	// QueueFrozen rejects new admissions.
	QueueFrozen QueueStatus = "frozen"
	// QueueAccepting admits eligible queued work.
	QueueAccepting QueueStatus = "accepting"
	// QueueDraining rejects admissions while existing processes finish.
	QueueDraining QueueStatus = "draining"
)

// Project records the isolated Git control-plane identity for a source checkout.
type Project struct {
	ID             string
	SourcePath     string
	RepositoryRoot string
	SourceBranch   string
	SourceRevision string
	RepositoryName string
	ClonePath      string
	CreatedAt      time.Time
	UpdatedAt      time.Time
}

// Item is the user-owned unit of Work selected for agent execution.
type Item struct {
	ID         string
	ExternalID string
	Title      string
	Task       string
	Repository string
}

// Run records one immutable native provider attempt.
type Run struct {
	ID                string
	WorkID            string
	ProjectID         string
	ParentRunID       string
	Provider          string
	Model             string
	SourceRevision    string
	DurableRevision   string
	ExecutionRevision string
	AcceptedRevision  string
	Branch            string
	Worktree          string
	CommandReceipt    string
	FailureReason     string
	Status            RunStatus
	Number            int
	Attempt           int
	ProcessID         int
	ExitCode          int
	QueuedAt          time.Time
	StartedAt         time.Time
	FinishedAt        time.Time
	UpdatedAt         time.Time
}

// Event is a durable structured lifecycle or provider event.
type Event struct {
	ID         string
	RunID      string
	WorkID     string
	Kind       string
	Title      string
	Detail     string
	DetailJSON string
	CreatedAt  time.Time
}

// LogChunk is one ordered stdout or stderr batch from a provider process.
type LogChunk struct {
	RunID     string
	Sequence  int64
	Stream    string
	Text      string
	CreatedAt time.Time
}

// Question is a provider request that can be answered before a resumed attempt.
type Question struct {
	ID        string
	RunID     string
	Text      string
	CreatedAt time.Time
}

// Answer is the user response linked to a later resumed run.
type Answer struct {
	ID          string
	QuestionID  string
	ResumeRunID string
	Text        string
	CreatedAt   time.Time
}

// Acceptance records review, validation, and source-application state.
type Acceptance struct {
	ID                  string
	WorkID              string
	RunID               string
	SourceBase          string
	AgentBase           string
	AgentTip            string
	IntegrationBranch   string
	IntegrationWorktree string
	ResultRevision      string
	Status              string
	ValidationJSON      string
	FailureReason       string
	CreatedAt           time.Time
	UpdatedAt           time.Time
}

// ProviderState stores durable queue backoff and rate-window state.
type ProviderState struct {
	Provider         string
	BackoffReason    string
	LastRunID        string
	BackoffUntil     time.Time
	LastStartedAt    time.Time
	WindowStartedAt  time.Time
	UpdatedAt        time.Time
	WindowAdmissions int
}

// QueueState stores the singleton durable queue admission state.
type QueueState struct {
	ID        string
	Status    QueueStatus
	Reason    string
	UpdatedAt time.Time
}

// Capability describes whether an orchestration operation is currently usable.
type Capability struct {
	Name      string
	Available bool
	Reason    string
}

// Snapshot is the ordered durable view used by an agent UI.
type Snapshot struct {
	Projects    []Project
	Runs        []Run
	Events      []Event
	Logs        []LogChunk
	Questions   []Question
	Acceptances []Acceptance
	Queue       QueueState
	Providers   []ProviderState
}

// Continuation contains the durable context needed for a child attempt.
type Continuation struct {
	Run      Run
	Task     string
	Logs     []LogChunk
	Question Question
	Answer   Answer
}

// DispatchRequest describes a reviewed root execution request.
type DispatchRequest struct {
	Work                    Item
	Provider                string
	Model                   string
	ConfirmedSourceRevision string
	UnsafeFlags             []string
}

// ResumeRequest describes a child attempt continuing earlier durable context.
type ResumeRequest struct {
	Work        Item
	ParentRunID string
	AnswerID    string
	Provider    string
	Model       string
}

// Transition validates a durable lifecycle change and returns the target state.
func Transition(from, to RunStatus) core.Result {
	if !validRunStatus(from) || !validRunStatus(to) {
		return core.Fail(core.NewError("agent work transition requires known non-empty statuses"))
	}
	if !allowedTransition(from, to) {
		return core.Fail(core.Errorf("agent work cannot transition from %q to %q", from, to))
	}
	return core.Ok(to)
}

// ValidateDispatch normalizes and validates a reviewed dispatch request.
func ValidateDispatch(request DispatchRequest) core.Result {
	normalized := DispatchRequest{
		Work: Item{
			ID:         core.Trim(request.Work.ID),
			ExternalID: core.Trim(request.Work.ExternalID),
			Title:      core.Trim(request.Work.Title),
			Task:       core.Trim(request.Work.Task),
			Repository: core.Trim(request.Work.Repository),
		},
		Provider:                core.Trim(request.Provider),
		Model:                   core.Trim(request.Model),
		ConfirmedSourceRevision: core.Trim(request.ConfirmedSourceRevision),
		UnsafeFlags:             append([]string(nil), request.UnsafeFlags...),
	}

	fields := []struct {
		name  string
		value string
	}{
		{name: "work ID", value: normalized.Work.ID},
		{name: "work title", value: normalized.Work.Title},
		{name: "work task", value: normalized.Work.Task},
		{name: "repository", value: normalized.Work.Repository},
		{name: "provider", value: normalized.Provider},
		{name: "confirmed source revision", value: normalized.ConfirmedSourceRevision},
	}
	for _, field := range fields {
		if field.value == "" {
			return core.Fail(core.Errorf("agent dispatch requires %s", field.name))
		}
	}

	return core.Ok(normalized)
}

func validRunStatus(status RunStatus) bool {
	switch status {
	case RunQueued, RunPreparing, RunRunning, RunWaiting, RunCancelling,
		RunCancelled, RunFailed, RunCompleted, RunInterrupted, RunAccepted, RunRejected:
		return true
	default:
		return false
	}
}

func allowedTransition(from, to RunStatus) bool {
	switch from {
	case RunQueued:
		return to == RunPreparing || to == RunCancelled || to == RunInterrupted
	case RunPreparing:
		return to == RunRunning || to == RunFailed || to == RunInterrupted
	case RunRunning:
		return to == RunWaiting || to == RunCancelling || to == RunFailed || to == RunCompleted || to == RunInterrupted
	case RunCancelling:
		return to == RunCancelled || to == RunFailed || to == RunInterrupted
	case RunCompleted:
		return to == RunAccepted || to == RunRejected
	default:
		return false
	}
}
