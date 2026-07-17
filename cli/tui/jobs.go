// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"context"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving/scheduler"
)

// modelLane owns the one scheduler wrapped around a loaded model. Every local
// session and every HTTP request uses scheduled, so starting or stopping the
// service never changes the model's concurrency policy.
type modelLane struct {
	base      inference.TextModel
	scheduled *scheduler.Model
	name      string

	closeOnce   sync.Once
	closeResult core.Result
}

// newModelLane creates the application's single serial generation lane.
func newModelLane(model inference.TextModel, name string) core.Result {
	if model == nil {
		return core.Fail(core.E("tui.newModelLane", "model is nil", nil))
	}
	scheduled, err := scheduler.New(model, scheduler.Config{
		Mode:            scheduler.ModeSerial,
		MaxConcurrent:   1,
		MaxQueue:        64,
		StreamBuffer:    16,
		RequestIDPrefix: "tui",
	})
	if err != nil {
		return core.Fail(core.E("tui.newModelLane", "create serial scheduler", err))
	}
	return core.Ok(&modelLane{base: model, scheduled: scheduled, name: name})
}

// Model returns the scheduled view shared by all generation consumers.
func (lane *modelLane) Model() inference.TextModel {
	if lane == nil {
		return nil
	}
	return lane.scheduled
}

// Close drains the scheduler and releases the base model exactly once.
func (lane *modelLane) Close() core.Result {
	if lane == nil {
		return core.Ok(nil)
	}
	lane.closeOnce.Do(func() {
		if lane.scheduled == nil {
			lane.closeResult = core.Ok(nil)
			return
		}
		lane.closeResult = lane.scheduled.Close()
	})
	return lane.closeResult
}

// generation identifies one in-flight session turn. Tags travel with every
// stream event so a late event can never be folded into another session.
type generation struct {
	SessionID string
	JobID     string
	cancel    context.CancelFunc
	events    chan streamEvent
}

// jobManager allows one in-flight generation per session while permitting
// independent sessions to queue on the shared model lane.
type jobManager struct {
	parent context.Context

	mu        sync.Mutex
	bySession map[string]*generation
}

func newJobManager(parent context.Context) *jobManager {
	if parent == nil {
		parent = context.Background()
	}
	return &jobManager{parent: parent, bySession: make(map[string]*generation)}
}

// Start registers and launches a tagged generation. The successful Result
// carries *generation for Bubble Tea's wait command.
func (jobs *jobManager) Start(sessionID, jobID string, model inference.TextModel, history []inference.Message, opts []inference.GenerateOption) core.Result {
	if jobs == nil {
		return core.Fail(core.E("tui.jobManager.Start", "job manager is nil", nil))
	}
	if core.Trim(sessionID) == "" {
		return core.Fail(core.E("tui.jobManager.Start", "session ID is empty", nil))
	}
	if core.Trim(jobID) == "" {
		return core.Fail(core.E("tui.jobManager.Start", "job ID is empty", nil))
	}
	if model == nil {
		return core.Fail(core.E("tui.jobManager.Start", "model is nil", nil))
	}

	jobs.mu.Lock()
	if jobs.bySession == nil {
		jobs.bySession = make(map[string]*generation)
	}
	if _, exists := jobs.bySession[sessionID]; exists {
		jobs.mu.Unlock()
		return core.Fail(core.E("tui.jobManager.Start", "session already has a running generation", nil))
	}
	parent := jobs.parent
	if parent == nil {
		parent = context.Background()
	}
	ctx, cancel := context.WithCancel(parent)
	generation := &generation{
		SessionID: sessionID,
		JobID:     jobID,
		cancel:    cancel,
		events:    make(chan streamEvent, 64),
	}
	jobs.bySession[sessionID] = generation
	jobs.mu.Unlock()

	go streamGeneration(ctx, generation, model, history, opts, func() {
		jobs.finish(sessionID, jobID)
	})
	return core.Ok(generation)
}

// Cancel stops only the named session's current generation.
func (jobs *jobManager) Cancel(sessionID string) core.Result {
	if jobs == nil {
		return core.Fail(core.E("tui.jobManager.Cancel", "job manager is nil", nil))
	}
	jobs.mu.Lock()
	generation, exists := jobs.bySession[sessionID]
	jobs.mu.Unlock()
	if !exists {
		return core.Fail(core.E("tui.jobManager.Cancel", "session has no running generation", nil))
	}
	generation.cancel()
	return core.Ok(nil)
}

// CancelAll stops every session generation without closing the model lane.
func (jobs *jobManager) CancelAll() core.Result {
	if jobs == nil {
		return core.Fail(core.E("tui.jobManager.CancelAll", "job manager is nil", nil))
	}
	jobs.mu.Lock()
	generations := make([]*generation, 0, len(jobs.bySession))
	for _, generation := range jobs.bySession {
		generations = append(generations, generation)
	}
	jobs.mu.Unlock()
	for _, generation := range generations {
		generation.cancel()
	}
	return core.Ok(len(generations))
}

func (jobs *jobManager) finish(sessionID, jobID string) {
	jobs.mu.Lock()
	defer jobs.mu.Unlock()
	if current := jobs.bySession[sessionID]; current != nil && current.JobID == jobID {
		delete(jobs.bySession, sessionID)
	}
}
