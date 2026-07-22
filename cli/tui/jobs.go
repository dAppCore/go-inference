// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"context"
	"iter"
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
	model     *laneTextModel
	name      string

	closeOnce   sync.Once
	closeResult core.Result
}

// laneTextModel keeps the scheduler's shared serial lane while capturing the
// base model's terminal Result before the next queued request may run. The
// private request-error seam lets the TUI attribute errors to one session
// without requiring a newer public inference option in the nested CLI module.
type laneTextModel struct {
	base      inference.TextModel
	scheduled *scheduler.Model
	gate      chan struct{}
	close     func() core.Result
	errMu     sync.Mutex
	lastErr   core.Result
}

type scopedErrorChatModel interface {
	chatWithErrorSink(context.Context, []inference.Message, func(error), ...inference.GenerateOption) iter.Seq[inference.Token]
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
	lane := &modelLane{base: model, scheduled: scheduled, name: name}
	lane.model = &laneTextModel{
		base:      model,
		scheduled: scheduled,
		gate:      make(chan struct{}, 1),
		lastErr:   core.Ok(nil),
	}
	lane.model.close = lane.Close
	return core.Ok(lane)
}

// Model returns the scheduled view shared by all generation consumers.
func (lane *modelLane) Model() inference.TextModel {
	if lane == nil {
		return nil
	}
	return lane.model
}

func (model *laneTextModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return model.stream(ctx, func() iter.Seq[inference.Token] {
		return model.scheduled.Generate(ctx, prompt, opts...)
	}, nil)
}

func (model *laneTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return model.chatWithErrorSink(ctx, messages, nil, opts...)
}

func (model *laneTextModel) chatWithErrorSink(ctx context.Context, messages []inference.Message, sink func(error), opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return model.stream(ctx, func() iter.Seq[inference.Token] {
		return model.scheduled.Chat(ctx, messages, opts...)
	}, sink)
}

func (model *laneTextModel) stream(ctx context.Context, source func() iter.Seq[inference.Token], sink func(error)) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if model == nil || model.base == nil || model.scheduled == nil {
			result := core.Fail(core.E("tui.laneTextModel.stream", "model lane is unavailable", nil))
			model.setResult(result, sink)
			return
		}
		if ctx == nil {
			ctx = context.Background()
		}
		select {
		case model.gate <- struct{}{}:
			defer func() { <-model.gate }()
		case <-ctx.Done():
			model.setResult(core.Fail(ctx.Err()), sink)
			return
		}
		for token := range source() {
			if !yield(token) {
				break
			}
		}
		result := model.base.Err()
		if result.OK && ctx.Err() != nil {
			result = core.Fail(ctx.Err())
		}
		model.setResult(result, sink)
	}
}

func (model *laneTextModel) setResult(result core.Result, sink func(error)) {
	if model != nil {
		model.errMu.Lock()
		model.lastErr = result
		model.errMu.Unlock()
	}
	if sink == nil {
		return
	}
	if result.OK {
		sink(nil)
		return
	}
	err := resultError(result)
	if err == nil {
		err = core.E("tui.laneTextModel.setResult", result.Error(), nil)
	}
	sink(err)
}

func (model *laneTextModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	if result := model.acquire(ctx); !result.OK {
		return result
	}
	defer model.release()
	return model.scheduled.Classify(ctx, prompts, opts...)
}

func (model *laneTextModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	if result := model.acquire(ctx); !result.OK {
		return result
	}
	defer model.release()
	return model.scheduled.BatchGenerate(ctx, prompts, opts...)
}

func (model *laneTextModel) acquire(ctx context.Context) core.Result {
	if model == nil || model.scheduled == nil {
		return core.Fail(core.E("tui.laneTextModel.acquire", "model lane is unavailable", nil))
	}
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case model.gate <- struct{}{}:
		return core.Ok(nil)
	case <-ctx.Done():
		return core.Fail(ctx.Err())
	}
}

func (model *laneTextModel) release() {
	if model != nil {
		<-model.gate
	}
}

func (model *laneTextModel) ModelType() string {
	if model == nil || model.base == nil {
		return ""
	}
	return model.base.ModelType()
}

func (model *laneTextModel) Info() inference.ModelInfo {
	if model == nil || model.base == nil {
		return inference.ModelInfo{}
	}
	return model.base.Info()
}

func (model *laneTextModel) Metrics() inference.GenerateMetrics {
	if model == nil || model.scheduled == nil {
		return inference.GenerateMetrics{}
	}
	return model.scheduled.Metrics()
}

func (model *laneTextModel) Err() core.Result {
	if model == nil {
		return core.Fail(core.E("tui.laneTextModel.Err", "model lane is unavailable", nil))
	}
	model.errMu.Lock()
	defer model.errMu.Unlock()
	return model.lastErr
}

func (model *laneTextModel) Close() core.Result {
	if model == nil || model.close == nil {
		return core.Ok(nil)
	}
	return model.close()
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

func (jobs *jobManager) Active(sessionID string) *generation {
	if jobs == nil {
		return nil
	}
	jobs.mu.Lock()
	defer jobs.mu.Unlock()
	return jobs.bySession[sessionID]
}

func (jobs *jobManager) ActiveCount() int {
	if jobs == nil {
		return 0
	}
	jobs.mu.Lock()
	defer jobs.mu.Unlock()
	return len(jobs.bySession)
}

func (jobs *jobManager) finish(sessionID, jobID string) {
	jobs.mu.Lock()
	defer jobs.mu.Unlock()
	if current := jobs.bySession[sessionID]; current != nil && current.JobID == jobID {
		delete(jobs.bySession, sessionID)
	}
}
