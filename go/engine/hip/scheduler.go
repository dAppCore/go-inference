// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"context"
	"iter"
	"sync"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	servingscheduler "dappco.re/go/inference/serving/scheduler"
)

// SchedulerConfig controls the package-first ROCm scheduler wrapper.
type SchedulerConfig struct {
	QueueSize     int // queued requests before backpressure rejects a submission
	OutputBuffer  int // per-request streamed-token buffer
	MaxConcurrent int // active lane cap when the model supports batched stepping
}

// ScheduledModel wraps a TextModel with request lifecycle ownership. Models
// with an available batch-step surface run through the shared continuous
// scheduler; other models retain the bounded single-worker path.
type ScheduledModel struct {
	model        inference.TextModel
	queue        chan *scheduledWork
	outputBuffer int
	continuous   *servingscheduler.Model
	nextID       atomic.Uint64

	mu       sync.Mutex
	cancel   map[string]context.CancelFunc
	sink     inference.ProbeSink
	closed   bool
	closeOne sync.Once
	closeErr error
	lastErr  error
	workerWG sync.WaitGroup
}

type scheduledWork struct {
	id       string
	req      inference.ScheduledRequest
	ctx      context.Context
	cancel   context.CancelFunc
	out      chan inference.ScheduledToken
	enqueued time.Time
}

type serializedHIPProbeSink struct {
	mu   sync.Mutex
	sink inference.ProbeSink
}

func (sink *serializedHIPProbeSink) EmitProbe(event inference.ProbeEvent) {
	if sink == nil || sink.sink == nil {
		return
	}
	sink.mu.Lock()
	defer sink.mu.Unlock()
	sink.sink.EmitProbe(event)
}

func serializeHIPProbeSink(sink inference.ProbeSink) inference.ProbeSink {
	if sink == nil {
		return nil
	}
	return &serializedHIPProbeSink{sink: sink}
}

// NewScheduledModel selects continuous lane scheduling when the loaded model
// can open a real lane set; otherwise it uses the bounded single-worker path.
func NewScheduledModel(model inference.TextModel, cfg SchedulerConfig) (*ScheduledModel, error) {
	if model == nil {
		return nil, core.E("rocm.NewScheduledModel", "model is nil", nil)
	}
	if cfg.QueueSize <= 0 {
		cfg.QueueSize = 1
	}
	if cfg.OutputBuffer <= 0 {
		cfg.OutputBuffer = 1
	}
	if cfg.MaxConcurrent <= 0 {
		cfg.MaxConcurrent = defaultHIPLaneSetMaxLanes
	}
	scheduled := &ScheduledModel{
		model:        model,
		outputBuffer: cfg.OutputBuffer,
		cancel:       map[string]context.CancelFunc{},
	}
	if batchStep, batchCapable := inference.As[inference.BatchStepModel](model); batchCapable && batchStep.BatchStepAvailable() {
		if _, tokenizable := inference.As[inference.TokenizerModel](model); tokenizable {
			continuous, err := servingscheduler.New(model, servingscheduler.Config{
				Mode:          servingscheduler.ModeInterleave,
				MaxConcurrent: cfg.MaxConcurrent,
				MaxQueue:      cfg.QueueSize,
				StreamBuffer:  cfg.OutputBuffer,
			})
			if err != nil {
				return nil, err
			}
			scheduled.continuous = continuous
			return scheduled, nil
		}
	}
	scheduled.queue = make(chan *scheduledWork, cfg.QueueSize)
	scheduled.workerWG.Add(1)
	go scheduled.run()
	return scheduled, nil
}

func (m *ScheduledModel) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	if m == nil {
		return inference.RequestHandle{}, nil, core.E("rocm.Schedule", "scheduler is nil", nil)
	}
	if m.model == nil {
		err := core.E("rocm.Schedule", "scheduled model is nil", nil)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
	if m.continuous == nil && (m.queue == nil || m.cancel == nil) {
		err := core.E("rocm.Schedule", "scheduler is not initialized", nil)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
	m.setErr(nil)
	if ctx == nil {
		ctx = context.Background()
	}
	req.ID = core.Trim(req.ID)
	if req.ID == "" {
		req.ID = core.Sprintf("rocm-%d", m.nextID.Add(1))
	}
	req.Messages = append([]inference.Message(nil), req.Messages...)
	req.Sampler = cloneSamplerConfig(req.Sampler)
	req.Labels = cloneStringMap(req.Labels)
	if err := ctx.Err(); err != nil {
		err = core.E("rocm.Schedule", "enqueue request", err)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
	if err := m.validateScheduledGemma4Context(&req); err != nil {
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
	if m.continuous != nil {
		handle, stream, err := m.continuous.Schedule(ctx, req)
		if err != nil {
			m.setErr(err)
		}
		return handle, stream, err
	}
	reqCtx, cancel := context.WithCancel(ctx)
	work := &scheduledWork{
		id:       req.ID,
		req:      req,
		ctx:      reqCtx,
		cancel:   cancel,
		out:      make(chan inference.ScheduledToken, m.outputBuffer),
		enqueued: time.Now(),
	}

	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		cancel()
		close(work.out)
		err := core.E("rocm.Schedule", "scheduler is closed", nil)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
	if _, exists := m.cancel[work.id]; exists {
		m.mu.Unlock()
		cancel()
		close(work.out)
		err := core.E("rocm.Schedule", "duplicate request id "+work.id, nil)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
	m.cancel[work.id] = cancel
	select {
	case m.queue <- work:
		m.mu.Unlock()
		m.emitSchedulerProbe(work.id, "queued", inference.ProbePhaseQueue, 0, 0, 0, false)
		return inference.RequestHandle{ID: work.id, Labels: cloneStringMap(req.Labels)}, work.out, nil
	default:
		delete(m.cancel, work.id)
		m.mu.Unlock()
		cancel()
		close(work.out)
		err := core.E("rocm.Schedule", "queue is full", nil)
		m.setErr(err)
		return inference.RequestHandle{}, nil, err
	}
}

func (m *ScheduledModel) CancelRequest(ctx context.Context, id string) (inference.RequestCancelResult, error) {
	if m == nil {
		return inference.RequestCancelResult{}, core.E("rocm.CancelRequest", "scheduler is nil", nil)
	}
	if m.model == nil {
		err := core.E("rocm.CancelRequest", "scheduled model is nil", nil)
		m.setErr(err)
		return inference.RequestCancelResult{}, err
	}
	if m.cancel == nil {
		err := core.E("rocm.CancelRequest", "scheduler is not initialized", nil)
		m.setErr(err)
		return inference.RequestCancelResult{}, err
	}
	id = core.Trim(id)
	if id == "" {
		err := core.E("rocm.CancelRequest", "request id is empty", nil)
		m.setErr(err)
		return inference.RequestCancelResult{}, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		m.setErr(err)
		return inference.RequestCancelResult{}, err
	}
	m.setErr(nil)
	if m.continuous != nil {
		result, err := m.continuous.CancelRequest(ctx, id)
		if err != nil {
			m.setErr(err)
		}
		return result, err
	}
	m.mu.Lock()
	cancel := m.cancel[id]
	m.mu.Unlock()
	if cancel == nil {
		if cancellable, ok := m.model.(inference.CancellableModel); ok {
			result, err := cancellable.CancelRequest(ctx, id)
			if err != nil {
				m.setErr(err)
			}
			return result, err
		}
		return inference.RequestCancelResult{ID: id, Cancelled: false, Reason: "request not found"}, nil
	}
	cancel()
	m.emitSchedulerProbe(id, "cancelled", inference.ProbePhaseQueue, 0, 0, 0, true)
	return inference.RequestCancelResult{ID: id, Cancelled: true}, nil
}

func (m *ScheduledModel) SetProbeSink(sink inference.ProbeSink) {
	if m == nil {
		return
	}
	delivery := serializeHIPProbeSink(sink)
	m.mu.Lock()
	m.sink = delivery
	m.mu.Unlock()
	if m.continuous != nil {
		m.continuous.SetProbeSink(delivery)
	}
	if probeable, ok := m.model.(inference.ProbeableModel); ok {
		probeable.SetProbeSink(delivery)
	}
}

func (m *ScheduledModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if m == nil || m.model == nil {
			if m != nil {
				m.setErr(core.E("rocm.Generate", "scheduled model is nil", nil))
			}
			return
		}
		m.setErr(nil)
		req := inference.ScheduledRequest{Prompt: prompt, Sampler: m.samplerConfigFromGenerateOptions(opts)}
		_, stream, err := m.Schedule(ctx, req)
		if err != nil {
			m.setErr(err)
			return
		}
		for scheduled := range stream {
			if !yield(scheduled.Token) {
				_, _ = m.CancelRequest(ctx, scheduled.RequestID)
				return
			}
		}
	}
}

func (m *ScheduledModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if m == nil || m.model == nil {
			if m != nil {
				m.setErr(core.E("rocm.Chat", "scheduled model is nil", nil))
			}
			return
		}
		m.setErr(nil)
		req := inference.ScheduledRequest{Messages: append([]inference.Message(nil), messages...), Sampler: m.samplerConfigFromGenerateOptions(opts)}
		_, stream, err := m.Schedule(ctx, req)
		if err != nil {
			m.setErr(err)
			return
		}
		for scheduled := range stream {
			if !yield(scheduled.Token) {
				_, _ = m.CancelRequest(ctx, scheduled.RequestID)
				return
			}
		}
	}
}

func (m *ScheduledModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	if m == nil || m.model == nil {
		err := core.E("rocm.Classify", "scheduled model is nil", nil)
		if m != nil {
			m.setErr(err)
		}
		return core.Fail(err)
	}
	m.setErr(nil)
	if err := rocmContextErr(ctx); err != nil {
		m.setErr(err)
		return core.Fail(err)
	}
	result := m.model.Classify(ctx, append([]string(nil), prompts...), opts...)
	if !result.OK {
		err, _ := result.Value.(error)
		m.setErr(err)
		return result
	}
	return core.Ok(cloneClassifyResults(result.Value.([]inference.ClassifyResult)))
}

func (m *ScheduledModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	if m == nil || m.model == nil {
		err := core.E("rocm.BatchGenerate", "scheduled model is nil", nil)
		if m != nil {
			m.setErr(err)
		}
		return core.Fail(err)
	}
	m.setErr(nil)
	if err := rocmContextErr(ctx); err != nil {
		m.setErr(err)
		return core.Fail(err)
	}
	result := m.model.BatchGenerate(ctx, append([]string(nil), prompts...), opts...)
	if !result.OK {
		err, _ := result.Value.(error)
		m.setErr(err)
		return result
	}
	results := cloneBatchResults(result.Value.([]inference.BatchResult))
	if resultErr := firstBatchResultError(results); resultErr != nil {
		m.setErr(resultErr)
	}
	return core.Ok(results)
}

func (m *ScheduledModel) ModelType() string {
	if m == nil || m.model == nil {
		return ""
	}
	return m.model.ModelType()
}

func (m *ScheduledModel) Info() inference.ModelInfo {
	if m == nil || m.model == nil {
		return inference.ModelInfo{}
	}
	return m.model.Info()
}

func (m *ScheduledModel) ModelIdentity() inference.ModelIdentity {
	if m == nil || m.model == nil {
		return inference.ModelIdentity{}
	}
	return rocmDecodeModelIdentity(m.model)
}

func (m *ScheduledModel) ModelProfile() ROCmModelProfile {
	if m == nil || m.model == nil {
		return ROCmModelProfile{}
	}
	if reporter, ok := m.model.(ROCmModelProfileReporter); ok {
		profile := reporter.ModelProfile()
		if profile.Matched() {
			return profile.clone()
		}
	}
	profile, ok := ResolveROCmModelProfileForModel(m.model)
	if !ok {
		return ROCmModelProfile{}
	}
	return profile
}

func (m *ScheduledModel) ModelRoutePlan() ROCmModelRoutePlan {
	if m == nil || m.model == nil {
		return ROCmModelRoutePlan{}
	}
	if reporter, ok := m.model.(ROCmModelRoutePlanReporter); ok {
		plan := reporter.ModelRoutePlan()
		if plan.Matched() {
			return rocmModelRoutePlanWithLiveCacheProfile(plan, m.model)
		}
	}
	profile := m.ModelProfile()
	if !profile.Matched() {
		return ROCmModelRoutePlan{}
	}
	return ROCmModelRoutePlanForProfileAndModel(profile, m.model)
}

func (m *ScheduledModel) Capabilities() inference.CapabilityReport {
	if m == nil || m.model == nil {
		return inference.CapabilityReport{Runtime: inference.RuntimeIdentity{Backend: "rocm"}}
	}
	report := rocmCapabilityReportForWrappedModel(m.model)
	report.Model = m.ModelIdentity()
	labels := map[string]string{
		"wrapper":                 "scheduled_model",
		"scheduler_wrapper":       "rocm",
		"scheduler_output_buffer": core.Sprintf("%d", m.outputBuffer),
	}
	if m.queue != nil {
		labels["scheduler_queue_size"] = core.Sprintf("%d", cap(m.queue))
	}
	m.mu.Lock()
	closed := m.closed
	m.mu.Unlock()
	labels["scheduler_closed"] = core.Sprintf("%t", closed)
	report.Labels = mergeStringMaps(report.Labels, labels)
	schedulerCapability := inference.SupportedCapability(inference.CapabilityScheduler, inference.CapabilityGroupRuntime)
	schedulerCapability.Labels = cloneStringMap(labels)
	cancelCapability := inference.SupportedCapability(inference.CapabilityRequestCancel, inference.CapabilityGroupRuntime)
	cancelCapability.Labels = cloneStringMap(labels)
	rocmCapabilityReportSetCapability(&report, schedulerCapability)
	rocmCapabilityReportSetCapability(&report, cancelCapability)
	return report
}

func (m *ScheduledModel) Metrics() inference.GenerateMetrics {
	if m == nil || m.model == nil {
		return inference.GenerateMetrics{}
	}
	if m.continuous != nil {
		return m.continuous.Metrics()
	}
	return m.model.Metrics()
}

func (m *ScheduledModel) Err() core.Result {
	if m == nil {
		return core.Ok(nil)
	}
	m.mu.Lock()
	err := m.lastErr
	m.mu.Unlock()
	if err != nil {
		return core.Fail(err)
	}
	if m.continuous != nil {
		return m.continuous.Err()
	}
	if m.model == nil {
		return core.Ok(nil)
	}
	return m.model.Err()
}

func (m *ScheduledModel) Close() core.Result {
	if m == nil {
		return core.Ok(nil)
	}
	m.closeOne.Do(func() {
		m.mu.Lock()
		m.closed = true
		for _, cancel := range m.cancel {
			cancel()
		}
		queue := m.queue
		model := m.model
		continuous := m.continuous
		m.mu.Unlock()
		if continuous != nil {
			if result := continuous.Close(); !result.OK {
				m.closeErr, _ = result.Value.(error)
			}
			return
		}
		if queue != nil {
			close(queue)
			m.workerWG.Wait()
		}
		if model != nil {
			if r := model.Close(); !r.OK {
				m.closeErr, _ = r.Value.(error)
			}
		}
	})
	return core.ResultOf(nil, m.closeErr)
}

func (m *ScheduledModel) run() {
	defer m.workerWG.Done()
	for work := range m.queue {
		m.process(work)
	}
}

func (m *ScheduledModel) process(work *scheduledWork) {
	defer func() {
		m.forget(work.id)
		close(work.out)
	}()

	queueLatency := time.Since(work.enqueued)
	if err := work.ctx.Err(); err != nil {
		m.emitSchedulerProbe(work.id, "cancelled_before_start", inference.ProbePhaseQueue, queueLatency, 0, time.Since(work.enqueued), true)
		return
	}
	m.emitSchedulerProbe(work.id, "started", inference.ProbePhasePrefill, queueLatency, 0, queueLatency, false)

	opts := generateOptionsFromSampler(work.req.Sampler)
	var stream iter.Seq[inference.Token]
	if len(work.req.Messages) > 0 {
		stream = m.model.Chat(work.ctx, append([]inference.Message(nil), work.req.Messages...), opts...)
	} else {
		stream = m.model.Generate(work.ctx, work.req.Prompt, opts...)
	}

	start := time.Now()
	var firstTokenLatency time.Duration
	var count int
	cancelled := false
streamLoop:
	for token := range stream {
		if count == 0 {
			firstTokenLatency = time.Since(start)
			m.emitSchedulerProbe(work.id, "first_token", inference.ProbePhaseDecode, queueLatency, firstTokenLatency, time.Since(work.enqueued), false)
		}
		count++
		select {
		case work.out <- inference.ScheduledToken{
			RequestID: work.id,
			Token:     token,
			Metrics:   m.model.Metrics(),
			Labels:    cloneStringMap(work.req.Labels),
		}:
		case <-work.ctx.Done():
			cancelled = true
			break streamLoop
		}
	}
	if work.ctx.Err() != nil {
		cancelled = true
	}
	event := "completed"
	if cancelled {
		event = "cancelled_during_decode"
	}
	m.emitSchedulerProbe(work.id, event, inference.ProbePhaseDecode, queueLatency, firstTokenLatency, time.Since(work.enqueued), cancelled)
}

func (m *ScheduledModel) forget(id string) {
	m.mu.Lock()
	delete(m.cancel, id)
	m.mu.Unlock()
}

func (m *ScheduledModel) emitSchedulerProbe(id, event string, phase inference.ProbePhase, queueLatency, firstTokenLatency, totalLatency time.Duration, cancelled bool) {
	if m == nil {
		return
	}
	m.mu.Lock()
	sink := m.sink
	queueDepth := len(m.queue)
	m.mu.Unlock()
	if sink == nil {
		return
	}
	sink.EmitProbe(inference.ProbeEvent{
		Kind:  inference.ProbeEventScheduler,
		Phase: phase,
		Labels: map[string]string{
			"request_id":             id,
			"event":                  event,
			"cancelled":              core.Sprintf("%t", cancelled),
			"queue_latency_ms":       core.Sprintf("%d", queueLatency.Milliseconds()),
			"first_token_latency_ms": core.Sprintf("%d", firstTokenLatency.Milliseconds()),
		},
		Scheduler: &inference.ProbeScheduler{
			RequestID:               id,
			Event:                   event,
			QueueDepth:              queueDepth,
			QueueLatencyMillis:      durationMilliseconds(queueLatency),
			FirstTokenLatencyMillis: durationMilliseconds(firstTokenLatency),
			TotalLatencyMillis:      durationMilliseconds(totalLatency),
			Cancelled:               cancelled,
		},
	})
}

func (m *ScheduledModel) setErr(err error) {
	if m == nil {
		return
	}
	m.mu.Lock()
	m.lastErr = err
	m.mu.Unlock()
}

func durationMilliseconds(duration time.Duration) float64 {
	return float64(duration) / float64(time.Millisecond)
}

func generateOptionsFromSampler(cfg inference.SamplerConfig) []inference.GenerateOption {
	opts := []inference.GenerateOption{}
	if cfg.MaxTokens > 0 {
		opts = append(opts, inference.WithMaxTokens(cfg.MaxTokens))
	}
	if cfg.Temperature != 0 {
		opts = append(opts, inference.WithTemperature(cfg.Temperature))
	}
	if cfg.TopK != 0 {
		opts = append(opts, inference.WithTopK(cfg.TopK))
	}
	if cfg.TopP != 0 {
		opts = append(opts, inference.WithTopP(cfg.TopP))
	}
	if cfg.MinP != 0 {
		opts = append(opts, inference.WithMinP(cfg.MinP))
	}
	if cfg.RepeatPenalty != 0 {
		opts = append(opts, inference.WithRepeatPenalty(cfg.RepeatPenalty))
	}
	if len(cfg.StopTokens) > 0 {
		opts = append(opts, inference.WithStopTokens(cfg.StopTokens...))
	}
	if cfg.ReturnLogits {
		opts = append(opts, inference.WithLogits())
	}
	return opts
}

func (m *ScheduledModel) samplerConfigFromGenerateOptions(opts []inference.GenerateOption) inference.SamplerConfig {
	cfg := cloneGenerateConfig(inference.ApplyGenerateOpts(opts))
	if m != nil && scheduledModelIsGemma4(m.model) {
		explicit := inference.GenerateConfig{}
		for _, opt := range opts {
			if opt != nil {
				opt(&explicit)
			}
		}
		if explicit.MaxTokens == 0 {
			cfg.MaxTokens = 0
		}
	}
	return inference.SamplerConfigFromGenerateConfig(cfg)
}

func (m *ScheduledModel) validateScheduledGemma4Context(req *inference.ScheduledRequest) error {
	if m == nil || !scheduledModelIsGemma4(m.model) {
		return nil
	}
	if req == nil {
		return core.E("rocm.Schedule", "scheduled request is required", nil)
	}
	contextLength := scheduledModelContextLength(m.model)
	promptTokens, promptKind := scheduledRequestPromptTokenCount(m.model, *req)
	remaining := contextLength - promptTokens
	if remaining <= 0 {
		return core.E("rocm.Schedule", promptKind+" reaches model context window", nil)
	}
	if req.Sampler.MaxTokens > remaining {
		return core.E("rocm.Schedule", "max tokens exceed remaining model context window", nil)
	}
	if req.Sampler.MaxTokens <= 0 {
		req.Sampler.MaxTokens = remaining
	}
	return nil
}

func scheduledModelIsGemma4(model inference.TextModel) bool {
	return isROCmGemma4Architecture(rocmDecodeModelIdentity(model).Architecture)
}

func scheduledModelContextLength(model inference.TextModel) int {
	if identity := rocmDecodeModelIdentity(model); identity.ContextLength > 0 {
		return identity.ContextLength
	}
	if provider, ok := model.(interface{ ContextLength() int }); ok {
		if contextLength := provider.ContextLength(); contextLength > 0 {
			return contextLength
		}
	}
	return defaultContextLengthCap
}

func scheduledRequestPromptTokenCount(model inference.TextModel, req inference.ScheduledRequest) (int, string) {
	if len(req.Messages) > 0 {
		if rocmModel, ok := model.(*rocmModel); ok && rocmModel != nil {
			return rocmModel.chatPromptTokenCount(req.Messages), "messages"
		}
		return rocmDecodePromptTokenCount(model, formatGemma4ChatTemplate(req.Messages)), "messages"
	}
	return rocmDecodePromptTokenCount(model, req.Prompt), "prompt"
}

func cloneSamplerConfig(cfg inference.SamplerConfig) inference.SamplerConfig {
	cfg.StopTokens = append([]int32(nil), cfg.StopTokens...)
	cfg.StopSequences = append([]string(nil), cfg.StopSequences...)
	return cfg
}
