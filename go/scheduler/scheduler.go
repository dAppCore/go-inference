// SPDX-Licence-Identifier: EUPL-1.2

// Package scheduler is the driver-neutral request scheduler for
// inference.TextModel. It wraps a model with bounded queueing,
// cancellation, streaming backpressure, and scheduler probe events.
//
//	model := scheduler.New(backend, scheduler.Config{
//	    MaxConcurrent: 4, MaxQueue: 16, StreamBuffer: 8,
//	    RequestIDPrefix: "ide", ProbeSink: sink,
//	})
//	handle, tokens, err := model.Schedule(ctx, request)
package scheduler

import (
	"context"
	"iter"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Config configures the package-first request scheduler.
type Config struct {
	MaxConcurrent   int
	MaxQueue        int
	StreamBuffer    int
	RequestIDPrefix string
	ProbeSink       inference.ProbeSink
}

// Model wraps an inference.TextModel with bounded queueing,
// cancellation, streaming backpressure, and scheduler probe events.
type Model struct {
	base            inference.TextModel
	queue           chan *job
	maxConcurrent   int
	streamBuffer    int
	requestIDPrefix string
	probeSink       inference.ProbeSink
	nextID          atomic.Uint64

	mu      sync.Mutex
	active  map[string]*job
	lastErr error
}

type job struct {
	req      inference.ScheduledRequest
	ctx      context.Context
	cancel   context.CancelFunc
	out      chan inference.ScheduledToken
	queuedAt time.Time
}

// New returns a scheduler wrapper for model. Nil models are accepted so
// callers can construct package surfaces before a backend loads.
//
//	scheduler := scheduler.New(model, scheduler.Config{MaxConcurrent: 4})
func New(model inference.TextModel, cfg Config) *Model {
	maxConcurrent := cfg.MaxConcurrent
	if maxConcurrent <= 0 {
		maxConcurrent = 1
	}
	maxQueue := cfg.MaxQueue
	if maxQueue < 0 {
		maxQueue = 0
	}
	streamBuffer := cfg.StreamBuffer
	if streamBuffer < 0 {
		streamBuffer = 0
	}
	prefix := core.Trim(cfg.RequestIDPrefix)
	if prefix == "" {
		prefix = "scheduler"
	}
	m := &Model{
		base:            model,
		queue:           make(chan *job, maxQueue),
		maxConcurrent:   maxConcurrent,
		streamBuffer:    streamBuffer,
		requestIDPrefix: prefix,
		probeSink:       cfg.ProbeSink,
		active:          map[string]*job{},
	}
	for worker := range maxConcurrent {
		go m.worker(worker)
	}
	return m
}

// Schedule enqueues a generation request and returns its streamed tokens.
//
//	handle, tokens, err := model.Schedule(ctx, request)
func (m *Model) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	if m == nil || m.base == nil {
		return inference.RequestHandle{}, nil, core.NewError("scheduler: model is nil")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return inference.RequestHandle{}, nil, err
	}
	if core.Trim(req.ID) == "" {
		req.ID = m.nextRequestID()
	}
	reqCtx, cancel := context.WithCancel(ctx)
	j := &job{
		req:      req,
		ctx:      reqCtx,
		cancel:   cancel,
		out:      make(chan inference.ScheduledToken, m.streamBuffer),
		queuedAt: time.Now(),
	}
	m.register(j)
	select {
	case m.queue <- j:
		m.emitProbe(j, "queued", 0, 0, false)
		return inference.RequestHandle{ID: req.ID, Model: inference.ModelIdentity{ID: req.Model}, Labels: cloneLabels(req.Labels)}, j.out, nil
	case <-ctx.Done():
		m.unregister(req.ID)
		cancel()
		close(j.out)
		return inference.RequestHandle{}, nil, ctx.Err()
	default:
		m.unregister(req.ID)
		cancel()
		close(j.out)
		return inference.RequestHandle{}, nil, core.NewError("scheduler: queue is full")
	}
}

// CancelRequest cancels a queued or running request by ID.
//
//	result, err := model.CancelRequest(ctx, id)
func (m *Model) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	if m == nil {
		return inference.RequestCancelResult{ID: id, Reason: "scheduler_nil"}, nil
	}
	if core.Trim(id) == "" {
		return inference.RequestCancelResult{Reason: "missing_id"}, nil
	}
	m.mu.Lock()
	j := m.active[id]
	m.mu.Unlock()
	if j == nil {
		if cancellable, ok := m.base.(inference.CancellableModel); ok {
			return cancellable.CancelRequest(context.Background(), id)
		}
		return inference.RequestCancelResult{ID: id, Reason: "not_found"}, nil
	}
	j.cancel()
	m.emitProbe(j, "cancel", time.Since(j.queuedAt), 0, true)
	return inference.RequestCancelResult{ID: id, Cancelled: true, Reason: "cancelled"}, nil
}

// Generate schedules a prompt request and yields tokens with scheduler
// backpressure semantics.
//
//	for token := range model.Generate(ctx, prompt) { … }
func (m *Model) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		req := inference.ScheduledRequest{Prompt: prompt, Sampler: inference.SamplerConfigFromGenerateConfig(inference.ApplyGenerateOpts(opts))}
		_, tokens, err := m.Schedule(ctx, req)
		if err != nil {
			m.setErr(err)
			return
		}
		for scheduled := range tokens {
			if !yield(scheduled.Token) {
				_, _ = m.CancelRequest(ctx, scheduled.RequestID)
				return
			}
		}
	}
}

// Chat schedules a chat request and yields tokens with scheduler
// backpressure semantics.
//
//	for token := range model.Chat(ctx, messages) { … }
func (m *Model) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		req := inference.ScheduledRequest{Messages: append([]inference.Message(nil), messages...), Sampler: inference.SamplerConfigFromGenerateConfig(inference.ApplyGenerateOpts(opts))}
		_, tokens, err := m.Schedule(ctx, req)
		if err != nil {
			m.setErr(err)
			return
		}
		for scheduled := range tokens {
			if !yield(scheduled.Token) {
				_, _ = m.CancelRequest(ctx, scheduled.RequestID)
				return
			}
		}
	}
}

// Classify delegates classification to the wrapped model.
//
//	results, err := model.Classify(ctx, prompts)
func (m *Model) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.ClassifyResult, error) {
	if m == nil || m.base == nil {
		return nil, core.NewError("scheduler: model is nil")
	}
	return m.base.Classify(ctx, prompts, opts...)
}

// BatchGenerate delegates batch generation to the wrapped model.
//
//	batches, err := model.BatchGenerate(ctx, prompts)
func (m *Model) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) ([]inference.BatchResult, error) {
	if m == nil || m.base == nil {
		return nil, core.NewError("scheduler: model is nil")
	}
	return m.base.BatchGenerate(ctx, prompts, opts...)
}

// ModelType returns the wrapped model's type name.
//
//	t := model.ModelType()
func (m *Model) ModelType() string {
	if m == nil || m.base == nil {
		return ""
	}
	return m.base.ModelType()
}

// Info returns the wrapped model's identity.
//
//	info := model.Info()
func (m *Model) Info() inference.ModelInfo {
	if m == nil || m.base == nil {
		return inference.ModelInfo{}
	}
	return m.base.Info()
}

// Metrics returns the wrapped model's last reported metrics.
//
//	metrics := model.Metrics()
func (m *Model) Metrics() inference.GenerateMetrics {
	if m == nil || m.base == nil {
		return inference.GenerateMetrics{}
	}
	return m.base.Metrics()
}

// Err returns the most recent error from the scheduler or the wrapped model.
//
//	if err := model.Err(); err != nil { … }
func (m *Model) Err() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.lastErr != nil {
		return m.lastErr
	}
	if m.base == nil {
		return nil
	}
	return m.base.Err()
}

// Close releases the wrapped model.
//
//	model.Close()
func (m *Model) Close() error {
	if m == nil || m.base == nil {
		return nil
	}
	return m.base.Close()
}

// SetProbeSink updates the scheduler probe sink.
//
//	model.SetProbeSink(sink)
func (m *Model) SetProbeSink(sink inference.ProbeSink) {
	if m == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.probeSink = sink
}

func (m *Model) worker(_ int) {
	for j := range m.queue {
		m.run(j)
	}
}

func (m *Model) run(j *job) {
	defer close(j.out)
	defer m.unregister(j.req.ID)
	queueLatency := time.Since(j.queuedAt)
	if err := j.ctx.Err(); err != nil {
		m.emitProbe(j, "cancelled", queueLatency, 0, true)
		return
	}
	startedAt := time.Now()
	m.emitProbe(j, "start", queueLatency, 0, false)
	// queueLatency is fixed for the whole request; format once and
	// reuse across every emitted token instead of paying a Sprintf per
	// token. firstTokenLatencyMS materialises the moment we see the
	// first token, then stays constant for the remainder of the stream.
	queueLatencyMS := millisString(queueLatency)
	firstTokenLatencyMS := ""
	firstToken := true
	requestLabelsCount := len(j.req.Labels)
	for token := range m.baseTokens(j) {
		firstLatency := time.Duration(0)
		if firstToken {
			firstLatency = time.Since(startedAt)
			firstToken = false
			m.emitProbe(j, "first_token", queueLatency, firstLatency, false)
			if firstLatency > 0 {
				firstTokenLatencyMS = millisString(firstLatency)
			}
		}
		// Build the per-token label map with a known final size so the
		// map grows without re-bucketing as we assign.
		extra := 1
		if firstTokenLatencyMS != "" {
			extra = 2
		}
		labels := make(map[string]string, requestLabelsCount+extra)
		for key, value := range j.req.Labels {
			labels[key] = value
		}
		labels["queue_latency_ms"] = queueLatencyMS
		if firstTokenLatencyMS != "" {
			labels["first_token_latency_ms"] = firstTokenLatencyMS
		}
		select {
		case <-j.ctx.Done():
			m.emitProbe(j, "cancelled", queueLatency, firstLatency, true)
			return
		case j.out <- inference.ScheduledToken{
			RequestID: j.req.ID,
			Token:     token,
			Metrics:   m.base.Metrics(),
			Labels:    labels,
		}:
		}
	}
	if err := m.base.Err(); err != nil {
		m.setErr(err)
	}
	m.emitProbe(j, "complete", queueLatency, 0, false)
}

func (m *Model) baseTokens(j *job) iter.Seq[inference.Token] {
	opts := generateOptions(j.req.Sampler)
	if len(j.req.Messages) > 0 {
		messages := append([]inference.Message(nil), j.req.Messages...)
		return m.base.Chat(j.ctx, messages, opts...)
	}
	return m.base.Generate(j.ctx, j.req.Prompt, opts...)
}

func (m *Model) register(j *job) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.active[j.req.ID] = j
}

func (m *Model) unregister(id string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.active, id)
}

func (m *Model) emitProbe(j *job, event string, queueLatency, firstTokenLatency time.Duration, cancelled bool) {
	m.mu.Lock()
	sink := m.probeSink
	queueDepth := len(m.queue)
	m.mu.Unlock()
	if sink == nil || j == nil {
		return
	}
	sink.EmitProbe(inference.ProbeEvent{
		Kind:  inference.ProbeEventScheduler,
		Phase: inference.ProbePhaseQueue,
		Labels: map[string]string{
			"request_id": j.req.ID,
			"event":      event,
			"model":      j.req.Model,
		},
		Scheduler: &inference.ProbeScheduler{
			RequestID:               j.req.ID,
			Event:                   event,
			QueueDepth:              queueDepth,
			QueueLatencyMillis:      millis(queueLatency),
			FirstTokenLatencyMillis: millis(firstTokenLatency),
			TotalLatencyMillis:      millis(time.Since(j.queuedAt)),
			Cancelled:               cancelled,
		},
	})
}

func (m *Model) setErr(err error) {
	if m == nil || err == nil {
		return
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.lastErr = err
}

func (m *Model) nextRequestID() string {
	// Hand-roll the "<prefix>-<seq>" format with strconv.AppendUint to
	// skip fmt's per-call reflection and intermediate allocations. One
	// alloc for the result string instead of three.
	id := m.nextID.Add(1)
	buf := make([]byte, 0, len(m.requestIDPrefix)+1+20)
	buf = append(buf, m.requestIDPrefix...)
	buf = append(buf, '-')
	buf = strconv.AppendUint(buf, id, 10)
	return string(buf)
}

func generateOptions(cfg inference.SamplerConfig) []inference.GenerateOption {
	opts := []inference.GenerateOption{}
	if cfg.MaxTokens > 0 {
		opts = append(opts, inference.WithMaxTokens(cfg.MaxTokens))
	}
	opts = append(opts, inference.WithTemperature(cfg.Temperature))
	if cfg.TopK > 0 {
		opts = append(opts, inference.WithTopK(cfg.TopK))
	}
	if cfg.TopP > 0 {
		opts = append(opts, inference.WithTopP(cfg.TopP))
	}
	if cfg.RepeatPenalty > 0 {
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

func cloneLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return map[string]string{}
	}
	out := make(map[string]string, len(labels))
	for key, value := range labels {
		out[key] = value
	}
	return out
}

func millisString(duration time.Duration) string {
	return core.Sprintf("%.3f", millis(duration))
}

func millis(duration time.Duration) float64 {
	if duration <= 0 {
		return 0
	}
	return float64(duration) / float64(time.Millisecond)
}
