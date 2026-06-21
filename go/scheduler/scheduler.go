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
	nextID          atomic.Uint64

	// probeSink is read on every scheduler event (queued / start /
	// first_token / cancel / cancelled / complete) and updated only
	// via SetProbeSink. An atomic.Pointer lets emitProbe load the
	// sink without contending m.mu — under burst dispatch we used to
	// pay one mu.Lock per probe event per producer (4 events × 64
	// producers = 256 lock acquisitions per bench iteration even
	// when no sink was attached).
	probeSink atomic.Pointer[probeSinkBox]

	// active holds in-flight jobs keyed by request ID. sync.Map fits
	// the access shape: CancelRequest's lookup is the contended
	// hot-path (32-goroutine parallel cancel-poll measured 4 orders
	// of magnitude slowdown vs the serial bench under a plain Mutex,
	// and ~2x worse under RWMutex due to its accounting overhead),
	// while register/unregister fire exactly twice per request and
	// are tolerant of sync.Map's slightly higher write cost.
	active sync.Map

	mu      sync.Mutex
	lastErr error
}

// probeSinkBox wraps the sink interface so it can be stored in an
// atomic.Pointer (atomic.Value rejects nil-typed interface stores;
// boxing avoids that constraint and keeps the load path branchless).
type probeSinkBox struct {
	sink inference.ProbeSink
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
	}
	if cfg.ProbeSink != nil {
		m.probeSink.Store(&probeSinkBox{sink: cfg.ProbeSink})
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
		// handle.Labels mirrors the request's caller-supplied Labels —
		// skip the map clone when the request has none. Saves one alloc
		// per Schedule in the burst-fan-out path where most producers
		// arrive without custom labels. When labels ARE present, we
		// still clone so callers can't mutate our run-loop view.
		var handleLabels map[string]string
		if len(req.Labels) > 0 {
			handleLabels = cloneLabels(req.Labels)
		}
		return inference.RequestHandle{ID: req.ID, Model: inference.ModelIdentity{ID: req.Model}, Labels: handleLabels}, j.out, nil
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
	value, ok := m.active.Load(id)
	if !ok {
		if cancellable, ok := m.base.(inference.CancellableModel); ok {
			return cancellable.CancelRequest(context.Background(), id)
		}
		return inference.RequestCancelResult{ID: id, Reason: "not_found"}, nil
	}
	j := value.(*job)
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
//	cr := model.Classify(ctx, prompts)
//	if !cr.OK { return cr }
//	results := cr.Value.([]inference.ClassifyResult)
func (m *Model) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	if m == nil || m.base == nil {
		return core.Fail(core.E("scheduler.Classify", "model is nil", nil))
	}
	return m.base.Classify(ctx, prompts, opts...)
}

// BatchGenerate delegates batch generation to the wrapped model.
//
//	br := model.BatchGenerate(ctx, prompts)
//	if !br.OK { return br }
//	batches := br.Value.([]inference.BatchResult)
func (m *Model) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	if m == nil || m.base == nil {
		return core.Fail(core.E("scheduler.BatchGenerate", "model is nil", nil))
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

// Err reports the most recent error from the scheduler or the wrapped model.
// The Result is OK with a nil Value when there is no error.
//
//	if r := model.Err(); !r.OK { … }
func (m *Model) Err() core.Result {
	if m == nil {
		return core.Ok(nil)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.lastErr != nil {
		return core.Fail(m.lastErr)
	}
	if m.base == nil {
		return core.Ok(nil)
	}
	return m.base.Err()
}

// Close releases the wrapped model. The Result is OK with a nil Value on
// success, or a failure carrying the error.
//
//	model.Close()
func (m *Model) Close() core.Result {
	if m == nil || m.base == nil {
		return core.Ok(nil)
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
	if sink == nil {
		m.probeSink.Store(nil)
		return
	}
	m.probeSink.Store(&probeSinkBox{sink: sink})
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
	// Build the per-request label map once. queue_latency_ms is fixed
	// at run() entry; first_token_latency_ms lands on first token and
	// is observability metadata about the request (not the individual
	// token), so we leave it in the shared map for the remainder of
	// the stream. Hoisting cloneLabels + millisString out of the
	// per-token loop is the biggest streaming alloc lift — 256-token
	// generates went from ~3 allocs/token to ~1.
	labels := cloneLabels(j.req.Labels)
	labels["queue_latency_ms"] = millisString(queueLatency)
	firstToken := true
	var firstLatency time.Duration
	for token := range m.baseTokens(j) {
		if firstToken {
			firstLatency = time.Since(startedAt)
			firstToken = false
			labels["first_token_latency_ms"] = millisString(firstLatency)
			m.emitProbe(j, "first_token", queueLatency, firstLatency, false)
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
	if r := m.base.Err(); !r.OK {
		if err, ok := r.Value.(error); ok {
			m.setErr(err)
		} else {
			m.setErr(core.NewError(r.Error()))
		}
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
	m.active.Store(j.req.ID, j)
}

func (m *Model) unregister(id string) {
	m.active.Delete(id)
}

func (m *Model) emitProbe(j *job, event string, queueLatency, firstTokenLatency time.Duration, cancelled bool) {
	if j == nil {
		return
	}
	// Lock-free fast path — burst-dispatch typically runs with no
	// sink attached; the atomic load + nil check returns in nanoseconds
	// and never contends the mutex that guards lastErr.
	box := m.probeSink.Load()
	if box == nil {
		return
	}
	sink := box.sink
	if sink == nil {
		return
	}
	// Channel len is internally atomic — safe to read without a lock.
	queueDepth := len(m.queue)
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
	// Fires per scheduled request. Hand-built via strconv.AppendInt
	// instead of Sprintf — Sprintf walks the fmt formatter pipeline
	// (~2 allocs); AppendInt into a pre-sized buffer + AsString is 1.
	id := m.nextID.Add(1)
	buf := make([]byte, 0, len(m.requestIDPrefix)+21)
	buf = append(buf, m.requestIDPrefix...)
	buf = append(buf, '-')
	buf = strconv.AppendUint(buf, id, 10)
	return core.AsString(buf)
}

// schedGreedyOpts is the cached single-option slice for the zero-value
// (greedy) sampler — the burst-dispatch case where callers leave
// Sampler unset. The closure forces Temperature to 0 (explicit greedy)
// and touches nothing else, so the base defaults survive. Caching the
// whole slice keeps that hot path at zero per-call allocation. The
// closure must never mutate cfg-derived state since it is shared.
var schedGreedyOpts = []inference.GenerateOption{func(c *inference.GenerateConfig) { c.Temperature = 0 }}

func generateOptions(cfg inference.SamplerConfig) []inference.GenerateOption {
	// Zero-value sampler (greedy, no overrides) is the burst-dispatch
	// default — serve it from the cached slice so it stays allocation-
	// free, exactly the old schedTempZeroOpt fast path. SamplerConfig
	// holds slice fields so it is not == comparable; check the fields
	// the applier would act on.
	if cfg.MaxTokens == 0 && cfg.Temperature == 0 && cfg.TopK == 0 &&
		cfg.TopP == 0 && cfg.RepeatPenalty == 0 && len(cfg.StopTokens) == 0 &&
		!cfg.ReturnLogits {
		return schedGreedyOpts
	}
	// One closure capturing the whole SamplerConfig instead of up to
	// seven separate WithX closures + a 7-cap slice. Each inference.WithX
	// returns a fresh func value that captures one field — heap-allocated
	// per call — so the previous shape paid 1-7 closure allocs plus the
	// backing-array alloc on every Schedule. The single applier preserves
	// the exact conditional semantics (only override a base default when
	// the sampler carries a meaningful value; Temperature is always set so
	// greedy/zero survives the base default), in one closure alloc + a
	// len-1 slice. Fires once per scheduled request.
	return []inference.GenerateOption{func(c *inference.GenerateConfig) {
		if cfg.MaxTokens > 0 {
			c.MaxTokens = cfg.MaxTokens
		}
		c.Temperature = cfg.Temperature
		if cfg.TopK > 0 {
			c.TopK = cfg.TopK
		}
		if cfg.TopP > 0 {
			c.TopP = cfg.TopP
		}
		if cfg.RepeatPenalty > 0 {
			c.RepeatPenalty = cfg.RepeatPenalty
		}
		if len(cfg.StopTokens) > 0 {
			c.StopTokens = core.SliceClone(cfg.StopTokens)
		}
		if cfg.ReturnLogits {
			c.ReturnLogits = true
		}
	}}
}

func cloneLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		// Preserve the original "empty/nil → fresh empty map" contract
		// callers relied on, but skip the unnecessary make+copy.
		return map[string]string{}
	}
	out := make(map[string]string, len(labels))
	for key, value := range labels {
		out[key] = value
	}
	return out
}

func millisString(duration time.Duration) string {
	// Sprintf("%.3f") was 2 allocs; FormatFloat returns the result
	// string directly without the formatter pipeline.
	return strconv.FormatFloat(millis(duration), 'f', 3, 64)
}

func millis(duration time.Duration) float64 {
	if duration <= 0 {
		return 0
	}
	return float64(duration) / float64(time.Millisecond)
}
