// SPDX-Licence-Identifier: EUPL-1.2

// Package scheduler is the driver-neutral request scheduler for
// inference.TextModel. It is the ONE scheduling home for lem serve: one
// constructor, one submission surface (inference.SchedulerModel.Schedule),
// and a Mode that selects the discipline —
//
//   - ModeSerial: a bounded-queue worker pool wrapping a TextModel with
//     cancellation, streaming backpressure, and scheduler probe events
//     (today's scheduler semantics — the most-tested base).
//   - ModeBatch: continuous in-flight batching — a running set advanced one
//     decode step per iteration under a dual concurrency + token budget
//     (the throughput core, lifted from the former serving/schedule).
//   - ModeInterleave: a live admission-budget scheduler — requests Submit at
//     any time, each admitted onto its own goroutine for per-request
//     backpressure + cancellation isolation (from the former serving/interleave).
//
// Mode-specific engine requirements are probed at construction: batch mode
// (and interleave configured with a token budget) needs the model to expose
// inference.TokenizerModel so prompt tokens can be counted for the
// MaxBatchTokens budget — a model that lacks it fails CLOSED at New rather
// than silently degrading the budget to a no-op. Serial mode needs only a
// TextModel and accepts a nil model (Schedule then reports the nil).
//
//	m, err := scheduler.New(backend, scheduler.Config{
//	    Mode: scheduler.ModeInterleave, MaxConcurrent: 4, MaxQueue: 16, StreamBuffer: 8,
//	})
//	if err != nil { return err }
//	handle, tokens, err := m.Schedule(ctx, request)
package scheduler

import (
	"context"
	"iter"
	"maps"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Mode selects the scheduling discipline. The zero value ("") is treated as
// ModeSerial so existing single-mode constructions keep their behaviour.
type Mode string

const (
	// ModeSerial wraps a TextModel with a bounded-queue worker pool.
	ModeSerial Mode = "serial"
	// ModeBatch runs continuous in-flight batching over a running set.
	ModeBatch Mode = "batch"
	// ModeInterleave admits requests live, each on its own goroutine.
	ModeInterleave Mode = "interleave"
)

// Config configures the unified request scheduler. Not every field is
// meaningful in every mode: MaxBatchTokens gates admission only in batch and
// interleave; RequestIDPrefix/ProbeSink drive serial's request IDs and probe
// stream. Non-positive sizings are clamped per mode so the scheduler always
// makes progress.
type Config struct {
	Mode            Mode                // scheduling discipline; "" = ModeSerial
	MaxConcurrent   int                 // serial workers / batch+interleave running-set cap
	MaxQueue        int                 // bounded queue depth / admission backpressure room
	MaxBatchTokens  int                 // batch+interleave running prompt-token budget; <= 0 = uncapped (batch requires > 0)
	StreamBuffer    int                 // per-request output channel buffer
	RequestIDPrefix string              // serial request-ID prefix (default "scheduler")
	ProbeSink       inference.ProbeSink // serial scheduler-probe sink (nil = no probes)
}

// Stats is a mode-neutral snapshot of a scheduler's counters, safe to read
// concurrently with Schedule/CancelRequest. Not every counter is populated in
// every mode (serial reports Submitted/Completed/Cancelled/Queued; batch adds
// Admitted/Active/MaxRunning; interleave populates all but MaxRunning).
type Stats struct {
	Submitted  int64 // total accepted submissions
	Admitted   int64 // total moved from queue into the running set (batch/interleave)
	Completed  int64 // total that ran to completion
	Cancelled  int64 // total retired by cancellation (queued or active)
	Active     int64 // requests currently running
	Queued     int64 // requests currently waiting for a slot
	MaxRunning int64 // largest running-set ever co-resident (batch witness)
}

// Model wraps an inference.TextModel and schedules requests through the mode
// selected at New. In serial mode it owns a bounded-queue worker pool; in
// batch/interleave mode it delegates Schedule/CancelRequest to the mode engine
// while still presenting the full TextModel surface (Generate/Chat/... route
// through Schedule; the accessors delegate to the wrapped model).
type Model struct {
	base inference.TextModel
	mode Mode

	// --- serial-mode state (owned by the worker pool) ---
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

	// serial-mode counters for Stats (off the per-token hot path).
	submitted atomic.Int64
	completed atomic.Int64
	cancelled atomic.Int64

	// --- non-serial engines (exactly one is non-nil off serial) ---
	batch *batchEngine
	inter *interleaveEngine
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

// New returns a scheduler for model in cfg.Mode, or an error when the mode is
// unknown or the model lacks a capability the mode requires. Serial mode
// accepts a nil model so callers can construct package surfaces before a
// backend loads; batch/interleave-with-budget require an inference.TokenizerModel.
//
//	m, err := scheduler.New(model, scheduler.Config{Mode: scheduler.ModeBatch, MaxConcurrent: 4, MaxBatchTokens: 8192})
func New(model inference.TextModel, cfg Config) (*Model, error) {
	mode := cfg.Mode
	if mode == "" {
		mode = ModeSerial
	}
	switch mode {
	case ModeSerial, ModeBatch, ModeInterleave:
	default:
		return nil, core.E("scheduler.New", "unknown scheduler mode "+string(mode)+" (want serial|batch|interleave)", nil)
	}

	// Mode-specific capability probe — fail closed, never a silent downgrade.
	// Batch's identity is its dual concurrency+token budget; interleave's
	// budget is optional. Either that needs the token budget must be able to
	// count a request's prompt tokens, which is the model's own tokeniser.
	needsTokenBudget := mode == ModeBatch || (mode == ModeInterleave && cfg.MaxBatchTokens > 0)
	if needsTokenBudget {
		if _, ok := model.(inference.TokenizerModel); !ok {
			return nil, core.E("scheduler.New", string(mode)+" mode with a token budget needs an inference.TokenizerModel to count prompt tokens; the model does not expose Encode — refusing to serve a budget that cannot engage", nil)
		}
	}

	m := &Model{
		base:            model,
		mode:            mode,
		maxConcurrent:   maxOrDefault(cfg.MaxConcurrent, 1),
		streamBuffer:    max(cfg.StreamBuffer, 0),
		requestIDPrefix: schedulerPrefix(cfg.RequestIDPrefix),
	}
	if cfg.ProbeSink != nil {
		m.probeSink.Store(&probeSinkBox{sink: cfg.ProbeSink})
	}

	switch mode {
	case ModeBatch:
		m.batch = newBatchEngine(m, cfg)
	case ModeInterleave:
		m.inter = newInterleaveEngine(cfg)
	default: // serial
		m.queue = make(chan *job, max(cfg.MaxQueue, 0))
		for worker := range m.maxConcurrent {
			go m.worker(worker)
		}
	}
	return m, nil
}

func maxOrDefault(v, def int) int {
	if v <= 0 {
		return def
	}
	return v
}

func schedulerPrefix(prefix string) string {
	prefix = core.Trim(prefix)
	if prefix == "" {
		return "scheduler"
	}
	return prefix
}

// Schedule enqueues a generation request and returns its streamed tokens,
// routing through the mode selected at New.
//
//	handle, tokens, err := model.Schedule(ctx, request)
func (m *Model) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	if m == nil || m.base == nil {
		return inference.RequestHandle{}, nil, core.NewError("scheduler: model is nil")
	}
	switch m.mode {
	case ModeBatch:
		return m.scheduleBatch(ctx, req)
	case ModeInterleave:
		return m.scheduleInterleave(ctx, req)
	}
	// serial
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
		m.submitted.Add(1)
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

// CancelRequest cancels a queued or running request by ID, routing through the
// active mode.
//
//	result, err := model.CancelRequest(ctx, id)
func (m *Model) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	if m == nil {
		return inference.RequestCancelResult{ID: id, Reason: "scheduler_nil"}, nil
	}
	if core.Trim(id) == "" {
		return inference.RequestCancelResult{Reason: "missing_id"}, nil
	}
	switch m.mode {
	case ModeBatch:
		return m.batch.cancel(id), nil
	case ModeInterleave:
		return m.inter.cancel(id), nil
	}
	// serial
	value, ok := m.active.Load(id)
	if !ok {
		if cancellable, ok := m.base.(inference.CancellableModel); ok {
			return cancellable.CancelRequest(context.Background(), id)
		}
		return inference.RequestCancelResult{ID: id, Reason: "not_found"}, nil
	}
	j := value.(*job)
	j.cancel()
	m.cancelled.Add(1)
	m.emitProbe(j, "cancel", time.Since(j.queuedAt), 0, true)
	return inference.RequestCancelResult{ID: id, Cancelled: true, Reason: "cancelled"}, nil
}

// Stats returns a mode-neutral snapshot of the scheduler's counters.
//
//	if s := model.Stats(); s.Submitted > 0 { … }
func (m *Model) Stats() Stats {
	if m == nil {
		return Stats{}
	}
	switch m.mode {
	case ModeBatch:
		return m.batch.stats()
	case ModeInterleave:
		return m.inter.stats()
	}
	queued := int64(0)
	if m.queue != nil {
		queued = int64(len(m.queue))
	}
	return Stats{
		Submitted: m.submitted.Load(),
		Completed: m.completed.Load(),
		Cancelled: m.cancelled.Load(),
		Queued:    queued,
	}
}

// Generate schedules a prompt request and yields tokens with scheduler
// backpressure semantics.
//
//	for token := range model.Generate(ctx, prompt) { … }
func (m *Model) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		// Skip the opts→SamplerConfig conversion when the caller supplied
		// none. ApplyGenerateOpts forces &cfg to escape (one heap alloc for
		// the DefaultGenerateConfig it builds) and its RepeatPenalty:1.0
		// default lands in req.Sampler, which then misses generateOptions'
		// zero-value greedy cache and pays a closure + slice (two more
		// allocs) in baseTokens. Leaving Sampler zero-valued routes the
		// no-opts case through the same cached greedy path Schedule(zero
		// request) already uses — byte-identical at the base because its
		// DefaultGenerateConfig re-applies RepeatPenalty:1.0 either way.
		req := inference.ScheduledRequest{Prompt: prompt}
		if len(opts) > 0 {
			req.Sampler = inference.SamplerConfigFromGenerateConfig(inference.ApplyGenerateOpts(opts))
		}
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
		// See Generate: the opts→SamplerConfig conversion is skipped when no
		// opts are supplied so the no-opts case keeps generateOptions on its
		// zero-value greedy cache (saves three allocs, byte-identical config
		// at the base).
		req := inference.ScheduledRequest{Messages: append([]inference.Message(nil), messages...)}
		if len(opts) > 0 {
			req.Sampler = inference.SamplerConfigFromGenerateConfig(inference.ApplyGenerateOpts(opts))
		}
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

// CloseEngine releases the mode engine's goroutines WITHOUT closing the wrapped
// model — for callers (e.g. the serve resolver) that own the model's lifecycle
// separately and only want the scheduler layer torn down.
//
//	model.CloseEngine()
func (m *Model) CloseEngine() {
	if m == nil {
		return
	}
	switch m.mode {
	case ModeBatch:
		if m.batch != nil {
			m.batch.close()
		}
	case ModeInterleave:
		if m.inter != nil {
			m.inter.close()
		}
	}
}

// Close releases the mode engine and the wrapped model. The Result is OK with
// a nil Value on success, or a failure carrying the error.
//
//	model.Close()
func (m *Model) Close() core.Result {
	if m == nil {
		return core.Ok(nil)
	}
	m.CloseEngine()
	if m.base == nil {
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
	m.completed.Add(1)
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
		cfg.TopP == 0 && cfg.MinP == 0 && cfg.RepeatPenalty == 0 && len(cfg.StopTokens) == 0 &&
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
		if cfg.MinP > 0 {
			c.MinP = cfg.MinP
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
	maps.Copy(out, labels)
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
