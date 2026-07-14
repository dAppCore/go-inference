// SPDX-Licence-Identifier: EUPL-1.2

// Batch mode — continuous in-flight batching. This is the throughput core
// lifted from the former serving/schedule package: a RUNNING SET of decoding
// requests, advanced one decode step per round in lockstep, admitting from a
// FIFO queue as slots and the token budget free, retiring finished sequences,
// admitting more, until queue and running set are both empty.
//
// Unlike serving/schedule's static Run(ctx, []Request, Stepper, onToken) — one
// blocking call over a pre-known slice — batchEngine is LIVE: requests Submit
// at any time and stream real inference.Token payloads on their own channel.
// It keeps schedule's dual admission budget (MaxConcurrency AND MaxBatchTokens)
// and its oversize-prompt / cap retirement rules, adopts interleave's bounded
// MaxQueue backpressure (the better queue bound of the two merged packages),
// and drives real per-request token sources (production: model.Generate/Chat;
// tests: fake generators) via iter.Pull instead of an injected int Stepper.
//
// The lockstep discipline is deliberate and is what distinguishes batch from
// interleave: one coordinator goroutine advances every running request exactly
// one token per round, so a round's admission decision sees the whole running
// set together (the batch witness Stats().MaxRunning records the largest set
// ever co-resident). The trade-off vs interleave is that batch shares a single
// backpressure/prefill lane — a newly-admitted request's first pull pays its
// prefill inside the round (briefly stalling its peers), and a stalled consumer
// backpressures the whole set until it drains or is cancelled. Interleave mode
// (per-request goroutines) is the choice when that isolation matters more than
// strict token-level lockstep.
package scheduler

import (
	"context"
	"iter"
	"sync"
	"sync/atomic"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// stream is one request's token generator — the exact shape
// inference.TextModel's Generate/Chat already returns. Shared by batch and
// interleave modes so both drive the model's real per-session stream unchanged.
type stream = iter.Seq[inference.Token]

// source constructs one request's stream bound to a per-request context (a
// child of the caller's Submit ctx). Cancelling that ctx cancels the stream,
// exactly as cancelling a direct Generate/Chat call would.
type source func(ctx context.Context) stream

// batchReq is one admitted-or-queued unit of work. Only the run loop mutates
// the queue/running bookkeeping, so no lock guards it.
type batchReq struct {
	id           string
	promptTokens int // counts against MaxBatchTokens
	maxNewTokens int // hard cap on generated tokens; <= 0 = run the source to its natural end
	src          source
	out          chan inference.ScheduledToken
	labels       map[string]string
	metrics      func() inference.GenerateMetrics
	ctx          context.Context
	cancel       context.CancelFunc

	// live pull state, set at admission:
	next      func() (inference.Token, bool)
	stop      func()
	generated int
}

// batchEngine runs one live continuous-batching loop. Construct with
// newBatchEngine; the returned engine is immediately live and must be closed to
// release its goroutine.
type batchEngine struct {
	cap          int // MaxConcurrency — running-set cap (clamped >= 1)
	maxQueue     int // MaxQueue — waiting-room for requests not yet admitted (clamped >= 1)
	maxTokens    int // MaxBatchTokens — running prompt-token budget (<= 0 = uncapped)
	streamBuffer int // per-request output channel buffer

	submitCh  chan *batchReq
	cancelCh  chan string
	closeCh   chan struct{}
	doneCh    chan struct{}
	closeOnce sync.Once

	submitted, admitted, completed, cancelled, active, queued, maxRunning atomic.Int64
}

// newBatchEngine builds a live batch engine from cfg and starts its loop.
func newBatchEngine(_ *Model, cfg Config) *batchEngine {
	e := &batchEngine{
		cap:          maxOrDefault(cfg.MaxConcurrent, 1),
		maxQueue:     maxOrDefault(cfg.MaxQueue, 1),
		maxTokens:    cfg.MaxBatchTokens,
		streamBuffer: max(cfg.StreamBuffer, 0),
		submitCh:     make(chan *batchReq),
		cancelCh:     make(chan string),
		closeCh:      make(chan struct{}),
		doneCh:       make(chan struct{}),
	}
	go e.run()
	return e
}

// scheduleBatch builds a batch request from a serve ScheduledRequest and
// submits it, counting prompt tokens through the model's own tokeniser for the
// budget.
func (m *Model) scheduleBatch(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	id := core.Trim(req.ID)
	if id == "" {
		id = m.nextRequestID()
	}
	opts := generateOptions(req.Sampler)
	if req.MetricsSink != nil {
		// Re-arm the request-scoped metrics sink dropped by the SamplerConfig
		// fold — the base engine delivers this request's own final metrics.
		opts = append(opts, inference.WithMetricsSink(req.MetricsSink))
	}
	messages := append([]inference.Message(nil), req.Messages...)
	prompt := req.Prompt
	src := func(rc context.Context) stream {
		if len(messages) > 0 {
			return m.base.Chat(rc, messages, opts...)
		}
		return m.base.Generate(rc, prompt, opts...)
	}
	br := &batchReq{
		id:           id,
		promptTokens: m.countPromptTokens(req),
		maxNewTokens: req.Sampler.MaxTokens,
		src:          src,
		labels:       cloneLabels(req.Labels),
		metrics:      m.base.Metrics,
	}
	out, err := m.batch.submit(ctx, br)
	if err != nil {
		return inference.RequestHandle{}, nil, err
	}
	var handleLabels map[string]string
	if len(req.Labels) > 0 {
		handleLabels = cloneLabels(req.Labels)
	}
	return inference.RequestHandle{ID: id, Model: inference.ModelIdentity{ID: req.Model}, Labels: handleLabels}, out, nil
}

// countPromptTokens counts a request's prompt tokens through the model's own
// tokeniser (guaranteed present for batch mode by the New capability probe).
// Messages are rendered through the chat template first so the count reflects
// what the model actually prefixes; a template failure falls back to the raw
// concatenation rather than failing the request.
func (m *Model) countPromptTokens(req inference.ScheduledRequest) int {
	// inference.As, not a direct assert: the base may be wrapped (serialModel
	// for a single-session model, or a welfare/policy decorator), which does not
	// re-declare TokenizerModel — As reaches it through Unwrap.
	tok, ok := inference.As[inference.TokenizerModel](m.base)
	if !ok {
		return 0
	}
	text := req.Prompt
	if len(req.Messages) > 0 {
		if rendered, err := tok.ApplyChatTemplate(req.Messages); err == nil {
			text = rendered
		} else {
			for _, msg := range req.Messages {
				text += msg.Content
			}
		}
	}
	return len(tok.Encode(text))
}

// submit enqueues a prepared batch request. It blocks only until the request
// reaches the queue (MaxQueue backpressure) or ctx/close fires first; once
// admitted, its source runs in the coordinator's lockstep rounds and its tokens
// stream on br.out.
func (e *batchEngine) submit(ctx context.Context, br *batchReq) (<-chan inference.ScheduledToken, error) {
	if br == nil || br.src == nil {
		return nil, core.E("scheduler.batch.submit", "nil source", nil)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	select {
	case <-e.closeCh:
		return nil, core.E("scheduler.batch.submit", "engine is closed", nil)
	default:
	}
	reqCtx, cancel := context.WithCancel(ctx)
	br.ctx = reqCtx
	br.cancel = cancel
	br.out = make(chan inference.ScheduledToken, e.streamBuffer)
	select {
	case e.submitCh <- br:
		e.submitted.Add(1)
		return br.out, nil
	case <-ctx.Done():
		cancel()
		return nil, ctx.Err()
	case <-e.closeCh:
		cancel()
		return nil, core.E("scheduler.batch.submit", "engine is closed", nil)
	}
}

// cancel cancels the request identified by id — a queued one never runs (its
// channel closes with zero tokens), an active one observes ctx.Done() at its
// next round. Unknown/finished ids are harmless no-ops.
func (e *batchEngine) cancel(id string) inference.RequestCancelResult {
	select {
	case e.cancelCh <- id:
	case <-e.doneCh:
	}
	return inference.RequestCancelResult{ID: id, Cancelled: true, Reason: "cancelled"}
}

// stats returns a snapshot of the engine's counters.
func (e *batchEngine) stats() Stats {
	return Stats{
		Submitted:  e.submitted.Load(),
		Admitted:   e.admitted.Load(),
		Completed:  e.completed.Load(),
		Cancelled:  e.cancelled.Load(),
		Active:     e.active.Load(),
		Queued:     e.queued.Load(),
		MaxRunning: e.maxRunning.Load(),
	}
}

// close stops admitting, cancels every queued and active request, waits for the
// loop to drain, and returns — no goroutine (loop or iter.Pull) survives.
func (e *batchEngine) close() {
	e.closeOnce.Do(func() { close(e.closeCh) })
	<-e.doneCh
}

// run is the single coordinator goroutine: it owns the queue and running set
// outright, reacting to submissions, cancellations and close between lockstep
// decode rounds.
func (e *batchEngine) run() {
	defer close(e.doneCh)

	var (
		queue        []*batchReq
		running      = make([]*batchReq, 0, e.cap)
		activeTokens int
	)

	retire := func(req *batchReq, cancelled bool) {
		req.cancel()
		if req.stop != nil {
			req.stop()
		}
		close(req.out)
		activeTokens -= req.promptTokens
		if cancelled {
			e.cancelled.Add(1)
		} else {
			e.completed.Add(1)
		}
	}

	admit := func() {
		for len(queue) > 0 && len(running) < e.cap {
			req := queue[0]
			// Already cancelled while queued → retire without ever running it.
			if req.ctx.Err() != nil {
				queue = queue[1:]
				req.cancel()
				close(req.out)
				e.cancelled.Add(1)
				continue
			}
			// Oversize prompt can never satisfy the token budget → retire it
			// (closed channel, no tokens) rather than wedge the queue head.
			if e.maxTokens > 0 && req.promptTokens > e.maxTokens {
				queue = queue[1:]
				req.cancel()
				close(req.out)
				e.cancelled.Add(1)
				continue
			}
			// Budget gate: admitting this prompt must keep the running
			// prompt-token total within MaxBatchTokens. If not, stop admitting
			// and let the running set drain first.
			if e.maxTokens > 0 && len(running) > 0 && activeTokens+req.promptTokens > e.maxTokens {
				break
			}
			queue = queue[1:]
			req.next, req.stop = iter.Pull(req.src(req.ctx))
			running = append(running, req)
			activeTokens += req.promptTokens
			e.admitted.Add(1)
			if int64(len(running)) > e.maxRunning.Load() {
				e.maxRunning.Store(int64(len(running)))
			}
		}
		e.active.Store(int64(len(running)))
		e.queued.Store(int64(len(queue)))
	}

	findAndCancel := func(id string) {
		for i, req := range queue {
			if req.id == id {
				req.cancel()
				queue = append(queue[:i], queue[i+1:]...)
				close(req.out)
				e.cancelled.Add(1)
				e.queued.Store(int64(len(queue)))
				return
			}
		}
		for _, req := range running {
			if req.id == id {
				req.cancel() // observed at its next round; retired there
				return
			}
		}
	}

	shutdown := func() {
		for drained := false; !drained; {
			select {
			case req := <-e.submitCh:
				queue = append(queue, req)
			default:
				drained = true
			}
		}
		for _, req := range queue {
			req.cancel()
			close(req.out)
			e.cancelled.Add(1)
		}
		queue = nil
		e.queued.Store(0)
		for _, req := range running {
			retire(req, true)
		}
		running = nil
		e.active.Store(0)
	}

	for {
		// submitSrc is nil (its case blocks forever) whenever the queue is
		// already at MaxQueue — real backpressure, adopted from interleave.
		var submitSrc chan *batchReq
		if len(queue) < e.maxQueue {
			submitSrc = e.submitCh
		}

		if len(running) == 0 {
			// Idle — block on a control event rather than busy-spin.
			select {
			case req := <-submitSrc:
				queue = append(queue, req)
				admit()
			case id := <-e.cancelCh:
				findAndCancel(id)
			case <-e.closeCh:
				shutdown()
				return
			}
			continue
		}

		// Drain one pending control event before the next round so live
		// submissions and cancels are honoured between decode steps (the
		// "continuous" in continuous batching).
		select {
		case req := <-submitSrc:
			queue = append(queue, req)
			admit()
			continue
		case id := <-e.cancelCh:
			findAndCancel(id)
			continue
		case <-e.closeCh:
			shutdown()
			return
		default:
		}

		// One lockstep decode round: advance every running request by one
		// token, deliver it, and retire those that finish (source EOS, cap, or
		// cancellation). running is mutated in place — retired requests are
		// spliced out, so on close it holds exactly the still-active set.
		if e.stepRound(&running, retire, shutdown) {
			return
		}
		e.active.Store(int64(len(running)))
		admit()
	}
}

// stepRound advances every running request one token. It returns true when the
// engine was closed mid-round (shutdown already ran, the caller must return).
func (e *batchEngine) stepRound(runningPtr *[]*batchReq, retire func(*batchReq, bool), shutdown func()) bool {
	running := *runningPtr
	i := 0
	for i < len(running) {
		req := running[i]
		if req.ctx.Err() != nil {
			retire(req, true)
			running = append(running[:i], running[i+1:]...)
			continue
		}
		tok, ok := req.next()
		if !ok {
			retire(req, req.ctx.Err() != nil)
			running = append(running[:i], running[i+1:]...)
			continue
		}
		req.generated++
		select {
		case req.out <- inference.ScheduledToken{RequestID: req.id, Token: tok, Metrics: req.metrics(), Labels: req.labels}:
		case <-req.ctx.Done():
			retire(req, true)
			running = append(running[:i], running[i+1:]...)
			continue
		case <-e.closeCh:
			// running still holds every not-yet-retired request (those before i
			// survived the round, req at i and those after are untouched), so a
			// plain shutdown over it retires the whole live set uniformly.
			*runningPtr = running
			shutdown()
			return true
		}
		if req.maxNewTokens > 0 && req.generated >= req.maxNewTokens {
			retire(req, false)
			running = append(running[:i], running[i+1:]...)
			continue
		}
		i++
	}
	*runningPtr = running
	return false
}
