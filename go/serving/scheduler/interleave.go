// SPDX-Licence-Identifier: EUPL-1.2

// Interleave mode — the live continuous-batching admission scheduler lifted
// from the former serving/interleave package. Requests Submit at any time —
// including while others are mid-decode — and are admitted into a bounded
// running set under a dual concurrency + token budget, FIFO otherwise. Each
// admitted request runs its source on its OWN goroutine, delivering tokens to
// its own bounded channel.
//
// This per-request-goroutine isolation is the deliberate difference from batch
// mode's single lockstep coordinator: a newly-admitted session's first pull
// (which also pays its prefill) never blocks any other active session's tokens,
// and one slow consumer backpressures only its own request. The cost is that
// fairness is admission-level, not strict token-level lockstep (batch mode is
// the choice when the stronger property is wanted). See
// docs/design-continuous-batching.md, slice 1, for why literal lockstep
// round-robin was rejected for this mode.
package scheduler

import (
	"context"
	"sync"
	"sync/atomic"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// interleaveReq is one admitted-or-queued unit of work; only the run loop
// mutates the queue/active bookkeeping that holds these, so no lock guards it.
type interleaveReq struct {
	id           string
	promptTokens int
	src          source
	out          chan inference.Token
	ctx          context.Context
	cancel       context.CancelFunc
}

// interleaveRetirement is how a request's driving goroutine reports back to
// run: it is the only channel through which the running-set bookkeeping is
// mutated from outside the run goroutine, keeping that state single-writer.
type interleaveRetirement struct {
	id           string
	promptTokens int
	cancelled    bool
}

// interleaveEngine runs a live admission loop over submitted requests.
// Construct with newInterleaveEngine; it is immediately live and must be closed
// to release its goroutine.
type interleaveEngine struct {
	maxActive      int
	maxQueue       int
	maxBatchTokens int
	streamBuffer   int

	submitCh  chan *interleaveReq
	cancelCh  chan string
	closeCh   chan struct{}
	doneCh    chan struct{}
	closeOnce sync.Once

	submitted, admitted, completed, cancelled, active, queued atomic.Int64
}

// newInterleaveEngine builds a live interleave engine from cfg and starts its
// admission loop.
func newInterleaveEngine(cfg Config) *interleaveEngine {
	e := &interleaveEngine{
		maxActive:      maxOrDefault(cfg.MaxConcurrent, 1),
		maxQueue:       maxOrDefault(cfg.MaxQueue, 1),
		maxBatchTokens: cfg.MaxBatchTokens,
		streamBuffer:   maxOrDefault(cfg.StreamBuffer, 1),
		// submitCh is deliberately UNBUFFERED: cfg.MaxQueue's actual
		// waiting-room capacity is enforced entirely by run's queue slice
		// (gated via submitSrc below). A buffered mailbox here would let a
		// Submit land a second, invisible pool of "room" in the channel
		// itself, on top of queue — real backpressure needs exactly one pool.
		submitCh: make(chan *interleaveReq),
		cancelCh: make(chan string),
		closeCh:  make(chan struct{}),
		doneCh:   make(chan struct{}),
	}
	go e.run()
	return e
}

// scheduleInterleave builds an interleave request from a serve ScheduledRequest,
// submits it, and adapts the request's inference.Token stream into the
// inference.ScheduledToken surface the serve mux consumes.
func (m *Model) scheduleInterleave(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
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
	if req.EnableThinking != nil {
		// Re-arm the thinking override the fold cannot hold — engine Chat
		// frames the turn with it (and continuity keys the conversation by it).
		opts = append(opts, inference.WithEnableThinking(req.EnableThinking))
	}
	messages := append([]inference.Message(nil), req.Messages...)
	prompt := req.Prompt
	src := func(rc context.Context) stream {
		if len(messages) > 0 {
			return m.base.Chat(rc, messages, opts...)
		}
		return m.base.Generate(rc, prompt, opts...)
	}
	promptTokens := 0
	if m.inter.maxBatchTokens > 0 {
		promptTokens = m.countPromptTokens(req)
	}
	tokens, err := m.inter.submit(ctx, id, promptTokens, src)
	if err != nil {
		return inference.RequestHandle{}, nil, err
	}
	labels := cloneLabels(req.Labels)
	out := make(chan inference.ScheduledToken, m.streamBuffer)
	go func() {
		defer close(out)
		for tok := range tokens {
			out <- inference.ScheduledToken{RequestID: id, Token: tok, Metrics: m.base.Metrics(), Labels: labels}
		}
		// The engine's post-generation snapshot is fresh HERE (the source seq
		// has fully drained), so the facade's last-stream Metrics stays honest
		// on the plain route too.
		final := m.base.Metrics()
		m.lastMetrics.Store(&final)
	}()
	var handleLabels map[string]string
	if len(req.Labels) > 0 {
		handleLabels = cloneLabels(req.Labels)
	}
	return inference.RequestHandle{ID: id, Model: inference.ModelIdentity{ID: req.Model}, Labels: handleLabels}, out, nil
}

// cbEligible reports whether a request can be served on the continuous-batching
// shared-lane-set path. Raw prompts ride at any sampling discipline (the lane
// owner runs per-lane sampler state token-identical to the plain sampled
// generate). A TEXT-ONLY chat turn rides when the model exposes the chat
// renderer capability (its own FormatChatPrompt / FormatChatPromptWithThinking
// — the same template string Chat itself encodes for the request's thinking
// flag); multimodal turns and renderer-less models keep the plain interleave
// path.
//
// Continuity handoff (the continuity×CB contract): when the engine carries a
// chat interceptor (conversation continuity, -state-conversations), a
// CONTINUATION-shaped conversation — one with a prior assistant turn — keeps
// the plain path, where base.Chat offers it to the interceptor to wake slept
// KV and prefill only the new turn. Fresh chats stay on CB: continuity would
// pay the identical full prefill for those anyway, and lanes batch it better.
// A conversation whose first turn rode CB pays one catch-up prefill at its
// second turn (continuity misses, prefills the whole history once, sleeps),
// then appends forever — either route is always token-correct; the split only
// decides who pays which prefill. Without an interceptor, continuations ride
// CB and simply full-prefill (correct, never wrong-token).
//
// NOTE: the serve-level welfare guard decorates TextModel.Chat, which the CB
// route does not call — the welfare×CB interplay is deliberately un-audited
// for now (tracked in the batching task); deployments running -welfare should
// keep chat off CB by not enabling the scheduler, or accept the gap.
func (m *Model) cbEligible(req inference.ScheduledRequest) bool {
	if len(req.Messages) == 0 {
		return core.Trim(req.Prompt) != ""
	}
	if m.cbRender == nil {
		return false
	}
	for _, msg := range req.Messages {
		if len(msg.Images) > 0 || len(msg.Audios) > 0 || len(msg.Videos) > 0 {
			return false
		}
	}
	if req.EnableThinking != nil && m.cbRenderThinking == nil {
		// A reasoning override needs the thinking-aware renderer to ride CB;
		// without it the plain path serves the override through engine Chat.
		return false
	}
	if m.cbIntercept != nil && m.cbIntercept.ChatInterceptorInstalled() && messagesCarryAssistantTurn(req.Messages) {
		return false
	}
	return true
}

// messagesCarryAssistantTurn reports whether the conversation already holds a
// model reply ("assistant" in the OpenAI/Anthropic wire convention, "model" in
// gemma's native spelling) — the shape whose prefix a prior continuity turn
// could have slept state for.
func messagesCarryAssistantTurn(messages []inference.Message) bool {
	for _, msg := range messages {
		switch core.Lower(core.Trim(msg.Role)) {
		case "assistant", "model":
			return true
		}
	}
	return false
}

// scheduleCBStep resolves a request to prompt tokens — a raw prompt directly,
// a text-only chat turn through the model's own template renderer — and drives
// it through the shared lane set, adapting the per-request inference.Token
// stream into the ScheduledToken surface the mux consumes — the CB-step twin
// of scheduleInterleave.
func (m *Model) scheduleCBStep(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	id := core.Trim(req.ID)
	if id == "" {
		id = m.nextRequestID()
	}
	tok, ok := inference.As[inference.TokenizerModel](m.base)
	if !ok { // gated at bind, but stay defensive rather than panic
		return m.scheduleInterleave(ctx, req)
	}
	prompt := req.Prompt
	if len(req.Messages) > 0 {
		if m.cbRenderThinking != nil {
			// The thinking-aware render: byte-identical to FormatChatPrompt for
			// a nil override, the request's reasoning flag applied otherwise —
			// exactly the framing engine Chat would encode for this request.
			prompt = m.cbRenderThinking.FormatChatPromptWithThinking(req.Messages, req.EnableThinking)
		} else {
			prompt = m.cbRender.FormatChatPrompt(req.Messages)
		}
	}
	promptIDs := tok.Encode(prompt)
	if len(promptIDs) == 0 {
		return inference.RequestHandle{}, nil, core.E("scheduler.scheduleCBStep", "prompt encoded to no tokens", nil)
	}
	// The full stop resolution the plain path arms (EOS, turn-close, template,
	// checkpoint-declared) — without it a lane decodes past EOS to its budget.
	stops := req.Sampler.StopTokens
	if m.cbStops != nil {
		stops = m.cbStops.ResolvedStopTokens(req.Sampler.StopTokens)
	}
	stopSet := make(map[int32]bool, len(stops))
	for _, s := range stops {
		stopSet[s] = true
	}
	// The streaming decode keeps word boundaries; Decode-of-one is the
	// fallback for models without it.
	decode := func(id int32) string { return tok.Decode([]int32{id}) }
	if m.cbDecode != nil {
		decode = m.cbDecode.DecodeToken
	}
	sink := req.MetricsSink
	// Per-REQUEST metrics: the base model's Metrics() is the last plain
	// generation's global snapshot — a lane never updates it, so the drive
	// loop counts per request and delivers at stream end (facade lastMetrics
	// + the request's MetricsSink; counts real, timings zero — the CB lane
	// timing gap is tracked). The drive loop delivers FINISHED
	// ScheduledTokens straight into the returned stream: no per-request
	// goroutine, one channel send per token end to end.
	out, err := m.cbEngine.submit(ctx, id, promptIDs, req.Sampler.MaxTokens, stops, req.Sampler, cbDeliver{
		decode:  decode,
		stopSet: stopSet,
		labels:  cloneLabels(req.Labels),
		metrics: inference.GenerateMetrics{PromptTokens: len(promptIDs)},
		finish: func(final inference.GenerateMetrics) {
			f := final
			m.lastMetrics.Store(&f)
			if sink != nil {
				sink(f)
			}
		},
	})
	if err != nil {
		return inference.RequestHandle{}, nil, err
	}
	var handleLabels map[string]string
	if len(req.Labels) > 0 {
		handleLabels = cloneLabels(req.Labels)
	}
	return inference.RequestHandle{ID: id, Model: inference.ModelIdentity{ID: req.Model}, Labels: handleLabels}, out, nil
}

// submit enqueues a request. promptTokens is the caller's prompt-length
// estimate, used only for MaxBatchTokens admission budgeting (0 is fine when
// the budget is unset). Submit blocks only while the queue is full (MaxQueue
// backpressure) or until ctx is cancelled first; once admitted, src runs on its
// own goroutine and its tokens stream on the returned channel, which closes
// when the stream ends, the request is cancelled, or the engine is closed.
func (e *interleaveEngine) submit(ctx context.Context, id string, promptTokens int, src source) (<-chan inference.Token, error) {
	if src == nil {
		return nil, core.E("scheduler.interleave.submit", "nil source", nil)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	// A closed engine's submitCh keeps its buffer (nothing forcibly closes it),
	// so a plain select below could land a send into a buffer nobody drains —
	// Go picks pseudo-randomly among ready cases, it does not prefer closeCh.
	// This up-front check makes rejection deterministic for the common case
	// (submit after close returned); the main select's own closeCh case still
	// catches the narrow window where close races concurrently with this call.
	select {
	case <-e.closeCh:
		return nil, core.E("scheduler.interleave.submit", "engine is closed", nil)
	default:
	}
	reqCtx, cancel := context.WithCancel(ctx)
	req := &interleaveReq{
		id:           id,
		promptTokens: promptTokens,
		src:          src,
		out:          make(chan inference.Token, e.streamBuffer),
		ctx:          reqCtx,
		cancel:       cancel,
	}
	select {
	case e.submitCh <- req:
		e.submitted.Add(1)
		return req.out, nil
	case <-ctx.Done():
		cancel()
		return nil, ctx.Err()
	case <-e.closeCh:
		cancel()
		return nil, core.E("scheduler.interleave.submit", "engine is closed", nil)
	}
}

// cancel cancels the request identified by id: a still-queued request never
// runs and its output channel closes with zero tokens; an active request's
// source observes ctx.Done() exactly as if the caller had cancelled its own
// submit context. Cancel on an unknown/finished/already-cancelled id is a
// harmless no-op. The result is always reported cancelled for the caller's
// bookkeeping (the engine's own Stats().Cancelled counts only real retirements).
func (e *interleaveEngine) cancel(id string) inference.RequestCancelResult {
	select {
	case e.cancelCh <- id:
	case <-e.doneCh:
	}
	return inference.RequestCancelResult{ID: id, Cancelled: true, Reason: "cancelled"}
}

// stats returns a snapshot of the engine's counters.
func (e *interleaveEngine) stats() Stats {
	return Stats{
		Submitted: e.submitted.Load(),
		Admitted:  e.admitted.Load(),
		Completed: e.completed.Load(),
		Cancelled: e.cancelled.Load(),
		Active:    e.active.Load(),
		Queued:    e.queued.Load(),
	}
}

// close stops admitting new requests and blocks until every already-active
// request's goroutine has observed cancellation and exited — no leaked
// goroutines survive a close. Already-queued requests are cancelled and their
// channels closed without ever running their source.
func (e *interleaveEngine) close() {
	e.closeOnce.Do(func() { close(e.closeCh) })
	<-e.doneCh
}

// run is the engine's single admission-loop goroutine: it owns the queue and
// running-set bookkeeping outright (single-writer, no lock needed), reacting to
// newly submitted requests, cancellations, retirements reported by driving
// goroutines, and close.
func (e *interleaveEngine) run() {
	defer close(e.doneCh)

	var (
		queue        []*interleaveReq
		active       = make(map[string]*interleaveReq, e.maxActive)
		activeTokens int
		wg           sync.WaitGroup
	)
	retireCh := make(chan interleaveRetirement, e.maxActive)

	// admit pulls from the front of the queue into the running set while a slot
	// is free and (when maxBatchTokens > 0) the request's promptTokens fits the
	// remaining budget. A request whose ctx is already cancelled while still
	// queued is retired rather than run.
	admit := func() {
		for len(queue) > 0 && len(active) < e.maxActive {
			req := queue[0]
			if req.ctx.Err() != nil {
				queue = queue[1:]
				close(req.out)
				e.cancelled.Add(1)
				continue
			}
			if e.maxBatchTokens > 0 && len(active) > 0 && activeTokens+req.promptTokens > e.maxBatchTokens {
				break // let the running set drain before admitting more
			}
			queue = queue[1:]
			active[req.id] = req
			activeTokens += req.promptTokens
			e.admitted.Add(1)
			e.active.Store(int64(len(active)))
			wg.Add(1)
			go e.drive(req, retireCh, &wg)
		}
		e.queued.Store(int64(len(queue)))
	}

	// findAndCancel cancels the request's ctx and, for a QUEUED match, removes
	// and closes it immediately rather than leaving that for admit to notice
	// later — admit only re-scans on a submit or retirement, which may never
	// happen again if the one active request never finishes.
	findAndCancel := func(id string) {
		if req, ok := active[id]; ok {
			req.cancel()
			return
		}
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
	}

	for {
		// submitSrc is nil (so its case blocks forever) whenever queue is
		// already at MaxQueue — real backpressure, not just a channel-buffer
		// size Submit could fill without run() having drained into queue.
		var submitSrc chan *interleaveReq
		if len(queue) < e.maxQueue {
			submitSrc = e.submitCh
		}

		select {
		case req := <-submitSrc:
			queue = append(queue, req)
			admit()

		case id := <-e.cancelCh:
			findAndCancel(id)

		case r := <-retireCh:
			delete(active, r.id)
			activeTokens -= r.promptTokens
			e.active.Store(int64(len(active)))
			if r.cancelled {
				e.cancelled.Add(1)
			} else {
				e.completed.Add(1)
			}
			admit()

		case <-e.closeCh:
			// Drain anything still in the submit mailbox that never reached
			// queue, so every Submit-returned channel is guaranteed to close.
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
			for _, req := range active {
				req.cancel()
			}
			for len(active) > 0 {
				r := <-retireCh
				delete(active, r.id)
				if r.cancelled {
					e.cancelled.Add(1)
				} else {
					e.completed.Add(1)
				}
			}
			e.active.Store(0)
			wg.Wait()
			return
		}
	}
}

// drive runs one admitted request's source to completion (or cancellation) on
// its own goroutine, delivering every token to req.out and reporting back on
// retireCh exactly once. It never blocks any other request: the only blocking
// point is the select below, guarded by the request's own ctx.
func (e *interleaveEngine) drive(req *interleaveReq, retireCh chan<- interleaveRetirement, wg *sync.WaitGroup) {
	defer wg.Done()
	defer close(req.out)
	for tok := range req.src(req.ctx) {
		select {
		case req.out <- tok:
		case <-req.ctx.Done():
			retireCh <- interleaveRetirement{id: req.id, promptTokens: req.promptTokens, cancelled: true}
			return
		}
	}
	retireCh <- interleaveRetirement{id: req.id, promptTokens: req.promptTokens, cancelled: req.ctx.Err() != nil}
}
