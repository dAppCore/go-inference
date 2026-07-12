// SPDX-Licence-Identifier: EUPL-1.2

// Package interleave is the live continuous-batching admission scheduler
// (slice 1 of docs/design-continuous-batching.md): requests can be Submitted
// at any time — including while other requests are mid-decode — and are
// admitted into a bounded running set under a dual concurrency+token budget,
// FIFO otherwise. It never runs a model: each admitted request's Source is
// the caller's EXISTING per-session token stream
// (engine.TextModel.Generate/Chat, or a continuity-woken session's
// handle.Generate — unmodified), so the package is purely a scheduling layer
// with no engine dependency.
//
// Each admitted request runs its Source on its own goroutine, delivering
// tokens to its own bounded output channel. This is deliberate over a single
// shared round-robin loop: a newly-admitted session's first pull also pays
// its prefill, and a synchronous per-round call on one shared loop would
// block every OTHER active session for that whole prefill — seeing that
// while designing this package is exactly why (see the design memo, slice 1
// vs slice 2). Per-request goroutines give per-request backpressure and
// cancellation isolation for free: one slow consumer or one long prefill
// never blocks anyone else's tokens.
//
//	e := interleave.New(interleave.Config{MaxActive: 8, MaxQueue: 64, StreamBuffer: 8})
//	defer e.Close()
//
//	tokens, err := e.Submit(ctx, "chat-42", promptTokens, interleave.Source(
//		func(rc context.Context) interleave.Stream { return model.Generate(rc, prompt) },
//	))
//	for tok := range tokens {
//		fmt.Print(tok.Text)
//	}
package interleave

import (
	"context"
	"iter"
	"sync"
	"sync/atomic"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Stream is one session's token generator — the shape engine.TextModel's
// Generate/Chat and a continuity-woken session's handle.Generate already
// return. Nothing about it changes to run under an Engine.
type Stream = iter.Seq[inference.Token]

// Source constructs one request's Stream bound to ctx — the Engine's own
// per-request context. Cancel(id) cancels it, and it is a child of the ctx
// passed to Submit, so the caller's own cancellation propagates too, exactly
// as it would calling the underlying Generate/Chat directly.
//
//	interleave.Source(func(rc context.Context) interleave.Stream {
//		return model.Generate(rc, prompt, opts...)
//	})
type Source func(ctx context.Context) Stream

// Config configures one Engine. Both admission limits gate together — a
// queued request is admitted only once BOTH a running-set slot is free and
// (when MaxBatchTokens > 0) its PromptTokens fits the remaining token
// budget. Non-positive MaxActive/StreamBuffer/MaxQueue are clamped to 1 so
// the engine always makes progress and Submit always has at least one slot
// of waiting room to fill before it blocks; MaxBatchTokens <= 0 disables the
// token budget (MaxActive alone gates).
type Config struct {
	MaxActive      int // concurrent running requests
	MaxQueue       int // Submit backpressure: requests waiting for a running-set slot
	MaxBatchTokens int // running PromptTokens budget across the active set; <= 0 = uncapped
	StreamBuffer   int // per-request output channel buffer
}

// Stats is a snapshot of one Engine's counters, safe to read concurrently
// with Submit/Cancel from any goroutine.
type Stats struct {
	Submitted int64 // total Submit calls that reached the queue
	Admitted  int64 // total that moved from the queue into the running set
	Completed int64 // total whose Source ran to completion
	Cancelled int64 // total retired by Cancel or ctx cancellation (queued or active)
	Active    int64 // requests currently running
	Queued    int64 // requests currently waiting for a slot
}

// request is one admitted-or-queued unit of work; only the run loop mutates
// the queue/active bookkeeping that holds these, so no lock guards it.
type request struct {
	id           string
	promptTokens int
	src          Source
	out          chan inference.Token
	ctx          context.Context
	cancel       context.CancelFunc
}

// retirement is how a request's driving goroutine reports back to run: it is
// the only channel through which the running-set bookkeeping is mutated from
// outside the run goroutine, keeping that state single-writer.
type retirement struct {
	id           string
	promptTokens int
	cancelled    bool
}

// Engine runs a live admission loop over Submitted requests. Construct with
// New; the returned Engine is immediately live and must be Close()d to
// release its goroutine.
type Engine struct {
	cfg Config

	submitCh  chan *request
	cancelCh  chan string
	closeCh   chan struct{}
	doneCh    chan struct{}
	closeOnce sync.Once

	submitted, admitted, completed, cancelled, active, queued atomic.Int64
}

// New builds a live Engine and starts its admission loop.
//
//	e := interleave.New(interleave.Config{MaxActive: 8, MaxQueue: 64, StreamBuffer: 8})
//	defer e.Close()
func New(cfg Config) *Engine {
	cfg.MaxActive = max(cfg.MaxActive, 1)
	cfg.StreamBuffer = max(cfg.StreamBuffer, 1)
	cfg.MaxQueue = max(cfg.MaxQueue, 1)
	e := &Engine{
		cfg: cfg,
		// submitCh is deliberately UNBUFFERED: cfg.MaxQueue's actual
		// waiting-room capacity is enforced entirely by run's queue slice
		// (gated via submitSrc below). A buffered mailbox here would let a
		// Submit land a second, invisible pool of "room" in the channel
		// itself, on top of queue — real backpressure needs exactly one
		// pool, not two.
		submitCh: make(chan *request),
		cancelCh: make(chan string),
		closeCh:  make(chan struct{}),
		doneCh:   make(chan struct{}),
	}
	go e.run()
	return e
}

// Submit enqueues a request. promptTokens is the caller's best estimate of
// the prompt length, used only for MaxBatchTokens admission budgeting (0 is
// fine when MaxBatchTokens is unset) — it does not gate src itself. Submit
// blocks only while the queue is full (MaxQueue backpressure) or until ctx is
// cancelled first, whichever comes first; once admitted, src runs on its own
// goroutine and its tokens stream on the returned channel, which closes when
// the Stream ends, the request is cancelled, or the Engine is Close()d.
//
//	tokens, err := e.Submit(ctx, "chat-42", 312, interleave.Source(
//		func(rc context.Context) interleave.Stream { return model.Generate(rc, prompt) },
//	))
func (e *Engine) Submit(ctx context.Context, id string, promptTokens int, src Source) (<-chan inference.Token, error) {
	if src == nil {
		return nil, core.E("interleave.Submit", "nil source", nil)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	// A closed Engine's submitCh keeps its buffer (nothing forcibly closes
	// it), so a plain select below could otherwise land a send into a
	// buffer nobody will ever drain — Go picks pseudo-randomly among
	// simultaneously ready cases, it does not prefer closeCh. This
	// up-front check makes rejection deterministic for the common case
	// (Submit called once Close has already returned); the main select's
	// own closeCh case still catches the narrow window where Close races
	// concurrently with this call.
	select {
	case <-e.closeCh:
		return nil, core.E("interleave.Submit", "engine is closed", nil)
	default:
	}
	reqCtx, cancel := context.WithCancel(ctx)
	req := &request{
		id:           id,
		promptTokens: promptTokens,
		src:          src,
		out:          make(chan inference.Token, e.cfg.StreamBuffer),
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
		return nil, core.E("interleave.Submit", "engine is closed", nil)
	}
}

// Cancel cancels the request identified by id: a still-queued request never
// runs and its output channel closes with zero tokens; an active request's
// Source observes ctx.Done() exactly as if the caller had cancelled its own
// Submit context. Cancel on an unknown, already-finished, or already-cancelled
// id is a harmless no-op.
func (e *Engine) Cancel(id string) {
	select {
	case e.cancelCh <- id:
	case <-e.doneCh:
	}
}

// Stats returns a snapshot of the engine's counters.
//
//	if s := e.Stats(); s.Active == 0 && s.Queued == 0 { /* idle */ }
func (e *Engine) Stats() Stats {
	return Stats{
		Submitted: e.submitted.Load(),
		Admitted:  e.admitted.Load(),
		Completed: e.completed.Load(),
		Cancelled: e.cancelled.Load(),
		Active:    e.active.Load(),
		Queued:    e.queued.Load(),
	}
}

// Close stops admitting new requests and blocks until every already-active
// request's goroutine has observed cancellation and exited — no leaked
// goroutines survive a Close(). Already-queued requests are cancelled and
// their channels closed without ever running their Source.
func (e *Engine) Close() {
	e.closeOnce.Do(func() { close(e.closeCh) })
	<-e.doneCh
}

// run is the Engine's single admission-loop goroutine: it owns the queue and
// running-set bookkeeping outright (single-writer, no lock needed), reacting
// to newly Submitted requests, Cancel requests, retirements reported by
// driving goroutines, and Close.
func (e *Engine) run() {
	defer close(e.doneCh)

	var (
		queue        []*request
		active       = make(map[string]*request, e.cfg.MaxActive)
		activeTokens int
		wg           sync.WaitGroup
	)
	retireCh := make(chan retirement, e.cfg.MaxActive)

	// admit pulls from the front of the queue into the running set while a
	// slot is free and (when MaxBatchTokens > 0) the request's PromptTokens
	// fits the remaining budget. A request whose ctx is already cancelled
	// while still queued (findAndCancel below removes a match immediately,
	// so this only fires for the CALLER's own outer ctx expiring while
	// queued, a Cancel(id) is never the source here) is retired rather than
	// run, so it never starts a Source needlessly.
	admit := func() {
		for len(queue) > 0 && len(active) < e.cfg.MaxActive {
			req := queue[0]
			if req.ctx.Err() != nil {
				queue = queue[1:]
				close(req.out)
				e.cancelled.Add(1)
				continue
			}
			if e.cfg.MaxBatchTokens > 0 && len(active) > 0 && activeTokens+req.promptTokens > e.cfg.MaxBatchTokens {
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

	// findAndCancel cancels the request's ctx and, for a QUEUED match,
	// removes and closes it immediately rather than leaving that for admit
	// to notice later — admit only re-scans the queue when a submit or a
	// retirement triggers it, which may never happen again if, say, the
	// one active request never finishes (a real scenario: a client cancels
	// a queued request while an unrelated long-running one occupies the
	// only active slot). It is a no-op for an id it cannot find (unknown,
	// already active-and-cancelled-again, or already retired) — Submit is
	// guaranteed to have returned before a caller can validly name an id
	// here, and this goroutine processes one select case at a time, so the
	// request is always already in active or queue by the time this runs.
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
		// submitSrc is nil (so its select case blocks forever) whenever
		// queue is already at MaxQueue — real backpressure, not just a
		// channel-buffer size that Submit could fill without run() ever
		// having drained anything into queue.
		var submitSrc chan *request
		if len(queue) < e.cfg.MaxQueue {
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
			// Drain anything still sitting in the submit mailbox that never
			// reached queue, so every Submit-returned channel is guaranteed
			// to close.
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

// drive runs one admitted request's Source to completion (or cancellation)
// on its own goroutine, delivering every token to req.out and reporting back
// on retireCh exactly once. It never blocks any other request: the only
// blocking point is the select below, guarded by the request's own ctx.
func (e *Engine) drive(req *request, retireCh chan<- retirement, wg *sync.WaitGroup) {
	defer wg.Done()
	defer close(req.out)
	for tok := range req.src(req.ctx) {
		select {
		case req.out <- tok:
		case <-req.ctx.Done():
			retireCh <- retirement{id: req.id, promptTokens: req.promptTokens, cancelled: true}
			return
		}
	}
	retireCh <- retirement{id: req.id, promptTokens: req.promptTokens, cancelled: req.ctx.Err() != nil}
}
