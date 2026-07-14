// SPDX-Licence-Identifier: EUPL-1.2

// Continuous-batching step coordinator — the interleave-mode fast path when the
// backend exposes an available inference.BatchStepModel (metal's multi-session
// owner). Where plain interleave runs each admitted request's decode on its OWN
// goroutine (K independent GPU submissions), this coordinator drives every
// admitted request through ONE shared inference.LaneSet: a single goroutine
// owns the lane set, admits requests as ragged lanes, and advances the whole
// running set by one token per round with a SINGLE batched forward
// (LaneSet.Step). Per-request tokens stream on per-request channels exactly as
// interleave's do.
//
// This coordinator serves every request the lane owner can serve
// byte-identically: a raw prompt OR a text-only chat turn (rendered through the
// model's own chat template — see scheduler.cbEligible / scheduleCBStep),
// decoded greedily OR with the request's own per-lane sampler. Streaming detok
// runs the model's DecodeToken, byte-identical to the plain path (stop tokens
// blanked the same way). A request the owner cannot serve byte-identically — a
// multimodal turn, a continuity-backed continuation, a renderer-less model —
// declines to the plain interleave engine, which is unchanged, so nothing is
// ever served on an unproven path.
//
// Backpressure is GLOBAL (a single shared drive loop), the same trade batch
// mode makes: one slow consumer stalls the round. Cancellation is per-request
// (a cancelled request's lane is retired before the next Step; other lanes
// continue).
package scheduler

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// cbReq is one admitted-or-queued unit of CB-step work. The drive loop
// delivers FINISHED ScheduledTokens straight into out — per-token text
// decode, stop blanking, running metrics and labels all happen in deliver —
// so a request costs ONE channel send per token end to end (the previous
// shape ran a per-request goroutine re-reading an intermediate token channel:
// one extra hop + goroutine wake per token, ~0.5ms/tok of the K=1 CB-vs-plain
// residual by the GPU-span split receipt).
type cbReq struct {
	id        string
	promptIDs []int32
	maxNew    int
	stops     []int32
	sampler   inference.SamplerConfig
	out       chan inference.ScheduledToken
	decode    func(int32) string // per-token stream text; nil = ids only
	stopSet   map[int32]bool     // terminator ids — their text stays blank
	labels    map[string]string
	metrics   inference.GenerateMetrics       // running per-request counts (drive-loop-owned)
	finishFn  func(inference.GenerateMetrics) // stream-end delivery (facade lastMetrics + MetricsSink)
	ctx       context.Context
	cancel    context.CancelFunc

	// Duration anchors (drive-loop-owned like metrics): start is the submit
	// instant (TotalDuration's zero), prefillDur the admission prefill's own
	// span (BeginPrepare / inline Prepare — the CB path knows it EXACTLY,
	// where the plain path only reports a coarse total), decodeStart the
	// admission-complete instant (DecodeDuration's zero).
	start       time.Time
	prefillDur  time.Duration
	decodeStart time.Time
}

// stampMetrics refreshes the request's duration + throughput fields from its
// anchors — called before every token send and at stream end, so streamed
// per-token snapshots grow monotonically and the final delivery carries the
// full honest split (counts were always real; timings were zero before #35's
// durations rung).
func (r *cbReq) stampMetrics() {
	r.metrics.PrefillDuration = r.prefillDur
	if !r.start.IsZero() {
		r.metrics.TotalDuration = time.Since(r.start)
	}
	if !r.decodeStart.IsZero() {
		r.metrics.DecodeDuration = time.Since(r.decodeStart)
	}
	if r.prefillDur > 0 && r.metrics.PromptTokens > 0 {
		r.metrics.PrefillTokensPerSec = float64(r.metrics.PromptTokens) / r.prefillDur.Seconds()
	}
	if r.metrics.DecodeDuration > 0 && r.metrics.GeneratedTokens > 0 {
		r.metrics.DecodeTokensPerSec = float64(r.metrics.GeneratedTokens) / r.metrics.DecodeDuration.Seconds()
	}
}

// end fires the stream-end delivery and closes the request's stream — the
// finish order (finishFn BEFORE close) is the happens-before edge the
// metrics-sink consumers rely on: the handler reads its sink-written locals
// only after its range over the stream ends.
func (r *cbReq) end() {
	r.stampMetrics()
	if r.finishFn != nil {
		r.finishFn(r.metrics)
	}
	close(r.out)
}

// cbStepEngine owns one inference.LaneSet and drives all admitted requests
// through it from a single goroutine. Construct with newCBStepEngine; it is
// immediately live and must be closed to release its goroutine and the owner.
type cbStepEngine struct {
	ls           inference.LaneSet
	maxActive    int
	maxQueue     int
	streamBuffer int
	defaultMax   int

	submitCh chan *cbReq
	cancelCh chan string
	closeCh  chan struct{}
	doneCh   chan struct{}
	once     sync.Once

	submitted, admitted, completed, cancelled, active, queued atomic.Int64
}

// newCBStepEngine opens the owner and starts the drive loop. Returns nil when
// the capability cannot actually bind a lane set right now (so the caller falls
// back to plain interleave — never a silent serial degrade of a claimed batch
// path).
func newCBStepEngine(bsm inference.BatchStepModel, cfg Config) *cbStepEngine {
	if bsm == nil || !bsm.BatchStepAvailable() {
		return nil
	}
	maxActive := maxOrDefault(cfg.MaxConcurrent, 1)
	ls, err := bsm.OpenLaneSet(inference.LaneSetConfig{MaxLanes: maxActive})
	if err != nil || ls == nil {
		return nil
	}
	e := &cbStepEngine{
		ls:           ls,
		maxActive:    maxActive,
		maxQueue:     maxOrDefault(cfg.MaxQueue, 1),
		streamBuffer: maxOrDefault(cfg.StreamBuffer, 1),
		defaultMax:   cbDefaultMaxNew,
		submitCh:     make(chan *cbReq),
		cancelCh:     make(chan string),
		closeCh:      make(chan struct{}),
		doneCh:       make(chan struct{}),
	}
	go e.run()
	return e
}

// cbDefaultMaxNew caps generation for a request that names no MaxTokens.
const cbDefaultMaxNew = 512

// cbDeliver is the per-request delivery spec the caller hands submit: how the
// drive loop turns lane tokens into the consumer's ScheduledTokens and what
// fires at stream end.
type cbDeliver struct {
	decode  func(int32) string
	stopSet map[int32]bool
	labels  map[string]string
	metrics inference.GenerateMetrics
	finish  func(inference.GenerateMetrics)
}

// submit enqueues a CB-step request. Blocks only while the queue is full or
// until ctx is cancelled; once admitted the request's lane is advanced by the
// shared Step loop and its tokens stream on the returned channel, which closes
// when the stream ends, the request is cancelled, or the engine is closed.
func (e *cbStepEngine) submit(ctx context.Context, id string, promptIDs []int32, maxNew int, stops []int32, sampler inference.SamplerConfig, d cbDeliver) (<-chan inference.ScheduledToken, error) {
	if len(promptIDs) == 0 {
		return nil, core.E("scheduler.cbstep.submit", "empty prompt", nil)
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if maxNew <= 0 {
		maxNew = e.defaultMax
	}
	select {
	case <-e.closeCh:
		return nil, core.E("scheduler.cbstep.submit", "engine is closed", nil)
	default:
	}
	reqCtx, cancel := context.WithCancel(ctx)
	req := &cbReq{
		id:        id,
		promptIDs: promptIDs,
		maxNew:    maxNew,
		stops:     stops,
		sampler:   sampler,
		out:       make(chan inference.ScheduledToken, e.streamBuffer),
		decode:    d.decode,
		stopSet:   d.stopSet,
		labels:    d.labels,
		metrics:   d.metrics,
		finishFn:  d.finish,
		ctx:       reqCtx,
		cancel:    cancel,
		start:     time.Now(),
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
		return nil, core.E("scheduler.cbstep.submit", "engine is closed", nil)
	}
}

// cancel cancels the request identified by id (queued or active). A no-op for an
// unknown/finished id; always reported cancelled for the caller's bookkeeping.
func (e *cbStepEngine) cancel(id string) inference.RequestCancelResult {
	select {
	case e.cancelCh <- id:
	case <-e.doneCh:
	}
	return inference.RequestCancelResult{ID: id, Cancelled: true, Reason: "cancelled"}
}

// stats snapshots the engine counters.
func (e *cbStepEngine) stats() Stats {
	return Stats{
		Submitted: e.submitted.Load(),
		Admitted:  e.admitted.Load(),
		Completed: e.completed.Load(),
		Cancelled: e.cancelled.Load(),
		Active:    e.active.Load(),
		Queued:    e.queued.Load(),
	}
}

// close stops admission and blocks until the drive loop has retired every lane
// and released the owner — no leaked goroutine, no orphaned request.
func (e *cbStepEngine) close() {
	e.once.Do(func() { close(e.closeCh) })
	<-e.doneCh
}

// cbPrepared is one BeginPrepare's result, delivered back to the drive loop:
// the prepared-but-unattached lane (or the error that ended the attempt).
type cbPrepared struct {
	req     *cbReq
	p       inference.PendingLane
	err     error
	prepDur time.Duration // the BeginPrepare span — the request's honest PrefillDuration
}

// run is the single drive-loop goroutine: it owns the lane set, the pending
// queue, and the lane→request map outright (single-writer, no lock).
//
// Admission overlap: when the lane set implements inference.
// LaneSetOverlappedAdmitter, each admission's heavy prefill (BeginPrepare)
// runs on its own goroutine while this loop keeps Stepping the in-flight
// lanes — a newcomer's multi-second prompt prefill no longer freezes every
// active stream. The finished prefill comes back on prepCh and the loop
// splices it in with CommitPrepare (µs). In-flight prefills count against the
// slot budget exactly as active lanes do, so a full set never pays a prefill
// CommitPrepare would have to throw away. Without the capability, admission
// is the original inline Prepare, unchanged.
func (e *cbStepEngine) run() {
	defer close(e.doneCh)
	defer func() { _ = e.ls.Close() }()

	overlap, _ := e.ls.(inference.LaneSetOverlappedAdmitter)

	var pending []*cbReq
	byLane := make(map[int]*cbReq, e.maxActive)
	preparing := make(map[string]*cbReq, e.maxActive)
	// prepCh is buffered to the slot budget: at most maxActive BeginPrepares
	// are ever in flight, so a prep goroutine's send NEVER blocks — it can
	// always run to completion, which is what lets drain() collect every
	// outstanding prefill before the deferred ls.Close() releases the owner.
	prepCh := make(chan cbPrepared, e.maxActive)

	// finish closes a request's stream and clears its bookkeeping.
	finish := func(laneID int, req *cbReq, counter *atomic.Int64) {
		req.cancel()
		req.end()
		delete(byLane, laneID)
		counter.Add(1)
		e.active.Store(int64(len(byLane)))
	}

	// admitInline is the capability-less path: the whole Prepare (prefill
	// included) runs here in the drive loop, stalling the round.
	admitInline := func(req *cbReq) {
		prepStart := time.Now()
		h, err := e.ls.Prepare(req.ctx, inference.LaneSpec{PromptIDs: req.promptIDs, MaxNew: req.maxNew, StopTokens: req.stops, Sampler: req.sampler})
		if err != nil || !h.Valid() {
			// Prefill failed (e.g. arch not ICB-eligible) — end the request's
			// stream cleanly rather than hang it; the caller sees an empty
			// completion, never a wrong-token one.
			req.cancel()
			req.end()
			e.cancelled.Add(1)
			return
		}
		req.prefillDur = time.Since(prepStart)
		req.decodeStart = time.Now()
		byLane[h.ID] = req
		e.admitted.Add(1)
		e.active.Store(int64(len(byLane)))
	}

	admit := func() {
		for len(pending) > 0 && e.ls.Active()+len(preparing) < e.maxActive {
			req := pending[0]
			pending = pending[1:]
			if req.ctx.Err() != nil {
				req.cancel()
				req.end()
				e.cancelled.Add(1)
				continue
			}
			if overlap == nil {
				admitInline(req)
				continue
			}
			preparing[req.id] = req
			go func(req *cbReq) {
				prepStart := time.Now()
				p, err := overlap.BeginPrepare(req.ctx, inference.LaneSpec{PromptIDs: req.promptIDs, MaxNew: req.maxNew, StopTokens: req.stops, Sampler: req.sampler})
				prepCh <- cbPrepared{req: req, p: p, err: err, prepDur: time.Since(prepStart)}
			}(req)
		}
		e.queued.Store(int64(len(pending)))
	}

	// handlePrepared splices a finished BeginPrepare into the running set (or
	// ends the request when the prefill failed / the consumer cancelled while
	// it ran — the pending lane is discarded, never attached).
	handlePrepared := func(pr cbPrepared) {
		delete(preparing, pr.req.id)
		if pr.err != nil || pr.p == nil {
			pr.req.cancel()
			pr.req.end()
			e.cancelled.Add(1)
			return
		}
		if pr.req.ctx.Err() != nil {
			pr.p.Discard()
			pr.req.cancel()
			pr.req.end()
			e.cancelled.Add(1)
			return
		}
		h, err := overlap.CommitPrepare(pr.p)
		if err != nil || !h.Valid() {
			pr.req.cancel()
			pr.req.end()
			e.cancelled.Add(1)
			return
		}
		pr.req.prefillDur = pr.prepDur
		pr.req.decodeStart = time.Now()
		byLane[h.ID] = pr.req
		e.admitted.Add(1)
		e.active.Store(int64(len(byLane)))
	}

	handleCancel := func(id string) {
		for i, req := range pending {
			if req.id == id {
				pending = append(pending[:i], pending[i+1:]...)
				req.cancel()
				req.end()
				e.cancelled.Add(1)
				e.queued.Store(int64(len(pending)))
				return
			}
		}
		if req := preparing[id]; req != nil {
			// Mid-prefill: cancel the request's ctx (BeginPrepare aborts or its
			// result arrives already-cancelled); handlePrepared ends the stream
			// and discards the lane when the result lands on prepCh.
			req.cancel()
			return
		}
		for laneID, req := range byLane {
			if req.id == id {
				_ = e.ls.Retire(inference.LaneHandle{ID: laneID})
				finish(laneID, req, &e.cancelled)
				return
			}
		}
	}

	drain := func() {
		for _, req := range pending {
			req.cancel()
			req.end()
		}
		pending = nil
		// Every in-flight BeginPrepare MUST complete before the deferred
		// ls.Close() runs (its session prefill is live GPU work against the
		// owner). Cancel their contexts, then collect each result off prepCh —
		// the buffered channel guarantees the goroutines all finish.
		for _, req := range preparing {
			req.cancel()
		}
		for len(preparing) > 0 {
			pr := <-prepCh
			delete(preparing, pr.req.id)
			if pr.p != nil {
				pr.p.Discard()
			}
			pr.req.cancel()
			pr.req.end()
			e.cancelled.Add(1)
		}
		for laneID, req := range byLane {
			_ = e.ls.Retire(inference.LaneHandle{ID: laneID})
			req.cancel()
			req.end()
			delete(byLane, laneID)
		}
	}

	for {
		admit()
		if len(byLane) == 0 {
			// Nothing running — block until a submit, a finished prefill, a
			// cancel, or a close arrives.
			select {
			case req := <-e.submitCh:
				pending = append(pending, req)
			case pr := <-prepCh:
				handlePrepared(pr)
			case id := <-e.cancelCh:
				handleCancel(id)
			case <-e.closeCh:
				drain()
				return
			}
			continue
		}
		// Lanes are running — service pending control messages and finished
		// prefills without blocking, then take one batched step.
		select {
		case req := <-e.submitCh:
			pending = append(pending, req)
			continue
		case pr := <-prepCh:
			handlePrepared(pr)
			continue
		case id := <-e.cancelCh:
			handleCancel(id)
			continue
		case <-e.closeCh:
			drain()
			return
		default:
		}
		steps, err := e.ls.Step(context.Background())
		if err != nil {
			// A forward failed — fail every active request cleanly.
			for laneID, req := range byLane {
				_ = e.ls.Retire(inference.LaneHandle{ID: laneID})
				req.cancel()
				req.end()
				delete(byLane, laneID)
			}
			e.active.Store(0)
			continue
		}
		e.deliver(steps, byLane, finish)
	}
}

// deliver routes each lane's step result to its request's stream and retires
// lanes that terminated or whose consumer cancelled mid-stream.
func (e *cbStepEngine) deliver(steps []inference.LaneStep, byLane map[int]*cbReq, finish func(int, *cbReq, *atomic.Int64)) {
	for _, s := range steps {
		req := byLane[s.Lane.ID]
		if req == nil {
			continue
		}
		cancelled := false
		if s.HasToken {
			t := inference.Token{ID: s.Token}
			if !req.stopSet[t.ID] && req.decode != nil {
				// Terminator text is never content — the plain path yields the
				// stop token with empty text (engine decodeFromPrefilled).
				t.Text = req.decode(t.ID)
			}
			req.metrics.GeneratedTokens++
			req.stampMetrics()
			select {
			case req.out <- inference.ScheduledToken{RequestID: req.id, Token: t, Metrics: req.metrics, Labels: req.labels}:
			case <-req.ctx.Done():
				cancelled = true
			}
		}
		switch {
		case cancelled:
			_ = e.ls.Retire(s.Lane)
			finish(s.Lane.ID, req, &e.cancelled)
		case s.Terminal:
			_ = e.ls.Retire(s.Lane)
			finish(s.Lane.ID, req, &e.completed)
		}
	}
}
