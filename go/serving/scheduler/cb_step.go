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
// This coordinator serves the requests the slice-1 owner can serve
// byte-identically: a raw prompt (tokenised via the model's Encode) decoded
// greedily. A chat request (needs template rendering the neutral inference.
// TextModel surface does not expose) or a non-greedy sampler falls back to the
// plain interleave engine, which is unchanged — so nothing regresses and no
// request is served on a path that was not proven byte-identical. See
// docs/design-continuous-batching.md for the pinned serve-integration rungs
// (chat rendering, incremental detokenisation, per-request sampler state).
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

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// cbReq is one admitted-or-queued unit of CB-step work.
type cbReq struct {
	id        string
	promptIDs []int32
	maxNew    int
	stops     []int32
	out       chan inference.Token
	ctx       context.Context
	cancel    context.CancelFunc
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

// submit enqueues a CB-step request. Blocks only while the queue is full or
// until ctx is cancelled; once admitted the request's lane is advanced by the
// shared Step loop and its tokens stream on the returned channel, which closes
// when the stream ends, the request is cancelled, or the engine is closed.
func (e *cbStepEngine) submit(ctx context.Context, id string, promptIDs []int32, maxNew int, stops []int32) (<-chan inference.Token, error) {
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
		out:       make(chan inference.Token, e.streamBuffer),
		ctx:       reqCtx,
		cancel:    cancel,
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

// run is the single drive-loop goroutine: it owns the lane set, the pending
// queue, and the lane→request map outright (single-writer, no lock).
func (e *cbStepEngine) run() {
	defer close(e.doneCh)
	defer func() { _ = e.ls.Close() }()

	var pending []*cbReq
	byLane := make(map[int]*cbReq, e.maxActive)

	// finish closes a request's stream and clears its bookkeeping.
	finish := func(laneID int, req *cbReq, counter *atomic.Int64) {
		req.cancel()
		close(req.out)
		delete(byLane, laneID)
		counter.Add(1)
		e.active.Store(int64(len(byLane)))
	}

	admit := func() {
		for len(pending) > 0 && e.ls.Active() < e.maxActive {
			req := pending[0]
			pending = pending[1:]
			if req.ctx.Err() != nil {
				req.cancel()
				close(req.out)
				e.cancelled.Add(1)
				continue
			}
			h, err := e.ls.Prepare(req.ctx, inference.LaneSpec{PromptIDs: req.promptIDs, MaxNew: req.maxNew, StopTokens: req.stops})
			if err != nil || !h.Valid() {
				// Prefill failed (e.g. arch not ICB-eligible) — end the request's
				// stream cleanly rather than hang it; the caller sees an empty
				// completion, never a wrong-token one.
				req.cancel()
				close(req.out)
				e.cancelled.Add(1)
				continue
			}
			byLane[h.ID] = req
			e.admitted.Add(1)
			e.active.Store(int64(len(byLane)))
		}
		e.queued.Store(int64(len(pending)))
	}

	handleCancel := func(id string) {
		for i, req := range pending {
			if req.id == id {
				pending = append(pending[:i], pending[i+1:]...)
				req.cancel()
				close(req.out)
				e.cancelled.Add(1)
				e.queued.Store(int64(len(pending)))
				return
			}
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
			close(req.out)
		}
		pending = nil
		for laneID, req := range byLane {
			_ = e.ls.Retire(inference.LaneHandle{ID: laneID})
			req.cancel()
			close(req.out)
			delete(byLane, laneID)
		}
	}

	for {
		admit()
		if len(byLane) == 0 {
			// Nothing running — block until a submit, cancel, or close arrives.
			select {
			case req := <-e.submitCh:
				pending = append(pending, req)
			case id := <-e.cancelCh:
				handleCancel(id)
			case <-e.closeCh:
				drain()
				return
			}
			continue
		}
		// Lanes are running — service pending control messages without blocking,
		// then take one batched step.
		select {
		case req := <-e.submitCh:
			pending = append(pending, req)
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
				close(req.out)
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
			select {
			case req.out <- inference.Token{ID: s.Token}:
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
