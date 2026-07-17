// SPDX-Licence-Identifier: EUPL-1.2

package scheduler

import (
	"context"
	"strconv"
	"testing"
	"time"

	"dappco.re/go/inference"
)

// heldSource yields deterministic tokens "<id>-0","<id>-1",… one per pull,
// never blocking a pull (so it does not stall batch's lockstep coordinator),
// until finish closes or the request's ctx is cancelled. It is the batch double
// for holding a request resident in the running set while admission is
// observed — the analogue of interleave's release-blocked fakeSource, adapted
// to lockstep (where a pull that blocks would freeze every peer).
func heldSource(id string, finish <-chan struct{}) source {
	return func(ctx context.Context) stream {
		return func(yield func(inference.Token) bool) {
			for i := 0; ; i++ {
				select {
				case <-finish:
					return
				case <-ctx.Done():
					return
				default:
				}
				if !yield(inference.Token{Text: id + "-" + strconv.Itoa(i)}) {
					return
				}
			}
		}
	}
}

// drainScheduled reads every scheduled token off ch until it closes, with a
// timeout, returning the token texts in arrival order.
func drainScheduled(t *testing.T, ch <-chan inference.ScheduledToken) []string {
	t.Helper()
	var got []string
	deadline := time.After(5 * time.Second)
	for {
		select {
		case tok, ok := <-ch:
			if !ok {
				return got
			}
			got = append(got, tok.Token.Text)
		case <-deadline:
			t.Fatalf("drainScheduled: timed out, got %v so far", got)
		}
	}
}

// discard consumes ch to completion in the background so the lockstep
// coordinator is never blocked delivering to an undrained active request.
func discard(ch <-chan inference.ScheduledToken) {
	go func() {
		for range ch {
		}
	}()
}

func submitReq(t *testing.T, e *batchEngine, id string, promptTokens, maxNew int, src source) <-chan inference.ScheduledToken {
	t.Helper()
	ch, err := e.submit(context.Background(), &batchReq{
		id:           id,
		promptTokens: promptTokens,
		maxNewTokens: maxNew,
		src:          src,
		metrics:      func() inference.GenerateMetrics { return inference.GenerateMetrics{} },
	})
	if err != nil {
		t.Fatalf("submit(%s) error = %v", id, err)
	}
	return ch
}

// TestBatch_Submit_Good: several requests admitted together each stream their
// own tokens to completion, in order.
func TestBatch_Submit_DrainScheduled_Good(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 4, MaxQueue: 8, StreamBuffer: 8})
	defer e.close()

	chans := map[string]<-chan inference.ScheduledToken{}
	for _, id := range []string{"a", "b", "c"} {
		chans[id] = submitReq(t, e, id, 4, 0, fakeSource(id, 3, nil, nil))
	}
	for id, ch := range chans {
		got := drainScheduled(t, ch)
		want := []string{id + "-0", id + "-1", id + "-2"}
		if len(got) != len(want) {
			t.Fatalf("%s tokens = %v, want %v", id, got, want)
		}
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("%s tokens[%d] = %q, want %q", id, i, got[i], want[i])
			}
		}
	}
	waitStats(t, e.stats, func(s Stats) bool { return s.Completed == 3 })
}

// TestBatch_MaxRunning: the batch witness — two requests admitted under a cap of
// 2 are co-resident in ONE running set (Stats().MaxRunning == 2), proving the
// engine actually batches rather than serialising.
func TestBatch_MaxRunning(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 2, MaxQueue: 8, StreamBuffer: 64})
	defer e.close()

	finish := make(chan struct{})
	discard(submitReq(t, e, "a", 1, 0, heldSource("a", finish)))
	discard(submitReq(t, e, "b", 1, 0, heldSource("b", finish)))

	s := waitStats(t, e.stats, func(s Stats) bool { return s.Active == 2 })
	if s.MaxRunning != 2 {
		t.Fatalf("MaxRunning = %d, want 2 (both requests co-resident)", s.MaxRunning)
	}
	close(finish)
	waitStats(t, e.stats, func(s Stats) bool { return s.Completed == 2 && s.Active == 0 })
}

// TestBatch_ContinuousAdmission: with a cap of 2, six requests all complete and
// the running set never exceeds the cap — queued requests take freed slots.
func TestBatch_ContinuousAdmission(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 2, MaxQueue: 16, StreamBuffer: 8})
	defer e.close()

	chans := make([]<-chan inference.ScheduledToken, 0, 6)
	for i := range 6 {
		id := "r" + strconv.Itoa(i)
		chans = append(chans, submitReq(t, e, id, 1, 0, fakeSource(id, 3, nil, nil)))
	}
	for _, ch := range chans {
		if got := drainScheduled(t, ch); len(got) != 3 {
			t.Fatalf("request tokens = %v, want 3", got)
		}
	}
	s := waitStats(t, e.stats, func(s Stats) bool { return s.Completed == 6 })
	if s.MaxRunning > 2 {
		t.Fatalf("MaxConcurrency violated: MaxRunning = %d (>2)", s.MaxRunning)
	}
}

// TestBatch_MaxBatchTokens: the token budget gates admission — a tight budget
// forces serialised admission so the running set stays at 1 despite a higher
// concurrency cap (the gate is deterministic, independent of timing).
func TestBatch_MaxBatchTokens(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 4, MaxQueue: 8, MaxBatchTokens: 15, StreamBuffer: 8})
	defer e.close()

	chans := make([]<-chan inference.ScheduledToken, 0, 3)
	for _, id := range []string{"a", "b", "c"} {
		chans = append(chans, submitReq(t, e, id, 10, 0, fakeSource(id, 2, nil, nil)))
	}
	for _, ch := range chans {
		drainScheduled(t, ch)
	}
	s := waitStats(t, e.stats, func(s Stats) bool { return s.Completed == 3 })
	if s.MaxRunning != 1 {
		t.Fatalf("token budget should serialise to 1 running, got MaxRunning = %d", s.MaxRunning)
	}
}

// TestBatch_OversizePrompt: a request whose prompt alone exceeds MaxBatchTokens
// is retired (channel closes with zero tokens) and never blocks the loop —
// other requests still complete.
func TestBatch_OversizePrompt(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 4, MaxQueue: 8, MaxBatchTokens: 16, StreamBuffer: 8})
	defer e.close()

	whale := submitReq(t, e, "whale", 1000, 0, fakeSource("whale", 4, nil, nil))
	ok := submitReq(t, e, "ok", 2, 0, fakeSource("ok", 2, nil, nil))

	if got := drainScheduled(t, whale); len(got) != 0 {
		t.Fatalf("oversize request tokens = %v, want none (retired unadmitted)", got)
	}
	if got := drainScheduled(t, ok); len(got) != 2 {
		t.Fatalf("non-oversize request tokens = %v, want 2", got)
	}
	waitStats(t, e.stats, func(s Stats) bool { return s.Cancelled >= 1 && s.Completed >= 1 })
}

// TestBatch_MaxNewTokens: a per-request generated-token cap stops a source that
// would otherwise run forever, exactly at the cap.
func TestBatch_MaxNewTokens(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 2, MaxQueue: 8, StreamBuffer: 8})
	defer e.close()

	finish := make(chan struct{})
	defer close(finish)
	ch := submitReq(t, e, "cap", 1, 3, heldSource("cap", finish))
	got := drainScheduled(t, ch)
	if len(got) != 3 {
		t.Fatalf("MaxNewTokens=3 should yield exactly 3 tokens, got %v", got)
	}
	waitStats(t, e.stats, func(s Stats) bool { return s.Completed == 1 })
}

// TestBatch_Cancel_Active: cancelling an active request retires it mid-stream —
// its channel closes and the cancellation is counted.
func TestBatch_Cancel_Active(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 2, MaxQueue: 8, StreamBuffer: 8})
	defer e.close()

	finish := make(chan struct{})
	defer close(finish)
	ch := submitReq(t, e, "a", 1, 0, heldSource("a", finish))

	// Wait for at least one token so the request is provably active, then cancel.
	select {
	case <-ch:
	case <-time.After(5 * time.Second):
		t.Fatal("no token from active request before cancel")
	}
	e.cancel("a")
	drainScheduled(t, ch) // must terminate once the cancellation lands
	waitStats(t, e.stats, func(s Stats) bool { return s.Cancelled == 1 && s.Active == 0 })
}

func TestBatch_cancel_BackpressuredActive_Ugly(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 1, MaxQueue: 1, StreamBuffer: 0})
	defer e.close()
	produced := make(chan int, 2)
	src := func(ctx context.Context) stream {
		return func(yield func(inference.Token) bool) {
			for index := 0; ; index++ {
				select {
				case <-ctx.Done():
					return
				case produced <- index:
				}
				if !yield(inference.Token{Text: strconv.Itoa(index)}) {
					return
				}
			}
		}
	}
	ch := submitReq(t, e, "backpressured", 1, 0, src)
	<-produced
	select {
	case <-ch:
	case <-time.After(time.Second):
		t.Fatal("first token did not arrive")
	}
	<-produced // the coordinator is now driving the next undrained token

	cancelled := make(chan struct{})
	go func() {
		e.cancel("backpressured")
		close(cancelled)
	}()
	select {
	case <-cancelled:
	case <-time.After(time.Second):
		t.Fatal("cancel blocked behind the request's full output channel")
	}
	drainScheduled(t, ch)
}

// TestBatch_Cancel_Queued: cancelling a QUEUED request retires it without ever
// running its source, while the active request occupying the slot streams on.
func TestBatch_Cancel_Queued(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 1, MaxQueue: 8, StreamBuffer: 8})
	defer e.close()

	finish := make(chan struct{})
	defer close(finish)
	discard(submitReq(t, e, "first", 1, 0, heldSource("first", finish))) // occupies the one slot, drained so the loop cycles

	secondStarted := make(chan string, 1)
	second := submitReq(t, e, "second", 1, 0, fakeSource("second", 1, secondStarted, nil))
	waitStats(t, e.stats, func(s Stats) bool { return s.Queued == 1 })
	e.cancel("second")

	if got := drainScheduled(t, second); len(got) != 0 {
		t.Fatalf("cancelled queued request tokens = %v, want none", got)
	}
	select {
	case <-secondStarted:
		t.Fatalf("cancelled queued request's source ran — it must never start")
	case <-time.After(20 * time.Millisecond):
	}
	waitStats(t, e.stats, func(s Stats) bool { return s.Cancelled == 1 })
}

// TestBatch_Submit_Bad: a nil source is rejected, and submit after close fails.
func TestBatch_Submit_Bad(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 2, MaxQueue: 2, StreamBuffer: 2})

	if _, err := e.submit(context.Background(), &batchReq{id: "a", src: nil}); err == nil {
		t.Fatalf("submit(nil source) error = nil, want non-nil")
	}
	e.close()
	if _, err := e.submit(context.Background(), &batchReq{id: "b", src: fakeSource("b", 1, nil, nil)}); err == nil {
		t.Fatalf("submit after close error = nil, want non-nil")
	}
}

// TestBatch_Defaults: a non-positive MaxConcurrency clamps to 1, so the running
// set never exceeds one even with several requests queued.
func TestBatch_Defaults(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 0, MaxQueue: 8, StreamBuffer: 8})
	defer e.close()

	a := submitReq(t, e, "a", 1, 0, fakeSource("a", 2, nil, nil))
	b := submitReq(t, e, "b", 1, 0, fakeSource("b", 2, nil, nil))
	drainScheduled(t, a)
	drainScheduled(t, b)
	s := waitStats(t, e.stats, func(s Stats) bool { return s.Completed == 2 })
	if s.MaxRunning != 1 {
		t.Fatalf("clamped MaxConcurrency should be 1, got MaxRunning = %d", s.MaxRunning)
	}
}

// TestBatch_Close: close retires the active and queued requests, leaves the
// engine idle, and rejects later submits.
func TestBatch_Close(t *testing.T) {
	e := newBatchEngine(nil, Config{MaxConcurrent: 1, MaxQueue: 8, StreamBuffer: 8})

	finish := make(chan struct{})
	defer close(finish)
	active := submitReq(t, e, "active", 1, 0, heldSource("active", finish))
	discard2 := make(chan struct{})
	go func() { defer close(discard2); drainScheduled(t, active) }()

	queued := submitReq(t, e, "queued", 1, 0, fakeSource("queued", 1, nil, nil))
	waitStats(t, e.stats, func(s Stats) bool { return s.Queued == 1 })

	e.close()
	<-discard2 // active channel closed

	if got := drainScheduled(t, queued); len(got) != 0 {
		t.Fatalf("queued request tokens after close = %v, want none", got)
	}
	if s := e.stats(); s.Active != 0 || s.Queued != 0 {
		t.Fatalf("Stats after close = %+v, want Active=0 Queued=0", s)
	}
	if _, err := e.submit(context.Background(), &batchReq{id: "after", src: fakeSource("after", 1, nil, nil)}); err == nil {
		t.Fatalf("submit after close error = nil, want non-nil")
	}
}
