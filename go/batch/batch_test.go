// SPDX-Licence-Identifier: EUPL-1.2

package batch

import (
	"context"
	"sync"
	"sync/atomic"
	"time"

	core "dappco.re/go"
)

// fakeCall is the deterministic stand-in for a real chat/embedding call
// (an expansion pipeline or a remote provider). Index N returns the result
// "r<N>" and usage {N, N, 2N}; any index in errOn fails with a typed error.
// It records the maximum number of in-flight Do calls observed so a test can
// assert the concurrency cap is never exceeded.
type fakeCall struct {
	errOn   map[int]bool  // indices that return an error
	delay   time.Duration // per-call work, so concurrency overlaps
	live    int64         // current in-flight (atomic)
	maxLive int64         // high-water mark of live (atomic)
	calls   int64         // total Do invocations (atomic)
}

// Do satisfies Call. It bumps the live counter, sleeps for delay so concurrent
// calls genuinely overlap, then returns a deterministic result or a typed error.
func (f *fakeCall) Do(ctx context.Context, index int, request any) (any, Usage, error) {
	atomic.AddInt64(&f.calls, 1)
	now := atomic.AddInt64(&f.live, 1)
	for {
		hi := atomic.LoadInt64(&f.maxLive)
		if now <= hi || atomic.CompareAndSwapInt64(&f.maxLive, hi, now) {
			break
		}
	}
	defer atomic.AddInt64(&f.live, -1)

	if f.delay > 0 {
		select {
		case <-time.After(f.delay):
		case <-ctx.Done():
			return nil, Usage{}, ctx.Err()
		}
	}

	if f.errOn[index] {
		return nil, Usage{}, core.E("batch", core.Sprintf("item %d failed", index), nil)
	}
	u := Usage{PromptTokens: index, CompletionTokens: index, TotalTokens: 2 * index}
	return "r" + core.Itoa(index), u, nil
}

// countingLimiter wraps a real Limiter and counts Wait calls so a test can
// assert every dispatched item was throttled.
type countingLimiter struct {
	inner Limiter
	waits int64
}

func (c *countingLimiter) Wait(ctx context.Context) error {
	atomic.AddInt64(&c.waits, 1)
	return c.inner.Wait(ctx)
}

func TestBatch_Run_Good(t *core.T) {
	// Ordered results: three requests fan out, come back in INPUT order with the
	// right per-item result, and usage aggregates across the batch.
	call := &fakeCall{}
	reqs := []any{"a", "b", "c"} // indices 0,1,2
	out := Run(context.Background(), reqs, Options{Concurrency: 3, Call: call})

	core.AssertEqual(t, 3, len(out.Items), "one result per request")
	for i, it := range out.Items {
		core.AssertEqual(t, i, it.Index, "results preserve input order")
		core.AssertNil(t, it.Err, "no item should error")
		core.AssertEqual(t, "r"+core.Itoa(i), it.Result, "deterministic result per index")
	}
	// usage = sum of {0,1,2} prompt + completion, {0,2,4} total
	core.AssertEqual(t, 3, out.Usage.PromptTokens, "prompt tokens summed")
	core.AssertEqual(t, 3, out.Usage.CompletionTokens, "completion tokens summed")
	core.AssertEqual(t, 6, out.Usage.TotalTokens, "total tokens summed")
}

func TestBatch_Run_Bad(t *core.T) {
	// A failing item is captured per-item — the rest still succeed, order holds,
	// and a failed item contributes no usage.
	call := &fakeCall{errOn: map[int]bool{1: true}}
	reqs := []any{"a", "b", "c"}
	out := Run(context.Background(), reqs, Options{Concurrency: 2, Call: call})

	core.AssertEqual(t, 3, len(out.Items))
	core.AssertNil(t, out.Items[0].Err, "item 0 succeeds")
	core.AssertError(t, out.Items[1].Err, "item 1") // typed error names its item
	core.AssertNil(t, out.Items[2].Err, "item 2 succeeds")
	core.AssertEqual(t, "r0", out.Items[0].Result)
	core.AssertEqual(t, nil, out.Items[1].Result, "a failed item has no result")
	core.AssertEqual(t, "r2", out.Items[2].Result)
	// only items 0 and 2 contribute: prompt 0+2=2, total 0+4=4
	core.AssertEqual(t, 2, out.Usage.PromptTokens, "failed item adds no usage")
	core.AssertEqual(t, 4, out.Usage.TotalTokens)
}

func TestBatch_Run_Ugly(t *core.T) {
	// Empty batch: no calls, empty results, zero usage — and a nil Call is a
	// programmer error reported per item rather than a panic.
	call := &fakeCall{}
	empty := Run(context.Background(), nil, Options{Concurrency: 4, Call: call})
	core.AssertEqual(t, 0, len(empty.Items), "empty batch yields no results")
	core.AssertEqual(t, 0, empty.Usage.TotalTokens, "empty batch has zero usage")
	core.AssertEqual(t, int64(0), atomic.LoadInt64(&call.calls), "empty batch never dispatches")

	// nil Call — every item fails closed with an error, no panic.
	nilOut := Run(context.Background(), []any{"x", "y"}, Options{Concurrency: 2})
	core.AssertEqual(t, 2, len(nilOut.Items))
	core.AssertError(t, nilOut.Items[0].Err, "no Call configured") // fails closed, never panics
	core.AssertError(t, nilOut.Items[1].Err, "no Call configured")
}

func TestBatch_Concurrency_Good(t *core.T) {
	// With a cap of 2 over 10 slow items, no more than 2 calls are ever in
	// flight at once.
	call := &fakeCall{delay: 20 * time.Millisecond}
	reqs := make([]any, 10)
	out := Run(context.Background(), reqs, Options{Concurrency: 2, Call: call})

	core.AssertEqual(t, 10, len(out.Items))
	core.AssertTrue(t, atomic.LoadInt64(&call.maxLive) <= 2,
		"observed concurrency must never exceed the cap of 2")
	core.AssertTrue(t, atomic.LoadInt64(&call.maxLive) >= 2,
		"with 10 slow items the cap should actually be reached")
}

func TestBatch_Concurrency_Bad(t *core.T) {
	// A non-positive concurrency must not mean "unbounded" — it clamps to 1, so
	// the work still completes serially without fanning out.
	call := &fakeCall{delay: 5 * time.Millisecond}
	reqs := make([]any, 6)
	out := Run(context.Background(), reqs, Options{Concurrency: 0, Call: call})

	core.AssertEqual(t, 6, len(out.Items))
	core.AssertTrue(t, atomic.LoadInt64(&call.maxLive) <= 1,
		"a zero/negative cap clamps to serial, never unbounded")
}

func TestBatch_Concurrency_Ugly(t *core.T) {
	// A cap larger than the batch is fine — concurrency is bounded by the work,
	// not the cap, and every item still completes exactly once.
	call := &fakeCall{delay: 5 * time.Millisecond}
	reqs := make([]any, 3)
	out := Run(context.Background(), reqs, Options{Concurrency: 100, Call: call})

	core.AssertEqual(t, 3, len(out.Items))
	core.AssertTrue(t, atomic.LoadInt64(&call.maxLive) <= 3,
		"can't run more in parallel than there are items")
	core.AssertEqual(t, int64(3), atomic.LoadInt64(&call.calls), "each item dispatched exactly once")
}

func TestBatch_Limiter_Good(t *core.T) {
	// The limiter throttles EVERY call: with burst 1 at 200/s, four calls take at
	// least ~3 intervals (15ms), and every dispatched item passed through Wait.
	lim := &countingLimiter{inner: NewTokenBucket(200, 1)} // 200/s → 5ms/token
	call := &fakeCall{}
	reqs := make([]any, 4)

	start := time.Now()
	out := Run(context.Background(), reqs, Options{Concurrency: 4, Call: call, Limiter: lim})
	elapsed := time.Since(start)

	core.AssertEqual(t, 4, len(out.Items))
	core.AssertEqual(t, int64(4), atomic.LoadInt64(&lim.waits), "every item is throttled through the limiter")
	core.AssertTrue(t, elapsed >= 12*time.Millisecond,
		"rate limiting serialises the burst — 4 tokens at 200/s can't all fire instantly")
}

func TestBatch_Refill_NonMonotonic_Ugly(t *core.T) {
	// refill's guard: if no time has elapsed since the last refill (last is at or
	// ahead of now — a non-advancing or non-monotonic clock, or two refills in the
	// same tick), it adds nothing and leaves the token count untouched. White-box:
	// pin last into the future so the elapsed delta is negative.
	tb := NewTokenBucket(1000, 5) // interval well above zero
	tb.mu.Lock()
	tb.tokens = 2
	tb.last = time.Now().Add(time.Hour) // last is in the future ⇒ elapsed < 0
	before := tb.tokens
	tb.refill()
	after := tb.tokens
	tb.mu.Unlock()
	core.AssertEqual(t, before, after, "a non-advancing clock refills nothing")
	core.AssertEqual(t, float64(2), after, "tokens are left exactly as they were")
}

func TestBatch_TokenBucket_Bad(t *core.T) {
	// A cancelled context unblocks a waiting bucket immediately with the context
	// error, rather than sleeping out the full interval.
	tb := NewTokenBucket(1, 1) // 1/s, burst 1
	ctx := context.Background()
	core.AssertNil(t, tb.Wait(ctx), "first token is free (burst)")

	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	core.AssertError(t, tb.Wait(cancelled)) // a cancelled context aborts the wait
}

// erroringLimiter always fails its Wait with a typed error WITHOUT touching the
// context — the seam for exercising runOne's "throttle wait" failure path
// distinctly from a context cancellation (which runOne checks first).
type erroringLimiter struct{}

func (erroringLimiter) Wait(ctx context.Context) error {
	return core.E("batchtest", "limiter refused", nil)
}

func TestBatch_RunAsCompleted_Empty_Ugly(t *core.T) {
	// An empty request slice to RunAsCompleted returns an already-closed channel
	// that yields nothing — the streaming twin of Run's empty-batch path.
	ch := RunAsCompleted(context.Background(), nil, Options{Concurrency: 4, Call: &fakeCall{}})
	count := 0
	for range ch {
		count++
	}
	core.AssertEqual(t, 0, count, "an empty as-completed batch delivers no items")
}

func TestBatch_Cancelled_Bad(t *core.T) {
	// A pre-cancelled context: the batch completes without dispatching the Call.
	// Whether an item is fed before the feed loop observes ctx.Done() is a race,
	// so the invariant under test is that NOTHING dispatches and any delivered
	// item carries the cancellation error — never a spurious success.
	call := &fakeCall{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancelled up front

	out := Run(ctx, []any{"a", "b", "c"}, Options{Concurrency: 1, Call: call})
	core.AssertEqual(t, 3, len(out.Items), "every request still yields a result slot")
	core.AssertEqual(t, int64(0), atomic.LoadInt64(&call.calls),
		"a cancelled batch never dispatches the Call")
	core.AssertEqual(t, 0, out.Usage.TotalTokens, "a cancelled batch has zero usage")
	for _, it := range out.Items {
		// Either the item was never fed (zero value, nil Err) or it reached runOne
		// and failed on the ctx guard — but it must never have succeeded.
		core.AssertNil(t, it.Result, "a cancelled batch yields no successful result")
	}
}

func TestBatch_RunOne_Cancelled_Bad(t *core.T) {
	// runOne's first guard: a context already cancelled when the item runs yields
	// a typed per-item error naming the item, and never invokes the Call. Calling
	// runOne directly removes the feed-loop race so the guard is hit every run.
	call := &fakeCall{}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	got := runOne(ctx, 7, "req", Options{Call: call, Limiter: NewTokenBucket(0, 1)})
	core.AssertEqual(t, 7, got.Index, "the result keeps the item index")
	core.AssertError(t, got.Err, "cancelled")
	core.AssertContains(t, got.Err.Error(), "item 7 cancelled", "the error names the cancelled item")
	core.AssertNil(t, got.Result, "a cancelled item has no result")
	core.AssertEqual(t, int64(0), atomic.LoadInt64(&call.calls), "the Call is never reached")
}

func TestBatch_LimiterRefused_Ugly(t *core.T) {
	// The limiter refuses every Wait (with a live context): runOne translates the
	// throttle failure into a per-item typed error rather than dispatching, and
	// the Call is never reached. The error names the throttle-wait stage.
	call := &fakeCall{}
	out := Run(context.Background(), []any{"x", "y"},
		Options{Concurrency: 2, Call: call, Limiter: erroringLimiter{}})

	core.AssertEqual(t, 2, len(out.Items))
	core.AssertContains(t, out.Items[0].Err.Error(), "throttle wait", "a refused throttle fails the item")
	core.AssertContains(t, out.Items[1].Err.Error(), "throttle wait", "a refused throttle fails the item")
	core.AssertEqual(t, int64(0), atomic.LoadInt64(&call.calls),
		"a refused throttle never dispatches the Call")
}

func TestBatch_TokenBucket_Unlimited_Good(t *core.T) {
	// A rate of zero means "no rate limit": every Wait returns at once (the
	// interval==0 fast path), so a burst clamps to at least 1 and never blocks.
	tb := NewTokenBucket(0, 0) // rate 0 → unlimited; burst 0 → clamped to 1
	for i := 0; i < 5; i++ {
		core.AssertNil(t, tb.Wait(context.Background()), "unlimited bucket never blocks")
	}
}

func TestBatch_TokenBucket_CancelDuringWait_Ugly(t *core.T) {
	// A bucket whose burst is spent makes the next Wait sleep for the refill
	// interval; cancelling the context during that sleep unblocks it with the
	// context error rather than waiting the interval out.
	tb := NewTokenBucket(1, 1) // 1/s, burst 1 → ~1s between tokens
	core.AssertNil(t, tb.Wait(context.Background()), "first token is free (burst)")

	ctx, cancel := context.WithCancel(context.Background())
	// Cancel shortly after Wait starts sleeping on the timer.
	go func() {
		time.Sleep(10 * time.Millisecond)
		cancel()
	}()
	start := time.Now()
	err := tb.Wait(ctx)
	core.AssertError(t, err) // a context cancelled mid-wait aborts the throttle
	core.AssertTrue(t, time.Since(start) < 500*time.Millisecond,
		"cancellation unblocks well before the full 1s interval")
}

func TestBatch_RunAsCompleted_Good(t *core.T) {
	// As-completed streams each result as it finishes (completion order, not
	// input order); the channel closes after the last, and every index arrives
	// exactly once with its usage.
	call := &fakeCall{}
	reqs := make([]any, 5)

	ch := RunAsCompleted(context.Background(), reqs, Options{Concurrency: 5, Call: call})

	seen := make(map[int]bool)
	var mu sync.Mutex
	total := 0
	for it := range ch {
		mu.Lock()
		core.AssertFalse(t, seen[it.Index], "each index arrives exactly once")
		seen[it.Index] = true
		total += it.Usage.TotalTokens
		mu.Unlock()
	}
	core.AssertEqual(t, 5, len(seen), "every item is delivered before the channel closes")
	// totals 0+2+4+6+8 = 20
	core.AssertEqual(t, 20, total, "as-completed carries per-item usage")
}
