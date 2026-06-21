// SPDX-Licence-Identifier: EUPL-1.2

package schedule

import (
	"context"
	"sync"
	"testing"

	core "dappco.re/go"
)

// fakeStepper is the test double for Stepper. It advances every running
// sequence by one token per Step: each seq gets a token equal to its current
// generated-count + a per-id base (so emitted tokens are deterministic and
// distinguishable per request), and a seq finishes once it reaches its
// per-id finishAfter count (modelling an EOS the engine can't predict in
// advance). It records the maximum running-set length it was ever called with
// — the witness that the scheduler honoured MaxConcurrency and MaxBatchTokens.
//
//	st := &fakeStepper{finishAfter: map[string]int{"a": 2, "b": 3}}
//	out, _ := New(Scheduler{MaxConcurrency: 2}).Run(ctx, reqs, st, nil)
type fakeStepper struct {
	mu          sync.Mutex
	finishAfter map[string]int // seq id -> generated count at which it finishes
	calls       int            // number of Step invocations
	maxRunning  int            // largest len(running) observed across calls
	seenIDs     [][]string     // running-set ids per call (order witness)
}

// Step advances each running seq by one token and finishes those that reach
// their finishAfter target. A seq with no finishAfter entry never finishes by
// EOS (it stops only at MaxNewTokens, enforced by the scheduler).
func (f *fakeStepper) Step(_ context.Context, running []*Seq) (StepResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls++
	if len(running) > f.maxRunning {
		f.maxRunning = len(running)
	}
	ids := make([]string, 0, len(running))
	res := StepResult{Tokens: make(map[string]int, len(running)), Finished: make(map[string]bool, len(running))}
	for _, s := range running {
		ids = append(ids, s.Request.ID)
		// Token value encodes (request, position) deterministically: 1000*ord
		// is unused here; we use generated count so per-request streams are
		// 0,1,2,... and the test can assert exact ordering.
		res.Tokens[s.Request.ID] = s.Generated
		if target, ok := f.finishAfter[s.Request.ID]; ok && s.Generated+1 >= target {
			res.Finished[s.Request.ID] = true
		}
	}
	f.seenIDs = append(f.seenIDs, ids)
	return res, nil
}

// errStepper fails on its Nth call (1-based) with a scoped error, advancing
// nothing — exercises the error-surfacing path.
type errStepper struct {
	failOn int
	calls  int
}

func (e *errStepper) Step(_ context.Context, _ []*Seq) (StepResult, error) {
	e.calls++
	if e.calls == e.failOn {
		return StepResult{}, core.E("schedule.test", "boom", nil)
	}
	// Never finish anything on a non-failing call so the loop must keep going
	// until the failing call (avoids accidental natural completion).
	return StepResult{Tokens: map[string]int{}, Finished: map[string]bool{}}, nil
}

// collectTokens returns a sink fn plus the map it fills, keyed by request id in
// emission order.
func collectTokens() (func(string, int), map[string][]int, *sync.Mutex) {
	mu := &sync.Mutex{}
	got := map[string][]int{}
	fn := func(id string, tok int) {
		mu.Lock()
		got[id] = append(got[id], tok)
		mu.Unlock()
	}
	return fn, got, mu
}

// resultByID indexes a Result slice by request id for assertions.
func resultByID(rs []Result) map[string]Result {
	m := make(map[string]Result, len(rs))
	for _, r := range rs {
		m[r.ID] = r
	}
	return m
}

// TestSchedule_Run_Good: every queued request runs to completion, the result
// set covers them all, and per-request tokens are collected in order.
func TestSchedule_Run_Good(t *testing.T) {
	st := &fakeStepper{finishAfter: map[string]int{"a": 2, "b": 3, "c": 1}}
	reqs := []Request{
		{ID: "a", PromptTokens: 4, MaxNewTokens: 8},
		{ID: "b", PromptTokens: 4, MaxNewTokens: 8},
		{ID: "c", PromptTokens: 4, MaxNewTokens: 8},
	}
	onTok, got, _ := collectTokens()

	out, err := New(Scheduler{MaxConcurrency: 4, MaxBatchTokens: 1 << 20}).Run(context.Background(), reqs, st, onTok)
	if err != nil {
		t.Fatalf("Run: unexpected error %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("want 3 results, got %d (%+v)", len(out), out)
	}
	by := resultByID(out)
	for _, id := range []string{"a", "b", "c"} {
		r, ok := by[id]
		if !ok {
			t.Fatalf("missing result for %q", id)
		}
		if !r.Finished || r.Err != nil {
			t.Fatalf("%q: want finished, no error; got %+v", id, r)
		}
	}
	// Token streams are 0,1,... up to finishAfter; assert exact per-request order.
	wantTok := map[string][]int{"a": {0, 1}, "b": {0, 1, 2}, "c": {0}}
	for id, want := range wantTok {
		if !equalInts(by[id].Tokens, want) {
			t.Fatalf("%q result tokens: want %v, got %v", id, want, by[id].Tokens)
		}
		if !equalInts(got[id], want) {
			t.Fatalf("%q onToken stream: want %v, got %v", id, want, got[id])
		}
	}
}

// TestSchedule_Run_Bad: a Stepper error surfaces from Run and aborts the loop.
func TestSchedule_Run_Bad(t *testing.T) {
	reqs := []Request{
		{ID: "a", PromptTokens: 2, MaxNewTokens: 100},
		{ID: "b", PromptTokens: 2, MaxNewTokens: 100},
	}
	st := &errStepper{failOn: 2} // succeed once, then fail
	out, err := New(Scheduler{MaxConcurrency: 4, MaxBatchTokens: 1 << 20}).Run(context.Background(), reqs, st, nil)
	if err == nil {
		t.Fatalf("want error from failing stepper, got nil (out=%+v)", out)
	}
	r := core.Fail(err)
	if r.OK {
		t.Fatalf("failed result should not be OK")
	}
}

// TestSchedule_Run_Ugly: an empty queue completes immediately with no results
// and no error, and a nil onToken callback is tolerated.
func TestSchedule_Run_Ugly(t *testing.T) {
	out, err := New(Scheduler{MaxConcurrency: 4, MaxBatchTokens: 1 << 20}).Run(context.Background(), nil, &fakeStepper{}, nil)
	if err != nil {
		t.Fatalf("empty queue: unexpected error %v", err)
	}
	if len(out) != 0 {
		t.Fatalf("empty queue: want 0 results, got %d", len(out))
	}

	// A single request with a nil onToken sink still completes.
	st := &fakeStepper{finishAfter: map[string]int{"solo": 1}}
	out, err = New(Scheduler{MaxConcurrency: 1, MaxBatchTokens: 1 << 20}).Run(context.Background(),
		[]Request{{ID: "solo", PromptTokens: 1, MaxNewTokens: 4}}, st, nil)
	if err != nil || len(out) != 1 || !out[0].Finished {
		t.Fatalf("solo nil-sink: want one finished result, got %+v err=%v", out, err)
	}
}

// TestSchedule_Admission_Good: the running set never exceeds MaxConcurrency, and
// admission is continuous — as sequences finish, queued ones take their slots.
func TestSchedule_Admission_Good(t *testing.T) {
	// 6 requests, cap 2, each finishes after a few tokens. With continuous
	// admission all 6 complete and the observed running set never exceeds 2.
	finish := map[string]int{}
	reqs := make([]Request, 0, 6)
	for _, id := range []string{"r0", "r1", "r2", "r3", "r4", "r5"} {
		finish[id] = 2
		reqs = append(reqs, Request{ID: id, PromptTokens: 1, MaxNewTokens: 8})
	}
	st := &fakeStepper{finishAfter: finish}
	out, err := New(Scheduler{MaxConcurrency: 2, MaxBatchTokens: 1 << 20}).Run(context.Background(), reqs, st, nil)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if len(out) != 6 {
		t.Fatalf("want 6 completed, got %d", len(out))
	}
	if st.maxRunning > 2 {
		t.Fatalf("MaxConcurrency violated: observed running set of %d (>2)", st.maxRunning)
	}
	if st.maxRunning != 2 {
		t.Fatalf("continuous admission should keep 2 running while work remains; max=%d", st.maxRunning)
	}
}

// TestSchedule_Admission_Bad: the token budget gates admission. Each running
// seq costs prompt+generated tokens; a tight MaxBatchTokens forces serialised
// admission so the running set is throttled below the concurrency cap.
func TestSchedule_Admission_Bad(t *testing.T) {
	// Cap allows 4 concurrent, but the byte budget only fits one 10-token
	// prompt at a time → running set must stay at 1 despite the higher cap.
	finish := map[string]int{"a": 1, "b": 1, "c": 1}
	reqs := []Request{
		{ID: "a", PromptTokens: 10, MaxNewTokens: 4},
		{ID: "b", PromptTokens: 10, MaxNewTokens: 4},
		{ID: "c", PromptTokens: 10, MaxNewTokens: 4},
	}
	st := &fakeStepper{finishAfter: finish}
	out, err := New(Scheduler{MaxConcurrency: 4, MaxBatchTokens: 15}).Run(context.Background(), reqs, st, nil)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if len(out) != 3 {
		t.Fatalf("want 3 completed, got %d", len(out))
	}
	if st.maxRunning != 1 {
		t.Fatalf("token budget should serialise to 1 running, got max=%d", st.maxRunning)
	}
}

// TestSchedule_Admission_Ugly: a request whose prompt alone exceeds
// MaxBatchTokens is rejected with a typed error in its Result and never blocks
// the loop — other requests still complete.
func TestSchedule_Admission_Ugly(t *testing.T) {
	st := &fakeStepper{finishAfter: map[string]int{"ok": 1}}
	reqs := []Request{
		{ID: "whale", PromptTokens: 1000, MaxNewTokens: 4}, // oversize prompt
		{ID: "ok", PromptTokens: 2, MaxNewTokens: 4},
	}
	out, err := New(Scheduler{MaxConcurrency: 4, MaxBatchTokens: 16}).Run(context.Background(), reqs, st, nil)
	if err != nil {
		t.Fatalf("oversize must not fail the whole loop, got %v", err)
	}
	by := resultByID(out)
	whale, ok := by["whale"]
	if !ok {
		t.Fatalf("oversize request must still appear in results")
	}
	if whale.Finished || whale.Err == nil {
		t.Fatalf("oversize: want not-finished with typed error, got %+v", whale)
	}
	if r := core.Fail(whale.Err); r.OK {
		t.Fatalf("oversize error should be a failed Result")
	}
	good, ok := by["ok"]
	if !ok || !good.Finished || good.Err != nil {
		t.Fatalf("the non-oversize request must complete normally, got %+v", good)
	}

	// Degenerate: a zero/negative MaxBatchTokens rejects every sized prompt but
	// the loop still terminates with a result per request.
	st2 := &fakeStepper{}
	out2, err2 := New(Scheduler{MaxConcurrency: 4, MaxBatchTokens: 0}).Run(context.Background(),
		[]Request{{ID: "x", PromptTokens: 1, MaxNewTokens: 1}}, st2, nil)
	if err2 != nil {
		t.Fatalf("zero budget: want no loop error, got %v", err2)
	}
	if len(out2) != 1 || out2[0].Err == nil {
		t.Fatalf("zero budget: want one rejected result, got %+v", out2)
	}
}

// TestSchedule_MaxNewTokens covers the cap-based finish: a seq with no EOS stops
// exactly at MaxNewTokens, and its Result carries that many tokens.
func TestSchedule_MaxNewTokens(t *testing.T) {
	st := &fakeStepper{finishAfter: map[string]int{}} // never EOS
	reqs := []Request{{ID: "cap", PromptTokens: 2, MaxNewTokens: 3}}
	onTok, got, _ := collectTokens()
	out, err := New(Scheduler{MaxConcurrency: 2, MaxBatchTokens: 1 << 20}).Run(context.Background(), reqs, st, onTok)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if len(out) != 1 || !out[0].Finished {
		t.Fatalf("want one finished result, got %+v", out)
	}
	if len(out[0].Tokens) != 3 {
		t.Fatalf("MaxNewTokens=3 should yield 3 tokens, got %v", out[0].Tokens)
	}
	if !equalInts(got["cap"], []int{0, 1, 2}) {
		t.Fatalf("token stream: want [0 1 2], got %v", got["cap"])
	}
}

// TestSchedule_ZeroMaxNewTokens covers a request asking for zero new tokens —
// it finishes immediately with an empty token list and no error.
func TestSchedule_ZeroMaxNewTokens(t *testing.T) {
	st := &fakeStepper{finishAfter: map[string]int{}}
	reqs := []Request{{ID: "noop", PromptTokens: 2, MaxNewTokens: 0}}
	out, err := New(Scheduler{MaxConcurrency: 2, MaxBatchTokens: 1 << 20}).Run(context.Background(), reqs, st, nil)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if len(out) != 1 || !out[0].Finished || len(out[0].Tokens) != 0 {
		t.Fatalf("zero MaxNewTokens: want finished empty-token result, got %+v", out)
	}
	if st.calls != 0 {
		t.Fatalf("a request needing no tokens should not invoke the stepper; calls=%d", st.calls)
	}
}

// TestSchedule_Cancel covers context cancellation: a cancelled context aborts
// the loop with a context error before completion.
func TestSchedule_Cancel(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	// Stepper that cancels the context on its first call and never finishes a
	// sequence, so without cancellation the loop would run forever.
	st := &cancelStepper{cancel: cancel}
	reqs := []Request{{ID: "a", PromptTokens: 1, MaxNewTokens: 1000000}}
	_, err := New(Scheduler{MaxConcurrency: 1, MaxBatchTokens: 1 << 20}).Run(ctx, reqs, st, nil)
	if err == nil {
		t.Fatalf("want context cancellation error, got nil")
	}
}

// TestSchedule_CancelBeforeStart covers an already-cancelled context: Run aborts
// before admitting anything.
func TestSchedule_CancelBeforeStart(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancelled up front
	st := &fakeStepper{finishAfter: map[string]int{"a": 1}}
	_, err := New(Scheduler{MaxConcurrency: 1, MaxBatchTokens: 1 << 20}).Run(ctx,
		[]Request{{ID: "a", PromptTokens: 1, MaxNewTokens: 4}}, st, nil)
	if err == nil {
		t.Fatalf("pre-cancelled context: want error, got nil")
	}
}

// cancelStepper cancels its captured context on the first Step call (then keeps
// advancing without ever finishing), so the scheduler must observe the
// cancellation to terminate.
type cancelStepper struct {
	cancel context.CancelFunc
	calls  int
}

func (c *cancelStepper) Step(_ context.Context, running []*Seq) (StepResult, error) {
	c.calls++
	if c.calls == 1 {
		c.cancel()
	}
	res := StepResult{Tokens: make(map[string]int), Finished: make(map[string]bool)}
	for _, s := range running {
		res.Tokens[s.Request.ID] = s.Generated
	}
	return res, nil
}

// TestSchedule_NilStepper covers the guard: Run with a nil Stepper returns a
// typed error rather than panicking.
func TestSchedule_NilStepper(t *testing.T) {
	_, err := New(Scheduler{MaxConcurrency: 1, MaxBatchTokens: 16}).Run(context.Background(),
		[]Request{{ID: "a", PromptTokens: 1, MaxNewTokens: 1}}, nil, nil)
	if err == nil {
		t.Fatalf("nil stepper: want typed error, got nil")
	}
}

// TestSchedule_Defaults covers config clamping: non-positive MaxConcurrency
// clamps to 1 so the loop still makes progress.
func TestSchedule_Defaults(t *testing.T) {
	st := &fakeStepper{finishAfter: map[string]int{"a": 1, "b": 1}}
	reqs := []Request{
		{ID: "a", PromptTokens: 1, MaxNewTokens: 4},
		{ID: "b", PromptTokens: 1, MaxNewTokens: 4},
	}
	out, err := New(Scheduler{MaxConcurrency: 0, MaxBatchTokens: 1 << 20}).Run(context.Background(), reqs, st, nil)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if len(out) != 2 {
		t.Fatalf("want 2 results with clamped cap, got %d", len(out))
	}
	if st.maxRunning != 1 {
		t.Fatalf("clamped MaxConcurrency should be 1, observed max=%d", st.maxRunning)
	}
}

// TestSchedule_ResultOrder covers result ordering: results come back in the
// order sequences finish, which the scheduler records deterministically.
func TestSchedule_ResultOrder(t *testing.T) {
	// c finishes first (after 1), a second (after 2), b last (after 3); with a
	// cap that runs all three together the finish order is c, a, b.
	st := &fakeStepper{finishAfter: map[string]int{"a": 2, "b": 3, "c": 1}}
	reqs := []Request{
		{ID: "a", PromptTokens: 1, MaxNewTokens: 8},
		{ID: "b", PromptTokens: 1, MaxNewTokens: 8},
		{ID: "c", PromptTokens: 1, MaxNewTokens: 8},
	}
	out, err := New(Scheduler{MaxConcurrency: 3, MaxBatchTokens: 1 << 20}).Run(context.Background(), reqs, st, nil)
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	order := make([]string, len(out))
	for i, r := range out {
		order[i] = r.ID
	}
	if !equalStrings(order, []string{"c", "a", "b"}) {
		t.Fatalf("finish order: want [c a b], got %v", order)
	}
}

// equalInts reports element-wise slice equality (nil and empty both match []).
func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// equalStrings reports element-wise string-slice equality.
func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
