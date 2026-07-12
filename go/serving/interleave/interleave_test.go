// SPDX-Licence-Identifier: EUPL-1.2

package interleave

import (
	"context"
	"strconv"
	"testing"
	"time"

	"dappco.re/go/inference"
)

// fakeSource builds a Source that reports to started (when non-nil) the
// instant its Stream begins running, then optionally blocks on release
// (when non-nil) before yielding n deterministic tokens "<id>-0".."<id>-(n-1)"
// — the controllable-timing double every admission/cancellation/backpressure
// test below drives, mirroring serving/scheduler's blockingModel double.
//
//	started := make(chan string, 1)
//	release := make(chan struct{})
//	src := fakeSource("a", 3, started, release)
//	<-started      // Stream is now running (or blocked on release)
//	close(release) // let it proceed to yield its 3 tokens
func fakeSource(id string, n int, started chan<- string, release <-chan struct{}) Source {
	return func(ctx context.Context) Stream {
		return func(yield func(inference.Token) bool) {
			if started != nil {
				started <- id
			}
			if release != nil {
				select {
				case <-release:
				case <-ctx.Done():
					return
				}
			}
			for i := range n {
				if ctx.Err() != nil {
					return
				}
				if !yield(inference.Token{Text: id + "-" + strconv.Itoa(i)}) {
					return
				}
			}
		}
	}
}

// drain reads every token off ch until it closes, with a timeout so a stuck
// test fails fast instead of hanging the suite.
func drain(t *testing.T, ch <-chan inference.Token) []string {
	t.Helper()
	var got []string
	deadline := time.After(5 * time.Second)
	for {
		select {
		case tok, ok := <-ch:
			if !ok {
				return got
			}
			got = append(got, tok.Text)
		case <-deadline:
			t.Fatalf("drain: timed out, got %v so far", got)
		}
	}
}

// waitStats polls e.Stats() until want reports true or the deadline passes —
// the run loop's bookkeeping updates asynchronously to a retirement/close, so
// tests that assert on it need this rather than an immediate read.
func waitStats(t *testing.T, e *Engine, want func(Stats) bool) Stats {
	t.Helper()
	deadline := time.After(5 * time.Second)
	for {
		s := e.Stats()
		if want(s) {
			return s
		}
		select {
		case <-deadline:
			t.Fatalf("stats never satisfied predicate, last = %+v", s)
		case <-time.After(time.Millisecond):
		}
	}
}

func TestEngine_Submit_Good(t *testing.T) {
	e := New(Config{MaxActive: 4, MaxQueue: 4, StreamBuffer: 4})
	defer e.Close()

	tokens, err := e.Submit(context.Background(), "a", 10, fakeSource("a", 3, nil, nil))
	if err != nil {
		t.Fatalf("Submit error = %v", err)
	}
	got := drain(t, tokens)
	want := []string{"a-0", "a-1", "a-2"}
	if len(got) != len(want) {
		t.Fatalf("tokens = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("tokens[%d] = %q, want %q (full: %v)", i, got[i], want[i], got)
		}
	}
	waitStats(t, e, func(s Stats) bool { return s.Completed == 1 && s.Active == 0 })
}

func TestEngine_Submit_Bad(t *testing.T) {
	e := New(Config{})
	defer e.Close()

	if _, err := e.Submit(context.Background(), "a", 0, nil); err == nil {
		t.Fatalf("Submit(nil source) error = nil, want non-nil")
	}

	e.Close()
	if _, err := e.Submit(context.Background(), "b", 0, fakeSource("b", 1, nil, nil)); err == nil {
		t.Fatalf("Submit after Close error = nil, want non-nil")
	}
}

func TestEngine_Submit_Ugly(t *testing.T) {
	// MaxActive: 1 occupied by a request that never releases, MaxQueue: 1
	// filled by a second request that can never be admitted either — a
	// third Submit has nowhere to go, so it must block on the caller's own
	// ctx rather than hang forever.
	e := New(Config{MaxActive: 1, MaxQueue: 1, StreamBuffer: 1})
	defer e.Close()

	started := make(chan string, 1)
	block := make(chan struct{}) // never closed — first request never proceeds past its release-wait
	if _, err := e.Submit(context.Background(), "first", 0, fakeSource("first", 1, started, block)); err != nil {
		t.Fatalf("Submit(first) error = %v", err)
	}
	<-started // first is now occupying the only active slot

	if _, err := e.Submit(context.Background(), "second", 0, fakeSource("second", 1, nil, nil)); err != nil {
		t.Fatalf("Submit(second) error = %v, want nil (fills the one queue slot)", err)
	}
	waitStats(t, e, func(s Stats) bool { return s.Queued == 1 })

	ctx, cancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer cancel()
	if _, err := e.Submit(ctx, "third", 0, fakeSource("third", 1, nil, nil)); err == nil {
		t.Fatalf("Submit(third) under backpressure error = nil, want ctx deadline error")
	}
}

func TestEngine_Cancel_Good(t *testing.T) {
	e := New(Config{MaxActive: 4, MaxQueue: 4, StreamBuffer: 4})
	defer e.Close()

	started := make(chan string, 1)
	block := make(chan struct{})
	tokens, err := e.Submit(context.Background(), "a", 0, fakeSource("a", 100, started, block))
	if err != nil {
		t.Fatalf("Submit error = %v", err)
	}
	<-started // active, blocked before its first yield
	e.Cancel("a")

	got := drain(t, tokens)
	if len(got) != 0 {
		t.Fatalf("tokens after cancelling a pre-yield active request = %v, want none", got)
	}
	waitStats(t, e, func(s Stats) bool { return s.Cancelled == 1 && s.Active == 0 })
}

func TestEngine_Cancel_Bad(t *testing.T) {
	// MaxActive: 1 occupied by a blocked request, so a second Submit sits
	// queued — Cancel on the QUEUED one must retire it without ever running
	// its Source.
	e := New(Config{MaxActive: 1, MaxQueue: 4, StreamBuffer: 4})
	defer e.Close()

	firstStarted := make(chan string, 1)
	firstBlock := make(chan struct{})
	if _, err := e.Submit(context.Background(), "first", 0, fakeSource("first", 1, firstStarted, firstBlock)); err != nil {
		t.Fatalf("Submit(first) error = %v", err)
	}
	<-firstStarted

	secondStarted := make(chan string, 1)
	tokens, err := e.Submit(context.Background(), "second", 0, fakeSource("second", 1, secondStarted, nil))
	if err != nil {
		t.Fatalf("Submit(second) error = %v", err)
	}
	e.Cancel("second")

	got := drain(t, tokens)
	if len(got) != 0 {
		t.Fatalf("tokens for a cancelled QUEUED request = %v, want none", got)
	}
	select {
	case <-secondStarted:
		t.Fatalf("cancelled queued request's Source ran — it must never start")
	case <-time.After(20 * time.Millisecond):
	}
	waitStats(t, e, func(s Stats) bool { return s.Cancelled == 1 })
}

func TestEngine_Cancel_Ugly(t *testing.T) {
	e := New(Config{MaxActive: 1, MaxQueue: 1, StreamBuffer: 1})
	defer e.Close()

	e.Cancel("does-not-exist") // must not panic or hang
	if s := e.Stats(); s.Cancelled != 0 {
		t.Fatalf("Cancel(unknown id) affected Stats = %+v, want unchanged", s)
	}
}

func TestEngine_Submit_MaxActive(t *testing.T) {
	e := New(Config{MaxActive: 1, MaxQueue: 4, StreamBuffer: 4})
	defer e.Close()

	firstStarted := make(chan string, 1)
	firstBlock := make(chan struct{})
	if _, err := e.Submit(context.Background(), "first", 0, fakeSource("first", 1, firstStarted, firstBlock)); err != nil {
		t.Fatalf("Submit(first) error = %v", err)
	}
	<-firstStarted

	secondStarted := make(chan string, 1)
	if _, err := e.Submit(context.Background(), "second", 0, fakeSource("second", 1, secondStarted, nil)); err != nil {
		t.Fatalf("Submit(second) error = %v", err)
	}
	waitStats(t, e, func(s Stats) bool { return s.Queued == 1 })

	select {
	case <-secondStarted:
		t.Fatalf("second request started while MaxActive: 1 was occupied — admission budget not honoured")
	case <-time.After(20 * time.Millisecond):
	}

	close(firstBlock) // let first finish and retire, freeing the slot
	select {
	case <-secondStarted:
	case <-time.After(5 * time.Second):
		t.Fatalf("second request never started after the slot freed")
	}
}

func TestEngine_Submit_MaxBatchTokens(t *testing.T) {
	e := New(Config{MaxActive: 4, MaxQueue: 4, MaxBatchTokens: 100, StreamBuffer: 4})
	defer e.Close()

	firstStarted := make(chan string, 1)
	firstBlock := make(chan struct{})
	if _, err := e.Submit(context.Background(), "first", 80, fakeSource("first", 1, firstStarted, firstBlock)); err != nil {
		t.Fatalf("Submit(first) error = %v", err)
	}
	<-firstStarted // 80/100 of the budget is now committed

	secondStarted := make(chan string, 1)
	if _, err := e.Submit(context.Background(), "second", 50, fakeSource("second", 1, secondStarted, nil)); err != nil {
		t.Fatalf("Submit(second) error = %v", err)
	}
	waitStats(t, e, func(s Stats) bool { return s.Queued == 1 })

	select {
	case <-secondStarted:
		t.Fatalf("second request (80+50 > 100 budget) started — MaxBatchTokens not honoured")
	case <-time.After(20 * time.Millisecond):
	}

	close(firstBlock) // first retires, freeing its 80 tokens of budget
	select {
	case <-secondStarted:
	case <-time.After(5 * time.Second):
		t.Fatalf("second request never started once the token budget freed")
	}
}

func TestEngine_Submit_BackpressureIsolation(t *testing.T) {
	e := New(Config{MaxActive: 4, MaxQueue: 4, StreamBuffer: 1})
	defer e.Close()

	// slow never drains its channel beyond the buffer, so its own goroutine
	// blocks delivering token index 1 (index 0 fills the size-1 buffer).
	slowTokens, err := e.Submit(context.Background(), "slow", 0, fakeSource("slow", 5, nil, nil))
	if err != nil {
		t.Fatalf("Submit(slow) error = %v", err)
	}

	fastTokens, err := e.Submit(context.Background(), "fast", 0, fakeSource("fast", 3, nil, nil))
	if err != nil {
		t.Fatalf("Submit(fast) error = %v", err)
	}
	got := drain(t, fastTokens)
	if len(got) != 3 {
		t.Fatalf("fast tokens = %v, want 3 tokens — a slow peer must not block it", got)
	}
	waitStats(t, e, func(s Stats) bool { return s.Completed >= 1 })

	// Cleanup: cancel the still-blocked slow request so Close doesn't have
	// to wait out its Source (t.Cleanup via Close would still terminate it,
	// this just keeps the test's own teardown fast).
	e.Cancel("slow")
	drain(t, slowTokens)
}

func TestEngine_Close_Good(t *testing.T) {
	e := New(Config{MaxActive: 1, MaxQueue: 4, StreamBuffer: 4})

	activeStarted := make(chan string, 1)
	activeBlock := make(chan struct{})
	activeTokens, err := e.Submit(context.Background(), "active", 0, fakeSource("active", 1, activeStarted, activeBlock))
	if err != nil {
		t.Fatalf("Submit(active) error = %v", err)
	}
	<-activeStarted

	queuedTokens, err := e.Submit(context.Background(), "queued", 0, fakeSource("queued", 1, nil, nil))
	if err != nil {
		t.Fatalf("Submit(queued) error = %v", err)
	}

	e.Close() // must return once every goroutine it started has exited

	if got := drain(t, activeTokens); len(got) != 0 {
		t.Fatalf("active request's tokens after Close = %v, want none (cancelled before its release)", got)
	}
	if got := drain(t, queuedTokens); len(got) != 0 {
		t.Fatalf("queued request's tokens after Close = %v, want none (never admitted)", got)
	}
	if s := e.Stats(); s.Active != 0 || s.Queued != 0 {
		t.Fatalf("Stats after Close = %+v, want Active=0 Queued=0", s)
	}

	if _, err := e.Submit(context.Background(), "after-close", 0, fakeSource("after-close", 1, nil, nil)); err == nil {
		t.Fatalf("Submit after Close error = nil, want non-nil")
	}
}
