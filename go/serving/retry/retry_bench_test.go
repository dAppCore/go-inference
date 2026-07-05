// SPDX-Licence-Identifier: EUPL-1.2

package retry

import (
	"context"
	"time"

	core "dappco.re/go"
)

// Package-level sinks defeat dead-code elimination so the benchmarked work is
// not optimised away by the compiler.
var (
	benchErrSink   error
	benchClassSink Class
)

// noopSleep is a zero-cost sleeper injected into Policy so the backoff-driving
// benchmarks exercise the attempt loop without waiting on a real clock — and
// without the per-call allocation recordSleeper's append would otherwise add.
func noopSleep(time.Duration) {}

// benchPolicy is a realistic production envelope (RFC §6.7 defaults) with the
// clock stubbed out — 200ms→10s exponential backoff, a one-minute budget, five
// attempts.
func benchPolicy() Policy {
	return Policy{
		InitialInterval: 200 * time.Millisecond,
		MaxInterval:     10 * time.Second,
		MaxElapsed:      time.Minute,
		MaxAttempts:     5,
		Multiplier:      2.0,
		sleep:           noopSleep,
	}
}

func BenchmarkClassify(b *core.B) {
	// Three representative paths: a mapped failure (429), success (2xx), and the
	// unmapped fail-closed fallback (418→ClassInternal).
	cases := []struct {
		name   string
		status int
	}{
		{"mapped", 429},
		{"success", 200},
		{"unmapped", 418},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *core.B) {
			b.ReportAllocs()
			var c Class
			for i := 0; i < b.N; i++ {
				c = Classify(tc.status)
			}
			benchClassSink = c
		})
	}
}

func BenchmarkRetryable(b *core.B) {
	// A retryable class (back off) and a permanent one (surface) — both O(1)
	// switch arms.
	cases := []struct {
		name string
		c    Class
	}{
		{"retryable", ClassRateLimited},
		{"permanent", ClassBadRequest},
	}
	for _, tc := range cases {
		b.Run(tc.name, func(b *core.B) {
			b.ReportAllocs()
			var ok bool
			for i := 0; i < b.N; i++ {
				ok = Retryable(tc.c)
			}
			benchClassSink = tc.c
			_ = ok
		})
	}
}

func BenchmarkDo_SuccessFirstTry(b *core.B) {
	// The hot path: fn succeeds on the first call, so no classify, no backoff,
	// no sleep. This should be allocation-free — Do touches only value types.
	ctx := context.Background()
	classify := classOf(ClassNone)
	p := benchPolicy()
	fn := func() error { return nil }
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchErrSink = Do(ctx, fn, classify, p)
	}
}

func BenchmarkDo_SuccessFirstTry_FreshClosure(b *core.B) {
	// The production shape: a fresh closure per request capturing a per-call
	// value. Exposes any escape of fn through Do — if Do leaks fn, this closure
	// heap-allocates once per op; if not, it stays on the caller stack (0 allocs).
	ctx := context.Background()
	classify := classOf(ClassNone)
	p := benchPolicy()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		n := i
		benchErrSink = Do(ctx, func() error { _ = n; return nil }, classify, p)
	}
}

func BenchmarkDo_SuccessAfterN(b *core.B) {
	// fn fails twice with a retryable class then succeeds: two backoff sleeps
	// (stubbed) and two interval grows. The counter is reset each iteration so
	// every op runs the full fail→fail→succeed schedule; the closure is built
	// once so the loop measures Do, not closure creation.
	ctx := context.Background()
	classify := classOf(ClassServiceUnavailable)
	p := benchPolicy()
	retryErr := core.E("provider", "503", nil)
	calls := 0
	fn := func() error {
		calls++
		if calls <= 2 {
			return retryErr
		}
		return nil
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		calls = 0
		benchErrSink = Do(ctx, fn, classify, p)
	}
}

func BenchmarkDo_Exhausted(b *core.B) {
	// fn always fails with a retryable class: Do runs MaxAttempts times and
	// returns the last error. Exercises the full backoff loop to the attempt cap.
	ctx := context.Background()
	classify := classOf(ClassRateLimited)
	p := benchPolicy()
	retryErr := core.E("provider", "429", nil)
	fn := func() error { return retryErr }
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchErrSink = Do(ctx, fn, classify, p)
	}
}

func BenchmarkDo_CancelledBeforeFirstAttempt(b *core.B) {
	// The one allocating path: an already-cancelled context short-circuits before
	// fn is ever called and Do returns a wrapped error. The single alloc/op is
	// the inherent &core.Err{} for that returned error (retry.go:58) — ctx.Err()
	// itself yields the context.Canceled sentinel, which does not allocate.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	classify := classOf(ClassNone)
	p := benchPolicy()
	fn := func() error { return nil }
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchErrSink = Do(ctx, fn, classify, p)
	}
}

func BenchmarkDo_Permanent(b *core.B) {
	// fn fails with a permanent class: Do classifies once and surfaces the error
	// immediately — one call, no backoff.
	ctx := context.Background()
	classify := classOf(ClassBadRequest)
	p := benchPolicy()
	permErr := core.E("provider", "400", nil)
	fn := func() error { return permErr }
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchErrSink = Do(ctx, fn, classify, p)
	}
}
