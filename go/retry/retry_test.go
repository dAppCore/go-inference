// SPDX-Licence-Identifier: EUPL-1.2

package retry

import (
	"context"
	"time"

	core "dappco.re/go"
)

// fakeFn returns a func that fails (with the given error) its first failures
// calls, then succeeds. It records how many times it was invoked so a test can
// assert the attempt count.
//
//	fn, calls := fakeFn(2, core.E("retry", "boom", nil))
//	_ = Do(ctx, fn, classify, policy)
//	core.AssertEqual(t, 3, *calls)  // 2 failures + 1 success
func fakeFn(failures int, err error) (func() error, *int) {
	calls := 0
	return func() error {
		calls++
		if calls <= failures {
			return err
		}
		return nil
	}, &calls
}

// recordSleeper returns a sleep func that records each requested duration
// without ever blocking — so backoff is asserted, never waited on.
func recordSleeper() (func(time.Duration), *[]time.Duration) {
	var slept []time.Duration
	return func(d time.Duration) { slept = append(slept, d) }, &slept
}

// errOf classifies by inspecting a sentinel error's code, so a fake fn can
// signal which class it is returning.
func classOf(c Class) func(error) Class {
	return func(error) Class { return c }
}

func TestRetry_Classify_Good(t *core.T) {
	// HTTP-ish statuses map onto the documented classes (§6.7).
	core.AssertEqual(t, ClassBadRequest, Classify(400))
	core.AssertEqual(t, ClassUnauthorised, Classify(401))
	core.AssertEqual(t, ClassPaymentRequired, Classify(402))
	core.AssertEqual(t, ClassForbidden, Classify(403))
	core.AssertEqual(t, ClassNotFound, Classify(404))
	core.AssertEqual(t, ClassPayloadTooLarge, Classify(413))
	core.AssertEqual(t, ClassUnprocessable, Classify(422))
	core.AssertEqual(t, ClassRateLimited, Classify(429))
	core.AssertEqual(t, ClassBadGateway, Classify(502))
	core.AssertEqual(t, ClassServiceUnavailable, Classify(503))
	core.AssertEqual(t, ClassInternal, Classify(500))

	// 2xx is not a failure class.
	core.AssertEqual(t, ClassNone, Classify(200))
}

func TestRetry_Retryable_Bad(t *core.T) {
	// Retryable classes per the RFC: rate-limited, provider-overloaded,
	// timeout, bad-gateway, service-unavailable.
	core.AssertTrue(t, Retryable(ClassRateLimited))
	core.AssertTrue(t, Retryable(ClassProviderOverloaded))
	core.AssertTrue(t, Retryable(ClassTimeout))
	core.AssertTrue(t, Retryable(ClassBadGateway))
	core.AssertTrue(t, Retryable(ClassServiceUnavailable))

	// Everything else surfaces immediately.
	core.AssertFalse(t, Retryable(ClassBadRequest))
	core.AssertFalse(t, Retryable(ClassUnauthorised))
	core.AssertFalse(t, Retryable(ClassPaymentRequired))
	core.AssertFalse(t, Retryable(ClassForbidden))
	core.AssertFalse(t, Retryable(ClassNotFound))
	core.AssertFalse(t, Retryable(ClassPayloadTooLarge))
	core.AssertFalse(t, Retryable(ClassUnprocessable))
	core.AssertFalse(t, Retryable(ClassInternal))
	core.AssertFalse(t, Retryable(ClassNone))
}

func TestRetry_Classify_Ugly(t *core.T) {
	// An unmapped / unknown status is treated as a permanent internal failure
	// rather than silently retried forever.
	core.AssertEqual(t, ClassInternal, Classify(418))
	core.AssertEqual(t, ClassInternal, Classify(0))
	core.AssertFalse(t, Retryable(Classify(418)))

	// A class beyond the known set is not retryable (fail closed).
	core.AssertFalse(t, Retryable(Class(9999)))
}

func TestRetry_Do_Good(t *core.T) {
	// Fails twice with a retryable class, then succeeds: Do returns nil and
	// the function was called exactly three times, with two backoff sleeps.
	sleep, slept := recordSleeper()
	fn, calls := fakeFn(2, core.E("provider", "503", nil))
	p := Policy{
		InitialInterval: 100 * time.Millisecond,
		MaxInterval:     2 * time.Second,
		MaxElapsed:      10 * time.Second,
		MaxAttempts:     5,
		Multiplier:      2.0,
		sleep:           sleep,
	}

	err := Do(context.Background(), fn, classOf(ClassServiceUnavailable), p)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 3, *calls, "two failures then a success")

	// Backoff sleeps between the three attempts: 100ms then 200ms (×2).
	core.AssertEqual(t, 2, len(*slept))
	core.AssertEqual(t, 100*time.Millisecond, (*slept)[0])
	core.AssertEqual(t, 200*time.Millisecond, (*slept)[1])
}

func TestRetry_Do_Bad(t *core.T) {
	// A permanent (non-retryable) class surfaces immediately: the function is
	// called once and never slept on.
	sleep, slept := recordSleeper()
	permanent := core.E("provider", "400 bad request", nil)
	fn, calls := fakeFn(99, permanent)
	p := Policy{
		InitialInterval: 50 * time.Millisecond,
		MaxAttempts:     5,
		sleep:           sleep,
	}

	err := Do(context.Background(), fn, classOf(ClassBadRequest), p)
	core.AssertError(t, err)
	core.AssertEqual(t, permanent, err, "the original error surfaces unchanged")
	core.AssertEqual(t, 1, *calls, "a permanent failure is not retried")
	core.AssertEqual(t, 0, len(*slept), "no backoff on a permanent failure")
}

func TestRetry_Do_Ugly(t *core.T) {
	// Attempts exhausted: a forever-failing retryable class is tried
	// MaxAttempts times and the LAST error is returned.
	sleep, slept := recordSleeper()
	boom := core.E("provider", "429 rate limited", nil)
	fn, calls := fakeFn(99, boom)
	p := Policy{
		InitialInterval: 10 * time.Millisecond,
		MaxInterval:     40 * time.Millisecond,
		MaxElapsed:      time.Hour, // generous — attempts is the binding limit here
		MaxAttempts:     4,
		Multiplier:      2.0,
		sleep:           sleep,
	}

	err := Do(context.Background(), fn, classOf(ClassRateLimited), p)
	core.AssertError(t, err)
	core.AssertEqual(t, 4, *calls, "MaxAttempts caps the retries")
	// 4 attempts → 3 sleeps between them, capped at MaxInterval=40ms:
	// 10ms, 20ms, 40ms (the 4th would-be 80ms is capped to 40ms).
	core.AssertEqual(t, 3, len(*slept))
	core.AssertEqual(t, 10*time.Millisecond, (*slept)[0])
	core.AssertEqual(t, 20*time.Millisecond, (*slept)[1])
	core.AssertEqual(t, 40*time.Millisecond, (*slept)[2], "backoff is capped at MaxInterval")

	// And a context already cancelled stops before the first call even starts.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	fn2, calls2 := fakeFn(99, boom)
	err2 := Do(ctx, fn2, classOf(ClassRateLimited), p)
	core.AssertError(t, err2)
	core.AssertEqual(t, 0, *calls2, "a cancelled context does not call fn")
}

// TestRetry_Do_Elapsed covers the elapsed-budget guard: when the next backoff
// would push total wall-clock past MaxElapsed, Do stops and returns the last
// error WITHOUT sleeping that final backoff. A MaxElapsed smaller than the very
// first interval trips the guard before any wait.
func TestRetry_Do_Elapsed(t *core.T) {
	sleep, slept := recordSleeper()
	boom := core.E("provider", "503", nil)
	fn, calls := fakeFn(99, boom) // always fails (retryable)
	p := Policy{
		InitialInterval: 5 * time.Second, // first backoff alone exceeds the budget
		MaxInterval:     10 * time.Second,
		MaxElapsed:      time.Millisecond, // tiny budget — the wait won't fit
		MaxAttempts:     5,
		Multiplier:      2.0,
		sleep:           sleep,
	}

	err := Do(context.Background(), fn, classOf(ClassServiceUnavailable), p)
	core.AssertError(t, err)
	core.AssertEqual(t, boom, err, "the last error surfaces when the budget is exhausted")
	// fn ran once; the budget guard fired before the backoff, so nothing slept.
	core.AssertEqual(t, 1, *calls, "the elapsed guard stops further attempts")
	core.AssertEqual(t, 0, len(*slept), "no backoff is slept once the budget is blown")
}

// TestRetry_Do_CancelDuringBackoff covers cancellation observed while waiting
// out a backoff: the injected sleeper cancels the context mid-wait, so the
// post-sleep cancellation check in waitOrCancel returns false and Do reports a
// "cancelled during backoff" error rather than retrying.
func TestRetry_Do_CancelDuringBackoff(t *core.T) {
	ctx, cancel := context.WithCancel(context.Background())
	boom := core.E("provider", "429", nil)
	fn, calls := fakeFn(99, boom) // always fails (retryable)

	// A sleeper that cancels the context as it "waits" — modelling the context
	// being cancelled during the backoff window.
	cancelDuringSleep := func(time.Duration) { cancel() }

	p := Policy{
		InitialInterval: 10 * time.Millisecond,
		MaxInterval:     time.Second,
		MaxElapsed:      time.Hour, // generous — cancellation, not budget, ends it
		MaxAttempts:     5,
		Multiplier:      2.0,
		sleep:           cancelDuringSleep,
	}

	err := Do(ctx, fn, classOf(ClassRateLimited), p)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "cancelled during backoff")
	// fn ran once, then the first backoff observed the cancellation.
	core.AssertEqual(t, 1, *calls, "Do stops once cancellation is seen in backoff")
}

// TestRetry_Do_DefaultSleeper covers the production default path of Do where no
// sleeper is injected (sleep == nil → time.Sleep). The call succeeds on the
// first attempt so the real clock is never actually waited on, exercising the
// nil-sleeper defaulting branch without slowing the test.
func TestRetry_Do_DefaultSleeper(t *core.T) {
	fn, calls := fakeFn(0, core.E("provider", "unused", nil)) // succeeds immediately
	p := Policy{
		InitialInterval: time.Hour, // would be ruinous if ever slept — it isn't
		MaxAttempts:     3,
		// sleep left nil on purpose: Do must default it to time.Sleep.
	}

	err := Do(context.Background(), fn, classOf(ClassNone), p)
	core.AssertNoError(t, err, "first-try success never reaches the sleeper")
	core.AssertEqual(t, 1, *calls)
}

// TestRetry_Do_ZeroAttempts covers the attempt-cap default: a Policy with
// MaxAttempts <= 0 is normalised to a single attempt. A retryable failure is
// therefore surfaced after exactly one call, with no backoff.
func TestRetry_Do_ZeroAttempts(t *core.T) {
	sleep, slept := recordSleeper()
	boom := core.E("provider", "503", nil)
	fn, calls := fakeFn(99, boom)
	p := Policy{
		InitialInterval: 10 * time.Millisecond,
		MaxAttempts:     0, // <=0 → one attempt
		Multiplier:      2.0,
		sleep:           sleep,
	}

	err := Do(context.Background(), fn, classOf(ClassServiceUnavailable), p)
	core.AssertError(t, err)
	core.AssertEqual(t, boom, err)
	core.AssertEqual(t, 1, *calls, "MaxAttempts<=0 means a single attempt")
	core.AssertEqual(t, 0, len(*slept), "a single attempt never backs off")

	// A negative MaxAttempts is normalised the same way.
	fnNeg, callsNeg := fakeFn(99, boom)
	pNeg := p
	pNeg.MaxAttempts = -5
	_ = Do(context.Background(), fnNeg, classOf(ClassServiceUnavailable), pNeg)
	core.AssertEqual(t, 1, *callsNeg, "a negative MaxAttempts is also one attempt")
}

// TestRetry_Do_CancelBeforeBackoff covers waitOrCancel's up-front guard: if the
// context is cancelled in the window between the failed attempt and the wait,
// the sleeper is never entered and Do reports "cancelled during backoff". Here
// fn cancels the context as it fails, so waitOrCancel sees a dead context on
// entry (its ctx.Err() != nil branch) and returns false without sleeping.
func TestRetry_Do_CancelBeforeBackoff(t *core.T) {
	ctx, cancel := context.WithCancel(context.Background())
	sleep, slept := recordSleeper()
	boom := core.E("provider", "429", nil)

	calls := 0
	fn := func() error {
		calls++
		cancel() // cancel during the attempt, before the backoff wait
		return boom
	}

	p := Policy{
		InitialInterval: 10 * time.Millisecond,
		MaxInterval:     time.Second,
		MaxElapsed:      time.Hour,
		MaxAttempts:     5,
		Multiplier:      2.0,
		sleep:           sleep,
	}

	err := Do(ctx, fn, classOf(ClassRateLimited), p)
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "cancelled during backoff")
	core.AssertEqual(t, 1, calls, "Do stops after the first attempt's backoff is cancelled")
	core.AssertEqual(t, 0, len(*slept), "the up-front cancel guard skips the sleep entirely")
}

// TestRetry_Do_ClampFirstInterval covers nextInterval's clamp branch: when the
// initial interval already exceeds MaxInterval, the first (and every) backoff is
// clamped down to the ceiling rather than sleeping the larger initial value.
func TestRetry_Do_ClampFirstInterval(t *core.T) {
	sleep, slept := recordSleeper()
	boom := core.E("provider", "503", nil)
	fn, calls := fakeFn(2, boom) // fail twice, then succeed
	p := Policy{
		InitialInterval: 5 * time.Second, // larger than the ceiling
		MaxInterval:     200 * time.Millisecond,
		MaxElapsed:      time.Hour,
		MaxAttempts:     5,
		Multiplier:      2.0,
		sleep:           sleep,
	}

	err := Do(context.Background(), fn, classOf(ClassServiceUnavailable), p)
	core.AssertNoError(t, err)
	core.AssertEqual(t, 3, *calls, "two failures then success")
	// Both backoffs are clamped to the 200ms ceiling — the 5s initial never sleeps.
	core.AssertEqual(t, 2, len(*slept))
	core.AssertEqual(t, 200*time.Millisecond, (*slept)[0], "first backoff clamped to MaxInterval")
	core.AssertEqual(t, 200*time.Millisecond, (*slept)[1], "growth stays clamped to MaxInterval")
}
