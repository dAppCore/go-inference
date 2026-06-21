// SPDX-Licence-Identifier: EUPL-1.2

package retry

import (
	"context"
	"time"

	core "dappco.re/go"
)

// Policy tunes the backoff loop (RFC §6.7). A zero Policy is usable —
// Do fills in conservative defaults (one attempt, no growth) — but a caller
// normally sets the interval/attempt envelope. sleep is unexported and injected
// only by tests, so production always waits on the real clock while a test
// records the schedule without blocking.
//
//	retry.Policy{
//	    InitialInterval: 200 * time.Millisecond, // first backoff
//	    MaxInterval:     10 * time.Second,        // ceiling per backoff
//	    MaxElapsed:      time.Minute,             // total budget across attempts
//	    MaxAttempts:     5,                       // hard attempt cap
//	    Multiplier:      2.0,                     // exponential growth factor
//	}
type Policy struct {
	InitialInterval time.Duration // backoff before the first retry
	MaxInterval     time.Duration // upper bound any single backoff is capped to
	MaxElapsed      time.Duration // total wall-clock budget; 0 = unbounded
	MaxAttempts     int           // maximum calls of fn; <=0 means one attempt
	Multiplier      float64       // backoff growth per retry; <=1 means constant

	// sleep waits for d. nil defaults to time.Sleep; tests inject a recorder
	// so the backoff schedule is asserted without real delay.
	sleep func(time.Duration)
}

// Do calls fn, classifying each failure with classify and retrying the
// retryable classes (§6.7 — 429, 502, 503, provider-overloaded, timeout) with
// exponential backoff. It stops — returning fn's last error — on the first
// success (nil), a permanent class, the attempt cap, the elapsed budget, or a
// cancelled context. A permanent failure surfaces immediately with no backoff.
//
//	err := retry.Do(ctx, func() error { return client.Chat(req) }, retry.ClassifyErr, p)
//	if err != nil { /* exhausted or permanent — fall out / fail */ }
func Do(ctx context.Context, fn func() error, classify func(error) Class, policy Policy) error {
	sleep := policy.sleep
	if sleep == nil {
		sleep = time.Sleep
	}
	attempts := policy.MaxAttempts
	if attempts <= 0 {
		attempts = 1
	}

	// A context already cancelled before the first call short-circuits — Do
	// never invokes fn under a dead context.
	if err := ctx.Err(); err != nil {
		return core.E("retry", "context cancelled before first attempt", err)
	}

	start := time.Now()
	interval := policy.InitialInterval
	var lastErr error

	for attempt := 1; ; attempt++ {
		lastErr = fn()
		if lastErr == nil {
			return nil
		}

		// A permanent class is surfaced as-is — no backoff, no further tries.
		if !Retryable(classify(lastErr)) {
			return lastErr
		}

		// Out of attempts — return the failure that exhausted the budget.
		if attempt >= attempts {
			return lastErr
		}

		// Compute this retry's backoff (capped at MaxInterval), then check it
		// against the remaining elapsed budget before waiting.
		wait := nextInterval(interval, policy.MaxInterval)
		if policy.MaxElapsed > 0 && time.Since(start)+wait > policy.MaxElapsed {
			return lastErr
		}

		// Honour cancellation while waiting out the backoff rather than
		// sleeping through a dead context.
		if !waitOrCancel(ctx, sleep, wait) {
			return core.E("retry", "context cancelled during backoff", ctx.Err())
		}

		interval = growInterval(interval, policy.Multiplier, policy.MaxInterval)
	}
}

// nextInterval clamps the current backoff to the ceiling. A zero or negative
// max leaves it unclamped.
func nextInterval(current, max time.Duration) time.Duration {
	if max > 0 && current > max {
		return max
	}
	return current
}

// growInterval advances the backoff by the multiplier, clamped to the ceiling.
// A multiplier <=1 keeps the interval constant (still capped).
func growInterval(current time.Duration, multiplier float64, max time.Duration) time.Duration {
	next := current
	if multiplier > 1 {
		next = time.Duration(float64(current) * multiplier)
	}
	if max > 0 && next > max {
		return max
	}
	return next
}

// waitOrCancel waits out d via the injected sleeper unless ctx is already done,
// returning false if the context was cancelled before the wait. The sleeper is
// synchronous (time.Sleep in production, a recorder in tests), so the
// cancellation check is taken up-front — a cancelled context never enters the
// sleep.
func waitOrCancel(ctx context.Context, sleep func(time.Duration), d time.Duration) bool {
	if ctx.Err() != nil {
		return false
	}
	sleep(d)
	return ctx.Err() == nil
}
