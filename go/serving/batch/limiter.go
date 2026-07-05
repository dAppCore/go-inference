// SPDX-Licence-Identifier: EUPL-1.2

package batch

import (
	"context"
	"sync"
	"time"
)

// Limiter throttles dispatch so a provider's request budget is never exceeded.
// Wait blocks until the next call may proceed, or returns the context error if
// the context is cancelled while waiting. The batch executor (§6.3) calls Wait
// once before EVERY item — batched or single — so the same per-provider /
// per-key budget governs both paths.
//
//	if err := opts.Limiter.Wait(ctx); err != nil { return err }
//	// ... safe to dispatch one request ...
type Limiter interface {
	Wait(ctx context.Context) error
}

// TokenBucket is a goroutine-safe token-bucket Limiter: it admits up to burst
// calls immediately, then refills one token every 1/ratePerSecond. It is the
// per-provider / per-key rate limiter of §6.3 — requests per second plus a
// burst size — so a batch fanning out under a concurrency cap still never
// outpaces the provider's limit.
//
//	tb := batch.NewTokenBucket(10, 5) // 10 req/s, burst of 5
//	tb.Wait(ctx)                       // blocks once the burst is spent
type TokenBucket struct {
	mu       sync.Mutex
	interval time.Duration // gap between refilled tokens (0 = unlimited)
	burst    float64       // maximum tokens the bucket can hold
	tokens   float64       // tokens currently available
	last     time.Time     // when tokens were last refilled
}

// NewTokenBucket builds a token bucket admitting burst calls immediately and
// then ratePerSecond calls per second thereafter. A ratePerSecond <= 0 means
// "no rate limit" (every Wait returns at once); a burst < 1 is clamped to 1 so
// at least one call can always proceed.
//
//	lim := batch.NewTokenBucket(200, 1) // 200/s, one at a time
func NewTokenBucket(ratePerSecond float64, burst int) *TokenBucket {
	b := float64(burst)
	if b < 1 {
		b = 1
	}
	tb := &TokenBucket{
		burst:  b,
		tokens: b, // start full so the first burst fires immediately
		last:   time.Now(),
	}
	if ratePerSecond > 0 {
		tb.interval = time.Duration(float64(time.Second) / ratePerSecond)
	}
	return tb
}

// Wait blocks until a token is available, then consumes it. With no rate limit
// (ratePerSecond <= 0) it returns immediately. It respects context
// cancellation: a cancelled or deadline-exceeded context unblocks the wait with
// that context's error rather than sleeping out the interval.
func (tb *TokenBucket) Wait(ctx context.Context) error {
	for {
		if err := ctx.Err(); err != nil {
			return err
		}

		tb.mu.Lock()
		if tb.interval == 0 {
			// Unlimited: nothing to throttle.
			tb.mu.Unlock()
			return nil
		}
		tb.refill()
		if tb.tokens >= 1 {
			tb.tokens--
			tb.mu.Unlock()
			return nil
		}
		// Not enough yet — work out how long until the next whole token.
		wait := time.Duration((1 - tb.tokens) * float64(tb.interval))
		tb.mu.Unlock()

		if wait <= 0 {
			wait = tb.interval
		}
		timer := time.NewTimer(wait)
		select {
		case <-timer.C:
			// Loop and re-check; another goroutine may have taken the token.
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		}
	}
}

// refill adds the tokens accrued since the last refill, capped at burst. The
// caller holds tb.mu.
func (tb *TokenBucket) refill() {
	now := time.Now()
	elapsed := now.Sub(tb.last)
	if elapsed <= 0 {
		return
	}
	tb.last = now
	tb.tokens += float64(elapsed) / float64(tb.interval)
	if tb.tokens > tb.burst {
		tb.tokens = tb.burst
	}
}
