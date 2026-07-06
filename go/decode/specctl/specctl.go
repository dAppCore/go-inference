// SPDX-Licence-Identifier: EUPL-1.2

// Package specctl is the adaptive speculative-length controller for the
// speculative decoding path. It pairs with pkg/ngram (the drafter): the drafter
// proposes draft tokens, the target model verifies and accepts a prefix of them,
// and this controller decides HOW MANY tokens to propose next time. Proposing
// too few wastes the target's batch verify; proposing too many wastes draft work
// the target throws away. The right number depends on how well recent drafts
// landed, which varies with the text — so the controller watches the acceptance
// rate and lengthens or shortens the draft to match (the same idea as SGLang's
// adaptive speculative-step policy, implemented as a clean continuous Go rule).
//
// Accept-rate method — EXPONENTIAL MOVING AVERAGE. Each Record folds the call's
// per-token acceptance ratio (accepted/proposed) into a running rate:
//
//	rate = (1-α)·rate + α·sample,  α = 2/(Window+1)
//
// α is the standard EMA smoothing factor for a Window-length average: a larger
// Window reacts more slowly (longer memory), a Window of 1 tracks the last
// sample alone. The rate lives in [0,1] and needs no history buffer.
//
// Length rule — LINEAR INTERPOLATION over [Min, Max]:
//
//	NextLength = clamp(round(Min + rate·(Max-Min)), Min, Max)
//
// Monotonic in the accept rate: rate 1.0 → Max (drafts are landing, speculate
// hard), rate 0.0 → Min (drafts are missing, stop wasting work), and a mid rate
// lands proportionally between. Cold start (no Record yet) seeds the rate at 1.0
// so a fresh controller speculates optimistically at Max until evidence lowers it
// — the same "explore higher first, let the average catch up" bias as SGLang.
//
//	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 8})
//	for {
//	    draft := drafter.DraftNext(c.NextLength()) // propose this many
//	    accepted := target.Verify(draft)           // target accepts a prefix
//	    c.Record(len(draft), len(accepted))         // feed the outcome back
//	}
package specctl

import (
	"sync"

	core "dappco.re/go"
)

// Controller configures the adaptive draft-length policy. Min and Max bound the
// recommended draft length (Min is clamped ≥ 1; Max is repaired to ≥ Min so the
// range never inverts). Window sizes the acceptance-rate EMA — larger reacts more
// slowly, smaller tracks recent samples more tightly (clamped ≥ 1). The zero
// Controller is a usable Min=1, Max=1, single-sample drafter rather than a dead
// one. New consumes a Controller config and returns the running *Adaptive.
//
//	specctl.Controller{Min: 1, Max: 8, Window: 8} // draft 1..8, ~8-sample EMA
type Controller struct {
	Min    int // lower draft-length bound (clamped ≥ 1)
	Max    int // upper draft-length bound (repaired to ≥ Min)
	Window int // EMA window for the accept rate (clamped ≥ 1)
}

// Adaptive runs one speculative-length policy. Construct with New. All methods
// take an internal lock, so a single Adaptive may be driven from many request
// goroutines (the verify loop and a metrics reader, say) without data races.
type Adaptive struct {
	mu    sync.Mutex
	min   int
	max   int
	alpha float64 // EMA smoothing factor, 2/(Window+1)
	rate  float64 // current acceptance rate in [0,1]
}

// New builds a running controller from a Controller config, clamping the config
// to sane bounds (Min ≥ 1, Max ≥ Min, Window ≥ 1) and seeding the accept rate at
// the optimistic cold-start default of 1.0.
//
//	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 8})
func New(cfg Controller) *Adaptive {
	minLen := max(cfg.Min, 1)
	maxLen := max(cfg.Max, minLen)
	window := max(cfg.Window, 1)
	return &Adaptive{
		min:   minLen,
		max:   maxLen,
		alpha: 2.0 / (float64(window) + 1.0),
		rate:  1.0,
	}
}

// Record folds one draft outcome into the acceptance-rate EMA: of `proposed`
// speculative tokens the target accepted `accepted`. `proposed <= 0` is a no-op
// (nothing was speculated, so there is nothing to learn). `accepted` is clamped
// to [0, proposed] so a caller passing a stale or oversized count cannot push the
// rate outside [0,1].
//
//	c.Record(len(draft), len(verified)) // e.g. proposed 8, accepted 5
func (a *Adaptive) Record(proposed, accepted int) {
	if proposed <= 0 {
		return // no speculation this round — nothing to record
	}
	accepted = core.Clamp(accepted, 0, proposed)
	sample := float64(accepted) / float64(proposed)

	a.mu.Lock()
	a.rate = (1.0-a.alpha)*a.rate + a.alpha*sample
	a.mu.Unlock()
}

// NextLength returns the recommended draft length in [Min, Max], interpolated
// linearly from the current accept rate: high acceptance → toward Max, low →
// toward Min. Always safe to call, including before any Record (cold start →
// Max).
//
//	n := c.NextLength() // how many tokens the drafter should propose next
func (a *Adaptive) NextLength() int {
	a.mu.Lock()
	rate := a.rate
	a.mu.Unlock()

	span := float64(a.max - a.min)
	length := int(core.Round(float64(a.min) + rate*span))
	return core.Clamp(length, a.min, a.max)
}

// AcceptRate returns the current acceptance-rate EMA in [0,1]. A fresh or freshly
// Reset controller reports the optimistic cold-start value of 1.0.
//
//	if c.AcceptRate() < 0.2 { /* drafts are mostly missing */ }
func (a *Adaptive) AcceptRate() float64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.rate
}

// Reset clears the learned acceptance rate back to the cold-start default of 1.0,
// so the controller speculates optimistically again (e.g. on a new request whose
// text shares nothing with the last). Bounds and window are unchanged.
//
//	c.Reset() // forget recent acceptance, start optimistic
func (a *Adaptive) Reset() {
	a.mu.Lock()
	a.rate = 1.0
	a.mu.Unlock()
}
