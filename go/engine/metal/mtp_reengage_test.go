// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"
	"time"
)

// reengageAfterPlain returns a policy state that has measured a plain stretch
// of plainRate tok/s and armed a probe window opening at token 0, with the
// probe clock backdated so `elapsed` has already passed.
func reengageAfterPlain(plainRate float64, elapsed time.Duration) *mtpReengage {
	r := &mtpReengage{}
	r.bailFresh()
	r.notePlainStretch(int(plainRate), 1.0, 0) // emitted/wall = plainRate
	r.probeT0 = time.Now().Add(-elapsed)
	return r
}

func TestMTPReengageBailFresh(t *testing.T) {
	r := &mtpReengage{cooldown: 128}
	r.bailFresh()
	if r.cooldown != nativeAssistantReengageCooldownMin {
		t.Fatalf("bailFresh cooldown = %d, want %d", r.cooldown, nativeAssistantReengageCooldownMin)
	}
}

func TestMTPReengageBailAgain(t *testing.T) {
	r := &mtpReengage{}
	r.bailAgain() // from zero: doubles the min
	if r.cooldown != 2*nativeAssistantReengageCooldownMin {
		t.Fatalf("first bailAgain cooldown = %d, want %d", r.cooldown, 2*nativeAssistantReengageCooldownMin)
	}
	for range 8 {
		r.bailAgain()
	}
	if r.cooldown != nativeAssistantReengageCooldownMax {
		t.Fatalf("bailAgain cap = %d, want %d", r.cooldown, nativeAssistantReengageCooldownMax)
	}
}

func TestMTPReengageNotePlainStretch(t *testing.T) {
	r := &mtpReengage{}
	r.notePlainStretch(64, 0.5, 100)
	if r.plainRate != 128 {
		t.Fatalf("plainRate = %v, want 128", r.plainRate)
	}
	if !r.probing() || r.probeLeft != nativeAssistantReengageProbeBlocks || r.probeTok0 != 100 {
		t.Fatalf("probe not armed: left=%d tok0=%d", r.probeLeft, r.probeTok0)
	}
	// zero-emitted stretch must not clobber the measured rate
	r.notePlainStretch(0, 0.5, 120)
	if r.plainRate != 128 {
		t.Fatalf("plainRate clobbered to %v by empty stretch", r.plainRate)
	}
}

func TestMTPReengageProbeCycleAbortsOnFullyRejected(t *testing.T) {
	r := reengageAfterPlain(100, 50*time.Millisecond)
	if bail := r.probeCycle(0, 1); !bail {
		t.Fatal("fully-rejected probe cycle must bail")
	}
	if r.probing() {
		t.Fatal("probe window must be closed after the abort")
	}
	if r.cooldown != 2*nativeAssistantReengageCooldownMin {
		t.Fatalf("failed probe cooldown = %d, want doubled %d", r.cooldown, 2*nativeAssistantReengageCooldownMin)
	}
}

func TestMTPReengageProbeCycleNoEngageOnCycleOne(t *testing.T) {
	// 60 tokens in 50ms = 1200 tok/s, far above any bar — but cycle 1 never
	// engages on its own (burst bias); the window stays open instead.
	r := reengageAfterPlain(100, 50*time.Millisecond)
	if bail := r.probeCycle(5, 60); bail {
		t.Fatal("hot cycle 1 must not bail")
	}
	if !r.probing() {
		t.Fatal("cycle 1 must leave the probe window open, engaged verdicts start at cycle 2")
	}
}

func TestMTPReengageProbeCycleEngagesFromCycleTwo(t *testing.T) {
	r := reengageAfterPlain(100, time.Millisecond)
	if bail := r.probeCycle(5, 300); bail {
		t.Fatal("cycle 1 bailed unexpectedly")
	}
	// cycle 2 at 600 tokens over a ~1ms-old window reads far above the bar
	// under any test-runner load: engage now.
	if bail := r.probeCycle(5, 600); bail {
		t.Fatal("above-bar cycle 2 must engage, not bail")
	}
	if r.probing() {
		t.Fatal("engage must close the probe window")
	}
	if r.cooldown != nativeAssistantReengageCooldownMin {
		t.Fatalf("engage must reset cooldown, got %d", r.cooldown)
	}
	if r.winN != 0 {
		t.Fatalf("engage must reset the engaged-rate window, winN=%d", r.winN)
	}
}

func TestMTPReengageProbeCycleMediocreRidesFullWindowThenFails(t *testing.T) {
	// 100 tok/s plain, margin 1.08 → bar 108. One token per cycle over a
	// 100ms-old window reads ≤ ~30 tok/s however fast the test runs: below
	// the bar but not fully rejected — the probe rides all cycles and fails
	// on the last with a doubled cooldown.
	r := reengageAfterPlain(100, 100*time.Millisecond)
	for i := range nativeAssistantReengageProbeBlocks - 1 {
		if bail := r.probeCycle(1, i+1); bail {
			t.Fatalf("mediocre cycle %d bailed before the window closed", i+1)
		}
	}
	if bail := r.probeCycle(1, nativeAssistantReengageProbeBlocks); !bail {
		t.Fatal("mediocre probe must fail on its last cycle")
	}
	if r.cooldown != 2*nativeAssistantReengageCooldownMin {
		t.Fatalf("failed probe cooldown = %d, want doubled", r.cooldown)
	}
}

func TestMTPReengageEngagedCycleInertWithoutPlainRate(t *testing.T) {
	r := &mtpReengage{}
	for range 10 {
		if bail := r.engagedCycle(1, 1.0, 10); bail {
			t.Fatal("engaged window must be inert before any plain stretch has run")
		}
	}
}

func TestMTPReengageEngagedCycleExitsBelowPlain(t *testing.T) {
	r := reengageAfterPlain(100, 0)
	r.probeLeft = 0 // engaged, not probing
	// window unfilled: no verdict yet
	for i := range len(r.winTok) - 1 {
		if bail := r.engagedCycle(1, 0.1, i); bail {
			t.Fatalf("cycle %d bailed before the window filled", i)
		}
	}
	// filled at 1 token per 100ms = 10 tok/s < 100: exit with doubled cooldown
	if bail := r.engagedCycle(1, 0.1, 9); !bail {
		t.Fatal("filled window below plain must exit")
	}
	if r.cooldown != 2*nativeAssistantReengageCooldownMin {
		t.Fatalf("rate-exit cooldown = %d, want doubled", r.cooldown)
	}
}

func TestMTPReengageEngagedCycleHoldsAtOrAbovePlain(t *testing.T) {
	r := reengageAfterPlain(100, 0)
	r.probeLeft = 0
	// 20 tokens per 100ms = 200 tok/s >= 100: never exits
	for i := range 10 {
		if bail := r.engagedCycle(20, 0.1, i*20); bail {
			t.Fatal("above-plain window must stay engaged")
		}
	}
}
