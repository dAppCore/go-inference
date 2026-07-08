// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"time"

	core "dappco.re/go"
)

// mtp_reengage.go — the MTP low-accept re-engagement policy (#299), shared by
// the greedy and sampled speculative loops. A patience bail no longer retires
// the drafter for the whole request: the loop runs PLAIN for a bounded
// cooldown (timed — the live plain rate), then re-probes drafting and keeps it
// only while the DELIVERED token rate holds against the plain rate. The gate
// is net rate, never acceptance %: a probe cycle still emits its accepted+1
// tokens, so a failed probe costs only the rate delta over those cycles.
//
// Policy pieces, each earned by a receipt (78b42db):
//   - probe: abort on a fully-rejected cycle (unambiguous, no clock); no
//     engage verdict on cycle 1 (right after a plain stretch the drafter's
//     proposals are at their most locally obvious — the burst bias); engaging
//     needs plainRate × the hysteresis margin (a fast target otherwise sits
//     exactly on the threshold and flaps).
//   - engaged stretch: a rolling rate window exits the moment it drops below
//     plain (the acceptance streak misses 2-of-5/3-of-5 alternation that
//     loses to plain every cycle).
//   - failed probes AND rate-exits double the cooldown (capped) — a
//     net-negative pair converges to a ~1% probe tax; a passing probe resets
//     it; a fresh patience bail restarts at the minimum (a new section going
//     weak is a new episode, not a failed probe).
//
// LTHN_MTP_REENGAGE=0 (mtpReengageDisabled) restores the permanent bail.
const (
	nativeAssistantReengageCooldownMin = 32
	nativeAssistantReengageCooldownMax = 256
	nativeAssistantReengageProbeBlocks = 3
)

// nativeAssistantDeepBootstrapPos is the context depth past which an engaged
// pair with NO measured plain rate spends one bounded plain stretch to arm the
// economics gate. The #299 policy only measures plainRate on the first
// acceptance-streak bail — a drafter that stays strong never bails, so the
// rolling rate exit stays inert, and deep-context verify costs (SDPA + KV sync
// scale with position; drafting does not) can leave the pair fully engaged at
// HALF the plain rate (#358: 26B @16K measured 58.4 vs plain 116.2 at 74%
// acceptance). Below this depth the economics never inverted on any receipt,
// so short runs pay nothing.
const nativeAssistantDeepBootstrapPos = 8192

// nativeAssistantDeepBootstrapTokens is the bootstrap stretch length — long
// enough for a stable rate on a ~30ms/token target, small enough that a deep
// pair that is genuinely winning (12B @16K: 71.5 vs plain 62.8) pays only a
// few percent once per generation.
const nativeAssistantDeepBootstrapTokens = 8

// needsDeepBootstrap reports whether the loop should spend the bootstrap
// stretch now: no plain rate measured, the attended context is deep, and
// enough budget remains for the measurement to pay back.
func (r *mtpReengage) needsDeepBootstrap(pos, emitted, maxNew int) bool {
	return r.plainRate == 0 && pos >= nativeAssistantDeepBootstrapPos &&
		emitted+4*nativeAssistantDeepBootstrapTokens <= maxNew
}

// nativeAssistantReengageMargin is the engage-side hysteresis: a probe must
// beat the plain rate by this factor to re-engage, while an engaged stretch
// only exits when it falls below the plain rate itself. Without the gap, a
// fast target sits exactly on the threshold (an e2b probe at 2-3 accepted
// tokens reads ~176 tok/s against a 174 plain) and the loop flaps — engaging
// on the burst, decaying, exiting — losing the margin each round trip.
const nativeAssistantReengageMargin = 1.08

// mtpReengage carries the policy state for one generate call. The zero value
// is ready: cooldown 0 marks "no plain stretch has run yet", which keeps the
// engaged-rate exit inert until the first bail measures a plain rate.
type mtpReengage struct {
	cooldown  int     // plain tokens to run on the next bail (doubles on failure)
	plainRate float64 // tok/s measured over the last plain stretch
	probeLeft int     // cycles remaining in the open probe window (0 = engaged)
	probeTok0 int     // emitted-token count at probe start
	probeT0   time.Time
	winTok    [3]int // rolling engaged-cycle window: tokens
	winSec    [3]float64
	winN      int
}

// bailFresh opens a NEW disengage episode (the patience trigger): the cooldown
// restarts at the minimum rather than continuing any previous doubling.
func (r *mtpReengage) bailFresh() {
	r.cooldown = nativeAssistantReengageCooldownMin
}

// bailAgain doubles the cooldown (capped) — a failed probe or a rate-exit is
// evidence the pair cannot sustain here, so the next probe waits longer.
func (r *mtpReengage) bailAgain() {
	r.cooldown = min(max(r.cooldown, nativeAssistantReengageCooldownMin)*2, nativeAssistantReengageCooldownMax)
}

// notePlainStretch records the measured rate of a completed plain stretch and
// arms the probe window that follows it. tokensNow is the emitted-token count
// as the probe opens.
func (r *mtpReengage) notePlainStretch(emitted int, wall float64, tokensNow int) {
	if emitted > 0 && wall > 0 {
		r.plainRate = float64(emitted) / wall
	}
	r.probeLeft = nativeAssistantReengageProbeBlocks
	r.probeTok0 = tokensNow
	r.probeT0 = time.Now()
}

// probing reports whether a probe window is open. Callers capture this BEFORE
// the cycle runs — probeCycle mutates it.
func (r *mtpReengage) probing() bool { return r.probeLeft > 0 }

// probeCycle closes out one drafted cycle of an open probe window and decides
// as early as the evidence allows. A cycle whose drafts were FULLY rejected
// aborts on the spot — one token for a whole draft+verify round is
// unambiguously below any plain rate, no clock needed. A window already above
// the engage bar engages on the spot — except on cycle 1 (burst bias). Only a
// mediocre window rides the full probe; its last cycle settles it either way.
// bail=true means the probe failed: the caller runs the next plain stretch.
func (r *mtpReengage) probeCycle(cycleAccepted, tokensNow int) (bail bool) {
	if r.probeLeft <= 0 {
		return false
	}
	r.probeLeft--
	if r.probeLeft == nativeAssistantReengageProbeBlocks-1 {
		// Cycle 1 is pure warmup — no verdict AND no clock, in both directions.
		// Shallow, its rate reads systematically HIGH (the burst bias:
		// post-plain proposals are locally obvious); deep, it reads
		// systematically LOW (the drafter re-uploads its target-KV mirror
		// after any plain stretch — ~40ms at 16K — and its first block after a
		// re-seed can be fully rejected even when the steady pair wins: #358's
		// 12B receipts showed both). The fully-rejected abort applies only to
		// the MEASURED cycles.
		r.probeT0 = time.Now()
		r.probeTok0 = tokensNow
		return false
	}
	probeRate := 0.0
	if wall := time.Since(r.probeT0).Seconds(); wall > 0 {
		probeRate = float64(tokensNow-r.probeTok0) / wall
	}
	engageBar := r.plainRate * nativeAssistantReengageMargin
	if cycleAccepted == 0 || probeRate >= engageBar {
		r.probeLeft = 0
	} else if r.probeLeft > 0 {
		return false
	}
	engaged := cycleAccepted > 0 && probeRate >= engageBar
	if mtpDiagForTest {
		nativeTraceLog(core.Sprintf("mtp-diag reengage probe: rate=%.1f plain=%.1f cooldown=%d engaged=%v\n",
			probeRate, r.plainRate, r.cooldown, engaged))
	}
	if engaged {
		r.cooldown = nativeAssistantReengageCooldownMin
		r.winN = 0
		return false
	}
	r.bailAgain()
	return true
}

// engagedCycle feeds one engaged (non-probe) cycle into the rolling window and
// reports whether the filled window's rate has dropped below the measured
// plain rate — the stretch exit. Inert until a plain stretch has run (the
// acceptance streak bootstraps the first bail). On exit the cooldown doubles:
// failing to sustain is the same evidence as a failed probe.
func (r *mtpReengage) engagedCycle(cycleTok int, cycleSec float64, tokensNow int) (bail bool) {
	if r.plainRate <= 0 {
		return false
	}
	r.winTok[r.winN%len(r.winTok)] = cycleTok
	r.winSec[r.winN%len(r.winSec)] = cycleSec
	r.winN++
	if r.winN < len(r.winTok) {
		return false
	}
	tok, sec := 0, 0.0
	for i := range r.winTok {
		tok += r.winTok[i]
		sec += r.winSec[i]
	}
	if sec <= 0 || float64(tok)/sec >= r.plainRate {
		return false
	}
	if mtpDiagForTest {
		nativeTraceLog(core.Sprintf("mtp-diag reengage rate-exit: emitted=%d\n", tokensNow))
	}
	r.bailAgain()
	return true
}
