// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"bytes"
	"io"
	"os"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
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
	if bail := r.probeCycle(0, 1); bail {
		t.Fatal("a fully-rejected WARMUP cycle must not decide anything")
	}
	if bail := r.probeCycle(0, 1); !bail {
		t.Fatal("a fully-rejected measured cycle must bail")
	}
	if r.probing() {
		t.Fatal("probe window must be closed after the abort")
	}
	if r.cooldown != 2*nativeAssistantReengageCooldownMin {
		t.Fatalf("failed probe cooldown = %d, want doubled %d", r.cooldown, 2*nativeAssistantReengageCooldownMin)
	}
}

func TestMTPReengageProbeCycle_ClosedWindow(t *testing.T) {
	r := &mtpReengage{probeLeft: 0, cooldown: 77}
	if r.probeCycle(0, 10) {
		t.Fatal("closed probe window requested a bail")
	}
	if r.cooldown != 77 || r.probeLeft != 0 {
		t.Fatalf("closed probe mutated state: cooldown=%d probeLeft=%d", r.cooldown, r.probeLeft)
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
	// the warmup reset the probe clock; backdate it so cycle 2's window is
	// never zero-width (two calls can land in one clock tick), then 300
	// tokens over ~1ms reads far above the bar: engage now.
	r.probeT0 = time.Now().Add(-time.Millisecond)
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
	// 100 tok/s plain, margin 1.08 → bar 108. Cycle 1 is warmup (resets the
	// probe clock); backdating the clock AFTER it makes the measured cycles
	// read one token per ~100ms — ≤ ~30 tok/s however fast the test runs:
	// below the bar but not fully rejected, so the probe rides the remaining
	// cycles and fails on the last with a doubled cooldown.
	r := reengageAfterPlain(100, 0)
	if bail := r.probeCycle(1, 1); bail {
		t.Fatal("warmup cycle bailed unexpectedly")
	}
	r.probeT0 = time.Now().Add(-100 * time.Millisecond)
	for i := range nativeAssistantReengageProbeBlocks - 2 {
		if bail := r.probeCycle(1, 2+i); bail {
			t.Fatalf("mediocre cycle %d bailed before the window closed", 2+i)
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

func TestMTPReengageEngagedCycle_ZeroDurationWindow(t *testing.T) {
	r := &mtpReengage{plainRate: 100}
	for i := range len(r.winTok) {
		if r.engagedCycle(1, 0, i+1) {
			t.Fatal("zero-duration window must not trigger a rate exit")
		}
	}
	if r.winN != len(r.winTok) {
		t.Fatalf("engagedCycle recorded %d rows, want %d", r.winN, len(r.winTok))
	}
}

func TestMTPReengageNeedsDeepBootstrap(t *testing.T) {
	r := &mtpReengage{}
	if r.needsDeepBootstrap(4096, 10, 512) {
		t.Fatal("shallow context must not bootstrap")
	}
	if !r.needsDeepBootstrap(nativeAssistantDeepBootstrapPos, 10, 512) {
		t.Fatal("deep context with no plain rate must bootstrap")
	}
	if r.needsDeepBootstrap(1<<20, 500, 512) {
		t.Fatal("must not bootstrap without budget to pay it back")
	}
	r.plainRate = 100
	if r.needsDeepBootstrap(1<<20, 10, 512) {
		t.Fatal("a measured plain rate must suppress the bootstrap")
	}
}

// --- Long multi-block integration: the #299 policy driven through the real
// Generate*FromSessionEach loops (assistant_load.go), not the bare struct
// above. The tests above pin mtpReengage's OWN decision logic with a fake
// clock; these pin the WIRING between that logic and the speculative
// generate loop -- probeCycle/engagedCycle actually get called from the
// right branch, runPlainStretch actually threads the carried lead and
// re-arms the probe, bailAgain actually escalates across several real
// bail/plain/probe rounds in one call.
//
// The cadence is DRIVEN, not hoped for: newNativeAssistantGenerateFixture's
// assistant carries all-zero weights end to end (embed/attn/mlp/head), so
// every drafted token is the same tie-broken index (0, greedyBF16Suppressed
// keeps the lowest-index winner on an exact tie) regardless of context or
// position -- see nativeAssistantPromptWhoseTargetTokensAvoid, already used
// by the low-accept-patience tests in assistant_load_test.go for the same
// reason. Pairing that fixed drafter with a prompt whose target continuation
// is proven (by a real reference Generate call) to avoid token 0 for the
// WHOLE run makes every drafted block a full, unambiguous reject: no
// content luck, no clock precision, so probeCycle's cycleAccepted==0 abort
// fires deterministically every time. What is NOT reachable this way is the
// engage verdict itself (probeRate >= engageBar): measured on this hardware,
// plain decode over this tiny fixture runs at several thousand tok/s while
// even a fully-accepted, fused speculative cycle tops out in the low
// hundreds (confirmed with the fused draft path forced eligible and with a
// target model built 16x more expensive on the Q/FF side -- neither closed
// the gap, since plain decode here is dispatch/cache-bound, not
// compute-bound). Engaging for real needs a target expensive enough that
// verify's one-command-buffer-per-block amortisation beats plain's
// per-token dispatch, which is a real-model-scale economy no synthetic
// fixture at unit-test speed can produce -- see the report for the file:line
// callsites this leaves uncovered (assistant_load.go's engagedCycle call
// sites), a needs-hook remainder precisely because it needs GPU wall-clock
// economics unit tests cannot fake honestly.

// captureNativeTraceLog redirects nativeTraceLog's (device.go) os.Stderr
// destination for the duration of fn and returns everything written --
// mtpDiagForTest's trace lines are the re-engagement machine's OWN fields
// (cooldown, measured plainRate, the engaged verdict) printed verbatim by
// mtp_reengage.go itself, so matching against them is reading the machine's
// state, not a self-computed pin. Process-wide os.Stderr swap: the caller
// must not run this concurrently with another such capture (none of this
// package's tests call t.Parallel()).
func captureNativeTraceLog(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	saved := os.Stderr
	os.Stderr = w
	captured := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		captured <- buf.String()
	}()
	fn()
	os.Stderr = saved
	if cerr := w.Close(); cerr != nil {
		t.Fatalf("close capture pipe writer: %v", cerr)
	}
	out := <-captured
	_ = r.Close()
	return out
}

// assertReengageCooldownEscalationTrace asserts the common behavioural
// signature shared by the greedy and sampled long-run tests below: a fresh
// patience bail at the exact streak constant, then the cooldown ladder
// doubling through at least two escalations, and never an engage verdict
// (this fixture/prompt pair cannot legitimately clear the rate bar -- see
// the package doc comment above).
func assertReengageCooldownEscalationTrace(t *testing.T, trace string) {
	t.Helper()
	if want := core.Sprintf("reengage bail: streak=%d", nativeAssistantLowAcceptPatience); !core.Contains(trace, want) {
		t.Fatalf("trace missing the patience bail %q:\n%s", want, trace)
	}
	if want := core.Sprintf("cooldown=%d", nativeAssistantReengageCooldownMin); !core.Contains(trace, want) {
		t.Fatalf("trace missing the initial cooldown %q:\n%s", want, trace)
	}
	if want := core.Sprintf("cooldown=%d", nativeAssistantReengageCooldownMin*2); !core.Contains(trace, want) {
		t.Fatalf("trace missing the first doubled cooldown %q:\n%s", want, trace)
	}
	if want := core.Sprintf("cooldown=%d", nativeAssistantReengageCooldownMin*4); !core.Contains(trace, want) {
		t.Fatalf("trace missing the second doubled cooldown %q:\n%s", want, trace)
	}
	if core.Contains(trace, "engaged=true") {
		t.Fatalf("trace reports an engage verdict this scenario cannot legitimately earn:\n%s", trace)
	}
}

// TestAssistantPairGenerateFromSessionEachReengageCooldownEscalatesOnRepeatedProbeFailure
// is the greedy lane's long multi-block drive: bailFresh -> the bounded
// plain cooldown stretch -> the re-armed probe -> repeated probe failure ->
// bailAgain doubling the cooldown, across several real rounds in one call.
func TestAssistantPairGenerateFromSessionEachReengageCooldownEscalatesOnRepeatedProbeFailure(t *testing.T) {
	requireNativeRuntime(t)
	const maxNew = 300
	const draftTokens = 2
	pair, mk := newNativeAssistantGenerateFixtureMaxLen(t, maxNew+20)
	defer pair.Close()
	prompt := nativeAssistantPromptWhoseTargetTokensAvoid(t, mk, 0, maxNew)
	target := mk()

	prevDiag := mtpDiagForTest
	mtpDiagForTest = true
	defer func() { mtpDiagForTest = prevDiag }()

	var got AssistantGenerateResult
	var err error
	trace := captureNativeTraceLog(t, func() {
		got, err = pair.GenerateFromSessionEach(target, prompt, maxNew, -1, draftTokens, nil, nil)
	})
	if err != nil {
		t.Fatalf("GenerateFromSessionEach: %v", err)
	}

	// Behavioural invariants on the machine's own result counters: the
	// prompt's guarantee makes every drafted block a full reject, so nothing
	// is ever accepted and the whole stream is target-sourced.
	if len(got.Tokens) != maxNew {
		t.Fatalf("generated %d tokens, want %d", len(got.Tokens), maxNew)
	}
	if got.AcceptedTokens != 0 {
		t.Fatalf("accepted tokens = %d, want 0 (target continuation avoids the drafter's only proposal)", got.AcceptedTokens)
	}
	if got.TargetTokens != maxNew {
		t.Fatalf("target tokens = %d, want %d (every committed token is target-sourced)", got.TargetTokens, maxNew)
	}
	if got.DraftCalls <= nativeAssistantLowAcceptPatience {
		t.Fatalf("draft calls = %d, want > %d patience blocks -- probing must have resumed after the plain stretch", got.DraftCalls, nativeAssistantLowAcceptPatience)
	}
	assertReengageCooldownEscalationTrace(t, trace)
}

// TestAssistantPairGenerateSampledFromSessionEachReengageCooldownEscalatesOnRepeatedProbeFailure
// is the sampled lane's twin. model.SampleParams{} (Temperature<=0, MinP<=0)
// routes model.Sampler through its greedy branch (model/sample.go
// sampleMapped), so the sampled verify matches the SAME target continuation
// the greedy test above proved avoids the drafter's only proposal -- reusing
// that guarantee rather than re-deriving one for the sampled lane.
func TestAssistantPairGenerateSampledFromSessionEachReengageCooldownEscalatesOnRepeatedProbeFailure(t *testing.T) {
	requireNativeRuntime(t)
	const maxNew = 300
	const draftTokens = 2
	pair, mk := newNativeAssistantGenerateFixtureMaxLen(t, maxNew+20)
	defer pair.Close()
	prompt := nativeAssistantPromptWhoseTargetTokensAvoid(t, mk, 0, maxNew)
	target := mk()

	prevDiag := mtpDiagForTest
	mtpDiagForTest = true
	defer func() { mtpDiagForTest = prevDiag }()

	params := model.SampleParams{}
	sampler := model.NewSampler(1)

	var got AssistantGenerateResult
	var err error
	trace := captureNativeTraceLog(t, func() {
		got, err = pair.GenerateSampledFromSessionEach(target, prompt, maxNew, nil, sampler, params, draftTokens, nil)
	})
	if err != nil {
		t.Fatalf("GenerateSampledFromSessionEach: %v", err)
	}

	if len(got.Tokens) != maxNew {
		t.Fatalf("generated %d tokens, want %d", len(got.Tokens), maxNew)
	}
	if got.AcceptedTokens != 0 {
		t.Fatalf("accepted tokens = %d, want 0 (target continuation avoids the drafter's only proposal)", got.AcceptedTokens)
	}
	if got.TargetTokens != maxNew {
		t.Fatalf("target tokens = %d, want %d (every committed token is target-sourced)", got.TargetTokens, maxNew)
	}
	if got.DraftCalls <= nativeAssistantLowAcceptPatience {
		t.Fatalf("draft calls = %d, want > %d patience blocks -- probing must have resumed after the plain stretch", got.DraftCalls, nativeAssistantLowAcceptPatience)
	}
	assertReengageCooldownEscalationTrace(t, trace)
}

// TestAssistantPairGenerateFromSessionEachCoversFirstAllAcceptedCycleBeforeAnyPlainRate
// covers the loop's OTHER major branch the cooldown-escalation tests above
// cannot reach: `if verify.AllAccepted`. Existing single-block accept tests
// (assistant_load_test.go's yield-stop-after-one-token cases) return before
// this check ever runs -- a stopped yield breaks the outer loop first -- so
// it stayed dark under the whole suite. nativeAssistantPromptWithAcceptedFirstDraft
// finds a prompt whose target continuation starts with the drafter's actual
// first proposal; with draftTokens=1 that single token IS the whole block,
// so it is unconditionally AllAccepted. reng is freshly zero-valued at the
// start of a call (no plain stretch has run), so wasProbing is false and
// engagedCycle's own plainRate==0 guard makes it inert (bail=false) --
// exercising dl.cycle(true, ...), the needsDeepBootstrap call (false at this
// shallow depth), and the "not probing" dispatch arm into engagedCycle.
func TestAssistantPairGenerateFromSessionEachCoversFirstAllAcceptedCycleBeforeAnyPlainRate(t *testing.T) {
	requireNativeRuntime(t)
	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	prompt := nativeAssistantPromptWithAcceptedFirstDraft(t, pair, mk)
	target := mk()

	const maxNew, draftTokens = 4, 1
	want, err := mk().Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("reference Generate: %v", err)
	}
	got, err := pair.GenerateFromSessionEach(target, prompt, maxNew, -1, draftTokens, nil, nil)
	if err != nil {
		t.Fatalf("GenerateFromSessionEach: %v", err)
	}
	if !idsEqual(got.Tokens, want) {
		t.Fatalf("tokens = %v, want target-identical %v", got.Tokens, want)
	}
	if got.AcceptedTokens == 0 {
		t.Fatal("accepted tokens = 0, want the guaranteed first-block accept (verify.AllAccepted never exercised)")
	}
}

// TestAssistantPairGenerateSampledFromSessionEachCoversFirstAllAcceptedCycleBeforeAnyPlainRate
// is the sampled lane's twin, reusing model.SampleParams{}'s zero-temperature
// greedy branch (see the cooldown-escalation tests' doc comment) so the same
// guaranteed-accept prompt applies unchanged.
func TestAssistantPairGenerateSampledFromSessionEachCoversFirstAllAcceptedCycleBeforeAnyPlainRate(t *testing.T) {
	requireNativeRuntime(t)
	pair, mk := newNativeAssistantGenerateFixture(t)
	defer pair.Close()
	prompt := nativeAssistantPromptWithAcceptedFirstDraft(t, pair, mk)
	target := mk()

	params := model.SampleParams{}
	const maxNew, draftTokens = 4, 1
	want, err := mk().GenerateSampledEach(prompt, maxNew, nil, model.NewSampler(1), params, nil, nil)
	if err != nil {
		t.Fatalf("reference GenerateSampledEach: %v", err)
	}
	got, err := pair.GenerateSampledFromSessionEach(target, prompt, maxNew, nil, model.NewSampler(1), params, draftTokens, nil)
	if err != nil {
		t.Fatalf("GenerateSampledFromSessionEach: %v", err)
	}
	if !idsEqual(got.Tokens, want) {
		t.Fatalf("tokens = %v, want target-identical %v", got.Tokens, want)
	}
	if got.AcceptedTokens == 0 {
		t.Fatal("accepted tokens = 0, want the guaranteed first-block accept (verify.AllAccepted never exercised)")
	}
}
