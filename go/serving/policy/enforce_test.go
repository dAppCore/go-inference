// SPDX-Licence-Identifier: EUPL-1.2

package policy

import (
	"context"
	"math/rand"
	"testing"

	core "dappco.re/go"
)

// runChunksMediated drives a mediating Enforcer (grade G2) over chunks and
// returns the concatenated output, the events, and whether a refuse stopped it.
func runChunksMediated(pol *Policy, m Mediator, chunks []string) (string, []Event, bool) {
	enf := pol.NewMediatingEnforcer(context.Background(), m)
	var b core.Builder
	var events []Event
	stopped := false
	for _, c := range chunks {
		out, ev, stop := enf.Feed(c)
		b.WriteString(out)
		events = append(events, ev...)
		if stop {
			stopped = true
			break
		}
	}
	if !stopped {
		out, ev, stop := enf.Close()
		b.WriteString(out)
		events = append(events, ev...)
		stopped = stop
	}
	return b.String(), events, stopped
}

// markSpan is a deterministic mediator: it wraps the span in guillemets. Because
// its output depends on the exact span bytes, a differential run over random
// chunkings proves the enforcer always mediates the COMPLETE span.
func markSpan(_ context.Context, _ int, span string) (string, error) {
	return "«" + span + "»", nil
}

func mustCompile(t testing.TB, jsonSrc string) *Policy {
	t.Helper()
	pol, err := Compile([]byte(jsonSrc))
	if err != nil {
		t.Fatalf("Compile: %v", err)
	}
	return pol
}

// runChunks drives an Enforcer over chunks and returns the concatenated output,
// the enforcement events, and whether a refuse stopped the stream.
func runChunks(pol *Policy, chunks []string) (string, []Event, bool) {
	enf := pol.NewEnforcer()
	var b core.Builder
	var events []Event
	stopped := false
	for _, c := range chunks {
		out, ev, stop := enf.Feed(c)
		b.WriteString(out)
		events = append(events, ev...)
		if stop {
			stopped = true
			break
		}
	}
	if !stopped {
		out, ev, stop := enf.Close()
		b.WriteString(out)
		events = append(events, ev...)
		stopped = stop
	}
	return b.String(), events, stopped
}

// randomChunks splits s into pieces of 1..4 bytes using r — the differential
// fuzz's boundary generator.
func randomChunks(r *rand.Rand, s string) []string {
	var chunks []string
	for i := 0; i < len(s); {
		step := 1 + r.Intn(4)
		if i+step > len(s) {
			step = len(s) - i
		}
		chunks = append(chunks, s[i:i+step])
		i += step
	}
	return chunks
}

// TestPolicy_Enforcer_Redact_Good pins a term redaction: the matched span is
// replaced, everything else is untouched.
func TestPolicy_Enforcer_Redact_Good(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"PROJECT-X","action":"redact","replacement":"[redacted]"}]}`)
	got, _, _ := runChunks(pol, []string{"the PROJECT-X launch is soon"})
	if got != "the [redacted] launch is soon" {
		t.Fatalf("redacted = %q", got)
	}
}

// TestPolicy_Enforcer_Term_CaseInsensitive pins ASCII case folding: any casing
// of the term matches.
func TestPolicy_Enforcer_Term_CaseInsensitive(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"client","action":"redact"}]}`)
	for _, in := range []string{"our Client here", "our CLIENT here", "our cLiEnT here"} {
		got, _, _ := runChunks(pol, []string{in})
		if got != "our [redacted] here" {
			t.Fatalf("input %q redacted to %q", in, got)
		}
	}
}

// TestPolicy_Enforcer_Refuse_Good pins refusal: the reply ends at the match, the
// message is the last visible text, and text after the match is never emitted.
func TestPolicy_Enforcer_Refuse_Good(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"SECRET","action":"refuse","message":"This deployment cannot discuss that."}]}`)
	got, events, stopped := runChunks(pol, []string{"the answer is SECRET plus more text"})
	if got != "the answer is This deployment cannot discuss that." {
		t.Fatalf("refused reply = %q", got)
	}
	if !stopped {
		t.Fatal("refuse must stop the stream")
	}
	if len(events) != 1 || events[0].Action != ActionRefuse || events[0].RuleIndex != 0 {
		t.Fatalf("events = %+v, want one refuse on rule #0", events)
	}
}

// TestPolicy_Enforcer_Pattern_Good pins pattern rules for both actions.
func TestPolicy_Enforcer_Pattern_Good(t *testing.T) {
	t.Run("redact", func(t *testing.T) {
		pol := mustCompile(t, `{"rules":[{"match":"pattern","value":"v[0-9]+\\.[0-9]+\\.[0-9]+-rc[0-9]+","action":"redact","replacement":"[unreleased]"}]}`)
		got, _, _ := runChunks(pol, []string{"ship v2.3.1-rc4 tomorrow"})
		if got != "ship [unreleased] tomorrow" {
			t.Fatalf("pattern redact = %q", got)
		}
	})
	t.Run("refuse", func(t *testing.T) {
		pol := mustCompile(t, `{"rules":[{"match":"pattern","value":"v[0-9]+\\.[0-9]+\\.[0-9]+-rc[0-9]+","action":"refuse","message":"No unreleased versions."}]}`)
		got, _, stopped := runChunks(pol, []string{"ship v2.3.1-rc4 tomorrow"})
		if got != "ship No unreleased versions." || !stopped {
			t.Fatalf("pattern refuse = %q stopped=%v", got, stopped)
		}
	})
}

// TestPolicy_Enforcer_BoundarySpanning pins the hard case: a match split across
// token boundaries — fed one byte at a time — is still caught and redacted.
func TestPolicy_Enforcer_BoundarySpanning(t *testing.T) {
	pol := mustCompile(t, `{"rules":[
		{"match":"term","value":"PROJECT-X","action":"redact"},
		{"match":"pattern","value":"rc[0-9]+","action":"redact","replacement":"[rc]"}
	]}`)
	text := "start PROJECT-X mid rc42 end"
	var oneByte []string
	for i := 0; i < len(text); i++ {
		oneByte = append(oneByte, text[i:i+1])
	}
	got, _, _ := runChunks(pol, oneByte)
	if got != "start [redacted] mid [rc] end" {
		t.Fatalf("byte-split enforcement = %q", got)
	}
}

// TestPolicy_Enforcer_NonMatchingPassthrough pins byte-exactness on a clean
// stream: text that cannot be part of any match streams through unchanged, and a
// chunk clear of any term-prefix is emitted whole in one Feed (nothing withheld).
func TestPolicy_Enforcer_NonMatchingPassthrough(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"SECRET","action":"redact"}]}`)
	enf := pol.NewEnforcer()
	chunk := "nothing to see in this line of text"
	out, events, stop := enf.Feed(chunk)
	if out != chunk {
		t.Fatalf("clean chunk emitted %q, want the whole chunk verbatim", out)
	}
	if events != nil || stop {
		t.Fatalf("clean chunk raised events=%v stop=%v", events, stop)
	}
}

// TestPolicy_Enforcer_Precedence pins the tie-break: the longest match wins
// (maximal munch), so an enclosing term beats the shorter one it contains.
func TestPolicy_Enforcer_Precedence(t *testing.T) {
	pol := mustCompile(t, `{"rules":[
		{"match":"term","value":"cat","action":"redact","replacement":"[short]"},
		{"match":"term","value":"category","action":"redact","replacement":"[long]"}
	]}`)
	got, _, _ := runChunks(pol, []string{"the category of cat"})
	if got != "the [long] of [short]" {
		t.Fatalf("precedence result = %q", got)
	}
}

// TestPolicy_Enforcer_Differential is the byte-identity proof: for a fixed text
// and policy, the whole-string result must equal the concatenation of the
// streamed result under 1000 independent random chunkings.
func TestPolicy_Enforcer_Differential(t *testing.T) {
	texts := map[string]string{
		"matching":     "intro PROJECT-X then client work on v1.2.3-rc9 and more client talk PROJECT-X end",
		"non-matching": "a wholly innocent paragraph with no sensitive terms whatsoever, just prose and prose",
		"adjacent":     "PROJECT-XPROJECT-Xclientclient v0.0.1-rc1v0.0.2-rc2",
		"prefix-tease": "this is a proj and a projec and a project and PROJECT-X finally",
	}
	pol := mustCompile(t, `{"rules":[
		{"match":"term","value":"PROJECT-X","action":"redact","replacement":"[P]"},
		{"match":"term","value":"client","action":"redact","replacement":"[C]"},
		{"match":"pattern","value":"v[0-9]+\\.[0-9]+\\.[0-9]+-rc[0-9]+","action":"redact","replacement":"[V]","window":20}
	]}`)

	for name, text := range texts {
		t.Run(name, func(t *testing.T) {
			want, _, _ := runChunks(pol, []string{text}) // whole-string reference
			r := rand.New(rand.NewSource(1))
			for iter := 0; iter < 1000; iter++ {
				got, _, _ := runChunks(pol, randomChunks(r, text))
				if got != want {
					t.Fatalf("chunking %d diverged\n  want %q\n  got  %q", iter, want, got)
				}
			}
		})
	}
}

// TestPolicy_Enforcer_Differential_Refuse proves byte-identity for the refuse
// action too — the stream must stop at the same point regardless of chunking.
func TestPolicy_Enforcer_Differential_Refuse(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"pattern","value":"v[0-9]+\\.[0-9]+\\.[0-9]+-rc[0-9]+","action":"refuse","message":"[stop]"}]}`)
	text := "we are close but v1.0.0-rc1 is unreleased so nothing after this appears"
	want, _, wantStop := runChunks(pol, []string{text})
	r := rand.New(rand.NewSource(7))
	for iter := 0; iter < 1000; iter++ {
		got, _, stop := runChunks(pol, randomChunks(r, text))
		if got != want || stop != wantStop {
			t.Fatalf("chunking %d diverged: got %q stop=%v, want %q stop=%v", iter, got, stop, want, wantStop)
		}
	}
}

// TestPolicy_Enforcer_ClosedSwallow pins that once a refuse fires, every further
// Feed is inert.
func TestPolicy_Enforcer_ClosedSwallow(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"halt","action":"refuse","message":"[end]"}]}`)
	enf := pol.NewEnforcer()
	_, _, stop := enf.Feed("please halt now")
	if !stop {
		t.Fatal("refuse should stop")
	}
	out, ev, stop := enf.Feed("more text")
	if out != "" || ev != nil || !stop {
		t.Fatalf("post-refuse Feed leaked out=%q ev=%v stop=%v", out, ev, stop)
	}
}

// TestPolicy_Enforcer_Rewrite_Good pins the grade-G2 happy path: a mediator that
// genuinely sanitises the span (its output carries no policy term) has that
// output emitted in place, everything else is untouched, and the Event is clean
// (not degraded) because re-enforcement finds nothing to redact.
func TestPolicy_Enforcer_Rewrite_Good(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"PROJECT-X","action":"rewrite"}]}`)
	reword := func(_ context.Context, _ int, _ string) (string, error) { return "our flagship", nil }
	got, events, _ := runChunksMediated(pol, reword, []string{"the PROJECT-X launch is soon"})
	if got != "the our flagship launch is soon" {
		t.Fatalf("mediated = %q", got)
	}
	if len(events) != 1 || events[0].Action != ActionRewrite || events[0].RuleIndex != 0 || events[0].Degraded {
		t.Fatalf("events = %+v, want one clean rewrite on rule #0", events)
	}
}

// TestPolicy_Enforcer_Rewrite_BoundarySpanning is the buffering-boundary proof:
// a span fed one byte at a time is mediated with the COMPLETE span, never a
// partial one — the mediator sees "PROJECT-X" whole.
func TestPolicy_Enforcer_Rewrite_BoundarySpanning(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"PROJECT-X","action":"rewrite"}]}`)
	var seen []string
	capture := func(_ context.Context, _ int, span string) (string, error) {
		seen = append(seen, span)
		return "[reworded]", nil // sanitised: no policy term, so re-enforcement is a no-op
	}
	text := "start PROJECT-X end"
	var oneByte []string
	for i := 0; i < len(text); i++ {
		oneByte = append(oneByte, text[i:i+1])
	}
	got, _, _ := runChunksMediated(pol, capture, oneByte)
	if got != "start [reworded] end" {
		t.Fatalf("byte-split mediation = %q", got)
	}
	if len(seen) != 1 || seen[0] != "PROJECT-X" {
		t.Fatalf("mediator saw %q, want exactly the whole span once", seen)
	}
}

// TestPolicy_Enforcer_Rewrite_MediatorError pins the fail-safe: a mediator error
// degrades to redact (the rule's replacement) — the original span never leaks.
func TestPolicy_Enforcer_Rewrite_MediatorError(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"SECRET","action":"rewrite","replacement":"[gone]"}]}`)
	boom := func(context.Context, int, string) (string, error) {
		return "SECRET-should-not-appear", core.E("test", "mediator down", nil)
	}
	got, _, _ := runChunksMediated(pol, boom, []string{"the SECRET value"})
	if got != "the [gone] value" {
		t.Fatalf("degraded = %q, want the redact fallback, never the original span", got)
	}
	if core.Contains(got, "SECRET") {
		t.Fatalf("degrade leaked the original span: %q", got)
	}
}

// TestPolicy_Enforcer_Rewrite_MediatorEmpty pins that an empty mediator result
// is treated as failure — it degrades to redact rather than deleting the span
// silently.
func TestPolicy_Enforcer_Rewrite_MediatorEmpty(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"SECRET","action":"rewrite"}]}`)
	empty := func(context.Context, int, string) (string, error) { return "", nil }
	got, _, _ := runChunksMediated(pol, empty, []string{"the SECRET value"})
	if got != "the [redacted] value" {
		t.Fatalf("empty-result = %q, want the redact fallback", got)
	}
}

// TestPolicy_Enforcer_Rewrite_MediatorTimeout pins that a mediator which blocks
// past the deadline (ignoring cancellation) still degrades to redact and lets the
// stream advance — the timeout is a guarantee of the layer, not the hook.
func TestPolicy_Enforcer_Rewrite_MediatorTimeout(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"SECRET","action":"rewrite"}],"mediate_timeout_ms":30}`)
	block := make(chan struct{})
	defer close(block) // release the abandoned mediator goroutine at test end
	stuck := func(context.Context, int, string) (string, error) {
		<-block // ignores ctx cancellation on purpose
		return "late", nil
	}
	got, _, _ := runChunksMediated(pol, stuck, []string{"the SECRET value"})
	if got != "the [redacted] value" {
		t.Fatalf("timeout = %q, want the redact fallback after the deadline", got)
	}
}

// TestPolicy_Enforcer_Rewrite_NoMediator pins the defensive path: a rewrite rule
// on a plain (non-mediating) Enforcer degrades to redact rather than emitting the
// span. The serving layer boots fatal before this can happen in production.
func TestPolicy_Enforcer_Rewrite_NoMediator(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"SECRET","action":"rewrite","replacement":"[x]"}]}`)
	got, _, _ := runChunks(pol, []string{"the SECRET value"})
	if got != "the [x] value" {
		t.Fatalf("no-mediator = %q, want the redact fallback", got)
	}
}

// TestPolicy_Enforcer_Rewrite_MidStream pins that mediation is per-span: when the
// mediator fails on one hit but sanitises a later one, each is handled
// independently — the first degrades to redact, the second emits clean — and the
// stream survives. The per-hit Degraded flags mirror that split.
func TestPolicy_Enforcer_Rewrite_MidStream(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"TAG","action":"rewrite","replacement":"[df]"}]}`)
	calls := 0
	flaky := func(_ context.Context, _ int, _ string) (string, error) {
		calls++
		if calls == 1 {
			return "", core.E("test", "first fails", nil)
		}
		return "reworded", nil
	}
	got, events, _ := runChunksMediated(pol, flaky, []string{"a TAG then another TAG here"})
	if got != "a [df] then another reworded here" {
		t.Fatalf("mid-stream = %q", got)
	}
	if len(events) != 2 || !events[0].Degraded || events[1].Degraded {
		t.Fatalf("events = %+v, want a degraded first hit and a clean second", events)
	}
}

// TestPolicy_Enforcer_Rewrite_Differential is the byte-identity proof for grade
// G2: with a content-dependent mediator, the whole-string result must equal the
// streamed result under 1000 random chunkings — so the enforcer always mediates
// the complete span regardless of where the token boundaries fall.
func TestPolicy_Enforcer_Rewrite_Differential(t *testing.T) {
	pol := mustCompile(t, `{"rules":[
		{"match":"term","value":"PROJECT-X","action":"rewrite"},
		{"match":"pattern","value":"rc[0-9]+","action":"rewrite","window":12}
	]}`)
	text := "intro PROJECT-X then rc42 and more PROJECT-X talk near rc9 end"
	want, _, _ := runChunksMediated(pol, markSpan, []string{text})
	r := rand.New(rand.NewSource(3))
	for iter := 0; iter < 1000; iter++ {
		got, _, _ := runChunksMediated(pol, markSpan, randomChunks(r, text))
		if got != want {
			t.Fatalf("chunking %d diverged\n  want %q\n  got  %q", iter, want, got)
		}
	}
}

// TestPolicy_Enforcer_Rewrite_ReEnforced_ResidualRedact pins the untrusted-
// mediator guard: mediator output that still contains a policy term (here one
// that matches a redact rule) has that residual hit redacted before emission, the
// original term never leaks, and the rewrite Event is marked degraded.
func TestPolicy_Enforcer_Rewrite_ReEnforced_ResidualRedact(t *testing.T) {
	pol := mustCompile(t, `{"rules":[
		{"match":"term","value":"PROJECT-X","action":"rewrite"},
		{"match":"term","value":"SECRET","action":"redact","replacement":"[S]"}
	]}`)
	inject := func(_ context.Context, _ int, _ string) (string, error) { return "our SECRET plan", nil }
	got, events, _ := runChunksMediated(pol, inject, []string{"the PROJECT-X launch"})
	if got != "the our [S] plan launch" {
		t.Fatalf("residual redact = %q", got)
	}
	if core.Contains(got, "SECRET") {
		t.Fatalf("re-enforcement leaked the residual term: %q", got)
	}
	if len(events) != 1 || events[0].Action != ActionRewrite || !events[0].Degraded {
		t.Fatalf("events = %+v, want one degraded rewrite", events)
	}
}

// TestPolicy_Enforcer_Rewrite_ReEnforced_ResidualRefuse pins that a residual hit
// on a REFUSE rule inside mediator output degrades to redact (the default
// redaction, since a refuse rule carries no replacement) WITHOUT stopping the
// stream — the stream-level refuse decision belongs to the original scan over the
// model's own output, not to a re-scan of mediator text.
func TestPolicy_Enforcer_Rewrite_ReEnforced_ResidualRefuse(t *testing.T) {
	pol := mustCompile(t, `{"rules":[
		{"match":"term","value":"PROJECT-X","action":"rewrite"},
		{"match":"term","value":"SECRET","action":"refuse","message":"stream must not stop here"}
	]}`)
	inject := func(_ context.Context, _ int, _ string) (string, error) { return "our SECRET plan", nil }
	got, events, stopped := runChunksMediated(pol, inject, []string{"the PROJECT-X here and more"})
	if got != "the our [redacted] plan here and more" {
		t.Fatalf("residual refuse = %q", got)
	}
	if stopped {
		t.Fatal("a residual refuse inside mediator output must NOT stop the stream")
	}
	if core.Contains(got, "SECRET") || core.Contains(got, "stream must not stop") {
		t.Fatalf("residual refuse leaked content or its message: %q", got)
	}
	if len(events) != 1 || events[0].Action != ActionRewrite || !events[0].Degraded {
		t.Fatalf("events = %+v, want one degraded rewrite", events)
	}
}

// TestPolicy_Enforcer_Rewrite_ReEnforced_NoRecursion pins that re-enforcement is
// a SINGLE non-recursive pass: a mediator that echoes the banned term back has
// that residual hit redacted, and the mediator is called exactly once — the
// re-scan never re-mediates.
func TestPolicy_Enforcer_Rewrite_ReEnforced_NoRecursion(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"PROJECT-X","action":"rewrite"}]}`)
	calls := 0
	echo := func(_ context.Context, _ int, span string) (string, error) {
		calls++
		return "our " + span + " team", nil
	}
	got, events, _ := runChunksMediated(pol, echo, []string{"ship PROJECT-X today"})
	if got != "ship our [redacted] team today" {
		t.Fatalf("no-recursion = %q", got)
	}
	if calls != 1 {
		t.Fatalf("mediator called %d times, want exactly one (re-enforcement must not re-mediate)", calls)
	}
	if len(events) != 1 || !events[0].Degraded {
		t.Fatalf("events = %+v, want one degraded rewrite", events)
	}
}

// TestPolicy_Enforcer_Rewrite_ReEnforced_Differential is the byte-identity proof
// for the re-enforcement pass: with a mediator that injects residual hits (a
// redact term and a refuse-rule token) around each span, the whole-string result
// must equal the streamed result under 1000 random chunkings — so the enforcer
// always mediates AND re-enforces the complete span, and the residual refuse
// never stops the stream regardless of where token boundaries fall.
func TestPolicy_Enforcer_Rewrite_ReEnforced_Differential(t *testing.T) {
	pol := mustCompile(t, `{"rules":[
		{"match":"term","value":"PROJECT-X","action":"rewrite"},
		{"match":"term","value":"SECRET","action":"redact","replacement":"[S]"},
		{"match":"pattern","value":"rc[0-9]+","action":"refuse","message":"[stop]","window":12}
	]}`)
	inject := func(_ context.Context, _ int, span string) (string, error) {
		return "SECRET " + span + " rc7", nil
	}
	text := "intro PROJECT-X then more PROJECT-X talk end"
	want, _, wantStop := runChunksMediated(pol, inject, []string{text})
	if wantStop {
		t.Fatal("a residual refuse must not stop the whole-string reference stream")
	}
	r := rand.New(rand.NewSource(5))
	for iter := 0; iter < 1000; iter++ {
		got, _, stop := runChunksMediated(pol, inject, randomChunks(r, text))
		if got != want || stop != wantStop {
			t.Fatalf("chunking %d diverged\n  want %q stop=%v\n  got  %q stop=%v", iter, want, wantStop, got, stop)
		}
	}
}

// TestPolicy_Event_Degraded pins the content-free Degraded flag across every path
// that sets it: a clean rewrite is NOT degraded; a mediator error, an empty
// result, a residual hit in the mediator output, and a missing mediator all mark
// the rewrite Event degraded.
func TestPolicy_Event_Degraded(t *testing.T) {
	const rewritePol = `{"rules":[{"match":"term","value":"X","action":"rewrite","replacement":"[r]"}]}`
	const residualPol = `{"rules":[{"match":"term","value":"X","action":"rewrite"},{"match":"term","value":"BAD","action":"redact"}]}`
	cases := []struct {
		name     string
		policy   string
		mediator Mediator
		plain    bool // drive a non-mediating Enforcer (no mediator wired)
		want     bool
	}{
		{"clean", rewritePol, func(context.Context, int, string) (string, error) { return "safe", nil }, false, false},
		{"mediator_error", rewritePol, func(context.Context, int, string) (string, error) { return "X", core.E("t", "boom", nil) }, false, true},
		{"empty_result", rewritePol, func(context.Context, int, string) (string, error) { return "", nil }, false, true},
		{"residual_hit", residualPol, func(context.Context, int, string) (string, error) { return "BAD", nil }, false, true},
		{"no_mediator", rewritePol, nil, true, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			pol := mustCompile(t, tc.policy)
			var events []Event
			if tc.plain {
				_, events, _ = runChunks(pol, []string{"a X b"})
			} else {
				_, events, _ = runChunksMediated(pol, tc.mediator, []string{"a X b"})
			}
			if len(events) != 1 || events[0].Action != ActionRewrite || events[0].Degraded != tc.want {
				t.Fatalf("events = %+v, want one rewrite Degraded=%v", events, tc.want)
			}
		})
	}
}

// TestPolicy_Rule_residualReplacement pins the per-action redact text used when a
// residual hit is degraded during re-enforcement: redact and rewrite rules use
// their own (configured or default) replacement; a refuse rule — which carries no
// replacement — falls back to the default redaction rather than deleting silently.
func TestPolicy_Rule_residualReplacement(t *testing.T) {
	pol := mustCompile(t, `{"rules":[
		{"match":"term","value":"A","action":"redact","replacement":"[a]"},
		{"match":"term","value":"B","action":"rewrite","replacement":"[b]"},
		{"match":"term","value":"C","action":"refuse","message":"no"}
	]}`)
	if got := pol.rules[0].residualReplacement(); got != "[a]" {
		t.Fatalf("redact residual = %q, want [a]", got)
	}
	if got := pol.rules[1].residualReplacement(); got != "[b]" {
		t.Fatalf("rewrite residual = %q, want [b]", got)
	}
	if got := pol.rules[2].residualReplacement(); got != DefaultReplacement {
		t.Fatalf("refuse residual = %q, want the default redaction", got)
	}
}

// BenchmarkPolicy_Enforcer_NoMatch measures the per-chunk overhead of the clean
// hot path on a term-only policy — the common deployment shape. The tail stays
// empty and each chunk streams through as a substring: the target is ~0 allocs
// steady-state.
func BenchmarkPolicy_Enforcer_NoMatch(b *testing.B) {
	pol := mustCompile(b, `{"rules":[
		{"match":"term","value":"PROJECT-X","action":"redact"},
		{"match":"term","value":"client","action":"redact"},
		{"match":"term","value":"confidential","action":"refuse","message":"no"}
	]}`)
	chunk := "the quick brown fox jumps over the lazy dog and keeps on running "
	enf := pol.NewEnforcer()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, _, _ := enf.Feed(chunk)
		if len(out) != len(chunk) {
			b.Fatalf("clean chunk should pass through whole, got %d/%d bytes", len(out), len(chunk))
		}
	}
}

// BenchmarkPolicy_Enforcer_NoMatch_Pattern measures the same clean stream on a
// policy that includes a pattern rule: a pattern's window is withheld every
// chunk, so this path buffers (join + one output string) — the honest cost of
// pattern support versus the term-only path above.
func BenchmarkPolicy_Enforcer_NoMatch_Pattern(b *testing.B) {
	pol := mustCompile(b, `{"rules":[
		{"match":"term","value":"client","action":"redact"},
		{"match":"pattern","value":"v[0-9]+\\.[0-9]+\\.[0-9]+-rc[0-9]+","action":"redact","window":40}
	]}`)
	chunk := "the quick brown fox jumps over the lazy dog and keeps on running "
	enf := pol.NewEnforcer()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		enf.Feed(chunk)
	}
}

// BenchmarkPolicy_MediatingEnforcer_NoMatch proves grade G2 keeps the clean hot
// path 0-alloc: a mediating enforcer on a rewrite policy, fed a stream with no
// hit, spawns no mediator goroutine and allocates nothing per chunk — the
// mediation cost is paid only when a rewrite span actually settles.
func BenchmarkPolicy_MediatingEnforcer_NoMatch(b *testing.B) {
	pol := mustCompile(b, `{"rules":[
		{"match":"term","value":"PROJECT-X","action":"rewrite"},
		{"match":"term","value":"client","action":"redact"}
	]}`)
	chunk := "the quick brown fox jumps over the lazy dog and keeps on running "
	enf := pol.NewMediatingEnforcer(context.Background(), markSpan)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, _, _ := enf.Feed(chunk)
		if len(out) != len(chunk) {
			b.Fatalf("clean chunk should pass through whole, got %d/%d bytes", len(out), len(chunk))
		}
	}
}
