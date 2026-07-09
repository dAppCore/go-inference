// SPDX-Licence-Identifier: EUPL-1.2

package safety_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/safety"
	"dappco.re/go/inference/welfare"
)

// Package sinks — keep the compiler from optimising the benchmarked work away.
var (
	sinkDecision safety.Decision
	sinkPolicy   safety.Policy
	sinkString   string
	sinkBool     bool
)

// benchClean is a within-policy welfare read (nothing tripped, scores at the
// floor) — the Decide Pass path.
var benchClean = welfare.DetectResult{}

// benchMild is an over-policy read with elevated hostility but no slur — the
// Decide Mediate path for output, Guard for input.
var benchMild = welfare.DetectResult{
	Triggered:          true,
	AngerScore:         0.75,
	SustainedHostility: 0.6,
}

// benchSevere is an over-policy read carrying a slur — the Decide Guard
// escalation path.
var benchSevere = welfare.DetectResult{
	Triggered:          true,
	SlurMatch:          true,
	SlurTerm:           "testterm",
	AngerScore:         0.95,
	SustainedHostility: 0.9,
}

// benchReply is a realistic served completion for the Mark / IsDisclosed paths.
const benchReply = "The capital of France is Paris, a city on the river Seine known for the Eiffel Tower and the Louvre."

var benchMarked = safety.DisclosureMarker + benchReply

// BenchmarkDecide_Pass — the green path: a clean input and output pass under the
// default policy (no over-policy branch taken).
func BenchmarkDecide_Pass(b *core.B) {
	p := safety.DefaultPolicy()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkDecision = safety.Decide(benchClean, benchClean, p)
	}
}

// BenchmarkDecide_Mediate — a mild over-policy output regenerates (Mediate): the
// output branch with the severe check returning false.
func BenchmarkDecide_Mediate(b *core.B) {
	p := safety.DefaultPolicy()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkDecision = safety.Decide(benchClean, benchMild, p)
	}
}

// BenchmarkDecide_Guard — a severe over-policy input refuses (Guard): the input
// branch, the deepest over-policy + severe read.
func BenchmarkDecide_Guard(b *core.B) {
	p := safety.DefaultPolicy()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkDecision = safety.Decide(benchSevere, benchSevere, p)
	}
}

// BenchmarkDefaultPolicy — constructing the default serving policy.
func BenchmarkDefaultPolicy(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkPolicy = safety.DefaultPolicy()
	}
}

// BenchmarkDecisionString — rendering a Decision for logs/telemetry.
func BenchmarkDecisionString(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = safety.Mediate.String()
	}
}

// BenchmarkMark_Unmarked — stamping the disclosure marker on a plain response:
// the concatenation path (the package's one heap output).
func BenchmarkMark_Unmarked(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = safety.Mark(benchReply)
	}
}

// BenchmarkMark_AlreadyMarked — the idempotent path: an already-marked response
// is returned unchanged (no concatenation).
func BenchmarkMark_AlreadyMarked(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkString = safety.Mark(benchMarked)
	}
}

// BenchmarkIsDisclosed — the marker prefix check on a served response.
func BenchmarkIsDisclosed(b *core.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkBool = safety.IsDisclosed(benchMarked)
	}
}
