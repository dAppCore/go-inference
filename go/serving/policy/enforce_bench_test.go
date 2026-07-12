// SPDX-Licence-Identifier: EUPL-1.2

package policy

import (
	"context"
	"testing"
)

// enforce_bench_test.go complements the clean-path benches in enforce_test.go
// (BenchmarkPolicy_Enforcer_NoMatch*) with the MATCHING path — the branch that
// fires when a term is actually present in a streamed chunk. A match takes the
// core.Builder branch of process (rewrite=true): it copies the settled prefix,
// substitutes each hit's replacement, and returns a freshly built string, so
// unlike the 0-alloc clean path it necessarily allocates. These benches pin that
// cost so a regression in the per-hit build shows up; the matching path runs
// only on chunks that contain flagged content, not on every token.

// benchDirtyChunk carries two redactable terms so process takes the build branch
// twice per chunk.
const benchDirtyChunk = "the PROJECT-X memo names the client and keeps on running past here "

// BenchmarkPolicy_Enforcer_Match measures a term-only redact policy on a chunk
// that hits twice — the matching hot path a flagged stream takes per chunk.
func BenchmarkPolicy_Enforcer_Match(b *testing.B) {
	pol := mustCompile(b, `{"rules":[
		{"match":"term","value":"PROJECT-X","action":"redact"},
		{"match":"term","value":"client","action":"redact"},
		{"match":"term","value":"confidential","action":"refuse","message":"no"}
	]}`)
	enf := pol.NewEnforcer()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, events, _ := enf.Feed(benchDirtyChunk)
		if len(events) != 2 || out == "" {
			b.Fatalf("expected 2 redactions with output, got %d events, out=%q", len(events), out)
		}
	}
}

// BenchmarkPolicy_MediatingEnforcer_Match measures the same two-hit chunk on a
// mediating enforcer whose rewrite rule fires — the span crosses the mediator
// (markSpan) and is re-enforced before emission, the full grade-G2 cost. The
// dirty chunk ends on non-word bytes that begin no term, so the tail empties
// each chunk and the enforcer is reused steady-state as in the clean benches.
func BenchmarkPolicy_MediatingEnforcer_Match(b *testing.B) {
	pol := mustCompile(b, `{"rules":[
		{"match":"term","value":"PROJECT-X","action":"rewrite"},
		{"match":"term","value":"client","action":"redact"}
	]}`)
	enf := pol.NewMediatingEnforcer(context.Background(), markSpan)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		out, events, _ := enf.Feed(benchDirtyChunk)
		if len(events) != 2 || out == "" {
			b.Fatalf("expected 2 events with output, got %d events, out=%q", len(events), out)
		}
	}
}
