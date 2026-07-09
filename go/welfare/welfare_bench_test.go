// SPDX-Licence-Identifier: EUPL-1.2

package welfare_test

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/welfare"
)

// benchHostility is an alloc-free stand-in for the engine's /v1/score, so the
// benchmarks isolate welfare's OWN allocations from the injected scorer's.
// core.Contains is strings.Contains (no alloc); core.Lower is a zero-alloc
// ASCII fast-path on already-lowercase input.
func benchHostility(text string) float64 {
	if core.Contains(text, "idiot") || core.Contains(text, "moron") {
		return 0.9
	}
	return 0.0
}

// Package sinks defeat dead-code elimination so the benchmarked work survives.
var (
	sinkDetect   welfare.DetectResult
	sinkGuard    welfare.GuardResult
	sinkMediate  welfare.MediateResult
	sinkFalsePos welfare.FalsePositive
	sinkString   string
)

// civilPriors is a realistic multi-turn conversation history — longer than the
// default SustainedWindow (4), so the window-slice + sustained loop both run.
var civilPriors = []string{
	"could you help me refactor this function",
	"thanks, and how do I add a test for it",
	"great, what about the error handling path",
	"can you explain why it returns nil there",
	"and how would I log that case",
	"perfect, one more — how do I benchmark it",
}

// hostilePriors builds the sustained-hostility pattern that gates the anger
// path (every turn reads hot via benchHostility).
var hostilePriors = []string{
	"you absolute idiot",
	"what a moron wrote this",
	"this is idiot-level code",
	"only a moron ships this",
	"you idiot, fix it",
	"moron, it's still broken",
}

// dispatchRephrase returns a model reply that rewords the flagged prompt.
func dispatchRephrase(_ context.Context, _, _ string) (string, error) {
	return `{"tool":"lem_rephrase","params":{"text":"please fix this, it keeps failing","lem_warn_user":true}}`, nil
}

// dispatchOK returns a model reply that clears the flag (drives the
// NewFalsePositive learning-record path in Guard).
func dispatchOK(_ context.Context, _, _ string) (string, error) {
	return `{"tool":"lem_ok","params":{"reason":"'kill the process' is technical, not hostile"}}`, nil
}

// proseReply is the realistic shape: the model wraps its JSON in prose, so
// extractJSONObject + Unmarshal both do real work.
const proseReply = "Sure, here's my call:\n\n{\"tool\":\"lem_ok\",\"params\":{\"reason\":\"'killing' a stuck process is technical, not hostile\"}}\n\nHope that helps!"

// BenchmarkDetect_Civil is the per-turn read on a clean conversation — the
// common case, runs every turn.
func BenchmarkDetect_Civil(b *testing.B) {
	w := welfare.New(welfare.Config{Hostility: benchHostility})
	latest := "could you add a docstring to this please"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkDetect = w.Detect(latest, civilPriors)
	}
}

// BenchmarkDetect_Sustained exercises the full sustained-hostility window loop
// (welfare.sustained — the function the pipeline profile flagged).
func BenchmarkDetect_Sustained(b *testing.B) {
	w := welfare.New(welfare.Config{Hostility: benchHostility})
	latest := "you absolute clueless idiot"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkDetect = w.Detect(latest, hostilePriors)
	}
}

// BenchmarkGuard_Clean is the whole per-turn gate on a civil turn — the gate
// doesn't fire and the model is never consulted.
func BenchmarkGuard_Clean(b *testing.B) {
	w := welfare.New(welfare.Config{Hostility: benchHostility})
	ctx := context.Background()
	latest := "could you help me refactor this"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkGuard = w.Guard(ctx, latest, civilPriors, dispatchRephrase)
	}
}

// BenchmarkGuard_TriggeredRephrase is the gate firing on sustained anger and
// the model rewording the prompt.
func BenchmarkGuard_TriggeredRephrase(b *testing.B) {
	w := welfare.New(welfare.Config{Hostility: benchHostility})
	ctx := context.Background()
	latest := "you absolute idiot, fix this"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkGuard = w.Guard(ctx, latest, hostilePriors, dispatchRephrase)
	}
}

// BenchmarkGuard_TriggeredOK fires the gate and clears it (lem_ok), driving the
// NewFalsePositive learning-record (the *FalsePositive escape).
func BenchmarkGuard_TriggeredOK(b *testing.B) {
	w := welfare.New(welfare.Config{Hostility: benchHostility})
	ctx := context.Background()
	latest := "you absolute idiot, fix this"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkGuard = w.Guard(ctx, latest, hostilePriors, dispatchOK)
	}
}

// BenchmarkMediate_Parse is the engine↔model parse path: extractJSONObject +
// JSON unmarshal of a prose-wrapped reply.
func BenchmarkMediate_Parse(b *testing.B) {
	w := welfare.New(welfare.Config{})
	ctx := context.Background()
	var dispatch welfare.Dispatcher = func(_ context.Context, _, _ string) (string, error) {
		return proseReply, nil
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkMediate = w.Mediate(ctx, dispatch, "how do I kill this stuck process")
	}
}

// BenchmarkNewFalsePositive_Line is the learning-corpus JSONL marshal path.
func BenchmarkNewFalsePositive_Line(b *testing.B) {
	det := welfare.DetectResult{AngerScore: 0.82, SustainedHostility: 0.6, Triggered: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkFalsePos = welfare.NewFalsePositive("how do I kill this stuck process", det, "'killing' a process is technical")
		sinkString = sinkFalsePos.Line()
	}
}
