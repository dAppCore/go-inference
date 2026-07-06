// SPDX-Licence-Identifier: EUPL-1.2

package ai

import (
	"testing"
	"time"

	core "dappco.re/go"
)

// AX-11 baseline benchmarks for the ai/metrics hot path.
//
// Metrics surfaces fire on every observable AI event — Record runs
// once per task completion, RAG query, security scan, etc.; Summary
// runs on every UI status refresh, every metrics endpoint hit, every
// status CLI command.
//
// No bench coverage existed before this file. AX-11 § "What counts
// as a hot path" lists "per-request observability writes" and
// "per-response aggregation reads" both at high priority. Landing
// these baselines IS the AX-11 contract for this package.
//
// Run:
//   go test -bench=. -benchmem -benchtime=300ms ./ai/...

// Sinks prevent the compiler from optimising bench bodies away.
var (
	metricsBenchSinkResult  core.Result
	metricsBenchSinkSummary map[string]any
	metricsBenchSinkEvent   Event
)

// --- fixtures ---

func benchEvent() Event {
	return Event{
		Type:    "agent.task.completed",
		Repo:    "core/the inference stack",
		AgentID: "agent-cladius",
		Data: map[string]any{
			"task":     "bench fixture",
			"duration": 1234,
		},
	}
}

func benchEventSlice(n int) []Event {
	events := make([]Event, n)
	for i := range n {
		events[i] = Event{
			Type:    "agent.task.completed",
			Repo:    "core/the inference stack",
			AgentID: "agent-cladius",
			Data: map[string]any{
				"task_index": i,
			},
		}
	}
	return events
}

// --- Record — file write per event ---

// The per-event observability write. Runs once per task completion;
// the alloc + ns/op of this loop directly govern how cheap "always-on"
// telemetry can be.
func BenchmarkMetrics_Record_Typical(b *testing.B) {
	benchSetupMetricsHome(b)
	event := benchEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metricsBenchSinkResult = Record(event)
	}
}

// --- Summary — aggregation over events ---

// Summary builds 3 count maps + clones the recent tail. The per-event
// cost matters when status pages fan out: every status refresh on the
// admin dashboard pays this proportional to event count.
func BenchmarkMetrics_Summary_100(b *testing.B) {
	events := benchEventSlice(100)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metricsBenchSinkSummary = Summary(events)
	}
}

func BenchmarkMetrics_Summary_1000(b *testing.B) {
	events := benchEventSlice(1000)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metricsBenchSinkSummary = Summary(events)
	}
}

func BenchmarkMetrics_Summary_Empty(b *testing.B) {
	var events []Event
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metricsBenchSinkSummary = Summary(events)
	}
}

// --- cloneEvent — used internally by Summary's recent tail copy ---

// cloneEvent fires once per recent event in every Summary. Hot when
// the recent tail is large (default cap is recentEventLimit).
func BenchmarkMetrics_cloneEvent_NoData(b *testing.B) {
	event := Event{
		Type:    "agent.task.completed",
		Repo:    "core/the inference stack",
		AgentID: "agent-cladius",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metricsBenchSinkEvent = cloneEvent(event)
	}
}

func BenchmarkMetrics_cloneEvent_WithData(b *testing.B) {
	event := benchEvent()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metricsBenchSinkEvent = cloneEvent(event)
	}
}

// --- ReadEvents — daily-file read path ---

// Read 24 hours of events. Hot when the metrics CLI / dashboard
// renders. Cost scales with file count (per-day) + event count.
func BenchmarkMetrics_ReadEvents_LastDay(b *testing.B) {
	benchSetupMetricsHome(b)
	for range 50 {
		Record(benchEvent())
	}
	since := time.Now().Add(-24 * time.Hour)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		metricsBenchSinkResult = ReadEvents(since)
	}
}

// benchSetupMetricsHome mirrors withTempMetricsHome from metrics_test.go
// (testing.TB-compatible variant for benchmarks).
func benchSetupMetricsHome(tb testing.TB) {
	tb.Helper()
	tempHome := tb.TempDir()
	tb.Setenv("CORE_HOME", "")
	tb.Setenv("DIR_HOME", "")
	tb.Setenv("HOME", tempHome)
}

// --- AX-11 alloc-budget gates ---

// TestAllocBudget_Metrics_Summary locks the per-event aggregation cost.
// Summary builds 3 count maps + 1 recent-copy slice + clones each event
// in the recent tail. Budget is set to current measured count + headroom
// so a regression that turns Summary into O(n²) by accident fails loud.
//
// Run: go test -run TestAllocBudget_Metrics . ./ai/...
func TestAllocBudget_Metrics_Summary(t *testing.T) {
	events := benchEventSlice(100)

	// Behavioural lock: empty input returns 4 keys (by_type, by_repo,
	// by_agent, recent) — never panics.
	out := Summary(nil)
	if _, ok := out["by_type"]; !ok {
		t.Fatalf("Summary missing by_type key on nil events")
	}
	if _, ok := out["by_repo"]; !ok {
		t.Fatalf("Summary missing by_repo key on nil events")
	}
	if _, ok := out["by_agent"]; !ok {
		t.Fatalf("Summary missing by_agent key on nil events")
	}
	if _, ok := out["recent"]; !ok {
		t.Fatalf("Summary missing recent key on nil events")
	}

	avg := testing.AllocsPerRun(5, func() {
		metricsBenchSinkSummary = Summary(events)
	})
	// Ceiling: 35 — current measured 30 (Apple M3 Ultra) + ~17%
	// headroom. Summary allocates: 3 count maps + grows, 1 recent
	// slice copy, cloneEvent per recent-tail event (Data map alloc
	// when present), outer map, 3 cloneCounts. The recent tail is
	// capped at recentEventLimit so the count is bounded regardless
	// of input size; both Summary_100 and Summary_1000 measure to
	// the same alloc count.
	const budget = 35.0
	if avg > budget {
		t.Fatalf("Summary alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Summary fires on every status/UI refresh — every dashboard tick pays this.\n"+
			"Profile: go test -bench=BenchmarkMetrics_Summary -benchmem -memprofile=/tmp/s.mem",
			avg, budget)
	}
}

// TestAllocBudget_Metrics_cloneEvent locks the per-recent-tail-event copy.
// cloneEvent fires inside Summary's recent loop — N calls per Summary.
// A regression here multiplies across the recent tail size on every
// dashboard tick.
func TestAllocBudget_Metrics_cloneEvent(t *testing.T) {
	event := benchEvent()

	// Behavioural lock: clone is value-equal but Data map is distinct
	// (mutating the clone's Data doesn't affect the original).
	cloned := cloneEvent(event)
	if cloned.Type != event.Type || cloned.Repo != event.Repo {
		t.Fatalf("cloneEvent dropped scalar fields")
	}
	cloned.Data["mutate"] = "test"
	if _, leaked := event.Data["mutate"]; leaked {
		t.Fatalf("cloneEvent did not deep-copy Data map — mutation leaked")
	}

	avg := testing.AllocsPerRun(5, func() {
		metricsBenchSinkEvent = cloneEvent(event)
	})
	// Ceiling: 3 — current measured 2 (Apple M3 Ultra: Data map +
	// internal allocator). benchEvent's Data has scalar values which
	// pass through cloneMetricValue untouched, so no per-value allocs.
	const budget = 3.0
	if avg > budget {
		t.Fatalf("cloneEvent alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"cloneEvent fires inside Summary's recent loop — N× per Summary.\n"+
			"Profile: go test -bench=BenchmarkMetrics_cloneEvent_WithData -benchmem",
			avg, budget)
	}
}
