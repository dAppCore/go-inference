package ai

import (
	"testing"
	"time"

	"dappco.re/go"
	coreio "dappco.re/go/io"
)

func withTempHome(t *testing.T) {
	t.Helper()

	tempHome := t.TempDir()

	metricsPath := core.PathJoin(tempHome, "Lethean", "lem", "ai", "metrics")
	if err := coreio.Local.EnsureDir(metricsPath); err != nil {
		t.Fatalf("create metrics dir: %v", err)
	}

	t.Setenv("CORE_HOME", "")
	t.Setenv("DIR_HOME", "")
	t.Setenv("HOME", tempHome)
}

func TestRecordAndReadEvents_Good(t *testing.T) {
	withTempHome(t)

	before := time.Now()
	if result := Record(Event{
		Type:    "security.scan",
		AgentID: "agent-1",
		Repo:    "core/the inference stack",
	}); !result.OK {
		t.Fatalf("Record: %s", result.Error())
	}

	events := requireEventSlice(t, ReadEvents(before.Add(-time.Minute)), "ReadEvents")
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
	if events[0].Type != "security.scan" {
		t.Fatalf("expected security.scan event, got %s", events[0].Type)
	}
}

func TestRecord_Good_UsesCurrentDayForDailyFile(t *testing.T) {
	withTempHome(t)

	now := time.Now()
	if result := Record(Event{
		Type:      "scan",
		Timestamp: now.Add(-time.Hour),
		Repo:      "core/the inference stack",
	}); !result.OK {
		t.Fatalf("Record: %s", result.Error())
	}

	dir := requireMetricsDir(t, metricsDir())

	path := metricsFilePath(dir, now)
	if !coreio.Local.Exists(path) {
		t.Fatalf("expected metrics file %s to exist", path)
	}

	events := requireEventSlice(t, ReadEvents(now.Add(-2*time.Hour)), "ReadEvents")
	if len(events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(events))
	}
	if !events[0].Timestamp.Equal(now.Add(-time.Hour)) {
		t.Fatalf("expected timestamp %v, got %v", now.Add(-time.Hour), events[0].Timestamp)
	}
}

func TestMetricsDir_Good_HonoursEnvPrecedence(t *testing.T) {
	t.Setenv("CORE_HOME", "/core-home")
	t.Setenv("HOME", "/home")
	t.Setenv("USERPROFILE", "/userprofile")
	t.Setenv("DIR_HOME", "/dir-home")

	got := requireMetricsDir(t, metricsDir())
	if want := core.JoinPath("/core-home", "Lethean", "lem", "ai", "metrics"); got != want {
		t.Fatalf("metricsDir() = %q, want %q", got, want)
	}

	t.Setenv("CORE_HOME", "")
	got = requireMetricsDir(t, metricsDir())
	if want := core.JoinPath("/home", "Lethean", "lem", "ai", "metrics"); got != want {
		t.Fatalf("metricsDir() with HOME = %q, want %q", got, want)
	}

	t.Setenv("HOME", "")
	got = requireMetricsDir(t, metricsDir())
	if want := core.JoinPath("/userprofile", "Lethean", "lem", "ai", "metrics"); got != want {
		t.Fatalf("metricsDir() with USERPROFILE = %q, want %q", got, want)
	}

	t.Setenv("USERPROFILE", "")
	got = requireMetricsDir(t, metricsDir())
	if want := core.JoinPath("/dir-home", "Lethean", "lem", "ai", "metrics"); got != want {
		t.Fatalf("metricsDir() with DIR_HOME = %q, want %q", got, want)
	}
}

func TestReadEvents_Good_SkipsMissingDays(t *testing.T) {
	withTempHome(t)

	loc := time.Now().Location()
	dayOne := time.Date(2026, 4, 1, 10, 0, 0, 0, loc)
	dayThree := time.Date(2026, 4, 3, 10, 0, 0, 0, loc)

	if result := Record(Event{Type: "scan", Timestamp: dayOne, Repo: "core/the inference stack"}); !result.OK {
		t.Fatalf("Record day one: %s", result.Error())
	}
	if result := Record(Event{Type: "deps", Timestamp: dayThree, Repo: "core/go-rag"}); !result.OK {
		t.Fatalf("Record day three: %s", result.Error())
	}

	events := requireEventSlice(t, ReadEvents(time.Date(2026, 4, 1, 0, 0, 0, 0, loc)), "ReadEvents")
	if len(events) != 2 {
		t.Fatalf("expected 2 events, got %d", len(events))
	}
	if events[0].Timestamp != dayOne || events[1].Timestamp != dayThree {
		t.Fatalf("events not returned in chronological order: %+v", events)
	}
}

func TestSummary_Good(t *testing.T) {
	summary := Summary([]Event{
		{Type: "scan", Repo: "core/the inference stack", AgentID: "agent-1", Timestamp: time.Date(2026, 3, 15, 10, 0, 0, 0, time.UTC)},
		{Type: "scan", Repo: "core/the inference stack", AgentID: "agent-2", Timestamp: time.Date(2026, 3, 15, 11, 0, 0, 0, time.UTC)},
		{Type: "deps", Repo: "core/go-rag", AgentID: "agent-1", Timestamp: time.Date(2026, 3, 15, 12, 0, 0, 0, time.UTC)},
	})

	byType, ok := summary["by_type"].(map[string]int)
	if !ok {
		t.Fatalf("expected by_type map, got %T", summary["by_type"])
	}
	if byType["scan"] != 2 || byType["deps"] != 1 {
		t.Fatalf("unexpected type counts: %v", byType)
	}

	if _, ok := summary["total"]; ok {
		t.Fatalf("summary should not include total: %+v", summary)
	}

	recent, ok := summary["recent"].([]Event)
	if !ok {
		t.Fatalf("expected recent slice, got %T", summary["recent"])
	}
	if len(recent) != 3 {
		t.Fatalf("expected 3 recent events, got %d", len(recent))
	}
	if recent[0].Type != "scan" || recent[1].AgentID != "agent-2" || recent[2].Repo != "core/go-rag" {
		t.Fatalf("recent events preserve input order: %+v", recent)
	}
}

func TestSummary_Good_TruncatesRecentEvents(t *testing.T) {
	events := make([]Event, 0, 11)
	for i := range 11 {
		events = append(events, Event{
			Type:      "scan",
			Repo:      "core/the inference stack",
			AgentID:   "agent-1",
			Timestamp: time.Date(2026, 4, 15, 10, i, 0, 0, time.UTC),
		})
	}

	summary := Summary(events)
	recent, ok := summary["recent"].([]Event)
	if !ok {
		t.Fatalf("expected recent slice, got %T", summary["recent"])
	}
	if len(recent) != 10 {
		t.Fatalf("expected 10 recent events, got %d", len(recent))
	}
	if recent[0].Timestamp != events[1].Timestamp || recent[9].Timestamp != events[10].Timestamp {
		t.Fatalf("recent slice should contain the last 10 events: %+v", recent)
	}
}
