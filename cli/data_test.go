// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
	"dappco.re/go/inference/serving/chathistory"
)

// dataTestHome redirects $HOME to a fresh t.TempDir() so every `lem data`
// verb under test resolves its own isolated ~/.lem/datasets.duckdb — the
// same seam cli/serve_test.go already uses for the admin-token path (there
// is no --home flag surface).
func dataTestHome(t *testing.T) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	return home
}

// runData is a small shim over runDataCommand that returns the exit code and
// the split stdout/stderr text, so test bodies stay one-liners.
func runData(args ...string) (int, string, string) {
	var stdout, stderr bytes.Buffer
	code := runDataCommand(context.Background(), args, &stdout, &stderr)
	return code, stdout.String(), stderr.String()
}

// TestRunDataCommand_Dispatch covers the router itself: no args and an
// unknown subcommand both fail with usage; -h/--help/help all succeed.
func TestRunDataCommand_Dispatch(t *testing.T) {
	dataTestHome(t)
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no args", nil, 2},
		{"unknown subcommand", []string{"frobnicate"}, 2},
		{"help word", []string{"help"}, 0},
		{"help flag", []string{"--help"}, 0},
		{"help short", []string{"-h"}, 0},
	} {
		t.Run(tc.name, func(t *testing.T) {
			code, _, stderr := runData(tc.args...)
			if code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr)
			}
		})
	}
}

// ---- create ----

func TestRunDataCreate_Good(t *testing.T) {
	dataTestHome(t)
	t.Run("explicit title", func(t *testing.T) {
		code, stdout, stderr := runData("create", "evening-vents", "--title", "Evening vents", "--purpose", "life stuff")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		if !core.Contains(stdout, "evening-vents") {
			t.Errorf("stdout missing slug: %q", stdout)
		}
	})
	t.Run("title defaults to slug", func(t *testing.T) {
		code, _, stderr := runData("create", "no-title-given", "--json")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		store, closeStore := openTestStore(t)
		defer closeStore()
		ds, err := resolveDatasetSlug(store, "no-title-given")
		if err != nil {
			t.Fatalf("resolve: %v", err)
		}
		if ds.Title != "no-title-given" {
			t.Errorf("Title = %q, want the slug", ds.Title)
		}
	})
	t.Run("flags before or after the slug", func(t *testing.T) {
		code, _, stderr := runData("create", "--title", "Flags First", "flags-first")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
	})
}

func TestRunDataCreate_Bad(t *testing.T) {
	dataTestHome(t)
	if code, _, _ := runData("create", "dupe-me"); code != 0 {
		t.Fatalf("seed create failed")
	}
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no positional", []string{"create"}, 2},
		{"invalid slug", []string{"create", "Not A Valid Slug!"}, 1},
		{"duplicate slug", []string{"create", "dupe-me"}, 1},
		{"unknown flag", []string{"create", "x", "--nonsense"}, 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			code, _, stderr := runData(tc.args...)
			if code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr)
			}
		})
	}
}

// ---- list ----

func TestRunDataList_Good(t *testing.T) {
	dataTestHome(t)
	t.Run("empty store", func(t *testing.T) {
		code, stdout, stderr := runData("list")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		if !core.Contains(stdout, "no datasets yet") {
			t.Errorf("stdout = %q, want the empty-store notice", stdout)
		}
	})
	mustCreate(t, "alpha")
	mustCreate(t, "beta")
	t.Run("table output", func(t *testing.T) {
		code, stdout, stderr := runData("list")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		if !core.Contains(stdout, "alpha") || !core.Contains(stdout, "beta") {
			t.Errorf("stdout missing datasets: %q", stdout)
		}
	})
	t.Run("json output", func(t *testing.T) {
		code, stdout, stderr := runData("list", "--json")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		var got []dataset.Dataset
		if r := core.JSONUnmarshalString(stdout, &got); !r.OK {
			t.Fatalf("decode json list: %v", r.Value)
		}
		if len(got) != 2 {
			t.Errorf("json list len = %d, want 2", len(got))
		}
	})
	t.Run("archived excluded then included", func(t *testing.T) {
		if code, _, _ := runData("archive", "alpha"); code != 0 {
			t.Fatalf("archive setup failed")
		}
		_, stdout, _ := runData("list")
		if core.Contains(stdout, "alpha") {
			t.Errorf("archived dataset leaked into default list: %q", stdout)
		}
		_, stdoutAll, _ := runData("list", "--archived")
		if !core.Contains(stdoutAll, "alpha") {
			t.Errorf("--archived did not include the archived dataset: %q", stdoutAll)
		}
	})
}

func TestRunDataList_Bad(t *testing.T) {
	dataTestHome(t)
	if code, _, _ := runData("list", "unexpected-arg"); code != 2 {
		t.Fatalf("exit %d, want 2", code)
	}
}

// ---- stats ----

func TestRunDataStats_Good(t *testing.T) {
	dataTestHome(t)
	mustCreate(t, "stats-ds")
	jsonlPath := writeJSONLFixture(t, `{"prompt":"hi","response":"hello"}`, `{"prompt":"how are you","response":"well thanks"}`)
	if code, _, stderr := runData("import", "stats-ds", "--jsonl", jsonlPath); code != 0 {
		t.Fatalf("import setup failed: %s", stderr)
	}

	code, stdout, stderr := runData("stats", "stats-ds", "--json")
	if code != 0 {
		t.Fatalf("exit %d; stderr=%s", code, stderr)
	}
	var got dataStats
	if r := core.JSONUnmarshalString(stdout, &got); !r.OK {
		t.Fatalf("decode stats json: %v", r.Value)
	}
	if got.Total != 2 {
		t.Errorf("Total = %d, want 2", got.Total)
	}
	if got.ByKind["pair"] != 2 {
		t.Errorf("ByKind[pair] = %d, want 2", got.ByKind["pair"])
	}
	if got.ByStatus["pending"] != 2 {
		t.Errorf("ByStatus[pending] = %d, want 2", got.ByStatus["pending"])
	}
}

func TestRunDataStats_Bad(t *testing.T) {
	dataTestHome(t)
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no positional", []string{"stats"}, 2},
		{"unknown slug", []string{"stats", "does-not-exist"}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if code, _, _ := runData(tc.args...); code != tc.wantCode {
				t.Fatalf("exit %d, want %d", code, tc.wantCode)
			}
		})
	}
}

// ---- import ----

func TestRunDataImport_Good(t *testing.T) {
	dataTestHome(t)
	mustCreate(t, "import-ds")

	t.Run("jsonl happy path", func(t *testing.T) {
		path := writeJSONLFixture(t, `{"prompt":"hi","response":"hello"}`, `{"messages":[{"role":"user","content":"hey"},{"role":"assistant","content":"hi there"}]}`)
		code, stdout, stderr := runData("import", "import-ds", "--jsonl", path)
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		if !core.Contains(stdout, "ingested=2") {
			t.Errorf("stdout = %q, want ingested=2", stdout)
		}
	})

	t.Run("re-import dedupes, still exits 0", func(t *testing.T) {
		path := writeJSONLFixture(t, `{"prompt":"hi","response":"hello"}`)
		code, stdout, stderr := runData("import", "import-ds", "--jsonl", path)
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		if !core.Contains(stdout, "deduped=1") {
			t.Errorf("stdout = %q, want deduped=1", stdout)
		}
	})

	t.Run("malformed rows are counted skips and exit non-zero", func(t *testing.T) {
		path := writeJSONLFixture(t, `{"prompt":"new one","response":"ok"}`, `not valid json`, `{"prompt":"","response":"x"}`)
		code, stdout, stderr := runData("import", "import-ds", "--jsonl", path)
		if code != 1 {
			t.Fatalf("exit %d, want 1 (truthful failure); stderr=%s", code, stderr)
		}
		if !core.Contains(stdout, "ingested=1") || !core.Contains(stdout, "skipped=2") {
			t.Errorf("stdout = %q, want ingested=1 skipped=2", stdout)
		}
	})

	t.Run("chats happy path", func(t *testing.T) {
		home := dataTestHome(t)
		mustCreate(t, "chat-import-ds")
		buildChatsFixture(t, home, "snider")
		code, stdout, stderr := runData("import", "chat-import-ds", "--chats", "snider")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		if !core.Contains(stdout, "ingested=1") {
			t.Errorf("stdout = %q, want ingested=1", stdout)
		}
	})
}

func TestRunDataImport_Bad(t *testing.T) {
	dataTestHome(t)
	mustCreate(t, "import-bad-ds")
	jsonlPath := writeJSONLFixture(t, `{"prompt":"hi","response":"hello"}`)
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no positional", []string{"import"}, 2},
		{"neither jsonl nor chats", []string{"import", "import-bad-ds"}, 2},
		{"both jsonl and chats", []string{"import", "import-bad-ds", "--jsonl", jsonlPath, "--chats", "snider"}, 2},
		{"session without chats", []string{"import", "import-bad-ds", "--jsonl", jsonlPath, "--session", "x"}, 2},
		{"unknown dataset", []string{"import", "does-not-exist", "--jsonl", jsonlPath}, 1},
		{"nonexistent jsonl file", []string{"import", "import-bad-ds", "--jsonl", "/no/such/file.jsonl"}, 1},
		{"chats user with no history", []string{"import", "import-bad-ds", "--chats", "nobody-ever-chatted"}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			code, _, stderr := runData(tc.args...)
			if code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr)
			}
		})
	}
}

// ---- score ----

func TestRunDataScore_Good(t *testing.T) {
	dataTestHome(t)
	mustCreate(t, "score-ds")
	path := writeJSONLFixture(t, `{"prompt":"hi","response":"hello there, how can I help today?"}`)
	if code, _, stderr := runData("import", "score-ds", "--jsonl", path); code != 0 {
		t.Fatalf("import setup failed: %s", stderr)
	}

	t.Run("plain score", func(t *testing.T) {
		code, stdout, stderr := runData("score", "score-ds")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		if !core.Contains(stdout, "scored=1") {
			t.Errorf("stdout = %q, want scored=1", stdout)
		}
	})

	t.Run("auto-approve wiring (always-true threshold)", func(t *testing.T) {
		code, stdout, stderr := runData("score", "score-ds", "--auto-approve", "lek>=0", "--json")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		var got dataScoreReport
		if r := core.JSONUnmarshalString(stdout, &got); !r.OK {
			t.Fatalf("decode score json: %v", r.Value)
		}
		if got.AutoApproved != 1 {
			t.Errorf("AutoApproved = %d, want 1", got.AutoApproved)
		}
		store, closeStore := openTestStore(t)
		defer closeStore()
		ds, err := resolveDatasetSlug(store, "score-ds")
		if err != nil {
			t.Fatalf("resolve: %v", err)
		}
		items := core.MustCast[[]dataset.Item](store.Items(dataset.ItemFilter{DatasetID: ds.ID}))
		review := core.MustCast[dataset.Review](store.ReviewLatest(items[0].ID))
		if review.Status != dataset.StatusApproved {
			t.Errorf("review status = %s, want approved", review.Status)
		}
	})

	t.Run("auto-reject never fires on an impossible threshold", func(t *testing.T) {
		code, stdout, stderr := runData("score", "score-ds", "--auto-reject", "lek>=1000000")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		if !core.Contains(stdout, "auto_rejected=0") {
			t.Errorf("stdout = %q, want auto_rejected=0", stdout)
		}
	})
}

func TestRunDataScore_Bad(t *testing.T) {
	dataTestHome(t)
	mustCreate(t, "score-bad-ds")
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no positional", []string{"score"}, 2},
		{"unknown kind", []string{"score", "score-bad-ds", "--kind", "nonsense"}, 2},
		{"judge kind not yet available", []string{"score", "score-bad-ds", "--kind", "judge:helpfulness"}, 2},
		{"bad auto-approve expression", []string{"score", "score-bad-ds", "--auto-approve", "not-an-expression"}, 2},
		{"bad filter expression", []string{"score", "score-bad-ds", "--filter", "nonsense-field=x"}, 2},
		{"unknown dataset", []string{"score", "does-not-exist"}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			code, _, stderr := runData(tc.args...)
			if code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr)
			}
		})
	}
}

// ---- export ----

func TestRunDataExport_Good(t *testing.T) {
	dataTestHome(t)
	mustCreate(t, "export-ds")
	path := writeJSONLFixture(t, `{"prompt":"hi","response":"hello"}`)
	if code, _, stderr := runData("import", "export-ds", "--jsonl", path); code != 0 {
		t.Fatalf("import setup failed: %s", stderr)
	}

	t.Run("default filter (status=approved) exports nothing before review", func(t *testing.T) {
		out := filepath.Join(t.TempDir(), "unreviewed.jsonl")
		code, stdout, stderr := runData("export", "export-ds", "--format", "pairs-jsonl", "--out", out)
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		if !core.Contains(stdout, "items=0") {
			t.Errorf("stdout = %q, want items=0 (nothing approved yet)", stdout)
		}
	})

	if code, _, stderr := runData("score", "export-ds", "--auto-approve", "lek>=0"); code != 0 {
		t.Fatalf("score setup failed: %s", stderr)
	}

	t.Run("sft-jsonl after approval", func(t *testing.T) {
		out := filepath.Join(t.TempDir(), "train.jsonl")
		code, stdout, stderr := runData("export", "export-ds", "--format", "sft-jsonl", "--out", out, "--json")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		var manifest dataset.ExportManifest
		if r := core.JSONUnmarshalString(stdout, &manifest); !r.OK {
			t.Fatalf("decode manifest json: %v", r.Value)
		}
		if manifest.ItemCount != 1 {
			t.Errorf("ItemCount = %d, want 1", manifest.ItemCount)
		}
		if _, err := os.Stat(out); err != nil {
			t.Errorf("export file missing: %v", err)
		}
		if _, err := os.Stat(out + ".manifest.json"); err != nil {
			t.Errorf("manifest sidecar missing: %v", err)
		}
	})

	t.Run("explicit filter overrides the default", func(t *testing.T) {
		out := filepath.Join(t.TempDir(), "all.jsonl")
		code, stdout, stderr := runData("export", "export-ds", "--format", "capture-jsonl", "--out", out, "--filter", "kind=pair")
		if code != 0 {
			t.Fatalf("exit %d; stderr=%s", code, stderr)
		}
		if !core.Contains(stdout, "items=1") {
			t.Errorf("stdout = %q, want items=1", stdout)
		}
	})
}

func TestRunDataExport_Bad(t *testing.T) {
	dataTestHome(t)
	mustCreate(t, "export-bad-ds")
	out := filepath.Join(t.TempDir(), "out.jsonl")
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no positional", []string{"export"}, 2},
		{"missing format and out", []string{"export", "export-bad-ds"}, 2},
		{"missing out", []string{"export", "export-bad-ds", "--format", "pairs-jsonl"}, 2},
		{"unknown format", []string{"export", "export-bad-ds", "--format", "nonsense", "--out", out}, 1},
		{"unknown dataset", []string{"export", "does-not-exist", "--format", "pairs-jsonl", "--out", out}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			code, _, stderr := runData(tc.args...)
			if code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr)
			}
		})
	}
}

// ---- archive ----

func TestRunDataArchive_Good(t *testing.T) {
	dataTestHome(t)
	mustCreate(t, "archive-ds")
	code, stdout, stderr := runData("archive", "archive-ds", "--json")
	if code != 0 {
		t.Fatalf("exit %d; stderr=%s", code, stderr)
	}
	var got dataset.Dataset
	if r := core.JSONUnmarshalString(stdout, &got); !r.OK {
		t.Fatalf("decode archived json: %v", r.Value)
	}
	if !got.Archived {
		t.Error("Archived = false, want true")
	}
}

func TestRunDataArchive_Bad(t *testing.T) {
	dataTestHome(t)
	for _, tc := range []struct {
		name     string
		args     []string
		wantCode int
	}{
		{"no positional", []string{"archive"}, 2},
		{"unknown dataset", []string{"archive", "does-not-exist"}, 1},
	} {
		t.Run(tc.name, func(t *testing.T) {
			code, _, stderr := runData(tc.args...)
			if code != tc.wantCode {
				t.Fatalf("exit %d, want %d; stderr=%s", code, tc.wantCode, stderr)
			}
		})
	}
}

// ---- review ----

// TestRunDataReview_Good drives runDataReview's own logic (slug parsing,
// the conditional headless-fallback message) through the injected
// runDataReviewTUI seam — never a real tui.RunDataReview/tea.NewProgram
// call, which would hang (or open a real interactive session) whenever
// the test process has a controlling terminal; see runDataReviewTUI's
// doc comment in data.go.
func TestRunDataReview_Good(t *testing.T) {
	original := runDataReviewTUI
	defer func() { runDataReviewTUI = original }()

	var gotCtx context.Context
	var gotSlug string
	runDataReviewTUI = func(ctx context.Context, slug string, stdout, stderr io.Writer) int {
		gotCtx, gotSlug = ctx, slug
		core.WriteString(stderr, "tui: could not open a new TTY (stubbed)\n")
		return 1
	}
	code, stdout, stderr := runData("review", "evening-vents")
	if code != 1 {
		t.Fatalf("exit %d, want 1; stderr=%s", code, stderr)
	}
	if gotCtx == nil || gotSlug != "evening-vents" {
		t.Fatalf("runDataReviewTUI args = ctx=%v slug=%q", gotCtx, gotSlug)
	}
	if !core.Contains(stdout, "data list") || !core.Contains(stdout, "data stats") {
		t.Errorf("stdout = %q, want the headless-verb fallback pointer on a TUI start failure", stdout)
	}

	runDataReviewTUI = func(context.Context, string, io.Writer, io.Writer) int { return 0 }
	code, stdout, _ = runData("review")
	if code != 0 {
		t.Fatalf("exit %d, want 0 on a clean TUI exit", code)
	}
	if core.Contains(stdout, "could not start") {
		t.Errorf("stdout = %q printed the fallback pointer despite a clean (code 0) TUI exit", stdout)
	}
}

func TestRunDataReview_Help(t *testing.T) {
	code, stdout, stderr := runData("review", "--help")
	if code != 0 {
		t.Fatalf("exit %d; stderr=%s", code, stderr)
	}
	if !core.Contains(stdout, "Usage:") {
		t.Errorf("stdout = %q, want a usage banner", stdout)
	}
}

// ---- parseItemFilter / splitFilterClause ----

func TestParseItemFilter(t *testing.T) {
	for _, tc := range []struct {
		name    string
		expr    string
		want    dataset.ItemFilter
		wantErr bool
	}{
		{"empty", "", dataset.ItemFilter{DatasetID: "ds1"}, false},
		{"status", "status=approved", dataset.ItemFilter{DatasetID: "ds1", Status: dataset.StatusApproved}, false},
		{"kind", "kind=pair", dataset.ItemFilter{DatasetID: "ds1", Kind: dataset.KindPair}, false},
		{"source", "source=ssd", dataset.ItemFilter{DatasetID: "ds1", Source: dataset.SourceSSD}, false},
		{"archived", "archived=true", dataset.ItemFilter{DatasetID: "ds1", IncludeArchived: true}, false},
		{"score expression", "lek>=80", dataset.ItemFilter{DatasetID: "ds1", Score: &dataset.ScoreExpression{Kind: dataset.ScoreKindLEK, Op: dataset.OpGTE, Threshold: 80}}, false},
		{"combined", "status=approved,lek>=80", dataset.ItemFilter{DatasetID: "ds1", Status: dataset.StatusApproved, Score: &dataset.ScoreExpression{Kind: dataset.ScoreKindLEK, Op: dataset.OpGTE, Threshold: 80}}, false},
		{"unknown field", "nonsense=x", dataset.ItemFilter{}, true},
		{"malformed clause", "totally-broken-clause-!!", dataset.ItemFilter{}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := parseItemFilter("ds1", tc.expr)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("parseItemFilter(%q) = %+v, nil, want an error", tc.expr, got)
				}
				return
			}
			if err != nil {
				t.Fatalf("parseItemFilter(%q) unexpected error: %v", tc.expr, err)
			}
			if got.DatasetID != tc.want.DatasetID || got.Status != tc.want.Status || got.Kind != tc.want.Kind ||
				got.Source != tc.want.Source || got.IncludeArchived != tc.want.IncludeArchived {
				t.Fatalf("parseItemFilter(%q) = %+v, want %+v", tc.expr, got, tc.want)
			}
			if (got.Score == nil) != (tc.want.Score == nil) {
				t.Fatalf("parseItemFilter(%q) Score presence mismatch: got %+v, want %+v", tc.expr, got.Score, tc.want.Score)
			}
			if got.Score != nil && *got.Score != *tc.want.Score {
				t.Fatalf("parseItemFilter(%q) Score = %+v, want %+v", tc.expr, *got.Score, *tc.want.Score)
			}
		})
	}
}

func TestSplitFilterClause(t *testing.T) {
	for _, tc := range []struct {
		name      string
		clause    string
		wantKey   string
		wantValue string
		wantOK    bool
	}{
		{"simple", "status=approved", "status", "approved", true},
		{"spaced", " status = approved ", "status", "approved", true},
		{"double-equals rejected", "lek==80", "", "", false},
		{"not-equals rejected", "lek!=80", "", "", false},
		{"no equals", "lek>=80", "", "", false},
		{"gt no equals at all", "lek>80", "", "", false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			key, value, ok := splitFilterClause(tc.clause)
			if ok != tc.wantOK || key != tc.wantKey || value != tc.wantValue {
				t.Fatalf("splitFilterClause(%q) = (%q, %q, %v), want (%q, %q, %v)", tc.clause, key, value, ok, tc.wantKey, tc.wantValue, tc.wantOK)
			}
		})
	}
}

// ---- test fixtures ----

// openTestStore opens the dataset store under the currently-redirected HOME
// (see dataTestHome) — assertion-side access to state a verb call already
// wrote, mirroring what the verbs themselves do via tui.OpenDatasetStore.
func openTestStore(t *testing.T) (dataset.Store, func()) {
	t.Helper()
	store, code := openDataStore(os.Stderr)
	if store == nil {
		t.Fatalf("openDataStore failed, code=%d", code)
	}
	return store, func() { _ = store.Close() }
}

// mustCreate creates a dataset with the given slug via the real verb path,
// failing the test on any error — the common setup step nearly every other
// verb's tests build on.
func mustCreate(t *testing.T, slug string) {
	t.Helper()
	if code, _, stderr := runData("create", slug); code != 0 {
		t.Fatalf("mustCreate(%q) failed: exit %d; stderr=%s", slug, code, stderr)
	}
}

// writeJSONLFixture writes lines (already-JSON-encoded row strings) as a
// newline-delimited file in a fresh temp dir and returns its path.
func writeJSONLFixture(t *testing.T, lines ...string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "fixture.jsonl")
	var buf bytes.Buffer
	for _, line := range lines {
		buf.WriteString(line)
		buf.WriteByte('\n')
	}
	if err := os.WriteFile(path, buf.Bytes(), 0o644); err != nil {
		t.Fatalf("write jsonl fixture: %v", err)
	}
	return path
}

// buildChatsFixture writes one two-turn conversation into
// <home>/Lethean/lem/users/<user>/chats.duckdb — the exact convention
// runDataImportChats resolves — so --chats import tests have real chat
// history to read.
func buildChatsFixture(t *testing.T, home, user string) {
	t.Helper()
	path := core.Path(home, "Lethean", "lem", "users", user, "chats.duckdb")
	h, err := chathistory.Open(user, path)
	if err != nil {
		t.Fatalf("open chathistory fixture: %v", err)
	}
	defer h.Close()
	convID, err := h.StartConversation(chathistory.NewConversation{Title: "fixture", ModelID: "lemer-lite"})
	if err != nil {
		t.Fatalf("start conversation: %v", err)
	}
	if _, err := h.WriteTurn(convID, chathistory.NewTurn{Role: "user", Content: "hello there"}); err != nil {
		t.Fatalf("write user turn: %v", err)
	}
	if _, err := h.WriteTurn(convID, chathistory.NewTurn{Role: "assistant", Content: "hi, how can I help?"}); err != nil {
		t.Fatalf("write assistant turn: %v", err)
	}
	if err := h.EndConversation(convID); err != nil {
		t.Fatalf("end conversation: %v", err)
	}
}
