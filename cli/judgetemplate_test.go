// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"os"
	"path/filepath"
	"testing"

	core "dappco.re/go"
)

// writeJudgeTemplateFile writes raw content to dir/<name>.md, creating dir
// if needed, and returns dir — the common fixture step judge template tests
// build on.
func writeJudgeTemplateFile(t *testing.T, dir, name, content string) string {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	path := filepath.Join(dir, name+".md")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return dir
}

const wantOnlyNumberInstruction = "Reply with ONLY the number."

func qualityTemplateFixture(body string) string {
	if body == "" {
		body = "Score {{response}} as a reply to {{prompt}}.\n\n" + wantOnlyNumberInstruction
	}
	return "---\n" +
		"name: quality\n" +
		"description: Overall response quality, 0-100.\n" +
		"range: 0-100\n" +
		"---\n" +
		body + "\n"
}

// ---- parseJudgeTemplate ----

func TestParseJudgeTemplate_Good(t *testing.T) {
	tpl, err := parseJudgeTemplate("quality", qualityTemplateFixture(""))
	if err != nil {
		t.Fatalf("parseJudgeTemplate: %v", err)
	}
	if tpl.Name != "quality" {
		t.Errorf("Name = %q, want quality", tpl.Name)
	}
	if tpl.Description != "Overall response quality, 0-100." {
		t.Errorf("Description = %q", tpl.Description)
	}
	if tpl.Min != 0 || tpl.Max != 100 {
		t.Errorf("range = %v-%v, want 0-100", tpl.Min, tpl.Max)
	}
	if !core.Contains(tpl.Body, "{{response}}") || !core.Contains(tpl.Body, "{{prompt}}") {
		t.Errorf("Body lost a placeholder: %q", tpl.Body)
	}
}

func TestParseJudgeTemplate_Bad(t *testing.T) {
	for _, tc := range []struct {
		name    string
		content string
	}{
		{"missing opening delimiter", "name: quality\ndescription: d\nrange: 0-100\n---\nbody\n"},
		{"missing closing delimiter", "---\nname: quality\ndescription: d\nrange: 0-100\nbody with no fence\n"},
		{"malformed front-matter line", "---\nname quality\ndescription: d\nrange: 0-100\n---\nbody\n"},
		{"missing name field", "---\ndescription: d\nrange: 0-100\n---\nbody\n"},
		{"missing description field", "---\nname: quality\nrange: 0-100\n---\nbody\n"},
		{"missing range field", "---\nname: quality\ndescription: d\n---\nbody\n"},
		{"name does not match resolution name", "---\nname: other\ndescription: d\nrange: 0-100\n---\nbody\n"},
		{"empty body", "---\nname: quality\ndescription: d\nrange: 0-100\n---\n\n"},
		{"bad range grammar", "---\nname: quality\ndescription: d\nrange: not-a-range-either\n---\nbody\n"},
		{"range max not greater than min", "---\nname: quality\ndescription: d\nrange: 100-0\n---\nbody\n"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := parseJudgeTemplate("quality", tc.content); err == nil {
				t.Fatalf("parseJudgeTemplate(%q) = nil error, want a failure", tc.name)
			}
		})
	}
}

// TestParseJudgeTemplate_Ugly proves a literal "---" line INSIDE the body
// (e.g. a markdown horizontal rule in the judge's prompt) is not mistaken
// for a second front-matter fence — parsing stops at the first closing
// delimiter and treats everything after it as body text verbatim.
func TestParseJudgeTemplate_Ugly(t *testing.T) {
	content := "---\n" +
		"name: quality\n" +
		"description: d\n" +
		"range: 0-100\n" +
		"---\n" +
		"Above the line.\n\n---\n\nBelow the line: {{prompt}} / {{response}}\n"
	tpl, err := parseJudgeTemplate("quality", content)
	if err != nil {
		t.Fatalf("parseJudgeTemplate: %v", err)
	}
	if !core.Contains(tpl.Body, "Above the line.") || !core.Contains(tpl.Body, "Below the line") {
		t.Fatalf("body lost text around the embedded '---': %q", tpl.Body)
	}
}

// ---- parseJudgeScoreRange ----

func TestParseJudgeScoreRange_Good(t *testing.T) {
	for _, tc := range []struct {
		expr     string
		min, max float64
	}{
		{"0-100", 0, 100},
		{"0-1", 0, 1},
		{"10-20", 10, 20},
		{" 0 - 100 ", 0, 100},
	} {
		min, max, err := parseJudgeScoreRange(tc.expr)
		if err != nil {
			t.Errorf("parseJudgeScoreRange(%q): %v", tc.expr, err)
			continue
		}
		if min != tc.min || max != tc.max {
			t.Errorf("parseJudgeScoreRange(%q) = %v-%v, want %v-%v", tc.expr, min, max, tc.min, tc.max)
		}
	}
}

func TestParseJudgeScoreRange_Bad(t *testing.T) {
	for _, expr := range []string{"", "100", "0-100-200", "abc-100", "0-abc", "100-0", "50-50"} {
		if _, _, err := parseJudgeScoreRange(expr); err == nil {
			t.Errorf("parseJudgeScoreRange(%q) = nil error, want a failure", expr)
		}
	}
}

// ---- renderJudgeTemplate ----

func TestRenderJudgeTemplate_Good(t *testing.T) {
	tpl := judgeTemplate{Body: "P: {{prompt}}\nR: {{response}}\nrepeat prompt: {{prompt}}"}
	got := renderJudgeTemplate(tpl, "what is 2+2?", "4")
	want := "P: what is 2+2?\nR: 4\nrepeat prompt: what is 2+2?"
	if got != want {
		t.Fatalf("renderJudgeTemplate = %q, want %q", got, want)
	}
	if core.Contains(got, "{{") {
		t.Fatalf("renderJudgeTemplate left a placeholder unfilled: %q", got)
	}
}

// ---- parseJudgeScore ----

func TestParseJudgeScore_Good(t *testing.T) {
	tpl := judgeTemplate{Min: 0, Max: 100}
	for _, tc := range []struct {
		reply string
		want  float64
	}{
		{"87", 87},
		{" 42 \n", 42},
		{"0", 0},
		{"100", 100},
		{"73.5", 73.5},
	} {
		got, err := parseJudgeScore(tc.reply, tpl)
		if err != nil {
			t.Errorf("parseJudgeScore(%q): %v", tc.reply, err)
			continue
		}
		if got != tc.want {
			t.Errorf("parseJudgeScore(%q) = %v, want %v", tc.reply, got, tc.want)
		}
	}
}

// TestParseJudgeScore_Bad covers an in-range-shaped number that falls
// OUTSIDE the template's declared range — a well-formed number is still a
// loud failure, never silently clamped.
func TestParseJudgeScore_Bad(t *testing.T) {
	tpl := judgeTemplate{Min: 0, Max: 100}
	for _, reply := range []string{"150", "-1", "100.01", "1000000"} {
		if _, err := parseJudgeScore(reply, tpl); err == nil {
			t.Errorf("parseJudgeScore(%q) = nil error, want an out-of-range failure", reply)
		}
	}
}

// TestParseJudgeScore_Ugly covers non-numeric garbage: prose, a number
// buried in a sentence, or an empty reply — the parser never best-effort
// extracts a number, it demands a bare one.
func TestParseJudgeScore_Ugly(t *testing.T) {
	tpl := judgeTemplate{Min: 0, Max: 100}
	for _, reply := range []string{"", "   ", "I'd say around 85", "Score: 87", "87/100", "eighty-seven", "NaN"} {
		if _, err := parseJudgeScore(reply, tpl); err == nil {
			t.Errorf("parseJudgeScore(%q) = nil error, want a non-numeric failure", reply)
		}
	}
}

// ---- readJudgeTemplateFile ----

func TestReadJudgeTemplateFile_Good(t *testing.T) {
	dir := t.TempDir()
	writeJudgeTemplateFile(t, dir, "quality", "content")
	content, ok := readJudgeTemplateFile(dir, "quality")
	if !ok || content != "content" {
		t.Fatalf("readJudgeTemplateFile = (%q, %v), want (\"content\", true)", content, ok)
	}
}

func TestReadJudgeTemplateFile_Bad(t *testing.T) {
	dir := t.TempDir()
	if _, ok := readJudgeTemplateFile(dir, "missing"); ok {
		t.Fatalf("readJudgeTemplateFile found a file that was never written")
	}
	if _, ok := readJudgeTemplateFile("", "quality"); ok {
		t.Fatalf("readJudgeTemplateFile(\"\", ...) = found, want not-found for an empty dir")
	}
}

// ---- resolveJudgeTemplateFrom: the resolution-order contract ----

// TestResolveJudgeTemplateFrom_Good proves the design's fixed order: an
// override wins over an in-repo default of the same name, and the in-repo
// default still resolves when no override is present.
func TestResolveJudgeTemplateFrom_Good(t *testing.T) {
	overrideDir := t.TempDir()
	inRepoDir := t.TempDir()

	t.Run("override wins when both are present", func(t *testing.T) {
		writeJudgeTemplateFile(t, overrideDir, "quality", "---\nname: quality\ndescription: override version\nrange: 0-100\n---\noverride body {{prompt}} {{response}}\n")
		writeJudgeTemplateFile(t, inRepoDir, "quality", "---\nname: quality\ndescription: in-repo version\nrange: 0-100\n---\nin-repo body {{prompt}} {{response}}\n")

		tpl, err := resolveJudgeTemplateFrom(overrideDir, inRepoDir, "quality")
		if err != nil {
			t.Fatalf("resolveJudgeTemplateFrom: %v", err)
		}
		if tpl.Description != "override version" {
			t.Fatalf("Description = %q, want the override to win", tpl.Description)
		}
	})

	t.Run("in-repo default resolves when no override exists", func(t *testing.T) {
		writeJudgeTemplateFile(t, inRepoDir, "factuality", "---\nname: factuality\ndescription: in-repo only\nrange: 0-100\n---\nbody {{prompt}} {{response}}\n")

		tpl, err := resolveJudgeTemplateFrom(overrideDir, inRepoDir, "factuality")
		if err != nil {
			t.Fatalf("resolveJudgeTemplateFrom: %v", err)
		}
		if tpl.Description != "in-repo only" {
			t.Fatalf("Description = %q, want the in-repo default", tpl.Description)
		}
	})

	t.Run("an empty override dir falls through cleanly", func(t *testing.T) {
		writeJudgeTemplateFile(t, inRepoDir, "refusal-correctness", "---\nname: refusal-correctness\ndescription: d\nrange: 0-100\n---\nbody {{prompt}} {{response}}\n")

		tpl, err := resolveJudgeTemplateFrom("", inRepoDir, "refusal-correctness")
		if err != nil {
			t.Fatalf("resolveJudgeTemplateFrom with no override dir: %v", err)
		}
		if tpl.Name != "refusal-correctness" {
			t.Fatalf("Name = %q, want refusal-correctness", tpl.Name)
		}
	})
}

func TestResolveJudgeTemplateFrom_Bad(t *testing.T) {
	overrideDir := t.TempDir()
	inRepoDir := t.TempDir()
	if _, err := resolveJudgeTemplateFrom(overrideDir, inRepoDir, "does-not-exist"); err == nil {
		t.Fatalf("resolveJudgeTemplateFrom found in neither dir = nil error, want an unknown-template failure")
	}
	if _, err := resolveJudgeTemplateFrom("", "", "does-not-exist"); err == nil {
		t.Fatalf("resolveJudgeTemplateFrom with two empty dirs = nil error, want an unknown-template failure")
	}
}

// ---- findInRepoJudgesDir ----

// TestFindInRepoJudgesDir_Good proves the bounded upward walk finds a
// worktree root (anchored on BOTH go.work and a judges/ subdirectory)
// several levels above a deeply nested starting directory.
func TestFindInRepoJudgesDir_Good(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.work"), []byte("go 1.26\n"), 0o644); err != nil {
		t.Fatalf("write go.work fixture: %v", err)
	}
	judgesDir := filepath.Join(root, "judges")
	if err := os.MkdirAll(judgesDir, 0o755); err != nil {
		t.Fatalf("mkdir judges fixture: %v", err)
	}
	start := filepath.Join(root, "a", "b", "c")
	if err := os.MkdirAll(start, 0o755); err != nil {
		t.Fatalf("mkdir nested start fixture: %v", err)
	}

	dir, ok := findInRepoJudgesDir(start, judgeTemplateMaxWalkUp)
	if !ok {
		t.Fatalf("findInRepoJudgesDir(%q) not found, want %q", start, judgesDir)
	}
	if dir != judgesDir {
		t.Fatalf("findInRepoJudgesDir(%q) = %q, want %q", start, dir, judgesDir)
	}
}

// TestFindInRepoJudgesDir_Bad covers two negative shapes: a bound too small
// to reach the anchor, and a judges/ directory present with no go.work
// beside it (a bare name match is never enough).
func TestFindInRepoJudgesDir_Bad(t *testing.T) {
	t.Run("bound too small to reach the anchor", func(t *testing.T) {
		root := t.TempDir()
		if err := os.WriteFile(filepath.Join(root, "go.work"), []byte("go 1.26\n"), 0o644); err != nil {
			t.Fatalf("write go.work fixture: %v", err)
		}
		if err := os.MkdirAll(filepath.Join(root, "judges"), 0o755); err != nil {
			t.Fatalf("mkdir judges fixture: %v", err)
		}
		start := filepath.Join(root, "a", "b", "c")
		if err := os.MkdirAll(start, 0o755); err != nil {
			t.Fatalf("mkdir nested start fixture: %v", err)
		}

		if _, ok := findInRepoJudgesDir(start, 3); ok {
			t.Fatalf("findInRepoJudgesDir found the anchor within a bound too small to reach it")
		}
	})

	t.Run("judges dir without a go.work anchor never matches", func(t *testing.T) {
		root := t.TempDir()
		if err := os.MkdirAll(filepath.Join(root, "judges"), 0o755); err != nil {
			t.Fatalf("mkdir judges fixture: %v", err)
		}
		if _, ok := findInRepoJudgesDir(root, judgeTemplateMaxWalkUp); ok {
			t.Fatalf("findInRepoJudgesDir matched a bare judges/ dir with no go.work beside it")
		}
	})
}

// TestDefaultInRepoJudgesDir_Good proves the real in-repo judges/ directory
// this worktree ships is actually discoverable from cli/'s own `go test`
// working directory (one level below the repo root), and that every
// shipped default template — quality.md, factuality.md,
// refusal-correctness.md — parses cleanly with both placeholders present.
// A synthetic-fixture-only test suite could pass while the real shipped
// files were subtly malformed; this closes that gap.
func TestDefaultInRepoJudgesDir_Good(t *testing.T) {
	dir, ok := defaultInRepoJudgesDir()
	if !ok {
		t.Fatalf("defaultInRepoJudgesDir did not find the repo-root judges/ directory from %v", func() string { wd, _ := os.Getwd(); return wd }())
	}
	for _, name := range []string{"quality", "factuality", "refusal-correctness"} {
		content, ok := readJudgeTemplateFile(dir, name)
		if !ok {
			t.Fatalf("judges/%s.md not found under %s", name, dir)
		}
		tpl, err := parseJudgeTemplate(name, content)
		if err != nil {
			t.Fatalf("parse judges/%s.md: %v", name, err)
		}
		if !core.Contains(tpl.Body, "{{prompt}}") || !core.Contains(tpl.Body, "{{response}}") {
			t.Errorf("judges/%s.md body is missing a placeholder", name)
		}
	}
}
