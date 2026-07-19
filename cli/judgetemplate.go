// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"math"
	"strconv"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// judgeTemplate is one parsed judge template — front matter (name,
// description, score range) plus the prompt body a judge driver renders
// {{prompt}}/{{response}} into. See judges/README.md at the repo root for
// the on-disk format this parses.
type judgeTemplate struct {
	Name        string
	Description string
	Min, Max    float64
	Body        string
}

const (
	// judgeTemplateDelim is the front-matter fence line, top and bottom.
	judgeTemplateDelim = "---"
	// judgeTemplateExt is the on-disk extension for both the in-repo
	// defaults (judges/<name>.md) and ~/.lem/judges/<name>.md overrides.
	judgeTemplateExt = ".md"
	// judgeTemplateMaxWalkUp bounds defaultInRepoJudgesDir's upward search
	// from the working directory — generous for any cwd this repo-resident
	// tool is realistically run from, while still guaranteed to terminate.
	judgeTemplateMaxWalkUp = 8
)

// parseJudgeTemplate parses one template file's raw content. name is the
// resolution name (the filename stem, e.g. "quality" for "quality.md") —
// the front-matter "name:" field must match it exactly; a mismatch is a
// loud error rather than a silently-tolerated rename/typo.
//
//	tpl, err := parseJudgeTemplate("quality", content)
func parseJudgeTemplate(name, content string) (judgeTemplate, error) {
	lines := core.Split(content, "\n")
	i := 0
	for i < len(lines) && core.Trim(lines[i]) == "" {
		i++
	}
	if i >= len(lines) || core.Trim(lines[i]) != judgeTemplateDelim {
		return judgeTemplate{}, core.E("main.parseJudgeTemplate", core.Sprintf("judge template %q: missing opening %q front-matter delimiter", name, judgeTemplateDelim), nil)
	}
	i++

	fields := map[string]string{}
	for i < len(lines) && core.Trim(lines[i]) != judgeTemplateDelim {
		line := core.Trim(lines[i])
		i++
		if line == "" {
			continue
		}
		idx := core.Index(line, ":")
		if idx < 0 {
			return judgeTemplate{}, core.E("main.parseJudgeTemplate", core.Sprintf("judge template %q: malformed front-matter line %q (want \"key: value\")", name, line), nil)
		}
		fields[core.Trim(line[:idx])] = core.Trim(line[idx+1:])
	}
	if i >= len(lines) {
		return judgeTemplate{}, core.E("main.parseJudgeTemplate", core.Sprintf("judge template %q: missing closing %q front-matter delimiter", name, judgeTemplateDelim), nil)
	}
	i++ // skip past the closing delimiter line
	body := core.Trim(core.Join("\n", lines[i:]...))

	tplName := fields["name"]
	description := fields["description"]
	rangeExpr := fields["range"]
	if tplName == "" || description == "" || rangeExpr == "" {
		return judgeTemplate{}, core.E("main.parseJudgeTemplate", core.Sprintf("judge template %q: front matter requires name, description, and range", name), nil)
	}
	if tplName != name {
		return judgeTemplate{}, core.E("main.parseJudgeTemplate", core.Sprintf("judge template %q: front-matter name %q does not match the template's file name", name, tplName), nil)
	}
	if body == "" {
		return judgeTemplate{}, core.E("main.parseJudgeTemplate", core.Sprintf("judge template %q: empty prompt body", name), nil)
	}

	min, max, rerr := parseJudgeScoreRange(rangeExpr)
	if rerr != nil {
		return judgeTemplate{}, core.E("main.parseJudgeTemplate", core.Sprintf("judge template %q: range %q", name, rangeExpr), rerr)
	}
	return judgeTemplate{Name: tplName, Description: description, Min: min, Max: max, Body: body}, nil
}

// parseJudgeScoreRange parses "MIN-MAX" — two non-negative numbers
// separated by a single '-', the minimal grammar judges/README.md
// documents (deliberately no support for a negative MIN — every default
// template ships a 0-based scale).
func parseJudgeScoreRange(expr string) (float64, float64, error) {
	parts := core.Split(expr, "-")
	if len(parts) != 2 {
		return 0, 0, core.NewError("want MIN-MAX")
	}
	min, err := strconv.ParseFloat(core.Trim(parts[0]), 64)
	if err != nil {
		return 0, 0, core.E("main.parseJudgeScoreRange", core.Sprintf("invalid minimum %q", parts[0]), err)
	}
	max, err := strconv.ParseFloat(core.Trim(parts[1]), 64)
	if err != nil {
		return 0, 0, core.E("main.parseJudgeScoreRange", core.Sprintf("invalid maximum %q", parts[1]), err)
	}
	if !(max > min) {
		return 0, 0, core.NewError("maximum must be greater than minimum")
	}
	return min, max, nil
}

// renderJudgeTemplate fills tpl's body placeholders with an item's prompt
// and response text — every occurrence, no escaping, no nested templating
// (the design's "keep the format minimal").
//
//	rendered := renderJudgeTemplate(tpl, "2+2?", "4")
func renderJudgeTemplate(tpl judgeTemplate, prompt, response string) string {
	rendered := core.Replace(tpl.Body, "{{prompt}}", prompt)
	rendered = core.Replace(rendered, "{{response}}", response)
	return rendered
}

// parseJudgeScore parses a judge model's raw reply per the contract's parse
// rules: the ENTIRE trimmed reply must be a bare number — no surrounding
// prose, no best-effort extraction of a number buried in a sentence — and
// it must fall within tpl's declared range. Either failure is a loud,
// descriptive error; this never returns a silent 0.
//
//	value, err := parseJudgeScore("87", tpl)
func parseJudgeScore(reply string, tpl judgeTemplate) (float64, error) {
	trimmed := core.Trim(reply)
	if trimmed == "" {
		return 0, core.NewError("judge reply is empty, want a bare number")
	}
	value, err := strconv.ParseFloat(trimmed, 64)
	if err != nil {
		return 0, core.E("main.parseJudgeScore", core.Sprintf("judge reply %q is not a bare number", trimmed), nil)
	}
	// strconv.ParseFloat happily accepts "NaN"/"Inf" as valid float64
	// values, and a NaN comparison is always false — without this check a
	// "NaN" reply would silently pass the range test below (neither < nor
	// > trips) and slip through as an apparently in-range score.
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return 0, core.E("main.parseJudgeScore", core.Sprintf("judge reply %q is not a finite number", trimmed), nil)
	}
	if value < tpl.Min || value > tpl.Max {
		return 0, core.E("main.parseJudgeScore", core.Sprintf("judge score %v is outside the template's declared range %v-%v", value, tpl.Min, tpl.Max), nil)
	}
	return value, nil
}

// resolveJudgeTemplateFrom is the pure resolution-order logic behind
// judge:<name> scoring: a template in overrideDir wins over one of the same
// name in inRepoDir. Either dir may be "" (not resolvable in this
// environment) without error — only "found in neither" fails. Kept free of
// any real filesystem/HOME dependency so resolution order is directly
// testable against two t.TempDir() fixtures.
//
//	tpl, err := resolveJudgeTemplateFrom(overrideDir, inRepoDir, "quality")
func resolveJudgeTemplateFrom(overrideDir, inRepoDir, name string) (judgeTemplate, error) {
	if content, ok := readJudgeTemplateFile(overrideDir, name); ok {
		return parseJudgeTemplate(name, content)
	}
	if content, ok := readJudgeTemplateFile(inRepoDir, name); ok {
		return parseJudgeTemplate(name, content)
	}
	return judgeTemplate{}, core.E("main.resolveJudgeTemplateFrom", core.Sprintf("unknown judge template %q (checked ~/.lem/judges/ and the in-repo judges/ default)", name), nil)
}

// readJudgeTemplateFile reads dir/<name>.md, reporting ok=false (never an
// error) for an empty dir or an absent file — both are ordinary "not found
// here, try the next source" outcomes for the caller's resolution order.
func readJudgeTemplateFile(dir, name string) (string, bool) {
	if core.Trim(dir) == "" {
		return "", false
	}
	path := core.Path(dir, name+judgeTemplateExt)
	if !coreio.Local.IsFile(path) {
		return "", false
	}
	content, err := coreio.Local.Read(path)
	if err != nil {
		return "", false
	}
	return content, true
}

// findInRepoJudgesDir is the pure, bounded upward walk from start looking
// for the worktree root this in-repo judges/ default ships beside —
// anchored on a directory carrying BOTH go.work and a judges/ subdirectory
// (the worktree root's own signature), never a bare "judges" name match
// against some unrelated ancestor.
func findInRepoJudgesDir(start string, maxLevels int) (string, bool) {
	dir := start
	for i := 0; i < maxLevels; i++ {
		anchor := core.Path(dir, "go.work")
		candidate := core.Path(dir, "judges")
		if coreio.Local.IsFile(anchor) && coreio.Local.IsDir(candidate) {
			return candidate, true
		}
		parent := core.PathDir(dir)
		if parent == dir {
			break
		}
		dir = parent
	}
	return "", false
}

// defaultInRepoJudgesDir walks up from the real working directory for the
// in-repo judges/ default judges/README.md documents. lem is built to
// <repo>/bin/lem and run via the Taskfile/lem.sh harness from within a
// checkout (the same repo-relative assumption cli/embed_metallib.go's
// plain-build fallback makes), so this is a reasonable, bounded search
// rather than a hard requirement — a binary copied elsewhere with no
// checkout in reach simply finds no in-repo defaults, and resolution falls
// through to a loud "unknown template" error rather than a wrong guess.
func defaultInRepoJudgesDir() (string, bool) {
	cwdResult := core.Getwd()
	if !cwdResult.OK {
		return "", false
	}
	cwd, ok := cwdResult.Value.(string)
	if !ok || core.Trim(cwd) == "" {
		return "", false
	}
	return findInRepoJudgesDir(cwd, judgeTemplateMaxWalkUp)
}
