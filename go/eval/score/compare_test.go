package score

import (
	"dappco.re/go"
)

func writeScoreFile(t *core.T, file string, value float64) {
	t.Helper()
	out := &ScorerOutput{ModelAverages: map[string]map[string]float64{"m": {"score": value}}}
	requireResultOK(t, WriteScores(file, out))
}

func TestCompare_RunCompare_Good(t *core.T) {
	dir := t.TempDir()
	oldFile := core.JoinPath(dir, "old.out")
	newFile := core.JoinPath(dir, "new.out")
	writeScoreFile(t, oldFile, 1)
	writeScoreFile(t, newFile, 2)
	assertResultOK(t, RunCompare(oldFile, newFile))
}

func TestCompare_RunCompare_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	assertResultError(t, RunCompare(core.JoinPath(t.TempDir(), "missing.out"), core.JoinPath(t.TempDir(), "new.out")))
}

func TestCompare_RunCompare_Ugly(t *core.T) {
	dir := t.TempDir()
	oldFile := core.JoinPath(dir, "old.out")
	newFile := core.JoinPath(dir, "new.out")
	writeScoreFile(t, oldFile, 0)
	writeScoreFile(t, newFile, 0)
	assertResultOK(t, RunCompare(oldFile, newFile))
}
