// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"context"
	"time"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// ---------------------------------------------------------------------------
// fakeTransport — in-memory RemoteTransport for testing
// ---------------------------------------------------------------------------

// fakeTransport implements RemoteTransport using canned responses keyed on
// a substring of the command string.  Commands are matched in insertion order
// so the first matching key wins.
type fakeTransport struct {
	commands []fakeCmd

	// copyFromCalls/copyToCalls count invocations; copyFromFailOn/copyToFailOn,
	// when non-zero, make the call with that 1-indexed number fail instead of
	// succeeding. This lets tests target e.g. the 2nd of two sequential
	// CopyFrom calls (processMLXNative/processWithConversion each scp the
	// safetensors file, then the adapter config) without a bespoke fake per
	// scenario. Zero value never fails, so every existing caller keeps its
	// current always-succeed behaviour.
	copyFromCalls  int
	copyFromFailOn int
	copyToCalls    int
	copyToFailOn   int
}

type fakeCmd struct {
	pattern string
	stdout  string
	err     error
}

func newFakeTransport() *fakeTransport { return &fakeTransport{} }

func (f *fakeTransport) On(pattern, stdout string, err error) {
	f.commands = append(f.commands, fakeCmd{pattern: pattern, stdout: stdout, err: err})
}

func (f *fakeTransport) Run(_ context.Context, cmd string) core.Result {
	for _, fc := range f.commands {
		if contains(cmd, fc.pattern) {
			return core.ResultOf(fc.stdout, fc.err)
		}
	}
	return core.Fail(core.NewError(core.Concat("fakeTransport: no match for command: ", cmd)))
}

func (f *fakeTransport) CopyFrom(_ context.Context, _, _ string) core.Result {
	f.copyFromCalls++
	if f.copyFromFailOn != 0 && f.copyFromCalls == f.copyFromFailOn {
		return core.Fail(core.NewError("fakeTransport: CopyFrom failed"))
	}
	return core.Ok(nil)
}

func (f *fakeTransport) CopyTo(_ context.Context, _, _ string) core.Result {
	f.copyToCalls++
	if f.copyToFailOn != 0 && f.copyToCalls == f.copyToFailOn {
		return core.Fail(core.NewError("fakeTransport: CopyTo failed"))
	}
	return core.Ok(nil)
}

// contains is a small helper to avoid importing strings just for this.
func contains(s, substr string) bool {
	return len(substr) == 0 || len(s) >= len(substr) && searchSubstr(s, substr)
}

func searchSubstr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// =========================================================================
// 1. AdapterMeta tests
// =========================================================================

func TestAdapterMetaKnownFamiliesGoodScenario(t *core.T) {
	tests := []struct {
		dirname  string
		wantTag  string
		wantPfx  string
		wantStem string
	}{
		// gemma-3-1b via "1b" prefix
		{"adapters-1b", "gemma-3-1b", "G1", "1b"},
		// gemma-3-27b via "27b" prefix
		{"adapters-27b", "gemma-3-27b", "G27", "27b"},
		// deepseek-r1-7b
		{"adapters-deepseek-r1-7b", "deepseek-r1-7b", "R1", "deepseek-r1-7b"},
		// gpt-oss
		{"adapters-gpt-oss", "gpt-oss-20b", "GPT", "gpt-oss"},
		// gemma-3-12b via "12b" prefix
		{"adapters-12b", "gemma-3-12b", "G12", "12b"},
		// gemma-3-4b via "4b" prefix
		{"adapters-4b", "gemma-3-4b", "G4", "4b"},
		// bench-1b
		{"adapters-bench-1b", "gemma-3-1b", "B1", "bench-1b"},
		// book
		{"adapters-book", "gemma-3-27b", "Book", "book"},
		// cross
		{"adapters-cross", "gemma-3-12b", "Cross", "cross"},
		// vi → gemma-3-1b
		{"adapters-vi", "gemma-3-1b", "Vi1", "vi"},
		// vi-12b → gemma-3-12b
		{"adapters-vi-12b", "gemma-3-12b", "Vi12", "vi-12b"},
		// lem-gpt-oss
		{"adapters-lem-gpt-oss", "gpt-oss-20b", "LGPT", "lem-gpt-oss"},
	}

	for _, tt := range tests {
		t.Run(tt.dirname, func(t *core.T) {
			tag, pfx, stem := AdapterMeta(tt.dirname)
			core.AssertEqual(t, tt.wantTag, tag, "model tag")
			core.AssertEqual(t, tt.wantPfx, pfx, "label prefix")
			core.AssertEqual(t, tt.wantStem, stem, "run ID stem")
		})
	}
}

func TestAdapterMetaWithVariantGoodScenario(t *core.T) {
	// "adapters-27b-reasoning" → 27b prefix matches, variant = "reasoning"
	tag, pfx, stem := AdapterMeta("adapters-27b-reasoning")
	core.AssertEqual(t, "gemma-3-27b", tag)
	core.AssertEqual(t, "G27-reasoning", pfx)
	core.AssertEqual(t, "27b-reasoning", stem)
}

func TestAdapterMetaWithoutVariantGoodScenario(t *core.T) {
	// "adapters-12b" → variant is empty → "base"
	tag, pfx, stem := AdapterMeta("adapters-12b")
	core.AssertEqual(t, "gemma-3-12b", tag)
	core.AssertEqual(t, "G12", pfx) // variant="base" produces short without suffix
	core.AssertEqual(t, "12b", stem)
}

func TestAdapterMetaSubdirectoryPatternGoodScenario(t *core.T) {
	// "adapters-15k/gemma-3-27b" → matches "15k/gemma-3-27b" prefix
	tag, pfx, stem := AdapterMeta("adapters-15k/gemma-3-27b")
	core.AssertEqual(t, "gemma-3-27b", tag)
	core.AssertEqual(t, "G27", pfx)
	// stem should replace "/" with "-"
	core.AssertEqual(t, "15k-gemma-3-27b", stem)
}

func TestAdapterMetaSubdirectoryWithVariantGoodScenario(t *core.T) {
	// "adapters-15k/gemma-3-1b-creative" → variant = "creative"
	tag, pfx, stem := AdapterMeta("adapters-15k/gemma-3-1b-creative")
	core.AssertEqual(t, "gemma-3-1b", tag)
	core.AssertEqual(t, "G1-creative", pfx)
	core.AssertEqual(t, "15k-gemma-3-1b-creative", stem)
}

func TestAdapterMeta_Unknown_Bad(t *core.T) {
	// Unknown dirname falls back: tag=name, short=name[:10], stem=name
	tag, pfx, stem := AdapterMeta("adapters-completelynewmodel42")
	core.AssertEqual(t, "completelynewmodel42", tag)
	core.AssertEqual(t, "completely", pfx) // truncated to 10 chars
	core.AssertEqual(t, "completelynewmodel42", stem)
}

func TestAdapterMetaUnknownShortGoodScenario(t *core.T) {
	// Short unknown name (< 10 chars) is not truncated.
	tag, pfx, stem := AdapterMeta("adapters-xyz")
	core.AssertEqual(t, "xyz", tag)
	core.AssertEqual(t, "xyz", pfx)
	core.AssertEqual(t, "xyz", stem)
}

func TestAdapterMetaNoPrefixGoodScenario(t *core.T) {
	// dirname without "adapters-" prefix — TrimPrefix does nothing useful,
	// but the function should still handle it gracefully.
	tag, pfx, stem := AdapterMeta("27b-fancy")
	core.AssertEqual(t, "gemma-3-27b", tag)
	core.AssertEqual(t, "G27-fancy", pfx)
	core.AssertEqual(t, "27b-fancy", stem)
}

// =========================================================================
// 2. FindUnscored tests
// =========================================================================

func TestFindUnscoredAllUnscoredGoodScenario(t *core.T) {
	checkpoints := []Checkpoint{
		{Dirname: "b-dir", Iteration: 200, RunID: "run-b", Label: "B @200"},
		{Dirname: "a-dir", Iteration: 100, RunID: "run-a", Label: "A @100"},
		{Dirname: "a-dir", Iteration: 50, RunID: "run-a", Label: "A @50"},
	}
	scored := map[[2]string]bool{}

	result := FindUnscored(checkpoints, scored)

	core.AssertLen(t, result, 3)
	// Should be sorted by (dirname, iteration)
	core.AssertEqual(t, "a-dir", result[0].Dirname)
	core.AssertEqual(t, 50, result[0].Iteration)
	core.AssertEqual(t, "a-dir", result[1].Dirname)
	core.AssertEqual(t, 100, result[1].Iteration)
	core.AssertEqual(t, "b-dir", result[2].Dirname)
	core.AssertEqual(t, 200, result[2].Iteration)
}

func TestFindUnscoredSomeScoredGoodScenario(t *core.T) {
	checkpoints := []Checkpoint{
		{Dirname: "dir", Iteration: 100, RunID: "run-1", Label: "L @100"},
		{Dirname: "dir", Iteration: 200, RunID: "run-1", Label: "L @200"},
		{Dirname: "dir", Iteration: 300, RunID: "run-1", Label: "L @300"},
	}
	scored := map[[2]string]bool{
		{"run-1", "L @100"}: true,
		{"run-1", "L @300"}: true,
	}

	result := FindUnscored(checkpoints, scored)

	core.AssertLen(t, result, 1)
	core.AssertEqual(t, 200, result[0].Iteration)
	core.AssertEqual(t, "L @200", result[0].Label)
}

func TestFindUnscoredAllScoredGoodScenario(t *core.T) {
	checkpoints := []Checkpoint{
		{Dirname: "dir", Iteration: 100, RunID: "run-1", Label: "L @100"},
		{Dirname: "dir", Iteration: 200, RunID: "run-1", Label: "L @200"},
	}
	scored := map[[2]string]bool{
		{"run-1", "L @100"}: true,
		{"run-1", "L @200"}: true,
	}

	result := FindUnscored(checkpoints, scored)
	core.AssertEmpty(t, result)
}

func TestFindUnscoredEmptyInputGoodScenario(t *core.T) {
	result := FindUnscored(nil, nil)
	core.AssertEmpty(t, result)

	result = FindUnscored([]Checkpoint{}, map[[2]string]bool{})
	core.AssertEmpty(t, result)
}

func TestFindUnscoredNilScoredGoodScenario(t *core.T) {
	// nil scored map should treat everything as unscored
	checkpoints := []Checkpoint{
		{Dirname: "a", Iteration: 1, RunID: "r", Label: "L @1"},
	}
	result := FindUnscored(checkpoints, nil)
	core.AssertLen(t, result, 1)
}

// =========================================================================
// 3. BufferInfluxResult / ReplayInfluxBuffer round-trip tests
// =========================================================================

func TestBufferInfluxResultRoundTripGoodScenario(t *core.T) {
	workDir := t.TempDir()

	cp := Checkpoint{
		RemoteDir: "/data/adapters-27b",
		Filename:  "0001000_adapters.safetensors",
		Dirname:   "adapters-27b",
		Iteration: 1000,
		ModelTag:  "gemma-3-27b",
		Label:     "G27 @1000",
		RunID:     "27b-capability-auto",
	}
	results := ProbeResult{
		Accuracy: 75.0,
		Correct:  3,
		Total:    4,
		ByCategory: map[string]CategoryResult{
			"math": {Correct: 2, Total: 2},
			"lang": {Correct: 1, Total: 2},
		},
		Probes: map[string]SingleProbeResult{
			"p1": {Passed: true, Response: "ok"},
			"p2": {Passed: false, Response: "wrong"},
		},
	}

	BufferInfluxResult(workDir, cp, results)

	// Verify the buffer file exists and contains valid JSONL
	bufPath := core.JoinPath(workDir, InfluxBufferFile)
	raw, err := coreio.Local.Read(bufPath)
	data := []byte(raw)
	core.RequireNoError(t, err)
	core.AssertNotEmpty(t, data)

	// Parse the JSONL entry and verify fields
	var entry bufferEntry
	mustJSONUnmarshalBytes(t, data[:len(data)-1], &entry) // trim trailing newline
	core.AssertEqual(t, cp.Label, entry.Checkpoint.Label)
	core.AssertEqual(t, cp.ModelTag, entry.Checkpoint.ModelTag)
	core.AssertEqual(t, cp.RunID, entry.Checkpoint.RunID)
	core.AssertEqual(t, results.Accuracy, entry.Results.Accuracy)
	core.AssertEqual(t, results.Correct, entry.Results.Correct)
	core.AssertEqual(t, results.Total, entry.Results.Total)
	core.AssertNotEmpty(t, entry.Timestamp)
}

func TestBufferInfluxResultMultipleEntriesGoodScenario(t *core.T) {
	workDir := t.TempDir()

	for i := range 3 {
		cp := Checkpoint{
			Dirname:   "dir",
			Iteration: i * 100,
			Label:     "L",
			RunID:     "run",
			ModelTag:  "tag",
		}
		results := ProbeResult{
			Accuracy: float64(i) * 25.0,
			Correct:  i,
			Total:    4,
			Probes:   map[string]SingleProbeResult{},
		}
		BufferInfluxResult(workDir, cp, results)
	}

	bufPath := core.JoinPath(workDir, InfluxBufferFile)
	raw, err := coreio.Local.Read(bufPath)
	data := []byte(raw)
	core.RequireNoError(t, err)

	// Count newlines — should be 3 JSONL lines
	lines := 0
	for _, b := range data {
		if b == '\n' {
			lines++
		}
	}
	core.AssertEqual(t, 3, lines)
}

func TestReplayInfluxBufferEmptyFileGoodScenario(t *core.T) {
	workDir := t.TempDir()

	// No buffer file exists — ReplayInfluxBuffer should be a no-op
	ReplayInfluxBuffer(workDir, nil)

	// Buffer file still shouldn't exist
	core.AssertFalse(t, coreio.Local.IsFile(core.JoinPath(workDir, InfluxBufferFile)))
}

func TestReplayInfluxBufferMissingFileGoodScenario(t *core.T) {
	// Calling with a nonexistent directory should not panic
	workDir := "/nonexistent/path/that/does/not/exist"
	ReplayInfluxBuffer(workDir, nil)
	core.AssertFalse(t, coreio.Local.IsFile(core.JoinPath(workDir, InfluxBufferFile)))
}

// =========================================================================
// 4. DiscoverCheckpoints tests (using fakeTransport)
// =========================================================================

func TestDiscoverCheckpointsHappyPathGoodScenario(t *core.T) {
	ft := newFakeTransport()

	base := "/data/training"

	// Command 1: list adapter directories (exact command from DiscoverCheckpoints)
	ft.On("ls -d "+base+"/adapters-* 2>/dev/null",
		base+"/adapters-27b\n"+base+"/adapters-1b\n", nil)

	// Command 2a: sub-directory check for adapters-27b — no gemma-3-* subdirs
	ft.On("ls -d "+base+"/adapters-27b/gemma-3-* 2>/dev/null", "", core.NewError("no match"))

	// Command 2b: sub-directory check for adapters-1b — no gemma-3-* subdirs
	ft.On("ls -d "+base+"/adapters-1b/gemma-3-* 2>/dev/null", "", core.NewError("no match"))

	// Command 3a: list safetensors in adapters-27b
	ft.On("ls "+base+"/adapters-27b/*_adapters.safetensors 2>/dev/null",
		base+"/adapters-27b/0001000_adapters.safetensors\n"+base+"/adapters-27b/0002000_adapters.safetensors\n", nil)

	// Command 3b: list safetensors in adapters-1b
	ft.On("ls "+base+"/adapters-1b/*_adapters.safetensors 2>/dev/null",
		base+"/adapters-1b/0000500_adapters.safetensors\n", nil)

	cfg := &AgentConfig{
		M3AdapterBase: base,
		Transport:     ft,
	}

	r := DiscoverCheckpoints(cfg)
	requireResultOK(t, r)
	checkpoints := r.Value.([]Checkpoint)
	core.AssertLen(t, checkpoints, 3)

	// Verify parsed checkpoint details
	found1000 := false
	found2000 := false
	found500 := false
	for _, cp := range checkpoints {
		switch {
		case cp.Dirname == "adapters-27b" && cp.Iteration == 1000:
			found1000 = true
			core.AssertEqual(t, "gemma-3-27b", cp.ModelTag)
			core.AssertEqual(t, "0001000_adapters.safetensors", cp.Filename)
			core.AssertContains(t, cp.Label, "@0001000")
			core.AssertContains(t, cp.RunID, "27b")
		case cp.Dirname == "adapters-27b" && cp.Iteration == 2000:
			found2000 = true
		case cp.Dirname == "adapters-1b" && cp.Iteration == 500:
			found500 = true
			core.AssertEqual(t, "gemma-3-1b", cp.ModelTag)
		}
	}
	core.AssertTrue(t, found1000, "should find iteration 1000")
	core.AssertTrue(t, found2000, "should find iteration 2000")
	core.AssertTrue(t, found500, "should find iteration 500")
}

func TestDiscoverCheckpointsWithSubDirsGoodScenario(t *core.T) {
	ft := newFakeTransport()

	base := "/data/training"

	// Command 1: list adapter directories
	ft.On("ls -d "+base+"/adapters-* 2>/dev/null",
		base+"/adapters-15k\n", nil)

	// Command 2: sub-directory check finds gemma-3-* subdirs
	ft.On("ls -d "+base+"/adapters-15k/gemma-3-* 2>/dev/null",
		base+"/adapters-15k/gemma-3-27b\n"+base+"/adapters-15k/gemma-3-1b\n", nil)

	// Command 3a: list safetensors in gemma-3-27b subdir
	ft.On("ls "+base+"/adapters-15k/gemma-3-27b/*_adapters.safetensors 2>/dev/null",
		base+"/adapters-15k/gemma-3-27b/0003000_adapters.safetensors\n", nil)

	// Command 3b: list safetensors in gemma-3-1b subdir
	ft.On("ls "+base+"/adapters-15k/gemma-3-1b/*_adapters.safetensors 2>/dev/null",
		base+"/adapters-15k/gemma-3-1b/0001500_adapters.safetensors\n", nil)

	cfg := &AgentConfig{
		M3AdapterBase: base,
		Transport:     ft,
	}

	r := DiscoverCheckpoints(cfg)
	requireResultOK(t, r)
	checkpoints := r.Value.([]Checkpoint)
	core.AssertLen(t, checkpoints, 2)

	// The dirname should include the subdirectory path relative to base
	for _, cp := range checkpoints {
		switch {
		case cp.Iteration == 3000:
			core.AssertEqual(t, "adapters-15k/gemma-3-27b", cp.Dirname)
			core.AssertEqual(t, "gemma-3-27b", cp.ModelTag)
		case cp.Iteration == 1500:
			core.AssertEqual(t, "adapters-15k/gemma-3-1b", cp.Dirname)
			core.AssertEqual(t, "gemma-3-1b", cp.ModelTag)
		default:
			t.Errorf("unexpected iteration %d", cp.Iteration)
		}
	}
}

func TestDiscoverCheckpointsNoAdaptersGoodScenario(t *core.T) {
	ft := newFakeTransport()
	base := "/data/training"

	// ls -d returns empty output
	ft.On("ls -d "+base+"/adapters-* 2>/dev/null", "", nil)

	cfg := &AgentConfig{
		M3AdapterBase: base,
		Transport:     ft,
	}

	r := DiscoverCheckpoints(cfg)
	requireResultOK(t, r)
	checkpoints := r.Value.([]Checkpoint)
	core.AssertEmpty(t, checkpoints)
}

func TestDiscoverCheckpointsSSHErrorBadScenario(t *core.T) {
	ft := newFakeTransport()
	base := "/data/training"

	ft.On("ls -d "+base+"/adapters-* 2>/dev/null", "", core.NewError("ssh: connection refused"))

	cfg := &AgentConfig{
		M3AdapterBase: base,
		Transport:     ft,
	}

	r := DiscoverCheckpoints(cfg)
	assertResultError(t, r)
	core.AssertContains(t, r.Error(), "list adapter dirs")
}

func TestDiscoverCheckpointsFilterPatternGoodScenario(t *core.T) {
	ft := newFakeTransport()
	base := "/data/training"

	// When Filter is set, the ls pattern changes to adapters-27b*
	ft.On("ls -d "+base+"/adapters-27b* 2>/dev/null",
		base+"/adapters-27b\n", nil)

	// No gemma-3-* subdirs
	ft.On("ls -d "+base+"/adapters-27b/gemma-3-* 2>/dev/null", "", core.NewError("no match"))

	ft.On("ls "+base+"/adapters-27b/*_adapters.safetensors 2>/dev/null",
		base+"/adapters-27b/0001000_adapters.safetensors\n", nil)

	cfg := &AgentConfig{
		M3AdapterBase: base,
		Transport:     ft,
		Filter:        "27b",
	}

	r := DiscoverCheckpoints(cfg)
	requireResultOK(t, r)
	checkpoints := r.Value.([]Checkpoint)
	core.AssertLen(t, checkpoints, 1)
	core.AssertEqual(t, 1000, checkpoints[0].Iteration)
}

func TestDiscoverCheckpointsNoSafetensorsGoodScenario(t *core.T) {
	ft := newFakeTransport()
	base := "/data/training"

	ft.On("ls -d "+base+"/adapters-* 2>/dev/null",
		base+"/adapters-27b\n", nil)
	ft.On("ls -d "+base+"/adapters-27b/gemma-3-* 2>/dev/null", "", core.NewError("no match"))

	// safetensors listing fails (no checkpoint files yet)
	ft.On("ls "+base+"/adapters-27b/*_adapters.safetensors 2>/dev/null", "", core.NewError("no match"))

	cfg := &AgentConfig{
		M3AdapterBase: base,
		Transport:     ft,
	}

	r := DiscoverCheckpoints(cfg)
	requireResultOK(t, r)
	checkpoints := r.Value.([]Checkpoint)
	core.AssertEmpty(t, checkpoints, "no safetensors means no checkpoints")
}

// --- v0.9.0 shape triplets ---

func TestAgent_NewAgent_Good(t *core.T) {
	cfg := &AgentConfig{WorkDir: t.TempDir()}
	agent := NewAgent(cfg)
	core.AssertEqual(t, cfg, agent.Config())
}

func TestAgent_NewAgent_Bad(t *core.T) {
	agent := NewAgent(nil)
	core.AssertNotNil(t, agent)
	core.AssertNil(t, agent.Config())
}

func TestAgent_NewAgent_Ugly(t *core.T) {
	cfg := &AgentConfig{OneShot: true, DryRun: true}
	agent := NewAgent(cfg)
	cfg.WorkDir = t.TempDir()
	core.AssertEqual(t, cfg.WorkDir, agent.Config().WorkDir)
}

func TestAgent_Agent_Config_Good(t *core.T) {
	cfg := &AgentConfig{WorkDir: t.TempDir()}
	agent := NewAgent(cfg)
	core.AssertEqual(t, cfg, agent.Config())
}

func TestAgent_Agent_Config_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	var agent Agent
	core.AssertNil(t, agent.Config())
}

func TestAgent_Agent_Config_Ugly(t *core.T) {
	cfg := &AgentConfig{Filter: "g1"}
	agent := NewAgent(cfg)
	agent.Config().Filter = "g2"
	core.AssertEqual(t, "g2", cfg.Filter)
}

func TestAgent_Agent_Execute_Good(t *core.T) {
	transport := newFakeTransport()
	cfg := &AgentConfig{WorkDir: t.TempDir(), OneShot: true, DryRun: true, Transport: transport}
	agent := NewAgent(cfg)
	core.AssertNotPanics(t, func() { agent.Execute(context.Background()) })
}

func TestAgent_Agent_Execute_Bad(t *core.T) {
	transport := newFakeTransport()
	cfg := &AgentConfig{WorkDir: t.TempDir(), OneShot: true, Transport: transport}
	agent := NewAgent(cfg)
	core.AssertNotPanics(t, func() { agent.Execute(context.Background()) })
}

func TestAgent_Agent_Execute_Ugly(t *core.T) {
	transport := newFakeTransport()
	cfg := &AgentConfig{WorkDir: t.TempDir(), OneShot: true, DryRun: true, Transport: transport}
	agent := NewAgent(&AgentConfig{WorkDir: t.TempDir(), OneShot: true, Transport: transport})
	core.AssertNotPanics(t, func() { agent.Execute(context.Background(), cfg) })
}

func TestAgent_Agent_Evaluate_Good(t *core.T) {
	agent := NewAgent(nil)
	err := agent.Evaluate(context.Background(), Checkpoint{})
	assertResultError(t, err, "config")

	// A configured agent resolves the Checkpoint target and reaches
	// ProcessOne — the fake transport cannot produce a real adapter, so the
	// eventual Result is still an error, but every statement between target
	// resolution and ProcessOne's own dispatch now runs.
	configured := NewAgent(&AgentConfig{Transport: newFakeTransport(), WorkDir: t.TempDir()})
	got := configured.Evaluate(context.Background(), Checkpoint{ModelTag: "gemma-3-1b"})
	core.AssertFalse(t, got.OK)
	core.AssertNotContains(t, got.Error(), "unsupported target type")
}

func TestAgent_Agent_Evaluate_Bad(t *core.T) {
	agent := NewAgent(&AgentConfig{})
	err := agent.Evaluate(context.Background(), 42)
	assertResultError(t, err, "unsupported")
}

func TestAgent_Agent_Evaluate_Ugly(t *core.T) {
	agent := NewAgent(&AgentConfig{})
	err := agent.Evaluate(context.Background(), (*Checkpoint)(nil))
	assertResultError(t, err, "nil checkpoint")
}

// =========================================================================
// resolveCheckpointTarget / resolveCheckpointPath / matchCheckpointTarget /
// equalsPathJoin — direct tests of Evaluate()'s target-resolution helpers.
// =========================================================================

func TestAgent_Agent_resolveCheckpointTarget_Good(t *core.T) {
	agent := NewAgent(&AgentConfig{})

	r := agent.resolveCheckpointTarget(context.Background(), Checkpoint{Label: "direct"})
	requireResultOK(t, r)
	core.AssertEqual(t, "direct", r.Value.(Checkpoint).Label)

	r2 := agent.resolveCheckpointTarget(context.Background(), &Checkpoint{Label: "pointer"})
	requireResultOK(t, r2)
	core.AssertEqual(t, "pointer", r2.Value.(Checkpoint).Label)
}

func TestAgent_Agent_resolveCheckpointTarget_Bad(t *core.T) {
	agent := NewAgent(&AgentConfig{})
	r := agent.resolveCheckpointTarget(context.Background(), 3.14)
	assertResultError(t, r, "unsupported")
}

func TestAgent_Agent_resolveCheckpointTarget_Ugly(t *core.T) {
	// A string target delegates to resolveCheckpointPath rather than
	// resolving directly.
	agent := NewAgent(&AgentConfig{Transport: newFakeTransport()})
	r := agent.resolveCheckpointTarget(context.Background(), "adapters-1b/adapter.safetensors")
	requireResultOK(t, r)
	core.AssertEqual(t, "adapters-1b", r.Value.(Checkpoint).Dirname)
}

func TestAgent_Agent_resolveCheckpointPath_Good(t *core.T) {
	ft := newFakeTransport()
	ft.On("ls -d /base/adapters-*", "/base/adapters-27b\n", nil)
	ft.On("ls -d /base/adapters-27b/gemma-3-*", "", core.AnError)
	ft.On("ls /base/adapters-27b/*_adapters.safetensors", "/base/adapters-27b/0001000_adapters.safetensors\n", nil)
	agent := NewAgent(&AgentConfig{M3AdapterBase: "/base", Transport: ft})

	// Exact match against a discovered checkpoint's RemoteDir wins over the
	// path-derived fallback.
	r := agent.resolveCheckpointPath(context.Background(), "/base/adapters-27b")
	requireResultOK(t, r)
	cp := r.Value.(Checkpoint)
	core.AssertEqual(t, "/base/adapters-27b", cp.RemoteDir)
	core.AssertEqual(t, 1000, cp.Iteration)
}

func TestAgent_Agent_resolveCheckpointPath_Bad(t *core.T) {
	agent := NewAgent(&AgentConfig{})
	r := agent.resolveCheckpointPath(context.Background(), "   ")
	assertResultError(t, r, "empty checkpoint path")
}

func TestAgent_Agent_resolveCheckpointPath_Ugly(t *core.T) {
	ft := newFakeTransport()
	ft.On("ls -d /base/adapters-*", "/base/adapters-27b\n", nil)
	ft.On("ls -d /base/adapters-27b/gemma-3-*", "", core.AnError)
	ft.On("ls /base/adapters-27b/*_adapters.safetensors", "/base/adapters-27b/0001000_adapters.safetensors\n", nil)
	agent := NewAgent(&AgentConfig{M3AdapterBase: "/base", Transport: ft})

	// Target matches none of the discovered checkpoints, so resolution
	// falls through to path-derived metadata instead of the discovery list.
	r := agent.resolveCheckpointPath(context.Background(), "/base/adapters-99b/x_adapters.safetensors")
	requireResultOK(t, r)
	cp := r.Value.(Checkpoint)
	core.AssertEqual(t, "adapters-99b", cp.Dirname)
	core.AssertEqual(t, "x_adapters.safetensors", cp.Filename)

	// A nil-cfg agent skips discovery entirely and still derives a checkpoint.
	var bare Agent
	r2 := bare.resolveCheckpointPath(context.Background(), "adapters-4b/x_adapters.safetensors")
	requireResultOK(t, r2)
	core.AssertEqual(t, "gemma-3-4b", r2.Value.(Checkpoint).ModelTag)

	// An absolute path outside M3AdapterBase is not stripped, so the
	// leading-"/" fallback reduces dirname to its basename.
	var noBase Agent
	r3 := noBase.resolveCheckpointPath(context.Background(), "/elsewhere/adapters-27b/x_adapters.safetensors")
	requireResultOK(t, r3)
	core.AssertEqual(t, "adapters-27b", r3.Value.(Checkpoint).Dirname)

	// A dirname that reduces to exactly "adapters-" leaves AdapterMeta's
	// label prefix and stem empty, exercising both fallback assignments.
	r4 := noBase.resolveCheckpointPath(context.Background(), "adapters-")
	requireResultOK(t, r4)
	cp4 := r4.Value.(Checkpoint)
	core.AssertEqual(t, "adapters-", cp4.Label)
	core.AssertEqual(t, "adapters--capability-auto", cp4.RunID)
}

func TestAgent_matchCheckpointTarget_Good(t *core.T) {
	checkpoints := []Checkpoint{
		{RemoteDir: "/data/adapters-27b", Dirname: "adapters-27b"},
		{RemoteDir: "/data/adapters-1b", Dirname: "adapters-1b"},
	}
	cp, ok := matchCheckpointTarget(checkpoints, "/data/adapters-1b")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "adapters-1b", cp.Dirname)

	// Target equals RemoteDir + "/" + Filename exactly, but matches neither
	// RemoteDir nor Dirname alone — the equalsPathJoin branch.
	joined := []Checkpoint{{RemoteDir: "/data/adapters-4b", Filename: "0000200_adapters.safetensors", Dirname: "adapters-4b"}}
	cp2, ok2 := matchCheckpointTarget(joined, "/data/adapters-4b/0000200_adapters.safetensors")
	core.AssertTrue(t, ok2)
	core.AssertEqual(t, "adapters-4b", cp2.Dirname)
}

func TestAgent_matchCheckpointTarget_Bad(t *core.T) {
	checkpoints := []Checkpoint{{RemoteDir: "/data/adapters-27b", Dirname: "adapters-27b"}}
	_, ok := matchCheckpointTarget(checkpoints, "/data/adapters-99b")
	core.AssertFalse(t, ok)
}

func TestAgent_matchCheckpointTarget_Ugly(t *core.T) {
	// Two checkpoints share a basename with neither an exact RemoteDir nor
	// Dirname match — the ambiguous result is rejected even though
	// candidates exist.
	ambiguous := []Checkpoint{
		{RemoteDir: "/a/nested/shared", Dirname: "something-else-a"},
		{RemoteDir: "/b/other/shared", Dirname: "something-else-b"},
	}
	_, ok := matchCheckpointTarget(ambiguous, "shared")
	core.AssertFalse(t, ok)

	// A single unique basename match resolves even without an exact match —
	// RemoteDir's basename misses, so the fallback checks Dirname's basename.
	// Dirname is deliberately not itself equal to target (that would hit the
	// exact-match branch instead), only its basename is.
	unique := []Checkpoint{{RemoteDir: "/only/one/xyz", Dirname: "prefix/adapters-x"}}
	cp, ok := matchCheckpointTarget(unique, "adapters-x")
	core.AssertTrue(t, ok)
	core.AssertEqual(t, "/only/one/xyz", cp.RemoteDir)
}

func TestAgent_equalsPathJoin_Good(t *core.T) {
	core.AssertTrue(t, equalsPathJoin("dir/file.txt", "dir", "file.txt"))
}

func TestAgent_equalsPathJoin_Bad(t *core.T) {
	core.AssertFalse(t, equalsPathJoin("short", "much/longer/dir", "file.txt"))
}

func TestAgent_equalsPathJoin_Ugly(t *core.T) {
	// Same total length as "ab" + "/" + "c" but the separator position holds
	// a different character, so the comparison must fail past the length
	// check rather than short-circuiting on it.
	core.AssertFalse(t, equalsPathJoin("abXc", "ab", "c"))
}

func TestAgent_Agent_ExecuteRemote_Good(t *core.T) {
	transport := newFakeTransport()
	transport.On("uptime", "ok", nil)
	agent := NewAgent(&AgentConfig{Transport: transport})
	r := agent.ExecuteRemote(context.Background(), "uptime")
	requireResultOK(t, r)
	out := r.Value.(string)
	core.AssertEqual(t, "ok", out)
}

func TestAgent_Agent_ExecuteRemote_Bad(t *core.T) {
	agent := NewAgent(&AgentConfig{})
	r := agent.ExecuteRemote(context.Background())
	assertResultError(t, r, "no command")
}

func TestAgent_Agent_ExecuteRemote_Ugly(t *core.T) {
	agent := NewAgent(&AgentConfig{})
	r := agent.ExecuteRemote(context.Background(), "a", "b")
	assertResultError(t, r, "expected")

	// The 3-arg (host, port, command) form builds a one-shot SSHTransport
	// from the agent's M3 credentials instead of using cfg.Transport.
	threeArg := NewAgent(&AgentConfig{M3SSHKey: "/missing", M3User: "nobody"})
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	r2 := threeArg.ExecuteRemote(ctx, "127.0.0.1", "1", "true")
	assertResultError(t, r2)
}

func TestAgent_Agent_CollectMetrics_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	agent := NewAgent(&AgentConfig{WorkDir: t.TempDir(), InfluxURL: "http://127.0.0.1:1", InfluxDB: "test"})
	assertResultOK(t, agent.CollectMetrics(context.Background()))
}

func TestAgent_Agent_CollectMetrics_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	agent := NewAgent(&AgentConfig{WorkDir: core.JoinPath(t.TempDir(), "missing"), InfluxURL: "http://127.0.0.1:1"})
	assertResultOK(t, agent.CollectMetrics(context.Background(), "http://127.0.0.1:1"))
}

func TestAgent_Agent_CollectMetrics_Ugly(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	agent := NewAgent(&AgentConfig{WorkDir: t.TempDir()})
	assertResultOK(t, agent.CollectMetrics(context.Background(), ""))
}

func TestAgent_Agent_DiscoverCheckpoints_Good(t *core.T) {
	transport := newFakeTransport()
	cfg := &AgentConfig{Transport: transport}
	agent := NewAgent(cfg)
	r := agent.DiscoverCheckpoints(context.Background())
	assertResultError(t, r)
}

func TestAgent_Agent_DiscoverCheckpoints_Bad(t *core.T) {
	agent := NewAgent(&AgentConfig{})
	r := agent.DiscoverCheckpoints(context.Background())
	assertResultError(t, r)
}

func TestAgent_Agent_DiscoverCheckpoints_Ugly(t *core.T) {
	transport := newFakeTransport()
	transport.On("ls -d", "", nil)
	agent := NewAgent(&AgentConfig{Transport: transport})
	r := agent.DiscoverCheckpoints(context.Background())
	requireResultOK(t, r)
	cps := r.Value.([]Checkpoint)
	core.AssertEmpty(t, cps)
}

func TestAgent_Agent_Influx_Good(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	agent := NewAgent(&AgentConfig{InfluxURL: "http://127.0.0.1:1", InfluxDB: "db"})
	core.AssertNotNil(t, agent.Influx())
}

func TestAgent_Agent_Influx_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	agent := NewAgent(&AgentConfig{})
	core.AssertNotNil(t, agent.Influx())
}

func TestAgent_Agent_Influx_Ugly(t *core.T) {
	agent := NewAgent(&AgentConfig{InfluxURL: "http://127.0.0.1:1"})
	first := agent.Influx()
	second := agent.Influx()
	core.AssertEqual(t, first, second)
}
