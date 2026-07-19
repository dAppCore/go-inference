// SPDX-Licence-Identifier: EUPL-1.2

package tune

import (
	"bytes"
	"context"
	"iter"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// TestTune_ParseDraftBlocks_Good pins the parse of the --depths surface: the
// empty default, spaces around entries, and skipped empty commas.
func TestTune_ParseDraftBlocks_Good(t *testing.T) {
	cases := []struct {
		in   string
		want []int
	}{
		{"", []int{4, 5, 6}},          // empty defaults to 4,5,6
		{"4,5,6", []int{4, 5, 6}},     // plain
		{" 3, 4 ,8 ", []int{3, 4, 8}}, // whitespace trimmed per entry
		{"5,,6", []int{5, 6}},         // empty entries skipped
		{"2", []int{2}},               // lower bound
		{"8", []int{8}},               // upper bound
	}
	for _, c := range cases {
		got, err := parseDraftBlocks(c.in)
		if err != nil {
			t.Fatalf("parseDraftBlocks(%q) = error %v, want %v", c.in, err, c.want)
		}
		if len(got) != len(c.want) {
			t.Fatalf("parseDraftBlocks(%q) = %v, want %v", c.in, got, c.want)
		}
		for i := range got {
			if got[i] != c.want[i] {
				t.Fatalf("parseDraftBlocks(%q) = %v, want %v", c.in, got, c.want)
			}
		}
	}
}

// TestTune_ParseDraftBlocks_Bad pins the rejections: out-of-range blocks
// (a block of 1 has no proposals to verify; >8 is out of MTP range), a
// non-numeric entry, and a value that yields no blocks at all.
func TestTune_ParseDraftBlocks_Bad(t *testing.T) {
	for _, in := range []string{"1", "9", "abc", ", ,"} {
		if got, err := parseDraftBlocks(in); err == nil {
			t.Fatalf("parseDraftBlocks(%q) = %v, nil error, want rejection", in, got)
		}
	}
}

// TestTune_DraftFlag_Good pins the --draft default: blank (or whitespace) means
// "auto" (the reactive ladder), an explicit path passes through untouched.
func TestTune_DraftFlag_Good(t *testing.T) {
	cases := []struct {
		in, want string
	}{
		{"", "auto"},
		{"   ", "auto"},
		{"/models/draft", "/models/draft"},
	}
	for _, c := range cases {
		if got := draftFlag(c.in); got != c.want {
			t.Fatalf("draftFlag(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestTune_ValidWorkload_Good pins the standard-set membership check: a workload
// from DefaultTuningWorkloads is accepted, an unknown one is rejected.
func TestTune_ValidWorkload_Good(t *testing.T) {
	if !validWorkload(inference.TuningWorkloadChat) {
		t.Fatalf("validWorkload(%q) = false, want true", inference.TuningWorkloadChat)
	}
	if validWorkload(inference.TuningWorkload("not-a-workload")) {
		t.Fatal("validWorkload(\"not-a-workload\") = true, want false")
	}
}

// TestTune_StandardTuningProfileDir_Good pins the profile-dir default shape:
// <HOME>/Lethean/lem/tuning, the path serve reads profiles from.
func TestTune_StandardTuningProfileDir_Good(t *testing.T) {
	dir := standardTuningProfileDir()
	want := core.PathJoin("Lethean", "lem", "tuning")
	if !core.HasSuffix(dir, want) {
		t.Fatalf("standardTuningProfileDir() = %q, want a path ending in %q", dir, want)
	}
}

// TestTune_RunTune_Bad pins the input-validation arms: a missing --model, an
// unsupported workload, and an out-of-range --depths each error before any
// drafter detection runs.
func TestTune_RunTune_Bad(t *testing.T) {
	if err := RunTune(context.Background(), Config{}); err == nil {
		t.Fatal("RunTune(empty model) = nil, want --model required error")
	}
	if err := RunTune(context.Background(), Config{ModelPath: "/tmp/x", Workload: "bogus"}); err == nil {
		t.Fatal("RunTune(bogus workload) = nil, want unsupported-workload error")
	}
	if err := RunTune(context.Background(), Config{ModelPath: "/tmp/x", Depths: "99"}); err == nil {
		t.Fatal("RunTune(out-of-range depths) = nil, want parse-depths error")
	}
}

// TestTune_RunTune_Ugly pins the no-drafter arm: a valid request whose model
// directory carries no detectable MTP drafter is reported as nothing-to-tune
// rather than silently succeeding.
func TestTune_RunTune_Ugly(t *testing.T) {
	// An empty directory is not a Gemma 4 family config, so the ladder stands
	// down and RunTune reports no drafter.
	err := RunTune(context.Background(), Config{ModelPath: t.TempDir()})
	if err == nil {
		t.Fatal("RunTune(no drafter) = nil, want no-MTP-drafter error")
	}
	if !core.Contains(err.Error(), "no MTP drafter") {
		t.Fatalf("RunTune(no drafter) error = %v, want a no-MTP-drafter message", err)
	}
}

// TestTune_RunTune_UglyDFlashDrafter pins the DFlash standdown: a detected
// DFlash block-diffusion drafter has no MTP verify-depth knob (its block size
// is fixed by the checkpoint), so RunTune reports the same honest DFlash
// notice generate/serve give and does not attempt the block sweep.
func TestTune_RunTune_UglyDFlashDrafter(t *testing.T) {
	modelDir := t.TempDir()
	writeFixture(t, core.PathJoin(modelDir, "config.json"), `{"model_type":"gemma4"}`)
	assistant := core.PathJoin(modelDir, "assistant")
	if r := core.MkdirAll(assistant, 0o755); !r.OK {
		t.Fatalf("mkdir assistant: %v", r.Value)
	}
	writeFixture(t, core.PathJoin(assistant, "config.json"), `{"speculators_model_type":"dflash","block_size":8}`)
	writeFixture(t, core.PathJoin(assistant, "model.safetensors"), "weights")

	var out bytes.Buffer
	err := RunTune(context.Background(), Config{ModelPath: modelDir, Depths: "4,5", Out: &out})
	if err != nil {
		t.Fatalf("RunTune(DFlash drafter) = %v, want nil", err)
	}
	if !core.Contains(out.String(), "does not apply to a DFlash drafter") {
		t.Fatalf("RunTune report missing the DFlash standdown notice:\n%s", out.String())
	}
}

// TestTune_RunTune_Good drives the seam-ABSENT path: a Gemma 4 target with an
// assistant/ drafter beside it resolves an ACTIVE drafter, but with no
// registered engine backend exposing inference.SpeculativePairBackend (this
// test binary never imports a concrete engine), RunTune reports that honestly
// and returns nil without writing a faked profile.
func TestTune_RunTune_Good(t *testing.T) {
	modelDir := t.TempDir()
	writeFixture(t, core.PathJoin(modelDir, "config.json"), `{"model_type":"gemma4"}`)
	assistant := core.PathJoin(modelDir, "assistant")
	if r := core.MkdirAll(assistant, 0o755); !r.OK {
		t.Fatalf("mkdir assistant: %v", r.Value)
	}
	writeFixture(t, core.PathJoin(assistant, "config.json"), `{"model_type":"gemma4"}`)
	writeFixture(t, core.PathJoin(assistant, "model.safetensors"), "weights")

	var out bytes.Buffer
	err := RunTune(context.Background(), Config{
		ModelPath: modelDir,
		Depths:    "4,5",
		Out:       &out,
	})
	if err != nil {
		t.Fatalf("RunTune(active drafter) = %v, want nil", err)
	}
	report := out.String()
	for _, want := range []string{
		"tune: target " + modelDir,
		"tune: drafter " + assistant,
		"no registered go-inference engine backend exposes a speculative-pair loader",
	} {
		if !core.Contains(report, want) {
			t.Fatalf("RunTune report missing %q\n--- got ---\n%s", want, report)
		}
	}
}

// TestTune_RunTune_GoodSeamPresent drives the seam-PRESENT path end to end
// with a fake backend var-swapped in for resolveSpeculativePairBackend (no
// global inference registry mutation — see the var's own doc): the sweep
// measures every requested block, picks the highest decode tok/s, and
// persists it as a tuning profile via serving.WriteTunedDraftBlockProfile.
func TestTune_RunTune_GoodSeamPresent(t *testing.T) {
	modelDir := t.TempDir()
	writeFixture(t, core.PathJoin(modelDir, "config.json"), `{"model_type":"gemma4"}`)
	assistant := core.PathJoin(modelDir, "assistant")
	if r := core.MkdirAll(assistant, 0o755); !r.OK {
		t.Fatalf("mkdir assistant: %v", r.Value)
	}
	writeFixture(t, core.PathJoin(assistant, "config.json"), `{"model_type":"gemma4"}`)
	writeFixture(t, core.PathJoin(assistant, "model.safetensors"), "weights")

	backend := &stubSpeculativePairBackend{perBlock: map[int]*stubSpeculativeTextModel{
		4: {metrics: inference.GenerateMetrics{DecodeTokensPerSec: 30}},
		5: {metrics: inference.GenerateMetrics{DecodeTokensPerSec: 45}}, // the winner
	}}
	orig := resolveSpeculativePairBackend
	resolveSpeculativePairBackend = func() (inference.SpeculativePairBackend, bool) { return backend, true }
	t.Cleanup(func() { resolveSpeculativePairBackend = orig })

	profileDir := t.TempDir()
	var out bytes.Buffer
	err := RunTune(context.Background(), Config{
		ModelPath:  modelDir,
		Depths:     "4,5",
		ProfileDir: profileDir,
		Prompt:     "hello",
		MaxTokens:  8,
		Out:        &out,
	})
	if err != nil {
		t.Fatalf("RunTune(seam present) = %v, want nil", err)
	}
	if !core.Contains(out.String(), "winner block 5") {
		t.Fatalf("RunTune report missing the winner line:\n%s", out.String())
	}
	if !backend.perBlock[4].closed || !backend.perBlock[5].closed {
		t.Fatal("every loaded pair must be closed after measurement")
	}

	files := core.PathGlob(core.JoinPath(profileDir, "*.json"))
	if len(files) != 1 {
		t.Fatalf("profile dir has %d *.json files, want exactly 1", len(files))
	}
	data := core.ReadFile(files[0])
	if !data.OK {
		t.Fatalf("read written profile: %v", data.Error())
	}
	var profile inference.TuningProfile
	if r := core.JSONUnmarshal(data.Value.([]byte), &profile); !r.OK {
		t.Fatalf("unmarshal written profile: %v", r.Error())
	}
	if profile.Key.Model.Path != modelDir {
		t.Fatalf("profile model path = %q, want %q", profile.Key.Model.Path, modelDir)
	}
	if profile.Candidate.Labels["mtp_draft_block"] != "5" {
		t.Fatalf("profile block label = %q, want %q", profile.Candidate.Labels["mtp_draft_block"], "5")
	}
}

func writeFixture(t *testing.T, path, content string) {
	t.Helper()
	if r := core.WriteFile(path, []byte(content), 0o644); !r.OK {
		t.Fatalf("write fixture %s: %v", path, r.Value)
	}
}

// stubSpeculativeTextModel is a minimal inference.TextModel for testing the
// sweep: Generate always yields one token (proving the drain loop runs), and
// Metrics/Err are pre-set per test case.
type stubSpeculativeTextModel struct {
	metrics inference.GenerateMetrics
	genErr  error // set: Err() reports it after Generate, as a failed run would
	closed  bool
}

func (m *stubSpeculativeTextModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		yield(inference.Token{ID: 1, Text: "x"})
	}
}
func (m *stubSpeculativeTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {}
}
func (m *stubSpeculativeTextModel) Classify(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.ClassifyResult(nil))
}
func (m *stubSpeculativeTextModel) BatchGenerate(ctx context.Context, prompts []string, opts ...inference.GenerateOption) core.Result {
	return core.Ok([]inference.BatchResult(nil))
}
func (m *stubSpeculativeTextModel) ModelType() string                  { return "stub" }
func (m *stubSpeculativeTextModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (m *stubSpeculativeTextModel) Metrics() inference.GenerateMetrics { return m.metrics }
func (m *stubSpeculativeTextModel) Err() core.Result {
	if m.genErr != nil {
		return core.Fail(m.genErr)
	}
	return core.Ok(nil)
}
func (m *stubSpeculativeTextModel) Close() core.Result { m.closed = true; return core.Ok(nil) }

var _ inference.TextModel = (*stubSpeculativeTextModel)(nil)

// stubSpeculativePairBackend implements inference.SpeculativePairBackend for
// tests: perBlock supplies the model to return for a given draft block,
// loadErr supplies a load failure for a given block; calls records every
// block LoadSpeculativePair was actually invoked with (so a test can assert a
// call was — or was not — made, e.g. after a cancelled context).
type stubSpeculativePairBackend struct {
	perBlock map[int]*stubSpeculativeTextModel
	loadErr  map[int]error
	calls    []int
}

func (b *stubSpeculativePairBackend) LoadSpeculativePair(targetPath, draftPath string, draftBlock int, opts ...inference.LoadOption) (inference.TextModel, error) {
	b.calls = append(b.calls, draftBlock)
	if err, ok := b.loadErr[draftBlock]; ok {
		return nil, err
	}
	if m, ok := b.perBlock[draftBlock]; ok {
		return m, nil
	}
	return nil, core.NewError("stubSpeculativePairBackend: unconfigured block")
}

var _ inference.SpeculativePairBackend = (*stubSpeculativePairBackend)(nil)

// TestTune_DefaultResolveSpeculativePairBackend_Bad pins the natural
// "nothing registered" state: train/tune never imports a concrete engine (the
// metal engine self-registers only behind the CLI's darwin/arm64 composition
// root), so the global inference registry is empty in this test binary and
// the default resolver reports ok=false.
func TestTune_DefaultResolveSpeculativePairBackend_Bad(t *testing.T) {
	if _, ok := defaultResolveSpeculativePairBackend(); ok {
		t.Fatal("defaultResolveSpeculativePairBackend() = ok=true with no engine registered, want false")
	}
}

// TestTune_MeasureDraftBlock_Good pins the successful measurement: the pair
// loads, Generate drains, and the engine's GenerateMetrics map onto
// TuningMeasurements field-for-field. The pair is closed afterwards.
func TestTune_MeasureDraftBlock_Good(t *testing.T) {
	model := &stubSpeculativeTextModel{metrics: inference.GenerateMetrics{
		PromptTokens: 10, GeneratedTokens: 20,
		PrefillTokensPerSec: 500, DecodeTokensPerSec: 45,
		TotalDuration: 2 * time.Second, PeakMemoryBytes: 1024, ActiveMemoryBytes: 512,
	}}
	backend := &stubSpeculativePairBackend{perBlock: map[int]*stubSpeculativeTextModel{5: model}}

	got, err := measureDraftBlock(context.Background(), backend, "/models/target", "/models/draft", "hi", 32, 5)
	if err != nil {
		t.Fatalf("measureDraftBlock = %v, want nil", err)
	}
	want := inference.TuningMeasurements{
		PromptTokens: 10, GeneratedTokens: 20,
		PrefillTokensPerSec: 500, DecodeTokensPerSec: 45,
		TotalMilliseconds: 2000, PeakMemoryBytes: 1024, ActiveMemoryBytes: 512,
	}
	if got != want {
		t.Fatalf("measureDraftBlock = %+v, want %+v", got, want)
	}
	if !model.closed {
		t.Fatal("measureDraftBlock must Close the loaded pair")
	}
	if len(backend.calls) != 1 || backend.calls[0] != 5 {
		t.Fatalf("LoadSpeculativePair calls = %v, want [5]", backend.calls)
	}
}

// TestTune_MeasureDraftBlock_Bad pins the load-failure arm: a backend that
// fails to load the pair (bad checkpoint, missing drafter) surfaces the error
// naming the load-pair step, with no metrics fabricated.
func TestTune_MeasureDraftBlock_Bad(t *testing.T) {
	backend := &stubSpeculativePairBackend{loadErr: map[int]error{5: core.NewError("attach drafter: no such file")}}

	_, err := measureDraftBlock(context.Background(), backend, "/models/target", "/models/draft", "hi", 32, 5)
	if err == nil {
		t.Fatal("measureDraftBlock over a load failure = nil, want an error")
	}
	if !core.Contains(err.Error(), "load speculative pair") {
		t.Fatalf("measureDraftBlock error = %v, want it to name the load-pair step", err)
	}
}

// TestTune_MeasureDraftBlock_Ugly pins the load-succeeds-but-generate-fails
// arm: the pair loads, but the run reports a failure via Err() (a decode
// error, a cancelled context) — measureDraftBlock surfaces it AND still closes
// the pair rather than leaking it.
func TestTune_MeasureDraftBlock_Ugly(t *testing.T) {
	model := &stubSpeculativeTextModel{genErr: core.NewError("decode: context cancelled")}
	backend := &stubSpeculativePairBackend{perBlock: map[int]*stubSpeculativeTextModel{5: model}}

	_, err := measureDraftBlock(context.Background(), backend, "/models/target", "/models/draft", "hi", 32, 5)
	if err == nil {
		t.Fatal("measureDraftBlock over a generate failure = nil, want an error")
	}
	if !model.closed {
		t.Fatal("measureDraftBlock must Close the pair even when generate fails")
	}
}

// TestTune_SweepDraftBlocks_Good pins the multi-block sweep: every block is
// measured in the given order, and a faster block scores higher.
func TestTune_SweepDraftBlocks_Good(t *testing.T) {
	backend := &stubSpeculativePairBackend{perBlock: map[int]*stubSpeculativeTextModel{
		4: {metrics: inference.GenerateMetrics{DecodeTokensPerSec: 30}},
		5: {metrics: inference.GenerateMetrics{DecodeTokensPerSec: 45}},
		6: {metrics: inference.GenerateMetrics{DecodeTokensPerSec: 40}},
	}}

	results := sweepDraftBlocks(context.Background(), backend, "/models/target", "/models/draft", "hi", 32, []int{4, 5, 6}, inference.TuningWorkloadChat)

	if len(results) != 3 {
		t.Fatalf("sweepDraftBlocks returned %d results, want 3", len(results))
	}
	for i, block := range []int{4, 5, 6} {
		if results[i].block != block {
			t.Fatalf("results[%d].block = %d, want %d (sweep order preserved)", i, results[i].block, block)
		}
		if results[i].err != nil {
			t.Fatalf("results[%d].err = %v, want nil", i, results[i].err)
		}
	}
	if results[1].score.Score <= results[0].score.Score {
		t.Fatalf("block 5 (45 tok/s) must outscore block 4 (30 tok/s): %+v vs %+v", results[1].score, results[0].score)
	}
}

// TestTune_SweepDraftBlocks_Bad pins the partial-failure arm: one block
// failing to load is recorded and skipped, and the remaining blocks are still
// measured rather than aborting the whole sweep.
func TestTune_SweepDraftBlocks_Bad(t *testing.T) {
	backend := &stubSpeculativePairBackend{
		perBlock: map[int]*stubSpeculativeTextModel{5: {metrics: inference.GenerateMetrics{DecodeTokensPerSec: 45}}},
		loadErr:  map[int]error{4: core.NewError("OOM at block 4")},
	}

	results := sweepDraftBlocks(context.Background(), backend, "/models/target", "/models/draft", "hi", 32, []int{4, 5}, inference.TuningWorkloadChat)

	if len(results) != 2 {
		t.Fatalf("sweepDraftBlocks returned %d results, want 2", len(results))
	}
	if results[0].err == nil {
		t.Fatal("block 4 must record its load failure")
	}
	if results[1].err != nil {
		t.Fatalf("block 5 must still measure despite block 4 failing: %v", results[1].err)
	}
}

// TestTune_SweepDraftBlocks_Ugly pins ctx cancellation: a context already
// cancelled before the sweep starts short-circuits every remaining block
// WITHOUT calling LoadSpeculativePair, each result carrying ctx's error.
func TestTune_SweepDraftBlocks_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	backend := &stubSpeculativePairBackend{} // any call would fail "unconfigured" — proving none happened

	results := sweepDraftBlocks(ctx, backend, "/models/target", "/models/draft", "hi", 32, []int{4, 5}, inference.TuningWorkloadChat)

	if len(results) != 2 {
		t.Fatalf("sweepDraftBlocks returned %d results, want 2 (still one per block)", len(results))
	}
	for i, r := range results {
		if r.err == nil {
			t.Fatalf("results[%d].err = nil, want context.Canceled", i)
		}
	}
	if len(backend.calls) != 0 {
		t.Fatalf("LoadSpeculativePair calls = %v, want none (cancellation short-circuits before loading)", backend.calls)
	}
}

// TestTune_BestBlockMeasurement_Good pins the winner selection: the highest
// score among successes wins, a failed candidate is never a contender.
func TestTune_BestBlockMeasurement_Good(t *testing.T) {
	results := []blockMeasurement{
		{block: 4, score: inference.TuningScore{Score: 30}},
		{block: 5, score: inference.TuningScore{Score: 45}},
		{block: 6, err: core.NewError("failed")},
	}
	best, ok := bestBlockMeasurement(results)
	if !ok {
		t.Fatal("bestBlockMeasurement = ok=false, want a winner among 2 successes")
	}
	if best.block != 5 {
		t.Fatalf("bestBlockMeasurement winner = block %d, want 5 (highest score)", best.block)
	}
}

// TestTune_BestBlockMeasurement_Bad pins the no-winner arms: an empty sweep
// and a sweep where every block failed both report ok=false.
func TestTune_BestBlockMeasurement_Bad(t *testing.T) {
	if _, ok := bestBlockMeasurement(nil); ok {
		t.Fatal("bestBlockMeasurement(nil) = ok=true, want false")
	}
	allFailed := []blockMeasurement{{block: 4, err: core.NewError("x")}, {block: 5, err: core.NewError("y")}}
	if _, ok := bestBlockMeasurement(allFailed); ok {
		t.Fatal("bestBlockMeasurement(all failed) = ok=true, want false")
	}
}
