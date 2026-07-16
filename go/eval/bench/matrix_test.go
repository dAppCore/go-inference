// SPDX-Licence-Identifier: EUPL-1.2

package bench

import (
	"context"
	"strings"
	"testing"
	"time"

	core "dappco.re/go"
)

// matrix_test.go proves the matrix layer without any engine: config parse + validation, model-ref
// resolution, and the RunMatrix loop's lane derivation, warmup discipline, per-row error isolation
// and report assembly — all over a fake loader.

// TestMatrix_LoadMatrixConfig_Good parses a two-run config and pins the defaults: prompt/tokens/
// warmup filled, lane derivation (plain-only without a draft, plain+mtp with one), name defaulting.
func TestMatrix_LoadMatrixConfig_Good(t *testing.T) {
	cfg, err := LoadMatrixConfig([]byte(`{
		"runs": [
			{"model": "org/model-a"},
			{"name": "b", "model": "/abs/path/b", "draft": "org/model-b-assistant", "tokens": 64}
		]
	}`))
	if err != nil {
		t.Fatalf("LoadMatrixConfig: %v", err)
	}
	if cfg.Tokens != 512 || cfg.Warmup != 16 || cfg.Prompt == "" {
		t.Fatalf("defaults not applied: tokens=%d warmup=%d prompt=%q", cfg.Tokens, cfg.Warmup, cfg.Prompt)
	}
	a, b := cfg.Runs[0], cfg.Runs[1]
	if a.Name != "model-a" {
		t.Fatalf("run 0 name = %q, want the ref base", a.Name)
	}
	if len(a.Lanes) != 1 || a.Lanes[0] != MatrixLanePlain {
		t.Fatalf("run 0 lanes = %v, want plain only (no draft)", a.Lanes)
	}
	if len(b.Lanes) != 2 || b.Lanes[0] != MatrixLanePlain || b.Lanes[1] != MatrixLaneMTP {
		t.Fatalf("run 1 lanes = %v, want plain+mtp (draft set)", b.Lanes)
	}
	if b.Tokens != 64 {
		t.Fatalf("run 1 tokens = %d, want the per-run override 64", b.Tokens)
	}
}

// TestMatrix_LoadMatrixConfig_Bad pins the rejections: malformed JSON, an empty runs array, a run
// without a model, and an mtp lane without a draft.
func TestMatrix_LoadMatrixConfig_Bad(t *testing.T) {
	for name, data := range map[string]string{
		"malformed json":    `{not json`,
		"no runs":           `{"runs": []}`,
		"run without model": `{"runs": [{"name": "x"}]}`,
		"mtp without draft": `{"runs": [{"model": "org/m", "lanes": ["mtp"]}]}`,
	} {
		if _, err := LoadMatrixConfig([]byte(data)); err == nil {
			t.Fatalf("%s: expected an error", name)
		}
	}
}

// TestMatrix_LoadMatrixConfig_Ugly pins the unknown-lane edge: a lane name outside plain/mtp is a
// clean config error naming the run, not a silent skip at run time.
func TestMatrix_LoadMatrixConfig_Ugly(t *testing.T) {
	_, err := LoadMatrixConfig([]byte(`{"runs": [{"name": "x", "model": "org/m", "lanes": ["turbo"]}]}`))
	if err == nil || !strings.Contains(err.Error(), "turbo") {
		t.Fatalf("expected the unknown-lane error to name the lane, got %v", err)
	}
}

// TestMatrix_ResolveModelRef_Good resolves an existing path as-is (the temp dir stands in for a
// snapshot directory).
func TestMatrix_ResolveModelRef_Good(t *testing.T) {
	dir := t.TempDir()
	got, err := ResolveModelRef(dir)
	if err != nil {
		t.Fatalf("ResolveModelRef(%s): %v", dir, err)
	}
	if got != dir {
		t.Fatalf("ResolveModelRef = %q, want the path unchanged %q", got, dir)
	}
}

// TestMatrix_ResolveModelRef_Bad pins the miss shape: a ref that is neither a path nor a cached repo
// errors with the cache location it tried (the "pull it" hint), and an empty ref errors.
func TestMatrix_ResolveModelRef_Bad(t *testing.T) {
	if _, err := ResolveModelRef(""); err == nil {
		t.Fatal("empty ref: expected an error")
	}
	_, err := ResolveModelRef("no-such-org/no-such-model-xyz")
	if err == nil || !strings.Contains(err.Error(), "models--no-such-org--no-such-model-xyz") {
		t.Fatalf("expected the miss to name the cache spelling tried, got %v", err)
	}
}

// fakeMatrixModel scripts one loaded model: it records drains and reports canned metrics; failDrain
// makes the TIMED drain fail (the warmup succeeds) to prove per-row error isolation.
type fakeMatrixModel struct {
	drains    []int
	failDrain bool
	spec      *MatrixSpec
	closed    bool
}

func (f *fakeMatrixModel) Drain(_ context.Context, _ string, maxTokens int) error {
	f.drains = append(f.drains, maxTokens)
	if f.failDrain && len(f.drains) > 1 {
		return context.DeadlineExceeded
	}
	return nil
}

func (f *fakeMatrixModel) Metrics() GenerationMetrics {
	return GenerationMetrics{GeneratedTokens: f.drains[len(f.drains)-1], DecodeTokensPerSec: 123.4, PrefillTokensPerSec: 456.7, DecodeDuration: time.Second}
}

func (f *fakeMatrixModel) Close() error {
	f.closed = true
	return nil
}

func (f *fakeMatrixModel) SpeculativeSummary() (MatrixSpec, bool) {
	if f.spec == nil {
		return MatrixSpec{}, false
	}
	return *f.spec, true
}

// TestMatrix_RunMatrix_Good drives a two-lane run through a fake loader and pins the discipline: the
// warmup drain precedes the timed drain (16 then the budget), rows carry the driver-reported rate,
// the mtp lane carries the acceptance summary, models are closed, and the streamed table names both
// lanes.
func TestMatrix_RunMatrix_Good(t *testing.T) {
	target := t.TempDir()
	draft := t.TempDir()
	var loaded []*fakeMatrixModel
	load := func(_ context.Context, modelPath, draftPath string, _ int) (MatrixModel, error) {
		f := &fakeMatrixModel{}
		if draftPath != "" {
			f.spec = &MatrixSpec{ProposedTokens: 10, AcceptedTokens: 6, AcceptanceRate: 0.6}
		}
		loaded = append(loaded, f)
		return f, nil
	}
	var out strings.Builder
	report, err := RunMatrix(context.Background(), MatrixConfig{
		Tokens: 32,
		Runs:   []MatrixRun{{Name: "m", Model: target, Draft: draft, Lanes: []string{MatrixLanePlain, MatrixLaneMTP}}},
	}, load, &out)
	if err != nil {
		t.Fatalf("RunMatrix: %v", err)
	}
	if len(report.Rows) != 2 {
		t.Fatalf("rows = %d, want 2 (plain + mtp)", len(report.Rows))
	}
	plain, mtp := report.Rows[0], report.Rows[1]
	if plain.Lane != MatrixLanePlain || mtp.Lane != MatrixLaneMTP {
		t.Fatalf("lane order = %s,%s", plain.Lane, mtp.Lane)
	}
	if plain.DecodeTokensPerSec != 123.4 || plain.GeneratedTokens != 32 {
		t.Fatalf("plain row = %+v, want the driver metrics through", plain)
	}
	if plain.Spec != nil {
		t.Fatal("plain lane must carry no speculative summary")
	}
	if mtp.Spec == nil || mtp.Spec.AcceptanceRate != 0.6 {
		t.Fatalf("mtp row spec = %+v, want the 60%% acceptance summary", mtp.Spec)
	}
	for i, f := range loaded {
		if len(f.drains) != 2 || f.drains[0] != 16 || f.drains[1] != 32 {
			t.Fatalf("model %d drains = %v, want warmup 16 then budget 32", i, f.drains)
		}
		if !f.closed {
			t.Fatalf("model %d was not closed", i)
		}
	}
	if !strings.Contains(out.String(), "plain") || !strings.Contains(out.String(), "mtp") {
		t.Fatalf("streamed table missing lanes:\n%s", out.String())
	}
}

// TestMatrix_RunMatrix_Bad pins per-row error isolation: a run whose model ref cannot resolve and a
// run whose timed drain fails each produce an error ROW, while the healthy run still measures — the
// grid never aborts on one hole.
func TestMatrix_RunMatrix_Bad(t *testing.T) {
	good := t.TempDir()
	load := func(_ context.Context, modelPath, _ string, _ int) (MatrixModel, error) {
		return &fakeMatrixModel{failDrain: strings.Contains(modelPath, "fail")}, nil
	}
	failDir := core.PathJoin(t.TempDir(), "fail")
	if r := core.MkdirAll(failDir, 0o755); !r.OK {
		t.Fatal(r.Error())
	}
	report, err := RunMatrix(context.Background(), MatrixConfig{Runs: []MatrixRun{
		{Name: "missing", Model: "no-such/repo-at-all"},
		{Name: "drainfail", Model: failDir},
		{Name: "healthy", Model: good},
	}}, load, nil)
	if err != nil {
		t.Fatalf("RunMatrix: %v", err)
	}
	if report.Rows[0].Err == "" || report.Rows[1].Err == "" {
		t.Fatalf("expected error rows, got %+v", report.Rows[:2])
	}
	if report.Rows[2].Err != "" || report.Rows[2].DecodeTokensPerSec == 0 {
		t.Fatalf("healthy row damaged by earlier failures: %+v", report.Rows[2])
	}
}

// TestMatrix_RunMatrix_Ugly pins cancellation: a cancelled context stops the grid between rows with
// the context error, keeping the rows already measured.
func TestMatrix_RunMatrix_Ugly(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	dir := t.TempDir()
	calls := 0
	load := func(context.Context, string, string, int) (MatrixModel, error) {
		calls++
		cancel() // cancel after the first row's load — the second row must never start
		return &fakeMatrixModel{}, nil
	}
	report, err := RunMatrix(ctx, MatrixConfig{Runs: []MatrixRun{
		{Name: "first", Model: dir},
		{Name: "second", Model: dir},
	}}, load, nil)
	if err == nil {
		t.Fatal("expected the context error")
	}
	if calls != 1 || len(report.Rows) != 1 {
		t.Fatalf("calls=%d rows=%d, want the grid stopped after the first row", calls, len(report.Rows))
	}
}
