// SPDX-Licence-Identifier: EUPL-1.2

// Package tune is the MTP-draft-block autotune business logic, rescued out of
// go-mlx's cmd/mlx tune command so it lives in a go-inference library rather
// than dying with go-mlx's cmd/. cmd/lem tune is thin flag-parsing over
// RunTune.
//
// go-mlx's tune loaded a speculative TARGET+DRAFT pair (mlx.LoadSpeculativePair),
// measured decode tok/s against each MTP draft block on the real model, and
// persisted the winner as a tuning profile serve auto-applies. go-inference now
// exposes that pair loader as inference.SpeculativePairBackend — a capability a
// registered engine backend declares (the metal engine implements it in
// engine/metal/inference_register.go by delegating to its existing
// assistant-pair / composed-pair loading machinery). RunTune discovers the
// capability by type assertion on the default registered backend, exactly as
// inference.LoadModel discovers a plain-load backend (resolveSpeculativePairBackend
// below) — no concrete engine import here, so the package stays engine-neutral.
// A build with no such backend registered (or one whose registered backend has
// not implemented the capability) reports the gap honestly and writes no
// profile, rather than faking a measurement.
//
//	tune.RunTune(ctx, tune.Config{ModelPath: dir, Depths: "4,5,6", Out: os.Stdout, Log: os.Stderr})
package tune

import (
	"context"
	"io"
	"slices"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving"
)

// Config is the declarative tune request mirroring go-mlx's tune flag surface.
type Config struct {
	ModelPath  string
	DraftPath  string // "auto" runs the ladder, a path forces the drafter
	Depths     string // comma-separated draft blocks to sweep (2..8)
	MaxTokens  int
	Prompt     string
	Workload   string
	ProfileDir string
	JSON       bool

	Out io.Writer // summary
	Log io.Writer // notices
}

// RunTune resolves the drafter for the target and, when the registered engine
// backend exposes inference.SpeculativePairBackend, sweeps each requested MTP
// draft block on the real speculative pair: loading the pair, decoding
// cfg.Prompt greedily up to cfg.MaxTokens, and scoring the run
// (inference.ScoreTuningMeasurements) for cfg.Workload. The highest-scoring
// block is persisted as a tuning profile serve auto-applies
// (serving.WriteTunedDraftBlockProfile). With no such backend registered, it
// validates inputs, detects the drafter, and reports the gap honestly rather
// than persisting a faked profile.
func RunTune(ctx context.Context, cfg Config) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if core.Trim(cfg.ModelPath) == "" {
		return core.NewError("tune: --model is required")
	}
	workload := inference.TuningWorkload(core.Trim(cfg.Workload))
	if workload == "" {
		workload = inference.TuningWorkloadChat
	}
	if !validWorkload(workload) {
		return core.E("tune", core.Sprintf("unsupported workload %q", cfg.Workload), nil)
	}
	blocks, err := parseDraftBlocks(cfg.Depths)
	if err != nil {
		return core.E("tune", "parse depths", err)
	}

	detection := serving.ResolveServeDraft(cfg.ModelPath, draftFlag(cfg.DraftPath), true)
	if !detection.Active() {
		return core.E("tune", core.Sprintf("no MTP drafter found for %s — nothing to tune (place an assistant/ beside the target or pass --draft)", cfg.ModelPath), nil)
	}

	dir := core.Trim(cfg.ProfileDir)
	if dir == "" {
		dir = standardTuningProfileDir()
	}

	core.Print(cfg.Out, "tune: target %s", cfg.ModelPath)
	core.Print(cfg.Out, "tune: drafter %s (%s)", detection.DraftPath, detection.Note)
	core.Print(cfg.Out, "tune: workload %s  ·  blocks %v  ·  profile-dir %s", workload, blocks, dir)

	if detection.IsDFlash() {
		// A DFlash block-diffusion drafter has no MTP verify-depth knob — its
		// block size is fixed by the checkpoint, not something a caller
		// chooses — so sweeping cfg.Depths against it would just reload the
		// same drafter N times and measure noise. Report the same honest
		// notice generate/serve give a DFlash drafter and stand down.
		core.Print(cfg.Out, "tune: %s", serving.DFlashDraftNotice(detection))
		core.Print(cfg.Out, "tune: the MTP draft-block sweep does not apply to a DFlash drafter — nothing to tune")
		return nil
	}

	backend, ok := resolveSpeculativePairBackend()
	if !ok {
		core.Print(cfg.Out, "tune: no registered go-inference engine backend exposes a speculative-pair loader (inference.SpeculativePairBackend) — no measurement was run and no profile was written")
		return nil
	}

	maxTokens := cfg.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 256
	}
	results := sweepDraftBlocks(ctx, backend, cfg.ModelPath, detection.DraftPath, cfg.Prompt, maxTokens, blocks, workload)
	for _, r := range results {
		if r.err != nil {
			core.Print(cfg.Log, "tune: block %d failed: %v", r.block, r.err)
			continue
		}
		core.Print(cfg.Out, "tune: block %d — %.1f decode tok/s (score %.2f)", r.block, r.measurements.DecodeTokensPerSec, r.score.Score)
	}
	best, ok := bestBlockMeasurement(results)
	if !ok {
		return core.E("tune", core.Sprintf("every swept draft block failed to load or measure for %s", cfg.ModelPath), nil)
	}

	path, werr := serving.WriteTunedDraftBlockProfile(dir, cfg.ModelPath, "", workload, best.block, best.measurements, best.score, time.Now().Unix())
	if werr != nil {
		return core.E("tune", "write tuning profile", werr)
	}
	core.Print(cfg.Out, "tune: winner block %d (%.1f decode tok/s) — wrote %s", best.block, best.measurements.DecodeTokensPerSec, path)
	return nil
}

// draftFlag defaults an empty --draft to "auto" (the reactive ladder).
func draftFlag(v string) string {
	if core.Trim(v) == "" {
		return "auto"
	}
	return v
}

// validWorkload reports whether workload is one of the standard set.
func validWorkload(workload inference.TuningWorkload) bool {
	return slices.Contains(inference.DefaultTuningWorkloads(), workload)
}

// parseDraftBlocks parses --depths into draft blocks, bounded to the MTP block
// semantics (2..8: a block of 1 has no proposals to verify). Ported from
// go-mlx's cmd/mlx parseTuneDraftBlocks.
func parseDraftBlocks(value string) ([]int, error) {
	value = core.Trim(value)
	if value == "" {
		value = "4,5,6"
	}
	parts := core.Split(value, ",")
	blocks := make([]int, 0, len(parts))
	for _, part := range parts {
		part = core.Trim(part)
		if part == "" {
			continue
		}
		parsed := core.ParseInt(part, 10, 32)
		if !parsed.OK {
			return nil, core.Errorf("invalid draft block %q", part)
		}
		block := int(parsed.Value.(int64))
		if block < 2 || block > 8 {
			return nil, core.Errorf("draft block %d out of range 2..8", block)
		}
		blocks = append(blocks, block)
	}
	if len(blocks) == 0 {
		return nil, core.NewError("no draft blocks to sweep")
	}
	return blocks, nil
}

// standardTuningProfileDir is ~/Lethean/lem/tuning — where tune would write and
// serve reads profiles (serving keeps its own unexported copy of this default).
func standardTuningProfileDir() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "lem", "tuning")
}

// resolveSpeculativePairBackend finds the registered engine backend that can
// load a target+drafter pair as one speculative TextModel: the
// inference.SpeculativePairBackend capability, discovered by type assertion on
// the preference-order default backend (metal -> rocm -> llama_cpp) — the same
// resolution inference.LoadModel uses for a plain load. A package var (not a
// plain func) so tests can inject a fake backend without touching the
// process-global inference registry — see the var-swap tests in tune_test.go,
// the same injection shape cli/engine_metal.go uses for speculativeLoader. No
// registered backend at all, or a registered backend that does not implement
// the capability, both report ok=false so RunTune can name the gap honestly
// instead of faking a measurement.
var resolveSpeculativePairBackend = defaultResolveSpeculativePairBackend

func defaultResolveSpeculativePairBackend() (inference.SpeculativePairBackend, bool) {
	result := inference.Default()
	if !result.OK {
		return nil, false
	}
	b, ok := result.Value.(inference.Backend)
	if !ok || b == nil {
		return nil, false
	}
	spl, ok := b.(inference.SpeculativePairBackend)
	return spl, ok
}

// blockMeasurement is one swept draft block's outcome. err set means the pair
// failed to load or generate for this block — recorded and skipped rather than
// aborting the whole sweep (a wide block is more likely to exhaust memory than
// a narrow one, and that alone should not fail blocks that DID measure).
type blockMeasurement struct {
	block        int
	measurements inference.TuningMeasurements
	score        inference.TuningScore
	err          error
}

// sweepDraftBlocks loads the target+drafter pair once per block through
// backend, decodes prompt greedily up to maxTokens on each load, and scores
// the run for workload. ctx cancellation stops any remaining blocks (each
// already-measured result is kept, recorded with ctx's error).
func sweepDraftBlocks(ctx context.Context, backend inference.SpeculativePairBackend, modelPath, draftPath, prompt string, maxTokens int, blocks []int, workload inference.TuningWorkload) []blockMeasurement {
	results := make([]blockMeasurement, 0, len(blocks))
	for _, block := range blocks {
		if err := ctx.Err(); err != nil {
			results = append(results, blockMeasurement{block: block, err: err})
			continue
		}
		measurements, err := measureDraftBlock(ctx, backend, modelPath, draftPath, prompt, maxTokens, block)
		m := blockMeasurement{block: block, measurements: measurements, err: err}
		if err == nil {
			m.score = inference.ScoreTuningMeasurements(workload, measurements)
		}
		results = append(results, m)
	}
	return results
}

// measureDraftBlock loads target+drafter at block through backend, decodes
// prompt greedily to maxTokens, and reads the engine's own GenerateMetrics back
// as the neutral TuningMeasurements the sweep scores. The pair is closed before
// returning regardless of outcome — a wide block's resident weights must not
// accumulate across the sweep.
func measureDraftBlock(ctx context.Context, backend inference.SpeculativePairBackend, modelPath, draftPath, prompt string, maxTokens, block int) (inference.TuningMeasurements, error) {
	tm, err := backend.LoadSpeculativePair(modelPath, draftPath, block)
	if err != nil {
		return inference.TuningMeasurements{}, core.E("tune.measureDraftBlock", core.Sprintf("load speculative pair (block %d)", block), err)
	}
	defer tm.Close()
	// Greedy (temperature 0): the pair's verify-exact lane at temp 0 is
	// byte-identical to plain decode, so this is the "plain AR decode" reference
	// go-mlx's tune measured, run once per block rather than as a separate
	// unpaired baseline load.
	for range tm.Generate(ctx, prompt, inference.WithMaxTokens(maxTokens), inference.WithTemperature(0)) {
		// drain — the measurement is Metrics() after the run, not the text
	}
	if r := tm.Err(); !r.OK {
		return inference.TuningMeasurements{}, core.E("tune.measureDraftBlock", core.Sprintf("generate (block %d)", block), r.Value.(error))
	}
	metrics := tm.Metrics()
	return inference.TuningMeasurements{
		PromptTokens:        metrics.PromptTokens,
		GeneratedTokens:     metrics.GeneratedTokens,
		PrefillTokensPerSec: metrics.PrefillTokensPerSec,
		DecodeTokensPerSec:  metrics.DecodeTokensPerSec,
		TotalMilliseconds:   float64(metrics.TotalDuration.Milliseconds()),
		PeakMemoryBytes:     metrics.PeakMemoryBytes,
		ActiveMemoryBytes:   metrics.ActiveMemoryBytes,
	}, nil
}

// bestBlockMeasurement picks the highest-scoring successful measurement among
// results. ok is false when every block failed (or results is empty).
func bestBlockMeasurement(results []blockMeasurement) (blockMeasurement, bool) {
	var best blockMeasurement
	found := false
	for _, r := range results {
		if r.err != nil {
			continue
		}
		if !found || r.score.Score > best.score.Score {
			best = r
			found = true
		}
	}
	return best, found
}
