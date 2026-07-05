// SPDX-Licence-Identifier: EUPL-1.2

// Package tune is the MTP-draft-block autotune business logic, rescued out of
// go-mlx's cmd/mlx tune command so it lives in a go-inference library rather
// than dying with go-mlx's cmd/. cmd/lem tune is thin flag-parsing over
// RunTune.
//
// go-mlx's tune loads a speculative TARGET+DRAFT pair (mlx.LoadSpeculativePair),
// measures plain AR decode against each MTP draft block on the real model, and
// persists the winner as a tuning profile serve auto-applies. go-inference does
// NOT yet expose a speculative-pair loader on any registered engine — the
// dappco.re/go/inference/generate package documents the same gap ("this engine
// exposes no speculative path"). So RunTune reproduces the full tune flag
// surface and its drafter detection + profile-dir plumbing, but reports the
// sweep honestly as blocked on that engine seam rather than faking a
// measurement. When a speculative loader lands, the sweep lights up here.
//
//	tune.RunTune(ctx, tune.Config{ModelPath: dir, Depths: "4,5,6", Out: os.Stdout, Log: os.Stderr})
package tune

import (
	"context"
	"io"

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

// RunTune resolves the drafter for the target and reports the tune plan. The
// MTP block sweep itself is blocked on a speculative-pair engine seam not yet in
// go-inference (see the package doc); RunTune validates inputs, detects the
// drafter, and reports honestly rather than persisting a faked profile.
func RunTune(ctx context.Context, cfg Config) error {
	_ = ctx
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
	core.Print(cfg.Out, "tune: the MTP draft-block sweep needs a speculative-pair loader; no registered go-inference engine exposes one yet, so no measurement was run and no profile was written")
	core.Print(cfg.Out, "tune: serve still auto-applies any profile already present in %s (--no-auto-profile opts out)", dir)
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
	for _, w := range inference.DefaultTuningWorkloads() {
		if w == workload {
			return true
		}
	}
	return false
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

// standardTuningProfileDir is ~/Lethean/data/tuning — where tune would write and
// serve reads profiles (serving keeps its own unexported copy of this default).
func standardTuningProfileDir() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "data", "tuning")
}
