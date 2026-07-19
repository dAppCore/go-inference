package agent

import (
	"context"
	"iter"
	"regexp"
	"slices"
	"strconv"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/eval/datapipe"
	coreio "dappco.re/go/io"
)

// checkpointNumRe extracts the numeric component of a checkpoint dir/file
// name. Hoisted to a package var so the pattern compiles once at init rather
// than on every DiscoverCheckpointsIter call.
var checkpointNumRe = regexp.MustCompile(`(\d+)`)

// RunAgentLoop is the main scoring agent loop.
func RunAgentLoop(cfg *AgentConfig) {
	core.Print(nil, repeatStr("=", LogSeparatorWidth))
	core.Print(nil, "ROCm Scoring Agent — Go Edition")
	core.Print(nil, "M3: %s@%s", cfg.M3User, cfg.M3Host)
	core.Print(nil, "Inference API: %s", cfg.APIURL)
	core.Print(nil, "Judge API: %s (%s)", cfg.JudgeURL, cfg.JudgeModel)
	core.Print(nil, "InfluxDB: %s/%s", cfg.InfluxURL, cfg.InfluxDB)
	if cfg.DBPath != "" {
		core.Print(nil, "DuckDB: %s", cfg.DBPath)
	}
	core.Print(nil, "Poll interval: %ds", cfg.PollInterval)
	core.Print(nil, repeatStr("=", LogSeparatorWidth))

	influx := datapipe.NewInfluxClient(cfg.InfluxURL, cfg.InfluxDB)
	coreio.Local.EnsureDir(cfg.WorkDir)

	for {
		ReplayInfluxBuffer(cfg.WorkDir, influx)

		core.Print(nil, "Discovering checkpoints on M3...")
		rDiscover := DiscoverCheckpoints(cfg)
		if !rDiscover.OK {
			core.Print(nil, "Discovery failed: %v", rDiscover.Error())
			if cfg.OneShot {
				return
			}
			time.Sleep(time.Duration(cfg.PollInterval) * time.Second)
			continue
		}
		checkpoints := rDiscover.Value.([]Checkpoint)
		core.Print(nil, "Found %d total checkpoints", len(checkpoints))

		var unscored []Checkpoint
		if cfg.Force {
			unscored = checkpoints
			core.Print(nil, "Force mode: scoring all %d checkpoints", len(unscored))
		} else {
			rScored := GetScoredLabels(influx)
			if !rScored.OK {
				core.Print(nil, "InfluxDB query failed: %v", rScored.Error())
			}
			var scored map[[2]string]bool
			if rScored.OK {
				scored = rScored.Value.(map[[2]string]bool)
			}
			core.Print(nil, "Already scored: %d (run_id, label) pairs", len(scored))
			unscored = FindUnscored(checkpoints, scored)
			core.Print(nil, "Unscored: %d checkpoints", len(unscored))
		}

		if len(unscored) == 0 {
			core.Print(nil, "Nothing to score. Sleeping %ds...", cfg.PollInterval)
			if cfg.OneShot {
				return
			}
			time.Sleep(time.Duration(cfg.PollInterval) * time.Second)
			continue
		}

		targets := unscored
		if !cfg.Force {
			targets = unscored[:1]
		}

		for i, target := range targets {
			core.Print(nil, "Grabbed: %s (%s) [%d/%d]", target.Label, target.Dirname, i+1, len(targets))

			if cfg.DryRun {
				core.Print(nil, "[DRY RUN] Would process: %s/%s", target.Dirname, target.Filename)
				continue
			}

			if r := ProcessOne(cfg, influx, target); !r.OK {
				core.Print(nil, "Error processing %s: %v", target.Label, r.Error())
			}
			time.Sleep(InterCheckpointDelay)
		}

		if cfg.DryRun || cfg.OneShot {
			return
		}
	}
}

// DiscoverCheckpoints lists all adapter directories and checkpoint files on M3 via SSH.
//
//	r := agent.DiscoverCheckpoints(cfg)
//	if !r.OK { return r }
//	cps := r.Value.([]agent.Checkpoint)
func DiscoverCheckpoints(cfg *AgentConfig) core.Result {
	var checkpoints []Checkpoint
	for cp, err := range DiscoverCheckpointsIter(cfg) {
		if err != nil {
			return core.Fail(err)
		}
		checkpoints = append(checkpoints, cp)
	}
	return core.Ok(checkpoints)
}

// DiscoverCheckpointsIter returns an iterator over discovered adapter checkpoints.
func DiscoverCheckpointsIter(cfg *AgentConfig) iter.Seq2[Checkpoint, error] {
	return func(yield func(Checkpoint, error) bool) {
		pattern := "adapters-*"
		if cfg.Filter != "" {
			pattern = "adapters-" + cfg.Filter + "*"
		}
		t := cfg.transport()
		ctx := context.Background()
		rOut := t.Run(ctx, core.Sprintf("ls -d %s/%s 2>/dev/null", cfg.M3AdapterBase, pattern))
		if !rOut.OK {
			yield(Checkpoint{}, core.E("agent.DiscoverCheckpointsIter", "list adapter dirs", rOut.Value.(error)))
			return
		}
		out := rOut.String()

		var adapterDirs []string
		for _, dirpath := range core.Split(core.Trim(out), "\n") {
			if dirpath == "" {
				continue
			}
			rSub := t.Run(ctx, core.Sprintf("ls -d %s/gemma-3-* 2>/dev/null", dirpath))
			if rSub.OK && core.Trim(rSub.String()) != "" {
				for _, sub := range core.Split(core.Trim(rSub.String()), "\n") {
					if sub != "" {
						adapterDirs = append(adapterDirs, sub)
					}
				}
			} else {
				adapterDirs = append(adapterDirs, dirpath)
			}
		}

		for _, dirpath := range adapterDirs {
			dirname := core.TrimPrefix(dirpath, core.Concat(cfg.M3AdapterBase, "/"))

			rFiles := t.Run(ctx, core.Sprintf("ls %s/*_adapters.safetensors 2>/dev/null", dirpath))
			if !rFiles.OK {
				continue
			}
			filesOut := rFiles.String()

			for _, fp := range core.Split(core.Trim(filesOut), "\n") {
				if fp == "" {
					continue
				}
				filename := fileBase(fp)

				match := checkpointNumRe.FindStringSubmatch(filename)
				if len(match) < 2 {
					continue
				}
				iteration, _ := strconv.Atoi(match[1])

				modelTag, labelPrefix, stem := AdapterMeta(dirname)
				label := core.Sprintf("%s @%s", labelPrefix, match[1])
				runID := core.Sprintf("%s-capability-auto", stem)

				if !yield(Checkpoint{
					RemoteDir: dirpath,
					Filename:  filename,
					Dirname:   dirname,
					Iteration: iteration,
					ModelTag:  modelTag,
					Label:     label,
					RunID:     runID,
				}, nil) {
					return
				}
			}
		}
	}
}

// GetScoredLabels returns all (run_id, label) pairs already scored in InfluxDB.
//
//	r := agent.GetScoredLabels(influx)
//	if !r.OK { return r }
//	scored := r.Value.(map[[2]string]bool)
func GetScoredLabels(influx *datapipe.InfluxClient) core.Result {
	r := influx.QuerySQL("SELECT DISTINCT run_id, label FROM " + MeasurementCapabilityScore)
	if !r.OK {
		return r
	}
	rows := r.Value.([]map[string]any)

	scored := make(map[[2]string]bool)
	for _, row := range rows {
		runID, _ := row["run_id"].(string)
		label, _ := row["label"].(string)
		if runID != "" && label != "" {
			scored[[2]string{runID, label}] = true
		}
	}
	return core.Ok(scored)
}

// FindUnscored filters checkpoints to only unscored ones, sorted by (dirname, iteration).
func FindUnscored(checkpoints []Checkpoint, scored map[[2]string]bool) []Checkpoint {
	// unscored is a subset of checkpoints, so len(checkpoints) is a safe
	// upper-bound capacity — one allocation instead of geometric regrowth
	// (each regrow also recopies the large Checkpoint structs).
	unscored := make([]Checkpoint, 0, len(checkpoints))
	for c := range FindUnscoredIter(checkpoints, scored) {
		unscored = append(unscored, c)
	}
	slices.SortFunc(unscored, func(a, b Checkpoint) int {
		if a.Dirname != b.Dirname {
			if a.Dirname < b.Dirname {
				return -1
			}
			return 1
		}
		return a.Iteration - b.Iteration
	})
	return unscored
}

// FindUnscoredIter returns an iterator over checkpoints that have not yet been scored.
func FindUnscoredIter(checkpoints []Checkpoint, scored map[[2]string]bool) iter.Seq[Checkpoint] {
	return func(yield func(Checkpoint) bool) {
		for _, c := range checkpoints {
			if !scored[[2]string{c.RunID, c.Label}] {
				if !yield(c) {
					return
				}
			}
		}
	}
}

// isMLXNative returns true if this model can be served directly on M3 via
// mlx_lm.server with --adapter, avoiding the MLX→PEFT conversion step.
func isMLXNative(modelTag string) bool {
	return core.HasPrefix(modelTag, "gemma-3-") || core.HasPrefix(modelTag, "gpt-oss")
}

// ProcessOne fetches, converts, scores, and pushes one checkpoint.
//
//	r := agent.ProcessOne(cfg, influx, cp)
//	if !r.OK { return r }
func ProcessOne(cfg *AgentConfig, influx *datapipe.InfluxClient, cp Checkpoint) core.Result {
	core.Print(nil, repeatStr("=", LogSeparatorWidth))
	core.Print(nil, "Processing: %s / %s [%s]", cp.Dirname, cp.Filename, cp.ModelTag)
	core.Print(nil, repeatStr("=", LogSeparatorWidth))

	if isMLXNative(cp.ModelTag) {
		return processMLXNative(cfg, influx, cp)
	}
	return processWithConversion(cfg, influx, cp)
}
