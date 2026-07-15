package score

import (
	"maps"
	"slices"

	"dappco.re/go"
)

// RunCompare reads two score files and prints a comparison table for each
// model showing Old, New, and Delta values for every metric.
func RunCompare(oldPath, newPath string) core.Result {
	oldResult := ReadScorerOutput(oldPath)
	if !oldResult.OK {
		return core.Fail(core.E("score.RunCompare", "read old file", oldResult.Value.(error)))
	}
	oldOutput := oldResult.Value.(*ScorerOutput)

	newResult := ReadScorerOutput(newPath)
	if !newResult.OK {
		return core.Fail(core.E("score.RunCompare", "read new file", newResult.Value.(error)))
	}
	newOutput := newResult.Value.(*ScorerOutput)

	// Collect all models present in both files.
	models := make(map[string]bool)
	for m := range oldOutput.ModelAverages {
		models[m] = true
	}
	for m := range newOutput.ModelAverages {
		models[m] = true
	}

	// Sort model names for deterministic output.
	for _, model := range slices.Sorted(maps.Keys(models)) {
		oldAvgs := oldOutput.ModelAverages[model]
		newAvgs := newOutput.ModelAverages[model]

		if oldAvgs == nil && newAvgs == nil {
			continue
		}

		core.Print(nil, "")
		core.Print(nil, "Model: %s", model)
		core.Print(nil, "%-25s %11s  %11s  %6s", "", "Old", "New", "Delta")

		// Collect all metrics from both old and new.
		metrics := make(map[string]bool)
		for k := range oldAvgs {
			metrics[k] = true
		}
		for k := range newAvgs {
			metrics[k] = true
		}

		for _, metric := range slices.Sorted(maps.Keys(metrics)) {
			oldVal := oldAvgs[metric]
			newVal := newAvgs[metric]
			delta := newVal - oldVal

			deltaStr := core.Sprintf("%+.2f", delta)

			core.Print(nil, "%-25s %11.2f  %11.2f  %6s", metric, oldVal, newVal, deltaStr)
		}
	}

	return core.Ok(nil)
}
