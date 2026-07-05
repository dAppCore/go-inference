package modelmgmt

import (
	"context"
	"time"

	"dappco.re/go"
	"dappco.re/go/inference/datapipe"
	"dappco.re/go/inference/score"
	"dappco.re/go/inference/serving"
	coreio "dappco.re/go/io"
)

// ExpandOutput is the JSONL output structure for expansion generation.
type ExpandOutput struct {
	ID             string  `json:"id"`
	Domain         string  `json:"domain,omitempty"`
	Prompt         string  `json:"prompt"`
	Response       string  `json:"response"`
	Model          string  `json:"model"`
	ElapsedSeconds float64 `json:"elapsed_seconds"`
	Chars          int     `json:"chars"`
}

// GetCompletedIDs queries InfluxDB for prompt IDs that have already been
// processed in the expansion_gen measurement.
func GetCompletedIDs(influx *datapipe.InfluxClient) core.Result {
	rRows := influx.QuerySQL("SELECT DISTINCT seed_id FROM expansion_gen")
	if !rRows.OK {
		return core.Fail(core.E("modelmgmt.GetCompletedIDs", "query expansion_gen", rRows.Value.(error)))
	}
	rows := rRows.Value.([]map[string]any)

	ids := make(map[string]bool, len(rows))
	for _, row := range rows {
		id := strVal(row, "seed_id")
		if id != "" {
			ids[id] = true
		}
	}

	return core.Ok(ids)
}

// ExpandPrompts generates responses for expansion prompts using the given
// backend and reports progress to InfluxDB. Already-completed prompts (per
// InfluxDB) are skipped. API errors for individual prompts are logged and
// skipped. InfluxDB reporting is best-effort.
func ExpandPrompts(ctx context.Context, backend serving.Backend, influx *datapipe.InfluxClient, prompts []score.Response,
	modelName, worker, outputDir string, dryRun bool, limit int) core.Result {

	remaining := prompts

	// Check InfluxDB for already-completed IDs.
	completedResult := GetCompletedIDs(influx)
	if !completedResult.OK {
		core.Print(nil, "warning: could not check completed IDs: %v", completedResult.Value)
	} else {
		completed := completedResult.Value.(map[string]bool)
		remaining = nil
		for _, p := range prompts {
			if !completed[p.ID] {
				remaining = append(remaining, p)
			}
		}

		skipped := len(prompts) - len(remaining)
		if skipped > 0 {
			core.Print(nil, "skipping %d already-completed prompts, %d remaining", skipped, len(remaining))
		}
	}

	if limit > 0 && limit < len(remaining) {
		remaining = remaining[:limit]
	}

	if len(remaining) == 0 {
		core.Print(nil, "all prompts already completed, nothing to do")
		return core.Ok(nil)
	}

	if dryRun {
		core.Print(nil, "dry-run: would process %d prompts with model %s (worker: %s)", len(remaining), modelName, worker)
		for i, p := range remaining {
			if i >= 10 {
				core.Print(nil, "  ... and %d more", len(remaining)-10)
				break
			}
			core.Print(nil, "  %s (domain: %s)", p.ID, p.Domain)
		}
		return core.Ok(nil)
	}

	outputPath := core.JoinPath(outputDir, core.Sprintf("expand-%s.jsonl", worker))
	f, err := coreio.Local.Append(outputPath)
	if err != nil {
		return core.Fail(core.E("modelmgmt.ExpandPrompts", "open output file", err))
	}
	defer f.Close()

	total := len(remaining)
	completedCount := 0

	for idx, p := range remaining {
		start := time.Now()
		rGen := backend.Generate(ctx, p.Prompt, serving.GenOpts{Temperature: 0.7, MaxTokens: 2048})
		elapsed := time.Since(start).Seconds()

		if !rGen.OK {
			core.Print(nil, "[%d/%d] id=%s ERROR: %s", idx+1, total, p.ID, rGen.Error())
			continue
		}
		res := rGen.Value.(serving.Result)

		response := res.Text
		chars := len(response)
		completedCount++

		out := ExpandOutput{
			ID:             p.ID,
			Domain:         p.Domain,
			Prompt:         p.Prompt,
			Response:       response,
			Model:          modelName,
			ElapsedSeconds: elapsed,
			Chars:          chars,
		}

		line := core.JSONMarshalString(out)
		if _, err := f.Write([]byte(core.Concat(line, "\n"))); err != nil {
			core.Print(nil, "[%d/%d] id=%s write error: %v", idx+1, total, p.ID, err)
			continue
		}

		genLine := core.Sprintf("expansion_gen,i=%d,w=%s,d=%s seed_id=\"%s\",gen_time=%f,chars=%di,model=\"%s\"",
			idx, datapipe.EscapeLp(worker), datapipe.EscapeLp(p.Domain),
			p.ID, elapsed, chars, modelName)

		pct := float64(completedCount) / float64(total) * 100.0
		progressLine := core.Sprintf("expansion_progress,worker=%s completed=%di,target=%di,pct=%f",
			datapipe.EscapeLp(worker), completedCount, total, pct)

		if rWrite := influx.WriteLp([]string{genLine, progressLine}); !rWrite.OK {
			core.Print(nil, "[%d/%d] id=%s influx write error: %s", idx+1, total, p.ID, rWrite.Error())
		}

		core.Print(nil, "[%d/%d] id=%s chars=%d time=%.1fs", idx+1, total, p.ID, chars, elapsed)
	}

	core.Print(nil, "expand complete: %d/%d prompts generated, output: %s", completedCount, total, outputPath)

	return core.Ok(nil)
}
