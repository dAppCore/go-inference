package agent

import (
	"bufio"
	"context"
	"io"
	"runtime"

	core "dappco.re/go"
	"dappco.re/go/inference/engine/capability"
	"dappco.re/go/inference/eval/datapipe"
	"dappco.re/go/inference/model/modelmgmt"
	"dappco.re/go/inference/eval/score"
	"dappco.re/go/inference/serving"
	coreio "dappco.re/go/io"
)

// ProbeResult holds the result of running all probes against a checkpoint.
type ProbeResult struct {
	Accuracy   float64                      `json:"accuracy"`
	Correct    int                          `json:"correct"`
	Total      int                          `json:"total"`
	ByCategory map[string]CategoryResult    `json:"by_category"`
	Probes     map[string]SingleProbeResult `json:"probes"`
}

// CategoryResult holds pass/fail counts for a probe category.
type CategoryResult struct {
	Correct int `json:"correct"`
	Total   int `json:"total"`
}

// SingleProbeResult holds the result of a single probe.
type SingleProbeResult struct {
	Passed   bool   `json:"passed"`
	Response string `json:"response"`
}

// ProbeCallback is called after each probe completes for real-time streaming.
type ProbeCallback func(probeID, category string, passed bool, response string, correct, total int)

// CapResponseEntry holds a capability probe response with its metadata for judge scoring.
type CapResponseEntry struct {
	ProbeID  string
	Category string
	Prompt   string
	Answer   string
	Response string
	Passed   bool
}

// ContentResponse holds a content probe response for later judging.
type ContentResponse struct {
	Probe    score.ContentProbe
	Response string
}

// probeRunnerResponse is the JSON response from the Python probe runner.
type probeRunnerResponse struct {
	Response string  `json:"response"`
	Error    string  `json:"error"`
	Elapsed  float64 `json:"elapsed"`
}

// probeRunnerRequest is the JSON request sent to the Python probe runner.
// Field order is the JSON-sorted key order (max_tokens, prompt, temp) so the
// marshalled bytes are identical to the previous map[string]any literal,
// while avoiding the per-probe map allocation and interface boxing.
type probeRunnerRequest struct {
	MaxTokens int     `json:"max_tokens"`
	Prompt    string  `json:"prompt"`
	Temp      float64 `json:"temp"`
}

// processMLXNative scores a checkpoint using Ollama on M3.
func processMLXNative(cfg *AgentConfig, influx *datapipe.InfluxClient, cp Checkpoint) core.Result {
	ollamaBase, ok := modelmgmt.OllamaBaseModelMap[cp.ModelTag]
	if !ok {
		return core.Fail(core.E("agent.processMLXNative", core.Sprintf("unknown Ollama model for tag %s", cp.ModelTag), nil))
	}
	hfBase := modelmgmt.HFBaseModelMap[cp.ModelTag]
	if hfBase == "" {
		hfBase = ollamaBase
	}

	tempModel := core.Sprintf("lem-%s-%d", cp.ModelTag, cp.Iteration)
	localAdapterDir := core.JoinPath(cfg.WorkDir, core.Concat("adapter-", cp.Dirname))
	peftDir := core.JoinPath(cfg.WorkDir, core.Concat("peft-", cp.Dirname))

	coreio.Local.EnsureDir(localAdapterDir)

	defer func() {
		coreio.Local.DeleteAll(localAdapterDir)
		coreio.Local.DeleteAll(peftDir)
		modelmgmt.OllamaDeleteModel(cfg.JudgeURL, tempModel)
	}()

	core.Print(nil, "Fetching adapter from M3 (%s)...", cp.Filename)
	remoteSF := core.Sprintf("%s/%s", cp.RemoteDir, cp.Filename)
	remoteCfg := core.Sprintf("%s/adapter_config.json", cp.RemoteDir)
	localSF := core.JoinPath(localAdapterDir, cp.Filename)
	localCfg := core.JoinPath(localAdapterDir, "adapter_config.json")

	ctx := context.Background()
	t := cfg.transport()
	if r := t.CopyFrom(ctx, remoteSF, localSF); !r.OK {
		return core.Fail(core.E("agent.processMLXNative", "scp safetensors", r.Value.(error)))
	}
	if r := t.CopyFrom(ctx, remoteCfg, localCfg); !r.OK {
		return core.Fail(core.E("agent.processMLXNative", "scp config", r.Value.(error)))
	}

	core.Print(nil, "Converting MLX → PEFT format...")
	if result := modelmgmt.ConvertMLXtoPEFT(localSF, localCfg, peftDir, hfBase); !result.OK {
		return core.Fail(core.E("agent.processMLXNative", "convert adapter", result.Value.(error)))
	}

	core.Print(nil, "Creating Ollama model %s (base: %s)...", tempModel, ollamaBase)
	if result := modelmgmt.OllamaCreateModel(cfg.JudgeURL, tempModel, ollamaBase, peftDir); !result.OK {
		return core.Fail(core.E("agent.processMLXNative", "ollama create", result.Value.(error)))
	}
	core.Print(nil, "Ollama model %s ready", tempModel)
	probeBackend := serving.NewHTTPBackend(cfg.JudgeURL, tempModel)

	results, fullResponses := RunCapabilityProbesFull(ctx, probeBackend, func(probeID, category string, passed bool, response string, correct, total int) {
		passedInt := 0
		if passed {
			passedInt = 1
		}
		ts := (EpochBase + int64(cp.Iteration)*1000 + int64(total+100)) * 1_000_000_000
		line := core.Sprintf(
			MeasurementProbeScore+",model=%s,run_id=%s,label=%s,probe_id=%s passed=%di,iteration=%di %d",
			datapipe.EscapeLp(cp.ModelTag), datapipe.EscapeLp(cp.RunID), datapipe.EscapeLp(cp.Label), datapipe.EscapeLp(probeID),
			passedInt, cp.Iteration, ts,
		)
		if r := influx.WriteLp([]string{line}); !r.OK {
			core.Print(nil, "  [%s] InfluxDB stream failed: %v", probeID, r.Error())
		}
	})

	core.Print(nil, "Capability: %s -- %.1f%% (%d/%d)",
		cp.Label, results.Accuracy, results.Correct, results.Total)

	if r := PushCapabilitySummary(influx, cp, results); !r.OK {
		core.Print(nil, "InfluxDB summary push failed, buffering: %v", r.Error())
		BufferInfluxResult(cfg.WorkDir, cp, results)
	}
	PushCapabilityResultsDB(cfg.DBPath, cp, results)

	judgeBackend := serving.NewHTTPBackend(cfg.JudgeURL, cfg.JudgeModel)
	judge := score.NewJudge(judgeBackend)

	core.Print(nil, "Judging %d capability responses (0-10 quality scoring)...", len(fullResponses))
	ScoreCapabilityAndPush(ctx, judge, influx, cp, fullResponses)

	core.Print(nil, "Running %d content probes (0-10 judge scoring)...", len(score.ContentProbes))
	contentResponses := RunContentProbesViaAPI(ctx, probeBackend)
	if len(contentResponses) > 0 {
		contentRunID := core.Replace(cp.RunID, "-capability-", "-content-")
		ScoreContentAndPush(ctx, judge, influx, cp, contentRunID, contentResponses)
	}

	return core.Ok(nil)
}

// processWithConversion fetches adapter locally, converts MLX→PEFT, and scores.
func processWithConversion(cfg *AgentConfig, influx *datapipe.InfluxClient, cp Checkpoint) core.Result {
	localAdapterDir := core.JoinPath(cfg.WorkDir, cp.Dirname)
	coreio.Local.EnsureDir(localAdapterDir)

	localSF := core.JoinPath(localAdapterDir, cp.Filename)
	localCfg := core.JoinPath(localAdapterDir, "adapter_config.json")

	defer func() {
		coreio.Local.Delete(localSF)
		coreio.Local.Delete(localCfg)
		peftDir := core.JoinPath(cfg.WorkDir, core.Sprintf("peft_%07d", cp.Iteration))
		coreio.Local.DeleteAll(peftDir)
	}()

	core.Print(nil, "Fetching adapter from M3...")
	remoteSF := core.Sprintf("%s/%s", cp.RemoteDir, cp.Filename)
	remoteCfg := core.Sprintf("%s/adapter_config.json", cp.RemoteDir)

	ctx := context.Background()
	t := cfg.transport()
	if r := t.CopyFrom(ctx, remoteSF, localSF); !r.OK {
		return core.Fail(core.E("agent.processWithConversion", "scp safetensors", r.Value.(error)))
	}
	if r := t.CopyFrom(ctx, remoteCfg, localCfg); !r.OK {
		return core.Fail(core.E("agent.processWithConversion", "scp config", r.Value.(error)))
	}

	core.Print(nil, "Converting MLX to PEFT format...")
	peftDir := core.JoinPath(cfg.WorkDir, core.Sprintf("peft_%07d", cp.Iteration))
	if result := modelmgmt.ConvertMLXtoPEFT(localSF, localCfg, peftDir, cfg.BaseModel); !result.OK {
		return core.Fail(core.E("agent.processWithConversion", "convert adapter", result.Value.(error)))
	}

	core.Print(nil, "Running %d capability probes...", len(capability.CapabilityProbes))
	modelName := cfg.Model
	if modelName == "" {
		modelName = cp.ModelTag
	}
	backend := serving.NewHTTPBackend(cfg.APIURL, modelName)

	results := RunCapabilityProbes(ctx, backend)

	core.Print(nil, "Result: %s -- %.1f%% (%d/%d)",
		cp.Label, results.Accuracy, results.Correct, results.Total)

	if r := PushCapabilityResults(influx, cp, results); !r.OK {
		core.Print(nil, "InfluxDB push failed, buffering: %v", r.Error())
		BufferInfluxResult(cfg.WorkDir, cp, results)
	}
	PushCapabilityResultsDB(cfg.DBPath, cp, results)

	return core.Ok(nil)
}

// RunCapabilityProbes runs all capability probes against a backend.
func RunCapabilityProbes(ctx context.Context, backend serving.Backend) ProbeResult {
	results := ProbeResult{
		ByCategory: make(map[string]CategoryResult),
		Probes:     make(map[string]SingleProbeResult),
	}

	correct := 0
	total := 0

	for _, probe := range capability.CapabilityProbes {
		rGen := backend.Generate(ctx, probe.Prompt, serving.GenOpts{Temperature: CapabilityTemperature, MaxTokens: CapabilityMaxTokens})
		if !rGen.OK {
			core.Print(nil, "  [%s] ERROR: %v", probe.ID, rGen.Error())
			results.Probes[probe.ID] = SingleProbeResult{Passed: false, Response: rGen.Error()}
			total++
			cat := results.ByCategory[probe.Category]
			cat.Total++
			results.ByCategory[probe.Category] = cat
			runtime.GC()
			continue
		}
		res := rGen.Value.(serving.Result)

		clean := capability.StripThinkBlocks(res.Text)
		passed := probe.Check(clean)
		total++
		if passed {
			correct++
		}

		cat := results.ByCategory[probe.Category]
		cat.Total++
		if passed {
			cat.Correct++
		}
		results.ByCategory[probe.Category] = cat

		stored := clean
		if len(stored) > MaxStoredResponseLen {
			stored = stored[:MaxStoredResponseLen]
		}
		results.Probes[probe.ID] = SingleProbeResult{Passed: passed, Response: stored}

		status := "FAIL"
		if passed {
			status = "PASS"
		}
		core.Print(nil, "  [%s] %s (expected: %s)", probe.ID, status, probe.Answer)
		runtime.GC()
	}

	if total > 0 {
		results.Accuracy = float64(correct) / float64(total) * 100
	}
	results.Correct = correct
	results.Total = total

	return results
}

// RunCapabilityProbesFull runs all probes via a backend and returns both
// aggregate results and full responses for judge scoring.
func RunCapabilityProbesFull(ctx context.Context, backend serving.Backend, onProbe ProbeCallback) (ProbeResult, []CapResponseEntry) {
	results := ProbeResult{
		ByCategory: make(map[string]CategoryResult),
		Probes:     make(map[string]SingleProbeResult),
	}
	fullResponses := make([]CapResponseEntry, 0, len(capability.CapabilityProbes))

	correct := 0
	total := 0

	for _, probe := range capability.CapabilityProbes {
		rGen := backend.Generate(ctx, probe.Prompt, serving.GenOpts{Temperature: CapabilityTemperature, MaxTokens: CapabilityMaxTokens})
		var response string
		if !rGen.OK {
			core.Print(nil, "  [%s] ERROR: %v", probe.ID, rGen.Error())
			response = core.Sprintf("ERROR: %v", rGen.Error())
		} else {
			response = rGen.Value.(serving.Result).Text
		}

		clean := capability.StripThinkBlocks(response)
		passed := probe.Check(clean)
		total++
		if passed {
			correct++
		}

		cat := results.ByCategory[probe.Category]
		cat.Total++
		if passed {
			cat.Correct++
		}
		results.ByCategory[probe.Category] = cat

		stored := clean
		if len(stored) > MaxStoredResponseLen {
			stored = stored[:MaxStoredResponseLen]
		}
		results.Probes[probe.ID] = SingleProbeResult{Passed: passed, Response: stored}

		fullResponses = append(fullResponses, CapResponseEntry{
			ProbeID:  probe.ID,
			Category: probe.Category,
			Prompt:   probe.Prompt,
			Answer:   probe.Answer,
			Response: clean,
			Passed:   passed,
		})

		status := "FAIL"
		if passed {
			status = "PASS"
		}
		core.Print(nil, "  [%s] %s (expected: %s)", probe.ID, status, probe.Answer)

		if onProbe != nil {
			onProbe(probe.ID, probe.Category, passed, stored, correct, total)
		}
		runtime.GC()
	}

	if total > 0 {
		results.Accuracy = float64(correct) / float64(total) * 100
	}
	results.Correct = correct
	results.Total = total

	return results, fullResponses
}

// RunContentProbesViaAPI runs content probes via a backend.
func RunContentProbesViaAPI(ctx context.Context, backend serving.Backend) []ContentResponse {
	responses := make([]ContentResponse, 0, len(score.ContentProbes))

	for _, probe := range score.ContentProbes {
		rGen := backend.Generate(ctx, probe.Prompt, serving.GenOpts{Temperature: ContentTemperature, MaxTokens: ContentMaxTokens})
		if !rGen.OK {
			core.Print(nil, "  [content:%s] ERROR: %v", probe.ID, rGen.Error())
			runtime.GC()
			continue
		}

		reply := capability.StripThinkBlocks(rGen.Value.(serving.Result).Text)
		core.Print(nil, "  [content:%s] got %d chars", probe.ID, len(reply))

		responses = append(responses, ContentResponse{
			Probe:    probe,
			Response: reply,
		})
		runtime.GC()
	}

	return responses
}

// RunContentProbes runs content probes via a backend.
//
// Deprecated: use RunContentProbesViaAPI. This alias remains for the older
// architecture/docs references that still use the shorter name.
func RunContentProbes(ctx context.Context, backend serving.Backend) []ContentResponse {
	return RunContentProbesViaAPI(ctx, backend)
}

// RunContentProbesViaRunner sends content probes through an SSH probe runner.
func RunContentProbesViaRunner(stdin io.WriteCloser, scanner *bufio.Scanner) []ContentResponse {
	responses := make([]ContentResponse, 0, len(score.ContentProbes))

	for _, probe := range score.ContentProbes {
		reqJSON := core.JSONMarshalString(probeRunnerRequest{
			MaxTokens: ContentMaxTokens,
			Prompt:    probe.Prompt,
			Temp:      ContentTemperature,
		})
		io.WriteString(stdin, core.Sprintf("%s\n", reqJSON))

		var response string
		if scanner.Scan() {
			var resp probeRunnerResponse
			if r := core.JSONUnmarshalString(string(scanner.Bytes()), &resp); !r.OK {
				core.Print(nil, "  [content:%s] parse error: %v", probe.ID, r.Value.(error))
				runtime.GC()
				continue
			} else if resp.Error != "" {
				core.Print(nil, "  [content:%s] ERROR: %s", probe.ID, resp.Error)
				runtime.GC()
				continue
			} else {
				response = resp.Response
			}
		} else {
			core.Print(nil, "  [content:%s] no response from runner", probe.ID)
			runtime.GC()
			continue
		}

		response = capability.StripThinkBlocks(response)
		core.Print(nil, "  [content:%s] got %d chars", probe.ID, len(response))

		responses = append(responses, ContentResponse{
			Probe:    probe,
			Response: response,
		})
		runtime.GC()
	}

	return responses
}
