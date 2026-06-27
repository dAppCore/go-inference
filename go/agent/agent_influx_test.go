package agent

import (
	"context"
	"net/http"

	core "dappco.re/go"
	"dappco.re/go/inference/datapipe"
	"dappco.re/go/inference/score"
	"dappco.re/go/inference/serving"
	coreio "dappco.re/go/io"
)

func capabilityJudge() *score.Judge {
	return score.NewJudge(&testBackend{result: serving.Result{Text: `{"reasoning":7.0,"correctness":8.0,"clarity":9.0}`}})
}

func contentJudge() *score.Judge {
	return score.NewJudge(&testBackend{result: serving.Result{Text: `{"ccp_compliance":5,"truth_telling":5,"engagement":4,"axiom_integration":4,"sovereignty_reasoning":5,"emotional_register":4}`}})
}

func TestAgentInflux_ScoreCapabilityAndPush_Good(t *core.T) {
	influx, rec := newFakeInflux(t, nil, 0)
	responses := []CapResponseEntry{{ProbeID: "p1", Category: "math", Prompt: "2+2", Answer: "4", Response: "4"}}
	ScoreCapabilityAndPush(context.Background(), capabilityJudge(), influx, sampleCheckpoint(), responses)
	core.AssertEqual(t, 1, rec.writeCount())
}

func TestAgentInflux_ScoreCapabilityAndPush_Bad(t *core.T) {
	influx, rec := newFakeInflux(t, nil, 0)
	ScoreCapabilityAndPush(context.Background(), score.NewJudge(&testBackend{err: core.AnError}), influx, sampleCheckpoint(), []CapResponseEntry{{ProbeID: "p1"}})
	core.AssertEqual(t, 0, rec.writeCount())
}

func TestAgentInflux_ScoreCapabilityAndPush_Ugly(t *core.T) {
	influx, rec := newFakeInflux(t, nil, 0)
	ScoreCapabilityAndPush(context.Background(), capabilityJudge(), influx, sampleCheckpoint(), nil)
	core.AssertEqual(t, 0, rec.writeCount())
}

func TestAgentInflux_ScoreContentAndPush_Good(t *core.T) {
	influx, rec := newFakeInflux(t, nil, 0)
	responses := []ContentResponse{{Probe: score.ContentProbes[0], Response: "answer"}}
	ScoreContentAndPush(context.Background(), contentJudge(), influx, sampleCheckpoint(), "content-run", responses)
	core.AssertEqual(t, 1, rec.writeCount())
}

func TestAgentInflux_ScoreContentAndPush_Bad(t *core.T) {
	influx, rec := newFakeInflux(t, nil, 0)
	ScoreContentAndPush(context.Background(), score.NewJudge(&testBackend{err: core.AnError}), influx, sampleCheckpoint(), "content-run", []ContentResponse{{Probe: score.ContentProbes[0]}})
	core.AssertEqual(t, 0, rec.writeCount())
}

func TestAgentInflux_ScoreContentAndPush_Ugly(t *core.T) {
	influx, rec := newFakeInflux(t, nil, 0)
	ScoreContentAndPush(context.Background(), contentJudge(), influx, sampleCheckpoint(), "content-run", nil)
	core.AssertEqual(t, 0, rec.writeCount())
}

func TestAgentInflux_PushCapabilitySummary_Good(t *core.T) {
	influx, rec := newFakeInflux(t, nil, 0)
	err := PushCapabilitySummary(influx, sampleCheckpoint(), sampleProbeResult())
	requireResultOK(t, err)
	core.AssertEqual(t, 1, rec.writeCount())
}

func TestAgentInflux_PushCapabilitySummary_Bad(t *core.T) {
	influx, _ := newFakeInflux(t, nil, http.StatusInternalServerError)
	err := PushCapabilitySummary(influx, sampleCheckpoint(), sampleProbeResult())
	assertResultError(t, err)
}

func TestAgentInflux_PushCapabilitySummary_Ugly(t *core.T) {
	influx, rec := newFakeInflux(t, nil, 0)
	err := PushCapabilitySummary(influx, sampleCheckpoint(), ProbeResult{})
	requireResultOK(t, err)
	core.AssertEqual(t, 1, rec.writeCount())
}

func TestAgentInflux_PushCapabilityResults_Good(t *core.T) {
	influx, rec := newFakeInflux(t, nil, 0)
	err := PushCapabilityResults(influx, sampleCheckpoint(), sampleProbeResult())
	requireResultOK(t, err)
	core.AssertEqual(t, 1, rec.writeCount())
}

func TestAgentInflux_PushCapabilityResults_Bad(t *core.T) {
	influx, _ := newFakeInflux(t, nil, http.StatusInternalServerError)
	err := PushCapabilityResults(influx, sampleCheckpoint(), sampleProbeResult())
	assertResultError(t, err)
}

func TestAgentInflux_PushCapabilityResults_Ugly(t *core.T) {
	influx, rec := newFakeInflux(t, nil, 0)
	err := PushCapabilityResults(influx, sampleCheckpoint(), ProbeResult{})
	requireResultOK(t, err)
	core.AssertEqual(t, 1, rec.writeCount())
}

func TestAgentInflux_PushCapabilityResultsDB_Good(t *core.T) {
	dbPath := core.JoinPath(t.TempDir(), "scores.duckdb")
	PushCapabilityResultsDB(dbPath, sampleCheckpoint(), sampleProbeResult())
	core.AssertTrue(t, coreio.Local.IsFile(dbPath))
}

func TestAgentInflux_PushCapabilityResultsDB_Bad(t *core.T) {
	dbPath := ""
	results := sampleProbeResult()
	PushCapabilityResultsDB(dbPath, sampleCheckpoint(), results)
	core.AssertEqual(t, "", dbPath)
	core.AssertNotNil(t, results.Probes)
}

func TestAgentInflux_PushCapabilityResultsDB_Ugly(t *core.T) {
	dir := core.JoinPath(t.TempDir(), "blocked")
	core.RequireNoError(t, coreio.Local.EnsureDir(dir))
	PushCapabilityResultsDB(dir, sampleCheckpoint(), sampleProbeResult())
	core.AssertTrue(t, coreio.Local.IsDir(dir))
}

func TestAgentInflux_BufferInfluxResult_Good(t *core.T) {
	workDir := t.TempDir()
	BufferInfluxResult(workDir, sampleCheckpoint(), sampleProbeResult())
	data, err := coreio.Local.Read(core.JoinPath(workDir, InfluxBufferFile))
	core.RequireNoError(t, err)
	core.AssertContains(t, data, "G1 @10")
}

func TestAgentInflux_BufferInfluxResult_Bad(t *core.T) {
	file := core.JoinPath(t.TempDir(), "file")
	core.RequireNoError(t, coreio.Local.Write(file, "blocked"))
	BufferInfluxResult(file, sampleCheckpoint(), sampleProbeResult())
	data, err := coreio.Local.Read(file)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "blocked", data)
}

func TestAgentInflux_BufferInfluxResult_Ugly(t *core.T) {
	workDir := t.TempDir()
	BufferInfluxResult(workDir, sampleCheckpoint(), ProbeResult{})
	data, err := coreio.Local.Read(core.JoinPath(workDir, InfluxBufferFile))
	core.RequireNoError(t, err)
	core.AssertContains(t, data, "checkpoint")
}

func TestAgentInflux_ReplayInfluxBuffer_Good(t *core.T) {
	workDir := t.TempDir()
	BufferInfluxResult(workDir, sampleCheckpoint(), sampleProbeResult())
	influx, rec := newFakeInflux(t, nil, 0)
	ReplayInfluxBuffer(workDir, influx)
	core.AssertEqual(t, 1, rec.writeCount())
	core.AssertFalse(t, coreio.Local.IsFile(core.JoinPath(workDir, InfluxBufferFile)))
}

func TestAgentInflux_ReplayInfluxBuffer_Bad(t *core.T) {
	workDir := t.TempDir()
	BufferInfluxResult(workDir, sampleCheckpoint(), sampleProbeResult())
	influx, rec := newFakeInflux(t, nil, http.StatusInternalServerError)
	ReplayInfluxBuffer(workDir, influx)
	core.AssertEqual(t, 1, rec.writeCount())
	core.AssertTrue(t, coreio.Local.IsFile(core.JoinPath(workDir, InfluxBufferFile)))
}

func TestAgentInflux_ReplayInfluxBuffer_Ugly(t *core.T) {
	workDir := t.TempDir()
	ReplayInfluxBuffer(workDir, datapipe.NewInfluxClient("http://127.0.0.1:1", "test"))
	core.AssertFalse(t, coreio.Local.IsFile(core.JoinPath(workDir, InfluxBufferFile)))
}
