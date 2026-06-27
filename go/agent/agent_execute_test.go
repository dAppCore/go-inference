package agent

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/datapipe"
)

func TestAgentExecute_RunAgentLoop_Good(t *core.T) {
	cfg := &AgentConfig{M3AdapterBase: "/missing", OneShot: true, Transport: newFakeTransport(), WorkDir: t.TempDir()}
	core.AssertNotPanics(t, func() { RunAgentLoop(cfg) })
	core.AssertTrue(t, cfg.OneShot)
}

func TestAgentExecute_RunAgentLoop_Bad(t *core.T) {
	cfg := &AgentConfig{M3AdapterBase: "/bad", OneShot: true, Transport: newFakeTransport(), WorkDir: t.TempDir()}
	core.AssertNotPanics(t, func() { RunAgentLoop(cfg) })
	core.AssertEqual(t, "/bad", cfg.M3AdapterBase)
}

func TestAgentExecute_RunAgentLoop_Ugly(t *core.T) {
	cfg := &AgentConfig{OneShot: true, DryRun: true, Transport: newFakeTransport(), WorkDir: t.TempDir()}
	core.AssertNotPanics(t, func() { RunAgentLoop(cfg) })
	core.AssertTrue(t, cfg.DryRun)
}

func TestAgentExecute_DiscoverCheckpoints_Good(t *core.T) {
	ft := newFakeTransport()
	ft.On("ls -d /base/adapters-*", "/base/adapters-27b\n", nil)
	ft.On("ls -d /base/adapters-27b/gemma-3-*", "", core.AnError)
	ft.On("ls /base/adapters-27b/*_adapters.safetensors", "/base/adapters-27b/0000010_adapters.safetensors\n", nil)
	r := DiscoverCheckpoints(&AgentConfig{M3AdapterBase: "/base", Transport: ft})
	requireResultOK(t, r)
	checkpoints := r.Value.([]Checkpoint)
	core.AssertLen(t, checkpoints, 1)
}

func TestAgentExecute_DiscoverCheckpoints_Bad(t *core.T) {
	r := DiscoverCheckpoints(&AgentConfig{M3AdapterBase: "/base", Transport: newFakeTransport()})
	assertResultError(t, r)
	core.AssertFalse(t, r.OK)
	core.AssertError(t, r.Value.(error))
}

func TestAgentExecute_DiscoverCheckpoints_Ugly(t *core.T) {
	ft := newFakeTransport()
	ft.On("ls -d /base/adapters-*", "", nil)
	r := DiscoverCheckpoints(&AgentConfig{M3AdapterBase: "/base", Transport: ft})
	requireResultOK(t, r)
	checkpoints := r.Value.([]Checkpoint)
	core.AssertEmpty(t, checkpoints)
}

func TestAgentExecute_DiscoverCheckpointsIter_Good(t *core.T) {
	ft := newFakeTransport()
	ft.On("ls -d /base/adapters-*", "/base/adapters-1b\n", nil)
	ft.On("ls -d /base/adapters-1b/gemma-3-*", "", core.AnError)
	ft.On("ls /base/adapters-1b/*_adapters.safetensors", "/base/adapters-1b/0000007_adapters.safetensors\n", nil)
	var checkpoints []Checkpoint
	for cp, err := range DiscoverCheckpointsIter(&AgentConfig{M3AdapterBase: "/base", Transport: ft}) {
		core.RequireNoError(t, err)
		checkpoints = append(checkpoints, cp)
	}
	core.AssertLen(t, checkpoints, 1)
}

func TestAgentExecute_DiscoverCheckpointsIter_Bad(t *core.T) {
	var gotErr error
	for _, err := range DiscoverCheckpointsIter(&AgentConfig{M3AdapterBase: "/base", Transport: newFakeTransport()}) {
		gotErr = err
	}
	core.AssertError(t, gotErr)
}

func TestAgentExecute_DiscoverCheckpointsIter_Ugly(t *core.T) {
	ft := newFakeTransport()
	ft.On("ls -d /base/adapters-*", "/base/adapters-1b\n", nil)
	ft.On("ls -d /base/adapters-1b/gemma-3-*", "", core.AnError)
	ft.On("ls /base/adapters-1b/*_adapters.safetensors", "/base/adapters-1b/no_iteration.safetensors\n", nil)
	count := 0
	for range DiscoverCheckpointsIter(&AgentConfig{M3AdapterBase: "/base", Transport: ft}) {
		count++
	}
	core.AssertEqual(t, 0, count)
}

func TestAgentExecute_GetScoredLabels_Good(t *core.T) {
	influx, _ := newFakeInflux(t, map[string][]map[string]any{"SELECT DISTINCT": {{"run_id": "r", "label": "l"}}}, 0)
	r := GetScoredLabels(influx)
	requireResultOK(t, r)
	labels := r.Value.(map[[2]string]bool)
	core.AssertTrue(t, labels[[2]string{"r", "l"}])
}

func TestAgentExecute_GetScoredLabels_Bad(t *core.T) {
	influx := datapipe.NewInfluxClient("http://127.0.0.1:1", "test")
	r := GetScoredLabels(influx)
	assertResultError(t, r)
}

func TestAgentExecute_GetScoredLabels_Ugly(t *core.T) {
	influx, _ := newFakeInflux(t, map[string][]map[string]any{"SELECT DISTINCT": {{"run_id": "", "label": "l"}}}, 0)
	r := GetScoredLabels(influx)
	requireResultOK(t, r)
	labels := r.Value.(map[[2]string]bool)
	core.AssertEmpty(t, labels)
}

func TestAgentExecute_FindUnscored_Good(t *core.T) {
	checkpoints := []Checkpoint{{RunID: "r", Label: "b", Dirname: "b"}, {RunID: "r", Label: "a", Dirname: "a"}}
	got := FindUnscored(checkpoints, map[[2]string]bool{{"r", "b"}: true})
	core.AssertLen(t, got, 1)
	core.AssertEqual(t, "a", got[0].Label)
}

func TestAgentExecute_FindUnscored_Bad(t *core.T) {
	got := FindUnscored(nil, nil)
	core.AssertEmpty(t, got)
	core.AssertEqual(t, 0, len(got))
}

func TestAgentExecute_FindUnscored_Ugly(t *core.T) {
	checkpoints := []Checkpoint{{RunID: "r", Label: "l"}}
	got := FindUnscored(checkpoints, map[[2]string]bool{{"r", "l"}: true})
	core.AssertEmpty(t, got)
}

func TestAgentExecute_FindUnscoredIter_Good(t *core.T) {
	checkpoints := []Checkpoint{{RunID: "r", Label: "l"}}
	count := 0
	for cp := range FindUnscoredIter(checkpoints, nil) {
		core.AssertEqual(t, "l", cp.Label)
		count++
	}
	core.AssertEqual(t, 1, count)
}

func TestAgentExecute_FindUnscoredIter_Bad(t *core.T) {
	count := 0
	for range FindUnscoredIter(nil, nil) {
		count++
	}
	core.AssertEqual(t, 0, count)
}

func TestAgentExecute_FindUnscoredIter_Ugly(t *core.T) {
	checkpoints := []Checkpoint{{RunID: "r", Label: "l"}}
	count := 0
	for range FindUnscoredIter(checkpoints, map[[2]string]bool{{"r", "l"}: true}) {
		count++
	}
	core.AssertEqual(t, 0, count)
}

func TestAgentExecute_ProcessOne_Good(t *core.T) {
	err := ProcessOne(&AgentConfig{Transport: newFakeTransport(), WorkDir: t.TempDir()}, datapipe.NewInfluxClient("http://127.0.0.1:1", "test"), Checkpoint{ModelTag: "unknown"})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "convert")
}

func TestAgentExecute_ProcessOne_Bad(t *core.T) {
	err := ProcessOne(&AgentConfig{Transport: newFakeTransport(), WorkDir: t.TempDir()}, datapipe.NewInfluxClient("http://127.0.0.1:1", "test"), Checkpoint{ModelTag: "gemma-3-1b"})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "convert")
}

func TestAgentExecute_ProcessOne_Ugly(t *core.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	core.AssertNotNil(t, ctx)
	err := ProcessOne(&AgentConfig{Transport: newFakeTransport(), WorkDir: t.TempDir()}, datapipe.NewInfluxClient("http://127.0.0.1:1", "test"), sampleCheckpoint())
	core.AssertError(t, err)
}
