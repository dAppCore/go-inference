package agent

import (
	"context"
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
	"dappco.re/go/inference/datapipe"
)

// runAgentLoopInfluxServer returns an httptest server that answers InfluxDB
// v3 query_sql calls with an empty result set — enough for GetScoredLabels
// to succeed with zero already-scored pairs.
func runAgentLoopInfluxServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v3/query_sql" {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(`[]`))
			return
		}
		w.WriteHeader(http.StatusNoContent)
	}))
}

func TestAgentExecute_RunAgentLoop_Good(t *core.T) {
	ft := newFakeTransport()
	ft.On("ls -d /base/adapters-*", "/base/adapters-27b\n", nil)
	ft.On("ls -d /base/adapters-27b/gemma-3-*", "", core.AnError)
	ft.On("ls /base/adapters-27b/*_adapters.safetensors", "/base/adapters-27b/0001000_adapters.safetensors\n", nil)

	influxSrv := runAgentLoopInfluxServer()
	defer influxSrv.Close()

	// Successful discovery + non-Force scoring lookup + DryRun means every
	// statement up to (but not including) the real ProcessOne dispatch runs,
	// without paying the InterCheckpointDelay sleep.
	cfg := &AgentConfig{
		M3AdapterBase: "/base", Transport: ft, WorkDir: t.TempDir(),
		InfluxURL: influxSrv.URL, InfluxDB: "test",
		DBPath: core.JoinPath(t.TempDir(), "scores.duckdb"),
		DryRun: true, OneShot: true,
	}
	core.AssertNotPanics(t, func() { RunAgentLoop(cfg) })
	core.AssertTrue(t, cfg.OneShot)
}

func TestAgentExecute_RunAgentLoop_Bad(t *core.T) {
	cfg := &AgentConfig{M3AdapterBase: "/bad", OneShot: true, Transport: newFakeTransport(), WorkDir: t.TempDir()}
	core.AssertNotPanics(t, func() { RunAgentLoop(cfg) })
	core.AssertEqual(t, "/bad", cfg.M3AdapterBase)

	// Discovery succeeds but finds no adapter directories at all — a
	// distinct "nothing to score" branch from the discovery failure above.
	ft := newFakeTransport()
	ft.On("ls -d /empty/adapters-*", "", nil)
	cfg2 := &AgentConfig{
		M3AdapterBase: "/empty", OneShot: true, Transport: ft, WorkDir: t.TempDir(),
		InfluxURL: "http://127.0.0.1:1",
	}
	core.AssertNotPanics(t, func() { RunAgentLoop(cfg2) })
}

func TestAgentExecute_RunAgentLoop_Ugly(t *core.T) {
	ft := newFakeTransport()
	ft.On("ls -d /base/adapters-*", "/base/adapters-27b\n", nil)
	ft.On("ls -d /base/adapters-27b/gemma-3-*", "", core.AnError)
	ft.On("ls /base/adapters-27b/*_adapters.safetensors", "/base/adapters-27b/0001000_adapters.safetensors\n", nil)

	// Force mode skips GetScoredLabels/FindUnscored entirely and processes
	// every discovered checkpoint directly; non-DryRun drives a real
	// ProcessOne call (it fails — no live M3/Ollama — exercising the
	// error-print branch) and its InterCheckpointDelay sleep. Deliberately
	// the one RunAgentLoop test that pays that real 5s cost.
	cfg := &AgentConfig{
		M3AdapterBase: "/base", Transport: ft, WorkDir: t.TempDir(),
		InfluxURL: "http://127.0.0.1:1",
		Force:     true, OneShot: true,
	}
	core.AssertNotPanics(t, func() { RunAgentLoop(cfg) })
	core.AssertTrue(t, cfg.Force)
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
	// A blank line between two real entries exercises the empty-fp skip.
	ft.On("ls /base/adapters-1b/*_adapters.safetensors",
		"/base/adapters-1b/0000007_adapters.safetensors\n\n/base/adapters-1b/0000009_adapters.safetensors\n", nil)
	var checkpoints []Checkpoint
	for cp, err := range DiscoverCheckpointsIter(&AgentConfig{M3AdapterBase: "/base", Transport: ft}) {
		core.RequireNoError(t, err)
		checkpoints = append(checkpoints, cp)
	}
	core.AssertLen(t, checkpoints, 2)

	// Stopping iteration early (the range body returns false to yield) must
	// halt cleanly instead of continuing to the second checkpoint.
	count := 0
	for range DiscoverCheckpointsIter(&AgentConfig{M3AdapterBase: "/base", Transport: ft}) {
		count++
		break
	}
	core.AssertEqual(t, 1, count)
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

	// An input where the last element sorts into the middle (rather than
	// bubbling all the way to the front) forces the sort comparator through
	// both the "less than" and "greater than" Dirname branches.
	reversed := []Checkpoint{
		{RunID: "r", Label: "c", Dirname: "c"},
		{RunID: "r", Label: "a", Dirname: "a"},
		{RunID: "r", Label: "b", Dirname: "b"},
	}
	sorted := FindUnscored(reversed, nil)
	core.AssertLen(t, sorted, 3)
	core.AssertEqual(t, "a", sorted[0].Dirname)
	core.AssertEqual(t, "b", sorted[1].Dirname)
	core.AssertEqual(t, "c", sorted[2].Dirname)
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

	// Stopping iteration early (yield returns false) halts the loop
	// instead of visiting the remaining unscored checkpoints.
	multi := []Checkpoint{{RunID: "r", Label: "one"}, {RunID: "r", Label: "two"}}
	seen := 0
	for range FindUnscoredIter(multi, nil) {
		seen++
		break
	}
	core.AssertEqual(t, 1, seen)
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
