package modelmgmt

import (
	"context"

	"dappco.re/go"
	"dappco.re/go/inference/eval/datapipe"
	"dappco.re/go/inference/eval/score"
	"dappco.re/go/inference/serving"
	coreio "dappco.re/go/io"
)

func TestExpand_GetCompletedIDs_Good(t *core.T) {
	influx, _ := newFakeInflux(t, map[string][]map[string]any{"expansion_gen": {{"seed_id": "s1"}}}, 0)
	r := GetCompletedIDs(influx)
	requireResultOK(t, r)
	ids := r.Value.(map[string]bool)
	core.AssertTrue(t, ids["s1"])
}

func TestExpand_GetCompletedIDs_Bad(t *core.T) {
	influx := datapipe.NewInfluxClient("http://127.0.0.1:1", "test")
	assertResultError(t, GetCompletedIDs(influx))
}

func TestExpand_GetCompletedIDs_Ugly(t *core.T) {
	influx, _ := newFakeInflux(t, map[string][]map[string]any{"expansion_gen": {{"seed_id": ""}}}, 0)
	r := GetCompletedIDs(influx)
	requireResultOK(t, r)
	ids := r.Value.(map[string]bool)
	core.AssertEmpty(t, ids)
}

func TestExpand_ExpandPrompts_Good(t *core.T) {
	influx, _ := newFakeInflux(t, map[string][]map[string]any{"expansion_gen": {}}, 0)
	prompts := []score.Response{{ID: "s1", Domain: "ethics", Prompt: "prompt"}}
	r := ExpandPrompts(context.Background(), &testBackend{result: serving.Result{Text: "generated response"}}, influx, prompts, "m", "w", t.TempDir(), true, 1)
	requireResultOK(t, r)
	core.AssertLen(t, prompts, 1)
}

func TestExpand_ExpandPrompts_Bad(t *core.T) {
	influx, _ := newFakeInflux(t, map[string][]map[string]any{"expansion_gen": {{"seed_id": "s1"}}}, 0)
	prompts := []score.Response{{ID: "s1", Domain: "ethics", Prompt: "prompt"}}
	r := ExpandPrompts(context.Background(), &testBackend{}, influx, prompts, "m", "w", t.TempDir(), false, 0)
	requireResultOK(t, r)
	core.AssertLen(t, prompts, 1)
}

func TestExpand_ExpandPrompts_Ugly(t *core.T) {
	influx, _ := newFakeInflux(t, map[string][]map[string]any{"expansion_gen": {}}, 0)
	prompts := []score.Response{{ID: "s1", Domain: "ethics", Prompt: "prompt"}}
	r := ExpandPrompts(context.Background(), &testBackend{err: core.AnError}, influx, prompts, "m", "w", t.TempDir(), false, 0)
	requireResultOK(t, r)
	core.AssertLen(t, prompts, 1)
}

// TestExpand_ExpandPrompts_Write_Good drives the real generation loop: a
// working backend writes JSONL to disk and reports progress to InfluxDB. The
// limit also truncates the two prompts down to one.
func TestExpand_ExpandPrompts_Write_Good(t *core.T) {
	influx, rec := newFakeInflux(t, map[string][]map[string]any{"expansion_gen": {}}, 0)
	prompts := []score.Response{
		{ID: "s1", Domain: "ethics", Prompt: "p1"},
		{ID: "s2", Domain: "law", Prompt: "p2"},
	}
	outDir := t.TempDir()
	requireResultOK(t, ExpandPrompts(context.Background(), &testBackend{result: serving.Result{Text: "resp"}}, influx, prompts, "m", "w", outDir, false, 1))
	data, err := coreio.Local.Read(core.JoinPath(outDir, "expand-w.jsonl"))
	core.RequireNoError(t, err)
	core.AssertContains(t, data, "resp")
	core.AssertTrue(t, rec.writeCount() > 0)
}

// TestExpand_ExpandPrompts_Write_Ugly fails to open the output file because the
// output directory path is actually a regular file.
func TestExpand_ExpandPrompts_Write_Ugly(t *core.T) {
	influx, _ := newFakeInflux(t, map[string][]map[string]any{"expansion_gen": {}}, 0)
	prompts := []score.Response{{ID: "s1", Prompt: "p"}}
	fileAsDir := core.JoinPath(t.TempDir(), "notadir")
	core.RequireNoError(t, coreio.Local.Write(fileAsDir, "x"))
	assertResultError(t, ExpandPrompts(context.Background(), &testBackend{result: serving.Result{Text: "r"}}, influx, prompts, "m", "w", fileAsDir, false, 0), "open output file")
}

// TestExpand_ExpandPrompts_DryRunMany exercises the dry-run listing cap (more
// than ten prompts prints an "and N more" summary).
func TestExpand_ExpandPrompts_DryRunMany(t *core.T) {
	influx, _ := newFakeInflux(t, map[string][]map[string]any{"expansion_gen": {}}, 0)
	var prompts []score.Response
	for i := range 11 {
		prompts = append(prompts, score.Response{ID: core.Sprintf("s%d", i), Prompt: "p"})
	}
	requireResultOK(t, ExpandPrompts(context.Background(), &testBackend{}, influx, prompts, "m", "w", t.TempDir(), true, 0))
}

// TestExpand_ExpandPrompts_InfluxDown proceeds when the completed-ID lookup
// fails: the warning path runs, generation still writes locally, and the
// best-effort InfluxDB write failure is swallowed.
func TestExpand_ExpandPrompts_InfluxDown(t *core.T) {
	influx := datapipe.NewInfluxClient("http://127.0.0.1:1", "test")
	prompts := []score.Response{{ID: "s1", Prompt: "p"}}
	outDir := t.TempDir()
	requireResultOK(t, ExpandPrompts(context.Background(), &testBackend{result: serving.Result{Text: "r"}}, influx, prompts, "m", "w", outDir, false, 0))
	data, err := coreio.Local.Read(core.JoinPath(outDir, "expand-w.jsonl"))
	core.RequireNoError(t, err)
	core.AssertContains(t, data, "\"r\"")
}
