package modelmgmt

import (
	"context"

	"dappco.re/go"
	"dappco.re/go/inference/eval/datapipe"
	"dappco.re/go/inference/eval/score"
	"dappco.re/go/inference/serving"
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
