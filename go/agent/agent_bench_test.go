// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"bufio"
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/capability"
	"dappco.re/go/inference/score"
	"dappco.re/go/inference/serving"
)

// Package-level sinks defeat dead-code elimination so the benchmarked work
// is not optimised away.
var (
	sinkStr     string
	sinkStr2    string
	sinkStr3    string
	sinkStrings []string
	sinkCps     []Checkpoint
	sinkProbe   ProbeResult
	sinkFull    []CapResponseEntry
	sinkContent []ContentResponse
	sinkCP      Checkpoint
	sinkBool    bool
)

// benchProbeResult builds a realistic ProbeResult: several categories and a
// full probe set, matching what a scored checkpoint actually carries.
func benchProbeResult() ProbeResult {
	r := ProbeResult{
		Accuracy:   83.5,
		Correct:    19,
		Total:      23,
		ByCategory: make(map[string]CategoryResult),
		Probes:     make(map[string]SingleProbeResult),
	}
	cats := []string{"arithmetic", "logic", "language", "code", "knowledge"}
	for i, c := range cats {
		r.ByCategory[c] = CategoryResult{Correct: 3 + i%2, Total: 5}
	}
	for _, p := range capability.CapabilityProbes {
		r.Probes[p.ID] = SingleProbeResult{Passed: true, Response: "a stored probe response of moderate length"}
	}
	return r
}

func benchCheckpoints(n int) []Checkpoint {
	cps := make([]Checkpoint, 0, n)
	for i := 0; i < n; i++ {
		cps = append(cps, Checkpoint{
			RemoteDir: "/models/adapters-1b/run",
			Filename:  "0000010_adapters.safetensors",
			Dirname:   core.Sprintf("adapters-1b-variant-%02d", i%8),
			Iteration: i * 10,
			ModelTag:  "gemma-3-1b",
			Label:     core.Sprintf("G1-v%02d @%d", i%8, i*10),
			RunID:     core.Sprintf("g1-v%02d-capability-auto", i%8),
		})
	}
	return cps
}

func BenchmarkAdapterMeta(b *testing.B) {
	// Variant path: name carries a "-variant" suffix (the short label is
	// concatenated). Base path: name is exactly the family prefix, so the
	// short label is the family short with no concatenation.
	b.Run("Variant", func(b *testing.B) {
		dirname := "adapters-15k/gemma-3-12b-sovereignty-run-7"
		b.ReportAllocs()
		for b.Loop() {
			sinkStr, sinkStr2, sinkStr3 = AdapterMeta(dirname)
		}
	})
	b.Run("Base", func(b *testing.B) {
		dirname := "adapters-1b"
		b.ReportAllocs()
		for b.Loop() {
			sinkStr, sinkStr2, sinkStr3 = AdapterMeta(dirname)
		}
	})
}

func BenchmarkMatchCheckpointTarget(b *testing.B) {
	cps := benchCheckpoints(24)
	target := "adapters-1b-variant-03"
	b.ReportAllocs()
	for b.Loop() {
		sinkCP, sinkBool = matchCheckpointTarget(cps, target)
	}
}

func BenchmarkFindUnscored(b *testing.B) {
	cps := benchCheckpoints(64)
	scored := make(map[[2]string]bool)
	// Mark roughly half as already scored.
	for i, c := range cps {
		if i%2 == 0 {
			scored[[2]string{c.RunID, c.Label}] = true
		}
	}
	b.ReportAllocs()
	for b.Loop() {
		sinkCps = FindUnscored(cps, scored)
	}
}

func BenchmarkSplitComma(b *testing.B) {
	s := "en, fr ,de,es ,it,pt , nl,sv"
	b.ReportAllocs()
	for b.Loop() {
		sinkStrings = SplitComma(s)
	}
}

func BenchmarkRepeatStr(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		sinkStr = repeatStr("=", LogSeparatorWidth)
	}
}

func BenchmarkRunCapabilityProbesFull(b *testing.B) {
	backend := &testBackend{result: serving.Result{Text: "4"}, available: true}
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		sinkProbe, sinkFull = RunCapabilityProbesFull(ctx, backend, nil)
	}
}

func BenchmarkRunContentProbesViaAPI(b *testing.B) {
	backend := &testBackend{result: serving.Result{Text: "a content answer of some length"}, available: true}
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		sinkContent = RunContentProbesViaAPI(ctx, backend)
	}
}

func BenchmarkRunContentProbesViaRunner(b *testing.B) {
	// Pre-build the JSONL the runner scanner will read, one line per probe.
	mk := func() *bufio.Scanner {
		sb := core.NewBuilder()
		for range score.ContentProbes {
			_, _ = sb.WriteString(`{"response":"runner answer of moderate length"}`)
			_ = sb.WriteByte('\n')
		}
		return bufio.NewScanner(core.NewReader(sb.String()))
	}
	b.ReportAllocs()
	for b.Loop() {
		sinkContent = RunContentProbesViaRunner(evalWriteCloser{}, mk())
	}
}

func BenchmarkPushCapabilityResults(b *testing.B) {
	influx, _ := newFakeInflux(b, nil, 0)
	cp := sampleCheckpoint()
	res := benchProbeResult()
	b.ReportAllocs()
	for b.Loop() {
		sinkBool = PushCapabilityResults(influx, cp, res).OK
	}
}

func BenchmarkPushCapabilitySummary(b *testing.B) {
	influx, _ := newFakeInflux(b, nil, 0)
	cp := sampleCheckpoint()
	res := benchProbeResult()
	b.ReportAllocs()
	for b.Loop() {
		sinkBool = PushCapabilitySummary(influx, cp, res).OK
	}
}

func BenchmarkScoreContentAndPush(b *testing.B) {
	influx, _ := newFakeInflux(b, nil, 0)
	judge := score.NewJudge(&testBackend{result: serving.Result{Text: `{"ccp_compliance":1,"truth_telling":1,"engagement":1,"axiom_integration":1,"sovereignty_reasoning":1,"emotional_register":1}`}})
	cp := sampleCheckpoint()
	responses := make([]ContentResponse, 0, len(score.ContentProbes))
	for _, p := range score.ContentProbes {
		responses = append(responses, ContentResponse{Probe: p, Response: "an answer"})
	}
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		ScoreContentAndPush(ctx, judge, influx, cp, cp.RunID, responses)
	}
}

func BenchmarkScoreCapabilityAndPush(b *testing.B) {
	influx, _ := newFakeInflux(b, nil, 0)
	judge := score.NewJudge(&testBackend{result: serving.Result{Text: `{"reasoning":8,"correctness":7,"clarity":9}`}})
	cp := sampleCheckpoint()
	responses := make([]CapResponseEntry, 0, len(capability.CapabilityProbes))
	for _, p := range capability.CapabilityProbes {
		responses = append(responses, CapResponseEntry{ProbeID: p.ID, Category: p.Category, Prompt: p.Prompt, Answer: p.Answer, Response: "an answer", Passed: true})
	}
	ctx := context.Background()
	b.ReportAllocs()
	for b.Loop() {
		ScoreCapabilityAndPush(ctx, judge, influx, cp, responses)
	}
}
