// SPDX-Licence-Identifier: EUPL-1.2

package fusion

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// quietModel is a zero-bookkeeping bench stand-in for a routed model: it returns
// a canned reply with no locking or prompt recording, so a Run/dispatchPanel
// benchmark measures fusion's own orchestration allocations rather than a fake's
// accounting. Pointer receivers keep the interface box a single pointer (no
// per-conversion copy of the struct).
type quietModel struct {
	id    string
	reply string
}

func (m *quietModel) Run(_ context.Context, _ string) (string, error) { return m.reply, nil }
func (m *quietModel) ID() string                                      { return m.id }

// benchReply is a realistic multi-hundred-character panel response — long enough
// that the O(n²) re-copy in a naive synthesis-prompt build shows up in B/op.
const benchReply = "The sky appears blue because shorter (blue) wavelengths of " +
	"sunlight are scattered far more strongly by the nitrogen and oxygen molecules " +
	"of the atmosphere than the longer red wavelengths — Rayleigh scattering, which " +
	"scales as one over wavelength to the fourth power. At sunrise and sunset the " +
	"path length is longer, so more blue is scattered out and the remaining light " +
	"skews red. Consensus across the panel; the only contested point is the role of " +
	"Mie scattering from aerosols near the horizon."

const benchPrompt = "Why is the sky blue during the day but red at sunrise and " +
	"sunset, and what role do atmospheric particles play in the colour we perceive?"

// benchPanel builds n survived panel responses as fusion's fan-out would collect
// them — distinct model IDs, each carrying a realistic reply, all Err == nil.
func benchPanel(n int) []PanelResponse {
	prs := make([]PanelResponse, n)
	for i := range prs {
		prs[i] = PanelResponse{
			ModelID: core.Concat("gemma-panel-member-", core.Itoa(i)),
			Text:    benchReply,
		}
	}
	return prs
}

func benchModels(n int) []Model {
	ms := make([]Model, n)
	for i := range ms {
		ms[i] = &quietModel{id: core.Concat("gemma-panel-member-", core.Itoa(i)), reply: benchReply}
	}
	return ms
}

// Package sinks — prevent the compiler eliminating the benchmarked work.
var (
	benchPromptSink string
	benchPanelSink  []PanelResponse
	benchResultSink Result
)

// BenchmarkBuildSynthesisPrompt is the hot synthesis-prompt assembly: one call
// per fusion request, looping every surviving panel member. The naive build
// re-copies the growing prompt per member (O(n²)); this is the allocation target.
func BenchmarkBuildSynthesisPrompt(b *testing.B) {
	panel := benchPanel(5)
	b.ReportAllocs()
	b.ResetTimer()
	var s string
	for i := 0; i < b.N; i++ {
		s = buildSynthesisPrompt(benchPrompt, panel)
	}
	benchPromptSink = s
}

// BenchmarkDispatchPanel measures the parallel fan-out: one output slice plus a
// goroutine per member. The output slice and the goroutines are inherent; the
// per-iteration closure environment is the avoidable part.
func BenchmarkDispatchPanel(b *testing.B) {
	models := benchModels(5)
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	var p []PanelResponse
	for i := 0; i < b.N; i++ {
		p = dispatchPanel(ctx, benchPrompt, models)
	}
	benchPanelSink = p
}

// BenchmarkRun is the whole success path: recursion mark, fan-out, survivor
// check, synthesis-prompt build, judge call, result assembly.
func BenchmarkRun(b *testing.B) {
	cfg := Config{AnalysisModels: benchModels(5), Judge: &quietModel{id: "judge", reply: benchReply}, Enabled: true}
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	var r Result
	for i := 0; i < b.N; i++ {
		r, _ = Run(ctx, benchPrompt, cfg)
	}
	benchResultSink = r
}
