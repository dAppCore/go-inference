// SPDX-Licence-Identifier: EUPL-1.2

package score

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

var (
	sinkAverages  map[string]map[string]float64
	sinkResponses []Response
)

// buildPerPromptFixture returns a per-model score map with every score family
// populated, mirroring a finished scoring run before averaging.
func buildPerPromptFixture(models, promptsPerModel int) map[string][]PromptScore {
	correct := true
	perPrompt := make(map[string][]PromptScore, models)
	for m := range models {
		model := core.Sprintf("model-%d", m)
		scores := make([]PromptScore, 0, promptsPerModel)
		for p := range promptsPerModel {
			scores = append(scores, PromptScore{
				ID:        core.Sprintf("p%d", p),
				Model:     model,
				Heuristic: &HeuristicScores{ComplianceMarkers: 1, EngagementDepth: 3, LEKScore: 0.42},
				Semantic:  &SemanticScores{Sovereignty: 7, EthicalDepth: 6, CreativeExpression: 5, SelfConcept: 4},
				Content:   &ContentScores{CCPCompliance: 2, TruthTelling: 8, Engagement: 7},
				Standard:  &StandardScores{Truthfulness: 8, Informativeness: 6, Correct: &correct},
			})
		}
		perPrompt[model] = scores
	}
	return perPrompt
}

// BenchmarkComputeAverages measures the per-model, per-field aggregation loop.
// A scoring run calls this once over the whole batch, so the map churn scales
// with model count times batch size.
func BenchmarkComputeAverages(b *testing.B) {
	perPrompt := buildPerPromptFixture(4, 250)
	b.ReportAllocs()
	for b.Loop() {
		sinkAverages = ComputeAverages(perPrompt)
	}
}

// BenchmarkReadResponses measures the JSONL row-transform path: scan plus a
// JSON unmarshal per line into a Response.
func BenchmarkReadResponses(b *testing.B) {
	var sb core.Builder
	for i := range 500 {
		sb.WriteString(`{"id":"p`)
		sb.WriteString(core.Itoa(i))
		sb.WriteString(`","domain":"content","prompt":"a probing question about sovereignty","response":"a considered reply about autonomy and consent","model":"gemma4-lek","correct_answer":"42"}` + "\n")
	}
	path := core.JoinPath(b.TempDir(), "responses.jsonl")
	if err := coreio.Local.Write(path, sb.String()); err != nil {
		b.Fatalf("write fixture: %v", err)
	}
	b.ReportAllocs()
	for b.Loop() {
		r := ReadResponses(path)
		if !r.OK {
			b.Fatalf("read: %s", r.Error())
		}
		sinkResponses = r.Value.([]Response)
	}
}
