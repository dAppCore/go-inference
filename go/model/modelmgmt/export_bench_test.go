// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/score"
)

// Dataset-shaping helpers run once per export over the whole response set —
// FilterResponses accumulates survivors, SplitData copies+shuffles.

var benchSinkResponses []score.Response

func benchResponses(n int) []score.Response {
	long := core.Concat("This response is comfortably past the fifty character keep ",
		"threshold so it survives filtering intact.")
	out := make([]score.Response, n)
	for i := range out {
		out[i] = score.Response{ID: core.Sprintf("r%d", i), Prompt: "p", Response: long}
	}
	return out
}

func BenchmarkFilterResponses(b *testing.B) {
	responses := benchResponses(1000)
	b.ReportAllocs()
	for b.Loop() {
		benchSinkResponses = FilterResponses(responses)
	}
}

func BenchmarkSplitData(b *testing.B) {
	responses := benchResponses(1000)
	b.ReportAllocs()
	for b.Loop() {
		train, _, _ := SplitData(responses, 80, 10, 10, 42)
		benchSinkResponses = train
	}
}
