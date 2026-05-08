// SPDX-Licence-Identifier: EUPL-1.2

package inference

import core "dappco.re/go"

func ExampleDatasetSample() {
	sample := DatasetSample{
		Messages: []Message{
			{Role: "user", Content: "Explain KV cache reuse"},
			{Role: "assistant", Content: "KV cache reuse avoids recomputing prior context."},
		},
		Reasoning: "focus on local inference state",
	}

	core.Println(len(sample.Messages), sample.Reasoning)
	// Output: 2 focus on local inference state
}

func ExampleBenchReport() {
	report := BenchReport{
		Model:               ModelIdentity{Architecture: "qwen3"},
		PrefillTokensPerSec: 1400,
		DecodeTokensPerSec:  42,
		PromptCacheHitRate:  0.75,
	}

	core.Println(report.Model.Architecture, report.DecodeTokensPerSec, report.PromptCacheHitRate)
	// Output: qwen3 42 0.75
}
