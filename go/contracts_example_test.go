// SPDX-Licence-Identifier: EUPL-1.2

package inference

import (
	"context"

	core "dappco.re/go"
)

func ExampleCacheService() {
	model := &contractModel{}
	stats, _ := any(model).(CacheService).CacheStats(context.Background())

	core.Println(stats.CacheMode)
	// Output: paged-q8
}

func ExampleEmbeddingModel() {
	model := &contractModel{}
	result, _ := any(model).(EmbeddingModel).Embed(context.Background(), EmbeddingRequest{Input: []string{"core"}})

	core.Println(len(result.Vectors))
	// Output: 1
}

func ExampleReasoningParser() {
	model := &contractModel{}
	result, _ := any(model).(ReasoningParser).ParseReasoning(nil, "visible")

	core.Println(result.Reasoning[0].Kind)
	// Output: think
}
