// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the ModelInfo projections — ModelInfoToBundle,
// ParserHint, ModelInfoToMemory. These are pure field projections that run
// once per model load / state-bundle compatibility check, not per token.
// All-scalar/string field copies with no slice or map work, so the
// expectation is 0 allocs/op; benched to confirm the projections stay
// allocation-free as ModelInfo grows.
//
// Run:    go test -bench='BenchmarkModelInfo_' -benchmem -run='^$' ./spine

package spine

import (
	"testing"

	"dappco.re/go/inference/kv/memory"
	"dappco.re/go/inference/model/bundle"
	"dappco.re/go/inference/decode/parser"
	"dappco.re/go/inference/train/lora"
)

// Sinks defeat compiler DCE.
var (
	modelInfoBenchSinkBundle bundle.ModelInfo
	modelInfoBenchSinkHint   parser.Hint
	modelInfoBenchSinkMemory memory.ModelInfo
)

// benchModelInfo is a representative loaded-model descriptor.
var benchModelInfo = ModelInfo{
	Architecture:  "gemma4",
	VocabSize:     262144,
	NumLayers:     48,
	NumHeads:      32,
	NumKVHeads:    8,
	HeadDim:       256,
	HiddenSize:    3840,
	QuantBits:     4,
	QuantGroup:    64,
	ContextLength: 131072,
	Adapter:       lora.AdapterInfo{Name: "lek2"},
}

func BenchmarkModelInfo_ToBundle(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		modelInfoBenchSinkBundle = ModelInfoToBundle(benchModelInfo)
	}
}

func BenchmarkModelInfo_ParserHint(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		modelInfoBenchSinkHint = ParserHint(benchModelInfo)
	}
}

func BenchmarkModelInfo_ToMemory(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		modelInfoBenchSinkMemory = ModelInfoToMemory(benchModelInfo)
	}
}
