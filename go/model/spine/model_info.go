// SPDX-Licence-Identifier: EUPL-1.2

package spine

import (
	"dappco.re/go/inference/kv/memory"
	"dappco.re/go/inference/model/bundle"
	"dappco.re/go/inference/decode/parser"
	"dappco.re/go/inference/train/lora"
)

// ModelInfo describes a loaded model.
type ModelInfo struct {
	Architecture          string
	VocabSize             int
	NumLayers             int
	NumHeads              int
	NumKVHeads            int
	HeadDim               int
	HiddenSize            int
	QuantBits             int
	QuantGroup            int
	ContextLength         int
	SlidingWindow         int
	ParallelSlots         int
	PromptCache           bool
	PromptCacheMinTokens  int
	CachePolicy           memory.KVCachePolicy
	CacheMode             memory.KVCacheMode
	KVCacheStorageDType   string
	PagedKVPageSize       int
	PagedKVPrealloc       bool
	FixedSlidingCacheSize int
	BatchSize             int
	PrefillChunkSize      int
	ExpectedQuantization  int
	MemoryLimitBytes      uint64
	CacheLimitBytes       uint64
	WiredLimitBytes       uint64
	Adapter               lora.AdapterInfo
}

// ModelInfoToBundle converts ModelInfo to bundle.ModelInfo for
// state-bundle compatibility checks.
//
//	out := spine.ModelInfoToBundle(info)
func ModelInfoToBundle(info ModelInfo) bundle.ModelInfo {
	return bundle.ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
		Adapter:       info.Adapter,
	}
}

// ParserHint builds the thinking-parser hint from a model description.
//
//	hint := spine.ParserHint(model.Info())
func ParserHint(info ModelInfo) parser.Hint {
	return parser.Hint{
		Architecture: info.Architecture,
		AdapterName:  info.Adapter.Name,
	}
}

// ModelInfoToMemory converts a ModelInfo into the structural mirror used
// by go-mlx/memory/, go-mlx/agent/, and other subpackages that work from
// the planner's view of a model.
//
//	out := spine.ModelInfoToMemory(info)
func ModelInfoToMemory(info ModelInfo) memory.ModelInfo {
	return memory.ModelInfo{
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
	}
}
