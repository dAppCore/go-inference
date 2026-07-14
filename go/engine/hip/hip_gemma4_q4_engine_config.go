// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

const (
	hipGemma4Q4PrefillDefaultUBatchTokens              = 512
	hipGemma4Q4DefaultPrefillAttentionQueryChunkTokens = 8
)

type hipGemma4Q4EngineConfig struct {
	DeviceKVMode               string
	DeviceKVBlockSize          int
	GlobalDeviceKVBlockSize    int
	ChunkedAttention           bool
	PageAlignedLocalKV         bool
	DisableInterleavedRowPages bool
	// DisableBatchedPrefill forces position-invariant token-at-a-time prompt
	// landing for quantized-KV prompt reuse and restored-state appends.
	DisableBatchedPrefill            bool
	DisableBatchedDecode             bool
	PrefillUBatchTokens              int
	PrefillAttentionQueryChunkTokens int
	// ForceBatchedProjection makes the prefill body launch the batched
	// projection kernel even for a single-token ubatch, instead of the
	// single-row fast path. The attached-drafter target prefill sets this so a
	// trailing one-token ubatch stays on the batched path; normal decode-style
	// single-token prefill leaves it false and keeps the single-row kernel.
	ForceBatchedProjection bool
}

func defaultHIPGemma4Q4EngineConfig() hipGemma4Q4EngineConfig {
	return hipGemma4Q4EngineConfig{
		DeviceKVMode:                     rocmKVCacheModeKQ8VQ4,
		DeviceKVBlockSize:                rocmGemma4Q4DeviceKVBlockSize,
		GlobalDeviceKVBlockSize:          rocmGemma4Q4GlobalDeviceKVBlockSize,
		ChunkedAttention:                 true,
		PrefillUBatchTokens:              hipGemma4Q4PrefillDefaultUBatchTokens,
		PrefillAttentionQueryChunkTokens: hipGemma4Q4DefaultPrefillAttentionQueryChunkTokens,
	}
}

func (cfg hipGemma4Q4EngineConfig) deviceKVMode() (string, error) {
	if !isROCmKVCacheMode(cfg.DeviceKVMode) {
		return "", core.E(hipGemma4Q4Layer0Operation, core.Sprintf("unsupported Gemma4 q4 device KV cache mode %q", cfg.DeviceKVMode), nil)
	}
	return cfg.DeviceKVMode, nil
}

func (cfg hipGemma4Q4EngineConfig) chunkedAttentionEnabled(promptTokens int) bool {
	return cfg.ChunkedAttention
}

func (cfg hipGemma4Q4EngineConfig) deviceKVBlockSize() int {
	if cfg.DeviceKVBlockSize > 0 {
		return cfg.DeviceKVBlockSize
	}
	return rocmGemma4Q4DeviceKVBlockSize
}

func (cfg hipGemma4Q4EngineConfig) globalDeviceKVBlockSize() int {
	if cfg.GlobalDeviceKVBlockSize > 0 {
		return cfg.GlobalDeviceKVBlockSize
	}
	if cfg.DeviceKVBlockSize > 0 {
		return cfg.DeviceKVBlockSize
	}
	return rocmGemma4Q4GlobalDeviceKVBlockSize
}

func (cfg hipGemma4Q4EngineConfig) interleavedRowPagesEnabled() bool {
	return !cfg.DisableInterleavedRowPages
}

func (cfg hipGemma4Q4EngineConfig) pageAlignedLocalKVEnabled() bool {
	return cfg.PageAlignedLocalKV
}

func (cfg hipGemma4Q4EngineConfig) deviceKVBlockSizeForSlidingWindow(slidingWindow int) int {
	if slidingWindow <= 0 {
		return cfg.globalDeviceKVBlockSize()
	}
	blockSize := cfg.deviceKVBlockSize()
	if blockSize > 1 && !cfg.interleavedRowPagesEnabled() {
		return rocmGemma4Q4DeviceKVBlockSize
	}
	return blockSize
}

func (cfg hipGemma4Q4EngineConfig) attentionWorkspaceNeeded(promptTokens int, generate inference.GenerateConfig) bool {
	return cfg.chunkedAttentionEnabled(promptTokens) || hipGemma4Q4DeviceCandidateSamplingRequested(generate) || hipGemma4Q4DeviceTopKSamplingRequested(generate)
}

func (cfg hipGemma4Q4EngineConfig) prefillUBatchTokens() (int, error) {
	if cfg.PrefillUBatchTokens <= 0 {
		return 0, core.E(hipGemma4Q4Layer0Operation, "Gemma4 q4 prefill ubatch tokens must be a positive integer", nil)
	}
	return cfg.PrefillUBatchTokens, nil
}

func (cfg hipGemma4Q4EngineConfig) prefillAttentionQueryChunkTokens() int {
	if cfg.PrefillAttentionQueryChunkTokens < 0 {
		return hipGemma4Q4DefaultPrefillAttentionQueryChunkTokens
	}
	return cfg.PrefillAttentionQueryChunkTokens
}

func hipGemma4Q4GenerateDeviceKVMode() (string, error) {
	return defaultHIPGemma4Q4EngineConfig().deviceKVMode()
}

func hipGemma4Q4ChunkedAttentionEnabled(promptTokens int) bool {
	return defaultHIPGemma4Q4EngineConfig().chunkedAttentionEnabled(promptTokens)
}

func hipGemma4Q4DeviceKVBlockSize() int {
	return defaultHIPGemma4Q4EngineConfig().deviceKVBlockSize()
}

func hipGemma4Q4GlobalDeviceKVBlockSize() int {
	return defaultHIPGemma4Q4EngineConfig().globalDeviceKVBlockSize()
}

func hipGemma4Q4DeviceKVBlockSizeForSlidingWindow(slidingWindow int) int {
	return defaultHIPGemma4Q4EngineConfig().deviceKVBlockSizeForSlidingWindow(slidingWindow)
}

func hipGemma4Q4AttentionWorkspaceNeeded(promptTokens int, generate inference.GenerateConfig) bool {
	return defaultHIPGemma4Q4EngineConfig().attentionWorkspaceNeeded(promptTokens, generate)
}

func hipGemma4Q4PrefillUBatchTokens() (int, error) {
	return defaultHIPGemma4Q4EngineConfig().prefillUBatchTokens()
}

func hipGemma4Q4PrefillAttentionQueryChunkTokens() int {
	return defaultHIPGemma4Q4EngineConfig().prefillAttentionQueryChunkTokens()
}
