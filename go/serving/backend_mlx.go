// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && !nomlx

package serving

import (
	"log/slog"

	"dappco.re/go"
	"dappco.re/go/inference"
)

// SetMLXMemoryLimits applies Metal cache and memory hard limits before the
// next call to NewMLXBackend / inference.LoadModel. Pass zero for either
// argument to leave that limit untouched. Spec §2.2 — memory management
// before loading.
//
//	ml.SetMLXMemoryLimits(4<<30, 32<<30) // 4 GB cache, 32 GB hard cap
//	ml.SetMLXMemoryLimits(0, 96<<30)     // memory only
func SetMLXMemoryLimits(cacheLimit, memoryLimit uint64) {
	if cacheLimit == 0 && memoryLimit == 0 {
		return
	}
	if _, ok := inference.SetRuntimeMemoryLimits("metal", inference.RuntimeMemoryLimits{
		CacheLimitBytes:  cacheLimit,
		MemoryLimitBytes: memoryLimit,
	}); !ok {
		slog.Warn("mlx: metal backend is not registered or does not expose memory limits")
	}
}

// NewMLXBackend loads a model via go-inference's Metal backend and wraps it
// in an InferenceAdapter for use as ml.Backend / ml.StreamingBackend.
//
// The application should import the concrete runtime package that registers
// "metal" with go-inference. Load options (context length, parallel slots,
// etc.) are forwarded directly to go-inference. Spec §2.2.
//
// Callers that need explicit Metal memory control should call
// ml.SetMLXMemoryLimits before NewMLXBackend; between probes use
// runtime.GC() to release unmanaged caches.
//
//	ml.SetMLXMemoryLimits(4<<30, 32<<30)
//	r := ml.NewMLXBackend("/models/gemma3-1b",
//	    inference.WithContextLen(8192),
//	)
func NewMLXBackend(modelPath string, loadOpts ...inference.LoadOption) core.Result {
	slog.Info("mlx: loading model via go-inference", "model_path", modelPath)

	opts := append(append([]inference.LoadOption(nil), loadOpts...), inference.WithBackend("metal"))
	result := inference.LoadModel(modelPath, opts...)
	if !result.OK {
		if err, ok := result.Value.(error); ok {
			return core.Fail(core.E("ml.NewMLXBackend", "metal backend", err))
		}
		return core.Fail(core.E("ml.NewMLXBackend", "metal backend failed to load model", nil))
	}
	m, ok := result.Value.(inference.TextModel)
	if !ok || m == nil {
		return core.Fail(core.E("ml.NewMLXBackend", "metal backend returned non-TextModel value", nil))
	}

	info := m.Info()
	slog.Info("mlx: model loaded",
		"arch", info.Architecture,
		"layers", info.NumLayers,
		"quant", info.QuantBits,
	)

	return core.Ok(NewInferenceAdapter(m, "mlx"))
}
