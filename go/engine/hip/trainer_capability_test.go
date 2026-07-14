// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
)

// TestHIPTrainerCapabilityBoundary_Good protects the honest training boundary:
// loaded Gemma4 models may expose experimental forward/loss helpers, but they
// must not advertise engine.TrainerModel until their adapter lifecycle is real.
func TestHIPTrainerCapabilityBoundary_Good(t *testing.T) {
	loaded := &hipLoadedModel{
		modelInfo:   inference.ModelInfo{Architecture: "gemma4", VocabSize: 107, NumLayers: 1, HiddenSize: 8, QuantBits: 4},
		contextSize: 4096,
	}
	tokenModel := newHipTokenModel(loaded, nil, "gemma4")
	if _, ok := any(tokenModel).(engine.TrainerModel); ok {
		t.Fatal("hipTokenModel must not advertise engine.TrainerModel before a real Gemma4 LoRA lifecycle exists")
	}

	shared, err := newHipEngineTextModel(loaded, nil, "gemma4")
	core.RequireNoError(t, err)
	_, err = shared.OpenTrainer(inference.TrainingConfig{LoRA: inference.LoRAConfig{Rank: 8, Alpha: 16}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "does not support training")
}
