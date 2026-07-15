// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/engine"
)

// TestHIPTrainerCapabilityBoundary_Good protects the linked shared trainer
// boundary while keeping unsupported model payloads as clean open failures.
func TestHIPTrainerCapabilityBoundary_Good(t *testing.T) {
	loaded := &hipLoadedModel{
		modelInfo:   inference.ModelInfo{Architecture: "gemma4", VocabSize: 107, NumLayers: 1, HiddenSize: 8, QuantBits: 4},
		contextSize: 4096,
	}
	tokenModel := newHipTokenModel(loaded, nil, "gemma4")
	if _, ok := any(tokenModel).(engine.TrainerModel); !ok {
		t.Fatal("hipTokenModel must advertise engine.TrainerModel through its retained Gemma4 LoRA lifecycle")
	}

	shared, err := newHipEngineTextModel(loaded, nil, "gemma4")
	core.RequireNoError(t, err)
	_, err = shared.OpenTrainer(inference.TrainingConfig{LoRA: inference.LoRAConfig{Rank: 8, Alpha: 16}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "linked Gemma4 runtime")

	report := rocmCapabilityReport(nativeDeviceInfo{}, inference.ModelIdentity{
		Path:          "/models/lmstudio-community-gemma-4-e2b-it-4bit",
		Architecture:  "gemma4",
		VocabSize:     262144,
		NumLayers:     35,
		HiddenSize:    1536,
		QuantBits:     4,
		QuantGroup:    64,
		ContextLength: 131072,
	}, inference.AdapterIdentity{}, true, defaultHIPKernelStatus(), rocmCapabilityReportOption{Gemma4Q4GenerateLinked: true})
	capability, ok := report.Capability(inference.CapabilityLoRATraining)
	if !ok || capability.Status != inference.CapabilityStatusExperimental {
		t.Fatalf("LoRA training capability = %+v ok=%v, want experimental linked lifecycle", capability, ok)
	}
	core.AssertEqual(t, "engine.TrainerModel", capability.Labels["training_interface"])
	core.AssertEqual(t, hipKernelStatusLinked, capability.Labels["training_kernel"])
}

func TestROCmModelOpenTrainerForwardsSharedEngine_Good(t *testing.T) {
	loaded := &hipLoadedModel{
		modelInfo:   inference.ModelInfo{Architecture: "gemma4", VocabSize: 107, NumLayers: 1, HiddenSize: 8, QuantBits: 4},
		contextSize: 4096,
	}
	shared, err := newHipEngineTextModel(loaded, nil, "gemma4")
	core.RequireNoError(t, err)
	model := &rocmModel{native: loaded, engineModel: shared}
	if _, ok := any(model).(engine.TrainerModel); !ok {
		t.Fatal("rocmModel must expose the shared engine.TrainerModel contract")
	}

	_, err = model.OpenTrainer(inference.TrainingConfig{LoRA: inference.LoRAConfig{Rank: 8, Alpha: 16}})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "linked Gemma4 runtime")
	_, err = (*rocmModel)(nil).OpenTrainer(inference.TrainingConfig{})
	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "shared engine model is not available")
}
