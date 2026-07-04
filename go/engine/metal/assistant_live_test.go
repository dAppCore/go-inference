// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"testing"

	"dappco.re/go/inference/internal/enginegate"
)

func TestRealE2BAssistantLoadMetadata(t *testing.T) {
	targetDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	assistantDir := enginegate.HFModelPath(t, "mlx-community/gemma-4-E2B-it-assistant-bf16")
	pair, err := LoadAssistantPairDirs(targetDir, assistantDir)
	if err != nil {
		t.Fatalf("LoadAssistantPairDirs(%s, %s): %v", targetDir, assistantDir, err)
	}
	defer pair.Close()

	assistant := pair.Assistant
	if assistant.ModelType() != "gemma4_assistant" {
		t.Fatalf("ModelType = %q, want gemma4_assistant", assistant.ModelType())
	}
	if assistant.NumLayers() != 4 {
		t.Fatalf("NumLayers = %d, want 4", assistant.NumLayers())
	}
	if assistant.BackboneHiddenSize <= 0 || assistant.Arch.Hidden <= 0 || assistant.Arch.Vocab <= 0 {
		t.Fatalf("assistant metadata = backbone %d arch %+v", assistant.BackboneHiddenSize, assistant.Arch)
	}
	if _, ok := assistant.Tensor("pre_projection.weight"); !ok {
		t.Fatal("pre_projection.weight was not retained")
	}
	if _, ok := assistant.Tensor("post_projection.weight"); !ok {
		t.Fatal("post_projection.weight was not retained")
	}
}
