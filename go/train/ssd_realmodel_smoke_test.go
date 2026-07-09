// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package train_test

import (
	"context"
	"os"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/train"
	"dappco.re/go/inference/train/dataset"

	_ "dappco.re/go/inference/engine/metal"  // registers the "metal" backend via init()
	_ "dappco.re/go/inference/model/builtin" // registers the built-in arches
)

// TestRunSSDModelRealGemma4ViaEngineMetal proves the SSD sampling path samples a
// REAL gemma4 checkpoint through the metal engine and captures the trace — the
// user path the synthetic fakes cannot exercise. SSD never trains, so this is
// generation + capture end to end on the GPU (green compile only proves it
// links). Default build/test never runs it: metal_runtime-gated + env-skip.
//
// Opt-in (needs the Apple GPU + a downloaded snapshot):
//
//	GO_INFERENCE_SMOKE_MODEL=~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/<hash> \
//	MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
//	go test -tags metal_runtime -run TestRunSSDModelRealGemma4ViaEngineMetal ./train/ -v
func TestRunSSDModelRealGemma4ViaEngineMetal(t *testing.T) {
	modelPath := os.Getenv("GO_INFERENCE_SMOKE_MODEL")
	if modelPath == "" {
		t.Skip("set GO_INFERENCE_SMOKE_MODEL to a gemma4 snapshot dir (with tokenizer.json) to run the real-model SSD smoke")
	}

	res := inference.LoadModel(modelPath, inference.WithBackend("metal"), inference.WithContextLen(4096))
	if !res.OK {
		t.Fatalf("LoadModel(%q): %v", modelPath, res.Value)
	}
	model := res.Value.(inference.TextModel)
	t.Cleanup(func() { model.Close() })

	ds := dataset.NewSliceDataset([]dataset.Sample{
		{Prompt: "Name a colour."},
		{Prompt: "Name a fruit."},
	})
	dir := t.TempDir()
	cfg := train.SSDConfig{
		SampleMaxTokens:   16,
		SampleTemperature: 0.7, // non-unit — diversity is the point of SSD sampling
		CheckpointDir:     dir,
		// A Score hook could be supplied here; go-inference has no scorer yet, so
		// this smoke proves the sample+capture lane (scoring later is archaeology).
	}

	result, err := train.RunSSDModel(context.Background(), model, ds, cfg, nil)
	if err != nil {
		t.Fatalf("RunSSDModel: %v", err)
	}
	if len(result.Samples) == 0 {
		t.Fatal("SSD produced no self-samples — the sampling path is broken")
	}
	for i, s := range result.Samples {
		t.Logf("SSD sample %d: prompt=%q response=%q", i, s.Prompt, strings.TrimSpace(s.Response))
		if strings.TrimSpace(s.Response) == "" {
			t.Fatalf("SSD sample %d has an empty response — generation produced nothing", i)
		}
	}
	if result.CaptureSidecar == "" {
		t.Fatal("SSD did not record a capture sidecar path")
	}
	captured := core.ReadFile(result.CaptureSidecar)
	if !captured.OK {
		t.Fatalf("read capture sidecar %q: %v", result.CaptureSidecar, captured.Value)
	}
	if len(captured.Value.([]byte)) == 0 {
		t.Fatal("capture sidecar is empty — a captured self-output never existed")
	}
	t.Logf("SSD smoke receipt: %d self-samples captured to %s on the real gemma4 metal path", len(result.Samples), result.CaptureSidecar)
}
