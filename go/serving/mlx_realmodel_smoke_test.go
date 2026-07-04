// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package serving_test

import (
	"context"
	"os"
	"strings"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/serving"

	_ "dappco.re/go/inference/engine/metal"  // registers the "metal" backend via init()
	_ "dappco.re/go/inference/model/builtin" // registers the built-in arches (composition wires arches, not the engine)
)

// TestServeRealGemma4ViaEngineMetal proves the metal engine serves a REAL
// gemma4 checkpoint end-to-end from go-inference alone — the user path the
// synthetic enginetest fixture cannot exercise. It is the "is the engine
// actually answering a user" check the conformance suite is a hypothesis for.
//
// Opt-in (needs the Apple GPU + a downloaded snapshot):
//
//	GO_INFERENCE_SMOKE_MODEL=~/.cache/huggingface/hub/models--mlx-community--gemma-4-e2b-it-4bit/snapshots/<hash> \
//	MLX_METALLIB_PATH=/Users/snider/Code/core/go-mlx/dist/lib/mlx.metallib \
//	go test -tags metal_runtime -run TestServeRealGemma4ViaEngineMetal ./serving/ -v
func TestServeRealGemma4ViaEngineMetal(t *testing.T) {
	modelPath := os.Getenv("GO_INFERENCE_SMOKE_MODEL")
	if modelPath == "" {
		t.Skip("set GO_INFERENCE_SMOKE_MODEL to a gemma4 snapshot dir (with tokenizer.json) to run the real-model serve smoke")
	}

	r := serving.NewMLXBackend(modelPath, inference.WithContextLen(4096))
	if !r.OK {
		t.Fatalf("NewMLXBackend(%q): %v", modelPath, r.Value)
	}
	backend := r.Value.(*serving.InferenceAdapter)
	t.Cleanup(func() { backend.Close() })

	// A fact a tiny model reliably answers without reasoning — this proves the
	// SERVE PATH (load → prefill → decode → stop), not the model's competence.
	noThink := false
	res := backend.Chat(context.Background(),
		[]serving.Message{{Role: "user", Content: "What is the capital of France? Answer in one word."}},
		serving.GenOpts{MaxTokens: 64, Temperature: 0, EnableThinking: &noThink},
	)
	if !res.OK {
		t.Fatalf("Chat: %v", res.Value)
	}
	got := strings.TrimSpace(res.Value.(serving.Result).Text)
	t.Logf("engine/metal answered: %q", got)
	if got == "" {
		t.Fatal("engine/metal returned an empty answer — the serve path is broken")
	}
	if !strings.Contains(strings.ToLower(got), "paris") {
		t.Errorf("expected a coherent answer containing 'Paris', got %q", got)
	}
}
