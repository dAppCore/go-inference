// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

// generate_supervised_test.go is the model-bound half of the generate coverage:
// it loads the REAL gemma-4 e2b checkpoint through the metal engine and drives
// RunGenerate's load → warm → decode path and its durable -state turn loop end to
// end — the arms (runBasicGenerate, runStateTurn, runStateSession, loadTextModel's
// success return) that a portable binary cannot reach. It follows the repo's
// supervised idiom: the file compiles only under -tags metal_runtime, and each
// test skips cleanly unless BOTH the checkpoint (HF cache) and the metallib
// (MLX_METALLIB_PATH → a loadable mlx.metallib on Apple GPU) are present.
//
// Opt-in:
//
//	MLX_METALLIB_PATH=/Users/snider/Code/core/go-inference/build/dist/lib/mlx.metallib \
//	go test -tags metal_runtime -run TestRunGenerate ./decode/generate/ -v

package generate_test

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/generate"
	"dappco.re/go/inference/internal/enginegate"

	_ "dappco.re/go/inference/engine/metal"  // registers the "metal" backend via init()
	_ "dappco.re/go/inference/model/builtin" // registers the built-in arches
)

// requireMetalReady skips unless the metal backend is registered AND available —
// an Apple GPU present and MLX_METALLIB_PATH resolving to a loadable metallib.
// It is the metallib half of the supervised gate; HFModelPath is the checkpoint
// half. Together they let this test run for real on a configured machine and
// skip cleanly everywhere else.
func requireMetalReady(t *testing.T) {
	t.Helper()
	b, ok := inference.Get("metal")
	if !ok {
		t.Skip("metal backend not registered")
	}
	if !b.Available() {
		t.Skip("metal backend unavailable — set MLX_METALLIB_PATH to a loadable mlx.metallib on Apple GPU hardware")
	}
}

// TestRunGenerate_BasicHappyPath_Real loads the real checkpoint and runs one tiny
// stateless generate, proving RunGenerate's load → warm → prefill → decode →
// render path produces non-empty output and a well-formed decode-metrics line.
func TestRunGenerate_BasicHappyPath_Real(t *testing.T) {
	requireMetalReady(t)
	modelPath := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")

	out := core.NewBuffer()
	err := generate.RunGenerate(context.Background(), generate.Config{
		ModelPath: modelPath,
		Prompt:    "Reply with a short greeting.",
		MaxTokens: 16,
		Temp:      0,
		Out:       out,
		Log:       core.NewBuffer(),
	})
	if err != nil {
		t.Fatalf("RunGenerate: %v", err)
	}
	body := out.String()
	if core.Trim(body) == "" {
		t.Fatal("RunGenerate produced empty output")
	}
	if !core.Contains(body, "decode") || !core.Contains(body, "tok/s") {
		t.Fatalf("output missing the decode-metrics line:\n%s", body)
	}
}

// TestRunGenerate_StateTurnResumes_Real proves the durable -state loop against a
// t.TempDir store: turn one opens a fresh session, generates, and sleeps KV
// blocks to disk (the store grows from nothing); turn two wakes that persisted
// state (no prompt replay) and resumes.
func TestRunGenerate_StateTurnResumes_Real(t *testing.T) {
	requireMetalReady(t)
	modelPath := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-4bit")
	storePath := core.PathJoin(t.TempDir(), "state", "agent.kv")

	turn := func(prompt string) string {
		out := core.NewBuffer()
		err := generate.RunGenerate(context.Background(), generate.Config{
			ModelPath:  modelPath,
			Prompt:     prompt,
			MaxTokens:  8,
			Temp:       0,
			StateName:  "gencov",
			StateStore: storePath,
			Out:        out,
			Log:        core.NewBuffer(),
		})
		if err != nil {
			t.Fatalf("RunGenerate(-state %q): %v", prompt, err)
		}
		return out.String()
	}

	first := turn("My name is Ada. Remember it.")
	if !core.Contains(first, "fresh state") {
		t.Fatalf("turn one should open fresh state:\n%s", first)
	}
	if !core.Contains(first, "slept") {
		t.Fatalf("turn one should sleep KV blocks to the store:\n%s", first)
	}
	read := core.ReadFile(storePath)
	if !read.OK {
		t.Fatalf("state store not written at %s: %v", storePath, read.Value)
	}
	if bytes, _ := read.Value.([]byte); len(bytes) == 0 {
		t.Fatal("state store is empty after turn one — nothing persisted")
	}

	second := turn("What is my name?")
	if !core.Contains(second, "woke") {
		t.Fatalf("turn two should wake the persisted state (no replay):\n%s", second)
	}
}
