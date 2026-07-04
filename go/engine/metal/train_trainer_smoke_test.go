// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"math"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/tokenizer"
)

// train_trainer_smoke_test.go is THE receipt for the train seam: a full SFT round trip on a real
// gemma-4 E2B (PLE) base through the new engine.Trainer seam — load → a synthetic {messages} dataset →
// a run of head-LoRA Steps on the metal train kernels → loss falls → Save the adapter → RELOAD it via
// engine/metal's AdapterPath → generate through the adapted head. It proves TRAIN and APPLY end to end
// on the GPU (green compile only proves it links). metal_runtime-gated (needs MLX_METALLIB_PATH + the
// cached bf16 E2B checkpoint); skips cleanly when either is absent.

// gemma4E2BBf16Dir resolves the cached mlx-community/gemma-4-E2B-it-bf16 snapshot, skipping when it is
// not present so the smoke is a no-op on a machine without the checkpoint.
func gemma4E2BBf16Dir(t *testing.T) string {
	t.Helper()
	base := filepath.Join(os.Getenv("HOME"),
		".cache/huggingface/hub/models--mlx-community--gemma-4-E2B-it-bf16/snapshots")
	entries, err := os.ReadDir(base)
	if err != nil || len(entries) == 0 {
		t.Skip("gemma-4-E2B-it-bf16 not cached")
	}
	return filepath.Join(base, entries[0].Name())
}

func TestLoRATrainerHeadSFTSmokeE2B(t *testing.T) {
	requireNativeRuntime(t)
	dir := gemma4E2BBf16Dir(t)
	// maxLen must exceed the model's sliding window (512 for E2B) so the sliding-window layers use the
	// ring KV cache — the same path normal generation takes with the 4096 default.
	const maxLen = 1024

	tok, err := tokenizer.LoadTokenizer(filepath.Join(dir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	// Tiny synthetic {messages} SFT dataset. Short sequences keep the full-vocab head cross-entropy
	// cheap; the trainer learns to predict each sequence's own next token (causal SFT).
	samples := []inference.DatasetSample{
		{Messages: []inference.Message{
			{Role: "user", Content: "What colour is the sky on a clear day?"},
			{Role: "assistant", Content: "The sky is blue."},
		}},
		{Messages: []inference.Message{
			{Role: "user", Content: "Name a fruit that is yellow."},
			{Role: "assistant", Content: "A banana is yellow."},
		}},
	}
	var tokenIDs [][]int32
	for _, s := range samples {
		text := ""
		for _, m := range s.Messages {
			text += m.Role + ": " + m.Content + "\n"
		}
		seq := tok.Encode(text)
		if len(seq) > maxLen {
			seq = seq[:maxLen]
		}
		if len(seq) < 2 {
			t.Fatalf("encoded sequence too short: %d tokens", len(seq))
		}
		tokenIDs = append(tokenIDs, seq)
	}
	batch := inference.Batch{TokenIDs: tokenIDs}

	// Open the trainer over the real E2B base and run the SFT loop.
	baseModel, err := LoadTokenModelDir(dir, maxLen)
	if err != nil {
		t.Fatalf("load token model: %v", err)
	}
	ntm, ok := baseModel.(*NativeTokenModel)
	if !ok {
		t.Fatalf("loader did not return a NativeTokenModel: %T", baseModel)
	}
	trainer, err := NewLoRATrainer(ntm, inference.TrainingConfig{
		LoRA:         inference.LoRAConfig{Rank: 8, Alpha: 16},
		LearningRate: 0.02,
	})
	if err != nil {
		t.Fatalf("open trainer: %v", err)
	}
	defer func() { _ = trainer.Close() }()

	loss0, err := trainer.Loss(batch)
	if err != nil {
		t.Fatalf("initial loss: %v", err)
	}
	if math.IsNaN(loss0) || math.IsInf(loss0, 0) {
		t.Fatalf("initial loss not finite: %v", loss0)
	}
	const steps = 40
	var lossLast float64
	for s := 0; s < steps; s++ {
		l, serr := trainer.Step(batch)
		if serr != nil {
			t.Fatalf("step %d: %v", s, serr)
		}
		if math.IsNaN(l) || math.IsInf(l, 0) {
			t.Fatalf("step %d loss not finite: %v", s, l)
		}
		lossLast = l
		if s%5 == 0 || s == steps-1 {
			t.Logf("SFT step %d: loss %.4f", s, l)
		}
	}
	if lossLast >= loss0 {
		t.Fatalf("head-LoRA SFT did not reduce loss: first=%.4f last=%.4f", loss0, lossLast)
	}

	// Save the trained adapter (go-mlx on-disk format).
	adapterDir := filepath.Join(t.TempDir(), "adapter")
	if err := trainer.Save(adapterDir); err != nil {
		t.Fatalf("save adapter: %v", err)
	}
	if _, serr := os.Stat(filepath.Join(adapterDir, "adapter.safetensors")); serr != nil {
		t.Fatalf("adapter.safetensors not written: %v", serr)
	}
	_ = trainer.Close() // free the trainer's base session before the reloads

	// A prefix of a trained sequence — greedy generation from it should DIFFER once the trained head
	// is applied, proving the adapter round-trips into inference.
	prompt := tokenIDs[0]
	if len(prompt) > 6 {
		prompt = prompt[:6]
	}
	baseGen := greedyGenerate(t, dir, maxLen, "", prompt, 16)            // no adapter
	adaptedGen := greedyGenerate(t, dir, maxLen, adapterDir, prompt, 16) // reload via AdapterPath

	// The registered backend must also accept the WithAdapterPath option end to end.
	res := metalBackend{}.LoadModel(dir, inference.WithAdapterPath(adapterDir))
	if !res.OK {
		t.Fatalf("metalBackend.LoadModel WithAdapterPath: %v", res.Value)
	}

	t.Logf("base greedy gen:    %v", baseGen)
	t.Logf("adapted greedy gen: %v", adaptedGen)
	if slices.Equal(baseGen, adaptedGen) {
		t.Fatalf("adapter did not change greedy generation — apply-at-inference was a no-op")
	}
	t.Logf("train seam receipt: head-LoRA SFT loss %.4f -> %.4f over %d steps; adapter saved + reloaded via AdapterPath and changed greedy generation on the real E2B GPU path",
		loss0, lossLast, steps)
}

// greedyGenerate loads the E2B model (optionally with adapterDir applied via AdapterPath), opens a fresh
// session, and greedily decodes maxNew tokens from prompt. It closes the model before returning.
func greedyGenerate(t *testing.T, dir string, maxLen int, adapterDir string, prompt []int32, maxNew int) []int32 {
	t.Helper()
	tm, err := LoadTokenModelDirWithConfig(dir, maxLen, TokenModelLoadConfig{AdapterPath: adapterDir})
	if err != nil {
		t.Fatalf("reload (adapter=%q): %v", adapterDir, err)
	}
	ntm := tm.(*NativeTokenModel)
	defer func() { _ = ntm.Close() }()
	stepper, err := ntm.OpenSession()
	if err != nil {
		t.Fatalf("open session (adapter=%q): %v", adapterDir, err)
	}
	sess := stepper.(*ArchSession)
	defer func() { _ = sess.Close() }()
	gen, err := sess.Generate(prompt, maxNew, -1)
	if err != nil {
		t.Fatalf("generate (adapter=%q): %v", adapterDir, err)
	}
	if len(gen) != maxNew {
		t.Fatalf("generate (adapter=%q) produced %d tokens, want %d", adapterDir, len(gen), maxNew)
	}
	return gen
}
