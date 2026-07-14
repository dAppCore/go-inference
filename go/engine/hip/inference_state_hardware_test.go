// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"bytes"
	"context"
	"os"
	"slices"
	"strings"
	"testing"

	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
)

// TestHIPGemma4ExactStateContinuityHardware_Good is the production ROCm
// receipt for the shared engine session path. It uses the linked Gemma-4 Q4
// model, rather than a host-only session fixture, to prove that a resident
// session can extend its prefetched prompt, export exact KV, range that KV, and
// continue token-identically after a fresh session imports the snapshot.
func TestHIPGemma4ExactStateContinuityHardware_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run the HIP Gemma-4 state receipt")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set GO_ROCM_MODEL_PATH to a linked Gemma-4 Q4 GGUF")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to the linked ROCm kernels HSACO")
	}
	if !ROCmAvailable() {
		t.Skip("ROCm runtime is not available on this host")
	}

	loaded := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(256))
	if !loaded.OK {
		t.Fatalf("production ROCm LoadModel(%q): %v", modelPath, loaded.Value)
	}
	model, ok := loaded.Value.(*rocmModel)
	if !ok {
		t.Fatalf("production ROCm LoadModel returned %T, want *rocmModel", loaded.Value)
	}
	defer func() {
		if result := model.Close(); !result.OK {
			t.Errorf("Close model: %v", result.Value)
		}
	}()
	if model.engineModel == nil {
		t.Fatal("production ROCm model did not compose the shared engine.TextModel")
	}

	ctx := context.Background()
	prompt := "Answer with one word: what color is a clear daytime sky?"
	appendPrompt := "\nNow answer with one word: what follows Monday?"
	appendIDs, err := model.Tokenize(appendPrompt)
	if err != nil {
		t.Fatalf("Tokenize append prompt: %v", err)
	}
	if len(appendIDs) == 0 {
		t.Fatal("append prompt tokenized to no tokens")
	}

	source := model.NewSession()
	if source == nil {
		t.Fatal("production ROCm model NewSession returned nil shared engine session")
	}
	defer func() { _ = source.Close() }()
	if err := source.Prefill(ctx, prompt); err != nil {
		t.Fatalf("Prefill: %v", err)
	}
	positioned, ok := source.(interface{ Pos() int })
	if !ok {
		t.Fatal("shared engine session does not expose its retained position")
	}
	prefilled := positioned.Pos()
	if prefilled == 0 {
		t.Fatal("Prefill left no retained prompt tokens")
	}
	if err := source.AppendPrompt(ctx, appendPrompt); err != nil {
		t.Fatalf("AppendPrompt on the same session: %v", err)
	}
	if got, want := positioned.Pos(), prefilled+len(appendIDs); got != want {
		t.Fatalf("same-session prompt reuse position = %d, want %d (retained prefix + appended tokens)", got, want)
	}

	if got := collectHIPHardwareTokens(source.Generate(ctx, inference.GenerateConfig{MaxTokens: 2})); len(got) == 0 {
		t.Fatalf("seed Generate produced no tokens: %v", source.Err())
	}
	if err := source.Err(); err != nil {
		t.Fatalf("seed Generate: %v", err)
	}

	snapshot, err := source.CaptureKV(ctx)
	if err != nil {
		t.Fatalf("CaptureKV: %v", err)
	}
	if snapshot == nil || snapshot.SeqLen == 0 || len(snapshot.Layers) == 0 {
		t.Fatalf("CaptureKV produced incomplete state: %+v", snapshot)
	}
	var rangedTokens int
	if err := source.RangeKVBlocks(ctx, 2, kv.CaptureOptions{}, func(block kv.Block) (bool, error) {
		if block.TokenCount <= 0 || block.Snapshot == nil {
			t.Fatalf("RangeKVBlocks produced invalid block: %+v", block)
		}
		if block.TokenStart != rangedTokens {
			t.Fatalf("RangeKVBlocks block start = %d, want contiguous %d", block.TokenStart, rangedTokens)
		}
		rangedTokens += block.TokenCount
		return true, nil
	}); err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	if rangedTokens != snapshot.SeqLen {
		t.Fatalf("RangeKVBlocks covered %d KV tokens, CaptureKV exported %d", rangedTokens, snapshot.SeqLen)
	}

	restored := model.NewSession()
	if restored == nil {
		t.Fatal("NewSession for restored state returned nil")
	}
	defer func() { _ = restored.Close() }()
	restorer, ok := restored.(inference.KVRestorer)
	if !ok {
		t.Fatal("shared engine session does not implement inference.KVRestorer")
	}
	if err := restorer.RestoreFromKV(ctx, snapshot); err != nil {
		t.Fatalf("RestoreFromKV: %v", err)
	}
	restoredSnapshot, err := restored.CaptureKV(ctx)
	if err != nil {
		t.Fatalf("CaptureKV after restore: %v", err)
	}
	assertHIPHardwareSnapshotEqual(t, snapshot, restoredSnapshot)

	continuation := "\nReply with exactly the next weekday after Tuesday:"
	if err := source.AppendPrompt(ctx, continuation); err != nil {
		t.Fatalf("source AppendPrompt continuation: %v", err)
	}
	if err := restored.AppendPrompt(ctx, continuation); err != nil {
		t.Fatalf("restored AppendPrompt continuation: %v", err)
	}
	want := collectHIPHardwareTokens(source.Generate(ctx, inference.GenerateConfig{MaxTokens: 4}))
	if err := source.Err(); err != nil {
		t.Fatalf("source continuation Generate: %v", err)
	}
	got := collectHIPHardwareTokens(restored.Generate(ctx, inference.GenerateConfig{MaxTokens: 4}))
	if err := restored.Err(); err != nil {
		t.Fatalf("restored continuation Generate: %v", err)
	}
	if len(got) != len(want) {
		t.Fatalf("continuation length after restore = %d, want %d", len(got), len(want))
	}
	for index := range want {
		if got[index] != want[index] {
			t.Fatalf("continuation token %d after restore = %d, want %d", index, got[index], want[index])
		}
	}

	canceled, cancel := context.WithCancel(ctx)
	cancel()
	if err := restored.RangeKVBlocks(canceled, 2, kv.CaptureOptions{}, func(kv.Block) (bool, error) {
		return true, nil
	}); err != context.Canceled {
		t.Fatalf("RangeKVBlocks with canceled context = %v, want %v", err, context.Canceled)
	}
}

func TestHIPGemma4SeededSamplingHardware_Good(t *testing.T) {
	if os.Getenv("GO_ROCM_RUN_MODEL_TESTS") != "1" {
		t.Skip("set GO_ROCM_RUN_MODEL_TESTS=1 to run the HIP Gemma-4 sampled-session receipt")
	}
	modelPath := strings.TrimSpace(os.Getenv("GO_ROCM_MODEL_PATH"))
	if modelPath == "" {
		t.Skip("set GO_ROCM_MODEL_PATH to a linked Gemma-4 Q4 GGUF")
	}
	if strings.TrimSpace(os.Getenv("GO_ROCM_KERNEL_HSACO")) == "" {
		t.Skip("set GO_ROCM_KERNEL_HSACO to the linked ROCm kernels HSACO")
	}
	if !ROCmAvailable() {
		t.Skip("ROCm runtime is not available on this host")
	}

	loaded := (&rocmBackend{}).LoadModel(modelPath, inference.WithContextLen(128))
	if !loaded.OK {
		t.Fatalf("production ROCm LoadModel(%q): %v", modelPath, loaded.Value)
	}
	model, ok := loaded.Value.(*rocmModel)
	if !ok {
		t.Fatalf("production ROCm LoadModel returned %T, want *rocmModel", loaded.Value)
	}
	defer func() {
		if result := model.Close(); !result.OK {
			t.Errorf("Close model: %v", result.Value)
		}
	}()

	ctx := context.Background()
	prompt := "Continue this short sequence with a few words: sunrise, morning, noon,"
	first := model.NewSession()
	second := model.NewSession()
	if first == nil || second == nil {
		t.Fatal("production ROCm model returned a nil shared engine session")
	}
	defer func() { _ = first.Close() }()
	defer func() { _ = second.Close() }()
	if err := first.Prefill(ctx, prompt); err != nil {
		t.Fatalf("first Prefill: %v", err)
	}
	if err := second.Prefill(ctx, prompt); err != nil {
		t.Fatalf("second Prefill: %v", err)
	}
	generate := inference.GenerateConfig{MaxTokens: 6, Temperature: 0.8, TopK: 40, TopP: 0.9, Seed: 0x5eed, SeedSet: true}
	want := collectHIPHardwareTokens(first.Generate(ctx, generate))
	if err := first.Err(); err != nil {
		t.Fatalf("first sampled Generate: %v", err)
	}
	got := collectHIPHardwareTokens(second.Generate(ctx, generate))
	if err := second.Err(); err != nil {
		t.Fatalf("second sampled Generate: %v", err)
	}
	if len(want) == 0 || !slices.Equal(got, want) {
		t.Fatalf("same-seed sampled tokens = %v, want %v", got, want)
	}

	snapshot, err := first.CaptureKV(ctx)
	if err != nil {
		t.Fatalf("CaptureKV after sampled generation: %v", err)
	}
	restored := model.NewSession()
	if restored == nil {
		t.Fatal("NewSession for sampled restore returned nil")
	}
	defer func() { _ = restored.Close() }()
	restorer, ok := restored.(inference.KVRestorer)
	if !ok {
		t.Fatal("sampled shared engine session does not implement inference.KVRestorer")
	}
	if err := restorer.RestoreFromKV(ctx, snapshot); err != nil {
		t.Fatalf("RestoreFromKV after sampled generation: %v", err)
	}
	continuation := " then comes"
	if err := first.AppendPrompt(ctx, continuation); err != nil {
		t.Fatalf("first AppendPrompt: %v", err)
	}
	if err := restored.AppendPrompt(ctx, continuation); err != nil {
		t.Fatalf("restored AppendPrompt: %v", err)
	}
	generate.Seed = 0xc0ffee
	generate.MaxTokens = 4
	want = collectHIPHardwareTokens(first.Generate(ctx, generate))
	if err := first.Err(); err != nil {
		t.Fatalf("first sampled continuation: %v", err)
	}
	got = collectHIPHardwareTokens(restored.Generate(ctx, generate))
	if err := restored.Err(); err != nil {
		t.Fatalf("restored sampled continuation: %v", err)
	}
	if len(want) == 0 || !slices.Equal(got, want) {
		t.Fatalf("restored same-seed sampled tokens = %v, want %v", got, want)
	}
}

func collectHIPHardwareTokens(seq func(func(inference.Token) bool)) []int32 {
	var ids []int32
	seq(func(token inference.Token) bool {
		ids = append(ids, token.ID)
		return true
	})
	return ids
}

func assertHIPHardwareSnapshotEqual(t testing.TB, want, got *kv.Snapshot) {
	t.Helper()
	if got == nil {
		t.Fatal("restored snapshot is nil")
	}
	if want.Architecture != got.Architecture || want.SeqLen != got.SeqLen || want.NumLayers != got.NumLayers || len(want.Layers) != len(got.Layers) {
		t.Fatalf("restored snapshot geometry = %+v, want %+v", got, want)
	}
	if !slices.Equal(want.Tokens, got.Tokens) {
		t.Fatalf("restored snapshot tokens = %v, want %v", got.Tokens, want.Tokens)
	}
	for index := range want.Layers {
		if want.Layers[index].CacheIndex != got.Layers[index].CacheIndex {
			t.Fatalf("restored snapshot layer %d device page block size = %d, want %d", index, got.Layers[index].CacheIndex, want.Layers[index].CacheIndex)
		}
		if want.Layers[index].CacheMode != got.Layers[index].CacheMode || len(want.Layers[index].TurboQuantPayloads) != len(got.Layers[index].TurboQuantPayloads) {
			t.Fatalf("restored snapshot layer %d device page representation = mode %q pages %d, want mode %q pages %d", index, got.Layers[index].CacheMode, len(got.Layers[index].TurboQuantPayloads), want.Layers[index].CacheMode, len(want.Layers[index].TurboQuantPayloads))
		}
		for page := range want.Layers[index].TurboQuantPayloads {
			if !bytes.Equal(want.Layers[index].TurboQuantPayloads[page], got.Layers[index].TurboQuantPayloads[page]) {
				t.Fatalf("restored snapshot layer %d device page %d differs from the exported exact KV state", index, page)
			}
		}
		if !bytes.Equal(want.Layers[index].KeyBytes, got.Layers[index].KeyBytes) ||
			!bytes.Equal(want.Layers[index].ValueBytes, got.Layers[index].ValueBytes) {
			t.Fatalf("restored snapshot layer %d differs from the exported exact KV state", index)
		}
	}
}
