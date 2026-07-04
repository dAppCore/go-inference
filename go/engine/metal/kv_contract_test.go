// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"slices"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	g4 "dappco.re/go/inference/model/gemma4"
	"dappco.re/go/inference/tokenizer"
)

// newKVContractTokenModel builds a hermetic bf16 NativeTokenModel with a tiny
// real tokenizer attached (the textTestTokenizerJSON fixture, so the
// string-prompt contract path has a working Encode without a checkpoint). Small
// synthetic gemma4 arch → arbitrary text; the gate is the kv.Snapshot round-trip,
// not coherence.
func newKVContractTokenModel(t *testing.T) (*NativeTokenModel, *tokenizer.Tokenizer) {
	t.Helper()
	dir := t.TempDir()
	path := core.PathJoin(dir, "tokenizer.json")
	if r := core.WriteFile(path, []byte(textTestTokenizerJSON), 0o644); !r.OK {
		t.Fatalf("WriteFile: %v", r.Value)
	}
	tok, err := tokenizer.LoadTokenizer(path)
	if err != nil {
		t.Fatalf("LoadTokenizer: %v", err)
	}
	const dModel, nHeads, nKV, headDim, dFF, vocab = 128, 2, 1, 64, 256, 102
	const maxLen = 24
	arch, err := g4.Config{
		HiddenSize: dModel, NumHiddenLayers: 2, IntermediateSize: dFF,
		NumAttentionHeads: nHeads, NumKeyValueHeads: nKV, HeadDim: headDim,
		VocabSize: vocab, RMSNormEps: 1e-6,
	}.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	layers := make([]DecodeLayerWeights, len(arch.Layer))
	for li := range layers {
		layers[li] = forwardLayer(dModel, nHeads, nKV, headDim, dFF, (li+1)*100)
	}
	g := &BF16Model{
		Layers:    layers,
		Embed:     toBF16Bytes(syntheticFloat32(vocab*dModel, 11)),
		FinalNorm: toBF16Bytes(syntheticFloat32(dModel, 7)),
	}
	g.LMHead, g.Tied = g.Embed, true
	tm, err := NewBF16TokenModel(g, arch, maxLen)
	if err != nil {
		t.Fatalf("NewBF16TokenModel: %v", err)
	}
	tm.AttachTokenizer(tok)
	return tm, tok
}

func kvContractOpenArchSession(t *testing.T, tm *NativeTokenModel, scope string) *ArchSession {
	t.Helper()
	stepper, err := tm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession (%s): %v", scope, err)
	}
	sess, ok := stepper.(*ArchSession)
	if !ok {
		t.Fatalf("OpenSession (%s): session is %T, want *ArchSession", scope, stepper)
	}
	return sess
}

// TestNativeTokenModelCaptureKVRestoreFromKVContinues is the #259 round-trip
// through the inference contracts: capture a prompt's KV state as a portable
// kv.Snapshot (inference.KVSnapshotter, probed off the model exactly as the
// composition layer does), restore it into a fresh session
// (inference.KVRestorer), continue decoding — and the continuation is
// token-identical to the uninterrupted greedy run. No metal.KVSnapshot, no kvconv.
func TestNativeTokenModelCaptureKVRestoreFromKVContinues(t *testing.T) {
	requireNativeRuntime(t)
	tm, tok := newKVContractTokenModel(t)
	ctx := context.Background()
	const prompt = "hello"
	const maxNew = 5

	// capture through the contract interface — the exact probe root composition runs.
	var snapshotter inference.KVSnapshotter = tm
	snap, err := snapshotter.CaptureKV(ctx, prompt, inference.KVSnapshotCaptureOptions{})
	if err != nil {
		t.Fatalf("CaptureKV: %v", err)
	}
	if snap == nil || len(snap.Tokens) == 0 || len(snap.Layers) != len(tm.arch.Layer) {
		t.Fatalf("CaptureKV returned malformed snapshot: %+v", snap)
	}
	if !idsEqual(snap.Tokens, tok.Encode(prompt)) {
		t.Fatalf("snapshot tokens = %v, want tokenised prompt %v", snap.Tokens, tok.Encode(prompt))
	}

	// restore through the contract interface into a fresh session, continue decoding.
	restored := kvContractOpenArchSession(t, tm, "restore")
	defer func() { _ = restored.Close() }()
	var restorer inference.KVRestorer = restored
	if err := restorer.RestoreFromKV(ctx, snap); err != nil {
		t.Fatalf("RestoreFromKV: %v", err)
	}
	if restored.Pos() != len(snap.Tokens) {
		t.Fatalf("restored pos = %d, want %d", restored.Pos(), len(snap.Tokens))
	}
	got, err := restored.GenerateFromCache(maxNew, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after RestoreFromKV: %v", err)
	}

	// uninterrupted reference: fresh session, cold greedy Generate over the same ids.
	cold := kvContractOpenArchSession(t, tm, "cold")
	defer func() { _ = cold.Close() }()
	want, err := cold.Generate(tok.Encode(prompt), maxNew, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("capture→RestoreFromKV continuation = %v, want uninterrupted %v", got, want)
	}
	t.Logf("CaptureKV(%q)→RestoreFromKV→GenerateFromCache == cold Generate: %v (token-identical, kv.Snapshot-native, no kvconv)", prompt, got)
}

// TestNativeTokenModelCaptureKVChunksContinues gates the streaming-chunk capture
// contract (inference.KVChunkSnapshotter): chunks tokenised in order prefill the
// same ids as their concatenation, and the restored continuation is
// token-identical to the uninterrupted run over those ids.
func TestNativeTokenModelCaptureKVChunksContinues(t *testing.T) {
	requireNativeRuntime(t)
	tm, tok := newKVContractTokenModel(t)
	ctx := context.Background()
	chunks := []string{"he", "llo"}
	const maxNew = 4

	wantIDs := append(append([]int32(nil), tok.Encode(chunks[0])...), tok.Encode(chunks[1])...)

	var chunkSnapshotter inference.KVChunkSnapshotter = tm
	snap, err := chunkSnapshotter.CaptureKVChunks(ctx, slices.Values(chunks), inference.KVSnapshotCaptureOptions{})
	if err != nil {
		t.Fatalf("CaptureKVChunks: %v", err)
	}
	if !idsEqual(snap.Tokens, wantIDs) {
		t.Fatalf("chunk snapshot tokens = %v, want per-chunk concatenation %v", snap.Tokens, wantIDs)
	}

	restored := kvContractOpenArchSession(t, tm, "chunk-restore")
	defer func() { _ = restored.Close() }()
	if err := restored.RestoreFromKV(ctx, snap); err != nil {
		t.Fatalf("RestoreFromKV: %v", err)
	}
	got, err := restored.GenerateFromCache(maxNew, -1)
	if err != nil {
		t.Fatalf("GenerateFromCache after chunk RestoreFromKV: %v", err)
	}

	cold := kvContractOpenArchSession(t, tm, "chunk-cold")
	defer func() { _ = cold.Close() }()
	want, err := cold.Generate(wantIDs, maxNew, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}
	if !idsEqual(got, want) {
		t.Fatalf("chunk capture→restore continuation = %v, want uninterrupted %v", got, want)
	}
}

// TestNativeTokenModelKVContractGuards covers the honest failure edges: the
// string-prompt capture needs an attached tokenizer, and the ctx-shaped shims
// reject a nil snapshot / empty prompt rather than pretending to succeed.
func TestNativeTokenModelKVContractGuards(t *testing.T) {
	requireNativeRuntime(t)
	tm, _ := newKVContractTokenModel(t)
	ctx := context.Background()

	// no tokenizer attached → string-prompt capture is unavailable (the gap, surfaced not hidden).
	tm.AttachTokenizer(nil)
	if _, err := tm.CaptureKV(ctx, "hello", inference.KVSnapshotCaptureOptions{}); err == nil {
		t.Fatalf("CaptureKV without tokenizer: want error, got nil")
	}
	if _, err := tm.CaptureKVChunks(ctx, slices.Values([]string{"he"}), inference.KVSnapshotCaptureOptions{}); err == nil {
		t.Fatalf("CaptureKVChunks without tokenizer: want error, got nil")
	}

	sess := kvContractOpenArchSession(t, tm, "guards")
	defer func() { _ = sess.Close() }()
	if err := sess.RestoreFromKV(ctx, nil); err == nil {
		t.Fatalf("RestoreFromKV(nil): want error, got nil")
	}
	cancelled, cancel := context.WithCancel(ctx)
	cancel()
	if err := sess.RestoreFromKV(cancelled, &kv.Snapshot{}); err == nil {
		t.Fatalf("RestoreFromKV(cancelled ctx): want ctx error, got nil")
	}
}
