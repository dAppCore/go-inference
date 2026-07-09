// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"slices"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	"dappco.re/go/inference/kv"
	g4 "dappco.re/go/inference/model/gemma4"
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

// TestSessionStateBlockNonNativeEncodingRoundTrip is the #290 gate: a non-native
// (RawKVOnly=false) block capture carries per-head float32 instead of the
// layer-level raw bf16 slab, so a q8/float32 KV snapshot Save no longer fails
// with errRawTensorNeedsNative and round-trips through Save→Load→Restore. float32
// is lossless → the restored continuation is token-identical to the uninterrupted
// run; q8 is lossy → it must round-trip without error and continue with in-range
// tokens. This is the direct codec proof (the generate -state WAKE glue is a
// separate, pre-existing gap; the block restore path itself is exercised here).
func TestSessionStateBlockNonNativeEncodingRoundTrip(t *testing.T) {
	requireNativeRuntime(t)
	tm, tok := newKVContractTokenModel(t)
	ctx := context.Background()
	const prompt = "hello"
	const maxNew = 5
	ids := tok.Encode(prompt)

	saved := kvContractOpenArchSession(t, tm, "nonnative-save")
	defer func() { _ = saved.Close() }()
	if err := saved.PrefillTokens(ids); err != nil {
		t.Fatalf("PrefillTokens: %v", err)
	}

	// Capture on the NON-NATIVE path (RawKVOnly=false) — the q8/float32-destined
	// block shape #290 wires. One block covers the short prompt.
	var blocks []kv.Block
	if err := saved.RangeKVBlocks(len(ids)+8, kv.CaptureOptions{RawKVOnly: false}, func(b kv.Block) (bool, error) {
		blocks = append(blocks, b)
		return true, nil
	}); err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	if len(blocks) != 1 || blocks[0].Snapshot == nil {
		t.Fatalf("expected 1 non-nil block for a short prompt, got %d", len(blocks))
	}
	snap := blocks[0].Snapshot

	// #290 shape: non-native capture carries per-head float32, NOT the layer-level
	// raw slab (which could only encode as native).
	for _, layer := range snap.Layers {
		if len(layer.KeyBytes) != 0 || len(layer.ValueBytes) != 0 {
			t.Fatalf("non-native block layer %d still carries the layer-level raw KV slab", layer.Layer)
		}
		if len(layer.Heads) == 0 {
			t.Fatalf("non-native block layer %d has no per-head tensors to quantise", layer.Layer)
		}
	}

	// Uninterrupted reference continuation.
	cold := kvContractOpenArchSession(t, tm, "nonnative-cold")
	defer func() { _ = cold.Close() }()
	want, err := cold.Generate(ids, maxNew, -1)
	if err != nil {
		t.Fatalf("cold Generate: %v", err)
	}

	dir := t.TempDir()
	roundTrip := func(enc kv.Encoding, name string) []int32 {
		path := core.PathJoin(dir, name+".kv")
		if err := snap.SaveWithOptions(path, kv.SaveOptions{KVEncoding: enc}); err != nil {
			t.Fatalf("Save %s: %v", name, err)
		}
		loaded, err := kv.Load(path)
		if err != nil {
			t.Fatalf("Load %s: %v", name, err)
		}
		restored := kvContractOpenArchSession(t, tm, name+"-restore")
		defer func() { _ = restored.Close() }()
		if err := restored.RestoreFromKV(ctx, loaded); err != nil {
			t.Fatalf("RestoreFromKV %s: %v", name, err)
		}
		got, err := restored.GenerateFromCache(maxNew, -1)
		if err != nil {
			t.Fatalf("GenerateFromCache %s: %v", name, err)
		}
		return got
	}

	// float32: lossless → token-identical continuation.
	if gotF32 := roundTrip(kv.KVSnapshotEncodingFloat32, "float32"); !idsEqual(gotF32, want) {
		t.Fatalf("float32 round-trip continuation = %v, want uninterrupted %v", gotF32, want)
	}

	// q8: the encoding that used to fail (errRawTensorNeedsNative). Lossy, so it
	// must round-trip WITHOUT error and continue with in-range tokens.
	gotQ8 := roundTrip(kv.EncodingQ8, "q8")
	if len(gotQ8) != maxNew {
		t.Fatalf("q8 continuation length = %d, want %d", len(gotQ8), maxNew)
	}
	for _, id := range gotQ8 {
		if id < 0 || int(id) >= tm.arch.Vocab {
			t.Fatalf("q8 continuation token %d out of vocab range", id)
		}
	}
	t.Logf("#290: float32 round-trip token-identical; q8 round-trips + continues %v (lossy)", gotQ8)
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
