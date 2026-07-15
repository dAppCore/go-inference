// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/composed"
	"dappco.re/go/inference/model/arch/Qwen/qwen3"
	state "dappco.re/go/inference/model/state"
)

// composed_state_test.go proves composedEngineSession's multi-turn -state (#379): a composed hybrid holds
// no persistent KV cache (it decodes stateless-replay, re-prefilling the token prefix each generate), so
// its save/restore is a TOKEN-PREFIX snapshot — CaptureKVWithOptions emits Tokens with no Layers, and a
// session restored from those tokens produces a continuation byte-identical to an unbroken one because the
// deterministic host-f32 forward recomputes byte-identical recurrent state from the identical prefix. The
// tiny model here is pure host f32 (no Metal device), so these run under the plain `go test ./...` gate.

// synComposedF32 is the deterministic weight filler model/composed's own unit tests use, so the tiny model
// below decodes deterministically (a fixed greedy continuation to compare across save/restore).
func synComposedF32(n, seed int) []float32 {
	out := make([]float32, n)
	for i := range out {
		out[i] = float32((i*seed+7)%101-50) * 0.02
	}
	return out
}

// newTinyComposedModel builds a minimal host-f32 composed hybrid (a gated-delta mixer + SwiGLU MLP per
// layer) as a model.SessionModel — enough to drive composedEngineSession's decode + snapshot path on the
// CPU. It mirrors model/composed's mkComposedModel fixture; D/vocab/FF are fixed small.
func newTinyComposedModel(nLayers int) model.SessionModel {
	const D, vocab, FF = 8, 32, 16
	cfg := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 8, ConvKernel: 4, Eps: 1e-5}
	vd := cfg.ValueHeads * cfg.HeadDim
	cd := 2*cfg.KeyHeads*cfg.HeadDim + cfg.ValueHeads*cfg.HeadDim
	layers := make([]composed.Layer, nLayers)
	for li := range layers {
		seed := li*13 + 20
		w := &qwen3.GatedDeltaWeights{
			InProjQKV:  synComposedF32(cd*D, seed+1),
			ConvWeight: synComposedF32(cd*cfg.ConvKernel, seed+2),
			ConvBias:   synComposedF32(cd, seed+3),
			InProjA:    synComposedF32(cfg.ValueHeads*D, seed+4),
			ALog:       synComposedF32(cfg.ValueHeads, seed+5),
			DtBias:     synComposedF32(cfg.ValueHeads, seed+6),
			InProjB:    synComposedF32(cfg.ValueHeads*D, seed+7),
			InProjZ:    synComposedF32(vd*D, seed+8),
			Norm:       synComposedF32(cfg.HeadDim, seed+9),
			OutProj:    synComposedF32(D*vd, seed+10),
		}
		layers[li] = composed.Layer{
			InputNorm:    synComposedF32(D, li*13+1),
			Mixer:        composed.NewGatedDeltaMixer(w, cfg),
			PostAttnNorm: synComposedF32(D, li*13+2),
			MLP:          &composed.MLP{Gate: synComposedF32(FF*D, li*13+3), Up: synComposedF32(FF*D, li*13+4), Down: synComposedF32(D*FF, li*13+5), FF: FF},
		}
	}
	m := &composed.ComposedModel{
		Embed: synComposedF32(vocab*D, 100), Layers: layers, NormF: synComposedF32(D, 101), Output: nil,
		D: D, Vocab: vocab, Eps: 1e-5,
	}
	return composed.NewTokenModel(m)
}

func newTinyComposedSession(nLayers int) *composedEngineSession {
	return &composedEngineSession{sm: newTinyComposedModel(nLayers), arch: "qwen3_next", numLayers: nLayers}
}

// TestComposedEngineSession_CaptureKVWithOptions_Good captures a prefilled+appended session and checks the
// snapshot is a well-formed token-prefix snapshot (the retained prefix, no KV Layers) that survives the
// binary format the -state path persists (MarshalBinary → UnmarshalBinary).
func TestComposedEngineSession_CaptureKVWithOptions_Good(t *testing.T) {
	const nLayers = 2
	sess := newTinyComposedSession(nLayers)
	prompt := []int32{1, 2, 3}
	turn := []int32{4, 5}
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if err := sess.AppendTokens(turn); err != nil {
		t.Fatalf("append: %v", err)
	}
	snap, err := sess.CaptureKVWithOptions(kv.CaptureOptions{})
	if err != nil {
		t.Fatalf("capture: %v", err)
	}
	want := []int32{1, 2, 3, 4, 5}
	assertTokens(t, "capture", snap.Tokens, want)
	if snap.TokenOffset != len(want) {
		t.Fatalf("TokenOffset %d, want %d", snap.TokenOffset, len(want))
	}
	if snap.Architecture != "qwen3_next" {
		t.Fatalf("Architecture %q, want qwen3_next", snap.Architecture)
	}
	if snap.NumLayers != nLayers {
		t.Fatalf("NumLayers %d, want %d", snap.NumLayers, nLayers)
	}
	if snap.Version != kv.SnapshotVersion {
		t.Fatalf("Version %d, want %d", snap.Version, kv.SnapshotVersion)
	}
	if len(snap.Layers) != 0 {
		t.Fatalf("composed snapshot must carry no KV layers, got %d", len(snap.Layers))
	}
	blob, err := snap.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var got kv.Snapshot
	if err := got.UnmarshalBinary(blob); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	assertTokens(t, "binary round-trip", got.Tokens, want)
}

// TestComposedEngineSession_CaptureKVWithOptions_Bad checks capture declines on a session with nothing
// prefilled — there is no resumable state, so it must error rather than emit an empty snapshot.
func TestComposedEngineSession_CaptureKVWithOptions_Bad(t *testing.T) {
	sess := newTinyComposedSession(2)
	if _, err := sess.CaptureKVWithOptions(kv.CaptureOptions{}); err == nil {
		t.Fatal("capture on an empty session must error")
	}
}

// TestComposedEngineSession_RestoreFromKV_Good restores a token-prefix snapshot into a fresh session and
// checks the prefix is reinstated (Pos matches) and copied, not aliased to the snapshot's slice.
func TestComposedEngineSession_RestoreFromKV_Good(t *testing.T) {
	tokens := []int32{3, 1, 4, 1, 5}
	snap := &kv.Snapshot{Version: kv.SnapshotVersion, Architecture: "qwen3_next", Tokens: tokens, TokenOffset: len(tokens)}
	sess := newTinyComposedSession(2)
	if err := sess.RestoreFromKV(context.Background(), snap); err != nil {
		t.Fatalf("restore: %v", err)
	}
	if sess.Pos() != len(tokens) {
		t.Fatalf("Pos %d, want %d", sess.Pos(), len(tokens))
	}
	snap.Tokens[0] = 99 // mutating the source must not perturb the restored session
	if sess.prompt[0] == 99 {
		t.Fatal("restored prompt aliases the snapshot tokens")
	}
}

// TestComposedEngineSession_RestoreFromKV_Bad checks the restore guards: a nil snapshot and a snapshot with
// no token prefix both decline (a composed snapshot resumes from tokens; without them there is nothing to
// resume).
func TestComposedEngineSession_RestoreFromKV_Bad(t *testing.T) {
	sess := newTinyComposedSession(2)
	if err := sess.RestoreFromKV(context.Background(), nil); err == nil {
		t.Fatal("restore of a nil snapshot must error")
	}
	if err := sess.RestoreFromKV(context.Background(), &kv.Snapshot{Version: kv.SnapshotVersion}); err == nil {
		t.Fatal("restore of a snapshot with no token prefix must error")
	}
}

// TestComposedEngineSession_RestoreFromKV_ContinuationParity is the acceptance semantics: a conversation
// saved (capture → binary round-trip) and resumed in a FRESH session produces a continuation byte-identical
// to the same conversation carried unbroken. Both re-prefill the identical prefix, so the deterministic
// host-f32 forward yields identical recurrent state and identical greedy tokens.
func TestComposedEngineSession_RestoreFromKV_ContinuationParity(t *testing.T) {
	const nLayers, maxNew = 3, 6
	prompt := []int32{1, 5, 9, 2}
	turn := []int32{7, 3}
	collect := func(int32) bool { return true }

	unbroken := newTinyComposedSession(nLayers)
	if err := unbroken.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill(unbroken): %v", err)
	}
	if err := unbroken.AppendTokens(turn); err != nil {
		t.Fatalf("append(unbroken): %v", err)
	}
	genUnbroken, err := unbroken.GenerateFromCacheEach(maxNew, -1, collect)
	if err != nil {
		t.Fatalf("generate(unbroken): %v", err)
	}

	saved := newTinyComposedSession(nLayers)
	if err := saved.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill(saved): %v", err)
	}
	if err := saved.AppendTokens(turn); err != nil {
		t.Fatalf("append(saved): %v", err)
	}
	snap, err := saved.CaptureKVWithOptions(kv.CaptureOptions{})
	if err != nil {
		t.Fatalf("capture: %v", err)
	}
	blob, err := snap.MarshalBinary()
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var restored kv.Snapshot
	if err := restored.UnmarshalBinary(blob); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	resumed := newTinyComposedSession(nLayers)
	if err := resumed.RestoreFromKV(context.Background(), &restored); err != nil {
		t.Fatalf("restore: %v", err)
	}
	genResumed, err := resumed.GenerateFromCacheEach(maxNew, -1, collect)
	if err != nil {
		t.Fatalf("generate(resumed): %v", err)
	}

	assertTokens(t, "continuation parity", genResumed, genUnbroken)
	t.Logf("resumed continuation byte-identical to unbroken over %d tokens: %v", maxNew, genResumed)
}

// collectComposedBlocks streams every kv.Block RangeKVBlocks yields into a slice for assertion.
func collectComposedBlocks(t *testing.T, sess *composedEngineSession, blockSize int, opts kv.CaptureOptions) []kv.Block {
	t.Helper()
	var blocks []kv.Block
	if err := sess.RangeKVBlocks(blockSize, opts, func(b kv.Block) (bool, error) {
		blocks = append(blocks, b)
		return true, nil
	}); err != nil {
		t.Fatalf("RangeKVBlocks: %v", err)
	}
	return blocks
}

// TestComposedEngineSession_RangeKVBlocks_Good checks the token-only block tiling: a prefilled session tiles
// its prefix onto the uniform blockSize grid (contiguous absolute Index / TokenStart, a partial final block),
// and every block carries a token-prefix snapshot (Tokens == the block slice, no KV Layers).
func TestComposedEngineSession_RangeKVBlocks_Good(t *testing.T) {
	const nLayers, blockSize = 2, 4
	sess := newTinyComposedSession(nLayers)
	prompt := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10} // 10 tokens -> blocks [0,4) [4,8) [8,10)
	if err := sess.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	blocks := collectComposedBlocks(t, sess, blockSize, kv.CaptureOptions{})
	if len(blocks) != 3 {
		t.Fatalf("blocks = %d, want 3 (ceil(10/4))", len(blocks))
	}
	wantStart, wantCount := []int{0, 4, 8}, []int{4, 4, 2}
	nextStart := 0
	for i, b := range blocks {
		if b.Index != i {
			t.Fatalf("block[%d].Index = %d, want %d (absolute grid index)", i, b.Index, i)
		}
		if b.TokenStart != wantStart[i] || b.TokenCount != wantCount[i] {
			t.Fatalf("block[%d] = start %d count %d, want start %d count %d", i, b.TokenStart, b.TokenCount, wantStart[i], wantCount[i])
		}
		if b.TokenStart != nextStart {
			t.Fatalf("block[%d].TokenStart = %d, want %d (contiguous)", i, b.TokenStart, nextStart)
		}
		nextStart += b.TokenCount
		if b.Snapshot == nil {
			t.Fatalf("block[%d] snapshot is nil", i)
		}
		if len(b.Snapshot.Layers) != 0 {
			t.Fatalf("block[%d] must be token-only, got %d KV layers", i, len(b.Snapshot.Layers))
		}
		end := b.TokenStart + b.TokenCount
		assertTokens(t, "block tokens", b.Snapshot.Tokens, prompt[b.TokenStart:end])
		if b.Snapshot.TokenOffset != end {
			t.Fatalf("block[%d].Snapshot.TokenOffset = %d, want %d (absolute end)", i, b.Snapshot.TokenOffset, end)
		}
		if b.Snapshot.SeqLen != b.TokenCount {
			t.Fatalf("block[%d].Snapshot.SeqLen = %d, want %d", i, b.Snapshot.SeqLen, b.TokenCount)
		}
		if b.Snapshot.Architecture != "qwen3_next" || b.Snapshot.NumLayers != nLayers {
			t.Fatalf("block[%d] snapshot arch/layers = %q/%d, want qwen3_next/%d", i, b.Snapshot.Architecture, b.Snapshot.NumLayers, nLayers)
		}
	}
	if nextStart != len(prompt) {
		t.Fatalf("blocks cover %d tokens, want %d", nextStart, len(prompt))
	}
}

// TestComposedEngineSession_RangeKVBlocks_Bad drives the guard arms: a nil yield, a non-positive block size,
// a negative block-start token, and a session with nothing prefilled all decline rather than emit blocks.
func TestComposedEngineSession_RangeKVBlocks_Bad(t *testing.T) {
	sess := newTinyComposedSession(2)
	if err := sess.PrefillTokens([]int32{1, 2, 3}); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	ok := func(kv.Block) (bool, error) { return true, nil }
	if err := sess.RangeKVBlocks(4, kv.CaptureOptions{}, nil); err == nil {
		t.Fatal("nil yield must error")
	}
	if err := sess.RangeKVBlocks(0, kv.CaptureOptions{}, ok); err == nil {
		t.Fatal("block size <= 0 must error")
	}
	if err := sess.RangeKVBlocks(4, kv.CaptureOptions{BlockStartToken: -1}, ok); err == nil {
		t.Fatal("negative block start token must error")
	}
	empty := newTinyComposedSession(2)
	if err := empty.RangeKVBlocks(4, kv.CaptureOptions{}, ok); err == nil {
		t.Fatal("empty session must error (nothing to sleep)")
	}
}

// TestComposedEngineSession_RangeKVBlocks_TrustedPrefix checks the trusted-prefix skip mirrors ArchSession's
// contract: whole blocks ending at or before opts.BlockStartToken are skipped, yielded Index / TokenStart
// stay ABSOLUTE in the grid, and a mid-block boundary re-emits the spanning block whole.
func TestComposedEngineSession_RangeKVBlocks_TrustedPrefix(t *testing.T) {
	const blockSize = 4
	sess := newTinyComposedSession(2)
	if err := sess.PrefillTokens([]int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}); err != nil { // 12 tokens -> 3 full blocks
		t.Fatalf("prefill: %v", err)
	}
	// Block-aligned boundary (2 whole blocks): only the third block [8,12) is emitted, at absolute index 2.
	aligned := collectComposedBlocks(t, sess, blockSize, kv.CaptureOptions{BlockStartToken: 8})
	if len(aligned) != 1 || aligned[0].Index != 2 || aligned[0].TokenStart != 8 || aligned[0].TokenCount != 4 {
		t.Fatalf("aligned boundary = %+v, want a single block index 2 start 8 count 4", aligned)
	}
	// Mid-block boundary: block [4,8) ends after 6 so it is NOT skipped — it is re-emitted whole from its
	// absolute start 4, then [8,12) follows at index 2.
	mid := collectComposedBlocks(t, sess, blockSize, kv.CaptureOptions{BlockStartToken: 6})
	if len(mid) != 2 || mid[0].Index != 1 || mid[0].TokenStart != 4 || mid[1].Index != 2 || mid[1].TokenStart != 8 {
		t.Fatalf("mid boundary = %+v, want blocks index 1 (start 4) then index 2 (start 8)", mid)
	}
}

// composedSleepToStore mirrors the serve sleep lane (Session.SaveKVBlocksToState): it streams the session's
// RangeKVBlocks through kv.SaveStateBlocksFromStream into store, honouring the trusted-prefix graft when a
// parent bundle is supplied. Returns the written bundle.
func composedSleepToStore(t *testing.T, ctx context.Context, store state.Writer, sess *composedEngineSession, blockSize int, uri string, parent *kv.StateBlockBundle) *kv.StateBlockBundle {
	t.Helper()
	opts := kv.StateBlockOptions{BlockSize: blockSize, KVEncoding: kv.EncodingNative, URI: uri}
	if parent != nil {
		opts.ReusePrefix = parent
		opts.ReusePrefixTokens = parent.TokenCount
		opts.ReusePrefixTrusted = true
	}
	captureOpts := kv.CaptureOptions{BlockStartToken: kv.TrustedReuseBoundary(opts, blockSize)}
	bundle, err := kv.SaveStateBlocksFromStream(ctx, store, opts, func(yield func(kv.Block) (bool, error)) error {
		return sess.RangeKVBlocks(blockSize, captureOpts, yield)
	})
	if err != nil {
		t.Fatalf("sleep %q: %v", uri, err)
	}
	return bundle
}

// composedWake reassembles a bundle into a token-only snapshot and restores it into a fresh session — the
// wake path a composed session takes (LoadFromStateBlocks -> RestoreFromKV; composed exposes no native
// block restorer, so WakeAgentMemory falls to the snapshot restore).
func composedWake(t *testing.T, ctx context.Context, store state.Store, bundle *kv.StateBlockBundle, nLayers int) *composedEngineSession {
	t.Helper()
	snap, err := kv.LoadFromStateBlocks(ctx, store, bundle)
	if err != nil {
		t.Fatalf("wake load: %v", err)
	}
	if len(snap.Layers) != 0 {
		t.Fatalf("woken composed snapshot must be token-only, got %d KV layers", len(snap.Layers))
	}
	sess := newTinyComposedSession(nLayers)
	if err := sess.RestoreFromKV(ctx, snap); err != nil {
		t.Fatalf("wake restore: %v", err)
	}
	return sess
}

// TestComposedEngineSession_BlockSleepWakeRoundTrip is the full CPU sleep->wake round trip through an
// in-memory state store: a prefilled session sleeps its prefix as token-only blocks, a fresh session wakes
// from the reassembled bundle, and its greedy continuation is byte-identical to the same conversation carried
// unbroken — the block-streaming sleep lane resuming through the deterministic host-f32 re-prefill.
func TestComposedEngineSession_BlockSleepWakeRoundTrip(t *testing.T) {
	const nLayers, blockSize, maxNew = 3, 4, 6
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	prompt := []int32{1, 5, 9, 2}
	turn := []int32{7, 3}
	full := append(append([]int32(nil), prompt...), turn...)

	slept := newTinyComposedSession(nLayers)
	if err := slept.PrefillTokens(prompt); err != nil {
		t.Fatalf("prefill: %v", err)
	}
	if err := slept.AppendTokens(turn); err != nil {
		t.Fatalf("append: %v", err)
	}
	bundle := composedSleepToStore(t, ctx, store, slept, blockSize, "mlx://composed/roundtrip", nil)
	if bundle.TokenCount != len(full) {
		t.Fatalf("bundle TokenCount = %d, want %d", bundle.TokenCount, len(full))
	}

	resumed := composedWake(t, ctx, store, bundle, nLayers)
	if resumed.Pos() != len(full) {
		t.Fatalf("woken Pos = %d, want %d", resumed.Pos(), len(full))
	}
	assertTokens(t, "woken prefix", resumed.prompt, full)

	collect := func(int32) bool { return true }
	genResumed, err := resumed.GenerateFromCacheEach(maxNew, -1, collect)
	if err != nil {
		t.Fatalf("generate(resumed): %v", err)
	}
	unbroken := newTinyComposedSession(nLayers)
	if err := unbroken.PrefillTokens(full); err != nil {
		t.Fatalf("prefill(unbroken): %v", err)
	}
	genUnbroken, err := unbroken.GenerateFromCacheEach(maxNew, -1, collect)
	if err != nil {
		t.Fatalf("generate(unbroken): %v", err)
	}
	assertTokens(t, "block sleep/wake continuation parity", genResumed, genUnbroken)
}

// TestComposedEngineSession_BlockSleepWakeRoundTrip_MultiTurn is the multi-turn re-sleep: sleep, wake, extend
// the conversation, then re-sleep declaring the first sleep's bundle as a trusted parent prefix. The re-sleep
// must GRAFT the parent's whole blocks by reference and tile the new turn's block contiguously against them —
// absolute Index / TokenStart continuing where the parent ended — and the final wake must reassemble the whole
// conversation and resume byte-identically.
func TestComposedEngineSession_BlockSleepWakeRoundTrip_MultiTurn(t *testing.T) {
	const nLayers, blockSize, maxNew = 3, 4, 6
	ctx := context.Background()
	store := state.NewInMemoryStore(nil)

	// Turn 1: a prefix that is a whole number of blocks (so it can be a trusted parent).
	turn1 := []int32{1, 5, 9, 2, 7, 3, 4, 8} // 8 tokens = 2 full blocks of 4
	first := newTinyComposedSession(nLayers)
	if err := first.PrefillTokens(turn1); err != nil {
		t.Fatalf("prefill(turn1): %v", err)
	}
	bundle1 := composedSleepToStore(t, ctx, store, first, blockSize, "mlx://composed/turn1", nil)
	if len(bundle1.Blocks) != 2 || bundle1.TokenCount != 8 {
		t.Fatalf("bundle1 = %d blocks / %d tokens, want 2 / 8", len(bundle1.Blocks), bundle1.TokenCount)
	}

	// Wake turn 1, extend with turn 2, re-sleep against the turn-1 bundle as a trusted parent.
	woken := composedWake(t, ctx, store, bundle1, nLayers)
	turn2 := []int32{11, 13, 15}
	if err := woken.AppendTokens(turn2); err != nil {
		t.Fatalf("append(turn2): %v", err)
	}
	bundle2 := composedSleepToStore(t, ctx, store, woken, blockSize, "mlx://composed/turn2", bundle1)

	// The two parent blocks are grafted by reference; only the new turn's block is freshly streamed.
	if bundle2.ReusedBlocks != 2 {
		t.Fatalf("bundle2 reused blocks = %d, want 2 (grafted turn-1 blocks)", bundle2.ReusedBlocks)
	}
	if len(bundle2.Blocks) != 3 || bundle2.TokenCount != 11 {
		t.Fatalf("bundle2 = %d blocks / %d tokens, want 3 / 11", len(bundle2.Blocks), bundle2.TokenCount)
	}
	if bundle2.Blocks[0].State.ChunkID != bundle1.Blocks[0].State.ChunkID || bundle2.Blocks[1].State.ChunkID != bundle1.Blocks[1].State.ChunkID {
		t.Fatal("bundle2's first two blocks must share the parent bundle's chunk refs")
	}
	// Contiguous tiling: absolute indices 0,1,2 with token starts 0,4,8 — the new block starts exactly where
	// the grafted parent ended.
	wantStart, wantCount := []int{0, 4, 8}, []int{4, 4, 3}
	for i, ref := range bundle2.Blocks {
		if ref.Index != i || ref.TokenStart != wantStart[i] || ref.TokenCount != wantCount[i] {
			t.Fatalf("bundle2.Blocks[%d] = index %d start %d count %d, want %d/%d/%d", i, ref.Index, ref.TokenStart, ref.TokenCount, i, wantStart[i], wantCount[i])
		}
	}

	// Final wake reassembles the whole conversation and resumes byte-identically to an unbroken run.
	full := append(append([]int32(nil), turn1...), turn2...)
	resumed := composedWake(t, ctx, store, bundle2, nLayers)
	if resumed.Pos() != len(full) {
		t.Fatalf("final woken Pos = %d, want %d", resumed.Pos(), len(full))
	}
	assertTokens(t, "multi-turn woken prefix", resumed.prompt, full)

	collect := func(int32) bool { return true }
	genResumed, err := resumed.GenerateFromCacheEach(maxNew, -1, collect)
	if err != nil {
		t.Fatalf("generate(resumed): %v", err)
	}
	unbroken := newTinyComposedSession(nLayers)
	if err := unbroken.PrefillTokens(full); err != nil {
		t.Fatalf("prefill(unbroken): %v", err)
	}
	genUnbroken, err := unbroken.GenerateFromCacheEach(maxNew, -1, collect)
	if err != nil {
		t.Fatalf("generate(unbroken): %v", err)
	}
	assertTokens(t, "multi-turn continuation parity", genResumed, genUnbroken)
}

// Example_composedEngineSessionState shows the composed -state round-trip: capture a prefilled session to a
// token-prefix snapshot (no KV Layers), then restore it into a fresh session that resumes from the same
// prefix.
func Example_composedEngineSessionState() {
	sess := newTinyComposedSession(2)
	_ = sess.PrefillTokens([]int32{1, 2, 3})
	snap, err := sess.CaptureKVWithOptions(kv.CaptureOptions{})
	if err != nil {
		return
	}
	resumed := newTinyComposedSession(2)
	_ = resumed.RestoreFromKV(context.Background(), snap)
	core.Println(core.Sprintf("tokens=%d layers=%d pos=%d", len(snap.Tokens), len(snap.Layers), resumed.Pos()))
	// Output: tokens=3 layers=0 pos=3
}

// Example_composedEngineSessionBlockSleep shows the serve-continuity block-sleep lane: RangeKVBlocks tiles a
// prefilled composed session's prefix into contiguous token-only blocks (Tokens, no KV Layers) on the
// blockSize grid, the same stream SaveKVBlocksToState persists.
func Example_composedEngineSessionBlockSleep() {
	sess := newTinyComposedSession(2)
	_ = sess.PrefillTokens([]int32{1, 2, 3, 4, 5}) // 5 tokens, blockSize 4 -> [0,4) then [4,5)
	_ = sess.RangeKVBlocks(4, kv.CaptureOptions{}, func(b kv.Block) (bool, error) {
		core.Println(core.Sprintf("index=%d start=%d count=%d layers=%d", b.Index, b.TokenStart, b.TokenCount, len(b.Snapshot.Layers)))
		return true, nil
	})
	// Output:
	// index=0 start=0 count=4 layers=0
	// index=1 start=4 count=1 layers=0
}

func assertTokens(t *testing.T, label string, got, want []int32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d, want %d", label, len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("%s: token[%d]=%d, want %d", label, i, got[i], want[i])
		}
	}
}
