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
	"dappco.re/go/inference/model/qwen3"
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

// TestComposedEngineSession_RangeKVBlocks_Unsupported checks the block-streaming sleep lane declines: a
// composed session holds no KV cache to stream as blocks, so it points the caller at the token-prefix
// snapshot path rather than emitting empty blocks.
func TestComposedEngineSession_RangeKVBlocks_Unsupported(t *testing.T) {
	sess := newTinyComposedSession(2)
	err := sess.RangeKVBlocks(16, kv.CaptureOptions{}, func(kv.Block) (bool, error) { return true, nil })
	if err == nil {
		t.Fatal("RangeKVBlocks must decline: composed holds no KV cache to stream as blocks")
	}
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
