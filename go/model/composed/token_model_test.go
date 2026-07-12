// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestComposedTokenModelSessionEqualsWhole verifies the SessionModel contract: decoding token by token
// through OpenSession's stepper produces the SAME hidden bytes as the whole-sequence DecodeForward, through
// the identical bf16 seam — so Generate can take the O(1)/token fast path with no output change.
func TestComposedTokenModelSessionEqualsWhole(t *testing.T) {
	m := mkComposedModel(3, 8, 32, 16)
	tm := NewTokenModel(m)
	tokens := []int32{3, 1, 4, 1, 5, 9}

	embs := make([][]byte, len(tokens))
	for i, tok := range tokens {
		e, err := tm.Embed(tok)
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		embs[i] = e
	}
	whole, err := tm.DecodeForward(embs)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	st, err := tm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	for i := range tokens {
		h, err := st.Step(embs[i])
		if err != nil {
			t.Fatalf("Step %d: %v", i, err)
		}
		if !bytes.Equal(h, whole[i]) {
			t.Fatalf("token %d: incremental Step != whole-sequence DecodeForward (fast path diverged)", i)
		}
	}
	t.Logf("composed SessionModel: incremental decode == whole-sequence over %d tokens (bf16 seam)", len(tokens))
}

// TestComposedStepperHeadFuseFastPath verifies composedStepper's session-level Head fast path: when the
// LAST layer's tail fuses the model's final RMSNorm + LM head GEMM in (ResidualNormMLPProjHeadDevice
// returns logits), Step caches them and the immediately-following Head call for THAT exact hidden reuses
// them verbatim — no second (real) headLogits computation. The cache is one-shot: a repeat Head call for
// the same hidden must fall through to the ordinary recompute rather than replaying stale data.
func TestComposedStepperHeadFuseFastPath(t *testing.T) {
	const D, vocab, FF = 1024, 32, 1024 // L·D·FF = 1,048,576 ≥ deviceMinWork with a single decode token
	m := mkComposedModel(1, D, vocab, FF)

	sentinel := make([]float32, vocab)
	for i := range sentinel {
		sentinel[i] = float32(i) + 0.5
	}
	calls := 0
	saved := ResidualNormMLPProjHeadDevice
	ResidualNormMLPProjHeadDevice = func(mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32, normF, head []float32, Vocab int) (y, logits []float32, err error) {
		calls++
		yOut := make([]float32, len(h))
		copy(yOut, h) // arbitrary but deterministic stand-in "tail output"
		return yOut, sentinel, nil
	}
	defer func() { ResidualNormMLPProjHeadDevice = saved }()
	// The head-fuse branch shares forwardEmb's outer isDense/floor gate with the plain proj-fused tail
	// (ResidualNormMLPProjDevice != nil is part of that gate's condition) — bind it to a stub that fails
	// the test if it's ever actually reached, proving the head-fuse short-circuits ahead of it.
	savedProjTail := ResidualNormMLPProjDevice
	ResidualNormMLPProjDevice = func(mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32) ([]float32, error) {
		t.Fatal("ResidualNormMLPProjDevice ran — the head-fuse should have short-circuited it via continue")
		return nil, nil
	}
	defer func() { ResidualNormMLPProjDevice = savedProjTail }()

	tm := NewTokenModel(m)
	sess, err := tm.OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	emb, err := tm.Embed(3)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	hidden, err := sess.Step(emb)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if calls != 1 {
		t.Fatalf("head-fuse hook called %d times, want 1 — did the tail's floor gate not open?", calls)
	}
	sh, ok := sess.(model.LMHead)
	if !ok {
		t.Fatal("composedStepper does not implement model.LMHead — model.generateStepwiseWithSession cannot find the fast path")
	}
	logits, err := sh.Head(hidden)
	if err != nil {
		t.Fatalf("Head: %v", err)
	}
	want := f32ToBF16Bytes(sentinel)
	if !bytes.Equal(logits, want) {
		t.Fatal("Head did not return the cached fused-device logits for the matching hidden")
	}

	// One-shot: a second Head call for the SAME hidden must NOT replay the cache (already consumed) —
	// it falls through to the ordinary recompute, which does not touch ResidualNormMLPProjHeadDevice at
	// all (that hook only ever runs inside forwardEmb's last-layer tail, never from Head).
	calls2 := 0
	ResidualNormMLPProjHeadDevice = func(mixerHidden, projW, h, normW, gate, up, down []float32, L, D, mixCols, FF int, eps float32, normF, head []float32, Vocab int) (y, logits []float32, err error) {
		calls2++
		return nil, nil, core.NewError("must not be called for a Head recompute")
	}
	logits2, err := sh.Head(hidden)
	if err != nil {
		t.Fatalf("Head (recompute): %v", err)
	}
	if bytes.Equal(logits2, want) {
		t.Fatal("Head reused the cached logits a second time — the cache must be one-shot")
	}
	if calls2 != 0 {
		t.Fatalf("ResidualNormMLPProjHeadDevice unexpectedly invoked during a Head recompute (calls2=%d)", calls2)
	}
	t.Logf("composedStepper.Head: cached fused-device logits reused exactly once (%d fuse call), then falls through cleanly", calls)
}

// TestComposedTokenModelHeadVocab checks the bookends.
func TestComposedTokenModelHeadVocab(t *testing.T) {
	m := mkComposedModel(2, 8, 32, 16)
	tm := NewTokenModel(m)
	if tm.Vocab() != 32 {
		t.Fatalf("Vocab = %d, want 32", tm.Vocab())
	}
	emb, err := tm.Embed(5)
	if err != nil || len(emb) != m.D*2 {
		t.Fatalf("Embed: len %d err %v", len(emb), err)
	}
	logits, err := tm.Head(emb)
	if err != nil || len(logits) != m.Vocab*2 {
		t.Fatalf("Head: len %d err %v (want %d bf16 bytes)", len(logits), err, m.Vocab*2)
	}
	t.Log("composed bookends: Embed→dModel bf16, Head→vocab bf16 logits, Vocab() correct")
}
