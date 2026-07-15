// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"bytes"
	"testing"
)

// TestTokenModelSessionEqualsWhole verifies the SessionModel contract on the wrapper: decoding token by
// token through OpenSession's incremental stepper produces the SAME hidden bytes as the whole-sequence
// DecodeForward — through the identical bf16 seam. This is what lets Generate prefer the O(1)/token fast
// path with no change in output.
func TestTokenModelSessionEqualsWhole(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 2)
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
			t.Fatalf("token %d: incremental Step hidden != whole-sequence DecodeForward (SessionModel fast path diverged)", i)
		}
	}
	t.Logf("mamba2 SessionModel: incremental decode == whole-sequence over %d tokens (bf16 seam consistent)", len(tokens))
}

// TestTokenModelHeadVocab checks the bookends: Embed yields dModel bf16 bytes, Head yields vocab bf16
// logits, Vocab reports the size.
func TestTokenModelHeadVocab(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 2)
	tm := NewTokenModel(m)
	if tm.Vocab() != 32 {
		t.Fatalf("Vocab = %d, want 32", tm.Vocab())
	}
	emb, err := tm.Embed(5)
	if err != nil || len(emb) != m.D*2 {
		t.Fatalf("Embed: len %d err %v (want %d bf16 bytes)", len(emb), err, m.D*2)
	}
	logits, err := tm.Head(emb)
	if err != nil {
		t.Fatalf("Head: %v", err)
	}
	if len(logits) != m.Vocab*2 {
		t.Fatalf("Head logits len %d, want %d bf16 bytes", len(logits), m.Vocab*2)
	}
	t.Log("mamba2 bookends: Embed→dModel bf16, Head→vocab bf16 logits, Vocab() correct")
}
