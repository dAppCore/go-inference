// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"bytes"
	"testing"
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
