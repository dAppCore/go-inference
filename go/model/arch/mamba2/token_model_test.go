// SPDX-Licence-Identifier: EUPL-1.2

package mamba2

import (
	"bytes"
	"testing"
)

func TestTokenModel_NewTokenModel_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 2)
	tm := NewTokenModel(m)
	if tm.m != m {
		t.Fatal("NewTokenModel did not retain the model reference")
	}
}

// TestTokenModel_NewTokenModel_Bad proves a zero-value model wraps without panicking — Vocab() honestly
// reports 0 rather than the wrapper fabricating or crashing.
func TestTokenModel_NewTokenModel_Bad(t *testing.T) {
	tm := NewTokenModel(&MambaModel{})
	if tm.Vocab() != 0 {
		t.Fatalf("Vocab = %d, want 0 for a zero-value model", tm.Vocab())
	}
}

// TestTokenModel_NewTokenModel_Ugly proves two wrappers over the SAME model are independently usable
// and deterministic — Embed on either produces identical bytes.
func TestTokenModel_NewTokenModel_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm1, tm2 := NewTokenModel(m), NewTokenModel(m)
	e1, err1 := tm1.Embed(3)
	e2, err2 := tm2.Embed(3)
	if err1 != nil || err2 != nil || !bytes.Equal(e1, e2) {
		t.Fatalf("two wrappers over the same model diverged: %v/%v err %v/%v", e1, e2, err1, err2)
	}
}

func TestTokenModel_MambaTokenModel_Vocab_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 2)
	if got := NewTokenModel(m).Vocab(); got != 32 {
		t.Fatalf("Vocab = %d, want 32", got)
	}
}

func TestTokenModel_MambaTokenModel_Vocab_Bad(t *testing.T) {
	if got := NewTokenModel(&MambaModel{Vocab: 0}).Vocab(); got != 0 {
		t.Fatalf("Vocab = %d, want 0", got)
	}
}

// TestTokenModel_MambaTokenModel_Vocab_Ugly proves Vocab() reads the underlying model LIVE, not a
// snapshot taken at NewTokenModel time.
func TestTokenModel_MambaTokenModel_Vocab_Ugly(t *testing.T) {
	m := &MambaModel{Vocab: 10}
	tm := NewTokenModel(m)
	m.Vocab = 99
	if got := tm.Vocab(); got != 99 {
		t.Fatalf("Vocab = %d, want 99 (must read live, not snapshot at construction)", got)
	}
}

func TestTokenModel_MambaTokenModel_HiddenSize_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 2)
	if got := NewTokenModel(m).HiddenSize(); got != 8 {
		t.Fatalf("HiddenSize = %d, want 8", got)
	}
}

// TestTokenModel_MambaTokenModel_HiddenSize_Ugly proves HiddenSize() reads the underlying model LIVE,
// not a snapshot taken at NewTokenModel time — same live-read contract as Vocab.
func TestTokenModel_MambaTokenModel_HiddenSize_Ugly(t *testing.T) {
	m := &MambaModel{D: 8}
	tm := NewTokenModel(m)
	m.D = 99
	if got := tm.HiddenSize(); got != 99 {
		t.Fatalf("HiddenSize = %d, want 99 (must read live, not snapshot at construction)", got)
	}
}

func TestTokenModel_MambaTokenModel_NumLayers_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 5)
	if got := NewTokenModel(m).NumLayers(); got != 5 {
		t.Fatalf("NumLayers = %d, want 5", got)
	}
}

// TestTokenModel_MambaTokenModel_NumLayers_Bad proves a zero-value model (no layers) reports 0 honestly
// rather than panicking on a nil Layers slice.
func TestTokenModel_MambaTokenModel_NumLayers_Bad(t *testing.T) {
	if got := NewTokenModel(&MambaModel{}).NumLayers(); got != 0 {
		t.Fatalf("NumLayers = %d, want 0 for a zero-value model", got)
	}
}

func TestTokenModel_MambaTokenModel_Embed_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm := NewTokenModel(m)
	emb, err := tm.Embed(5)
	if err != nil || len(emb) != m.D*2 {
		t.Fatalf("Embed(5): len %d err %v, want %d bf16 bytes", len(emb), err, m.D*2)
	}
	want := f32ToBF16Bytes(m.Embed[5*m.D : 6*m.D])
	if !bytes.Equal(emb, want) {
		t.Fatal("Embed bytes do not match the model's embedding row")
	}
}

func TestTokenModel_MambaTokenModel_Embed_Bad(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm := NewTokenModel(m)
	if _, err := tm.Embed(-1); err == nil {
		t.Fatal("negative id accepted")
	}
	if _, err := tm.Embed(int32(m.Vocab)); err == nil {
		t.Fatal("id == Vocab accepted (one past the last valid id)")
	}
}

// TestTokenModel_MambaTokenModel_Embed_Ugly pins the boundary: id == Vocab-1 (the LAST valid id) must
// be accepted — distinct from _Bad's id == Vocab (one past) rejection.
func TestTokenModel_MambaTokenModel_Embed_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm := NewTokenModel(m)
	if _, err := tm.Embed(int32(m.Vocab - 1)); err != nil {
		t.Fatalf("Embed(Vocab-1): last valid id rejected: %v", err)
	}
}

func TestTokenModel_MambaTokenModel_Head_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm := NewTokenModel(m)
	hidden := f32ToBF16Bytes(syn(m.D, 7))
	logits, err := tm.Head(hidden)
	if err != nil || len(logits) != m.Vocab*2 {
		t.Fatalf("Head: len %d err %v, want %d bf16 bytes", len(logits), err, m.Vocab*2)
	}
}

func TestTokenModel_MambaTokenModel_Head_Bad(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm := NewTokenModel(m)
	if _, err := tm.Head([]byte{1, 2, 3}); err == nil {
		t.Fatal("wrong-length hidden accepted")
	}
}

// TestTokenModel_MambaTokenModel_Head_Ugly proves the tied-vs-explicit LM head branch actually takes
// effect: a model with an EXPLICIT LMHead produces different logits than an otherwise-identical tied
// model (mkModel's fixed seeds give both the same Embed/NormF) — distinct from _Bad's malformed-input
// rejection.
func TestTokenModel_MambaTokenModel_Head_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	tied := mkModel(cfg, 8, 32, 1)
	explicit := mkModel(cfg, 8, 32, 1)
	explicit.LMHead = syn(32*8, 999) // distinct explicit head, NOT tied to Embed
	hidden := f32ToBF16Bytes(syn(8, 7))
	lt, err := NewTokenModel(tied).Head(hidden)
	if err != nil {
		t.Fatalf("tied Head: %v", err)
	}
	le, err := NewTokenModel(explicit).Head(hidden)
	if err != nil {
		t.Fatalf("explicit Head: %v", err)
	}
	if bytes.Equal(lt, le) {
		t.Fatal("tied and explicit-LMHead logits identical — LMHead branch not taking effect")
	}
}

func TestTokenModel_MambaTokenModel_DecodeForward_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 2)
	tm := NewTokenModel(m)
	tokens := []int32{3, 1, 4}
	embs := make([][]byte, len(tokens))
	for i, tok := range tokens {
		e, err := tm.Embed(tok)
		if err != nil {
			t.Fatalf("Embed: %v", err)
		}
		embs[i] = e
	}
	out, err := tm.DecodeForward(embs)
	if err != nil {
		t.Fatalf("DecodeForward: %v", err)
	}
	if len(out) != len(tokens) {
		t.Fatalf("DecodeForward returned %d hiddens, want %d", len(out), len(tokens))
	}
	for i, h := range out {
		if len(h) != m.D*2 {
			t.Fatalf("hidden %d len %d, want %d bf16 bytes", i, len(h), m.D*2)
		}
	}
}

func TestTokenModel_MambaTokenModel_DecodeForward_Bad(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm := NewTokenModel(m)
	if _, err := tm.DecodeForward([][]byte{{1, 2, 3}}); err == nil {
		t.Fatal("wrong-length input embedding accepted")
	}
}

// TestTokenModel_MambaTokenModel_DecodeForward_Ugly proves the empty-input edge: zero inputs returns
// (nil, nil) — no error, no panic — distinct from _Bad's malformed-length rejection.
func TestTokenModel_MambaTokenModel_DecodeForward_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	tm := NewTokenModel(m)
	out, err := tm.DecodeForward(nil)
	if err != nil || out != nil {
		t.Fatalf("DecodeForward(nil) = %v, %v; want nil, nil", out, err)
	}
}

func TestTokenModel_MambaTokenModel_OpenSession_Bad(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 1)
	st, err := NewTokenModel(m).OpenSession()
	if err != nil {
		t.Fatalf("OpenSession: %v", err)
	}
	if _, err := st.Step([]byte{1, 2, 3}); err == nil {
		t.Fatal("wrong-length step embedding accepted")
	}
}

// TestTokenModel_MambaTokenModel_OpenSession_Ugly proves session independence: two steppers opened
// from the same model both start from FRESH (zero) recurrent state — stepping the identical token
// through each produces identical output, proving OpenSession doesn't leak state between calls.
func TestTokenModel_MambaTokenModel_OpenSession_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, HeadDim: 8, StateDim: 8, NumGroups: 1, ConvKernel: 4, Eps: 1e-5}
	m := mkModel(cfg, 8, 32, 2)
	tm := NewTokenModel(m)
	e, err := tm.Embed(7)
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	st1, _ := tm.OpenSession()
	st2, _ := tm.OpenSession()
	h1, err := st1.Step(e)
	if err != nil {
		t.Fatalf("Step 1: %v", err)
	}
	h2, err := st2.Step(e)
	if err != nil {
		t.Fatalf("Step 2: %v", err)
	}
	if !bytes.Equal(h1, h2) {
		t.Fatal("two fresh sessions stepping the same token diverged — state leaked between OpenSession calls")
	}
}

// TestTokenModel_MambaTokenModel_OpenSession_Good verifies the SessionModel contract on the wrapper:
// decoding token by token through OpenSession's incremental stepper produces the SAME hidden bytes as
// the whole-sequence DecodeForward — through the identical bf16 seam. This is what lets Generate prefer
// the O(1)/token fast path with no change in output.
func TestTokenModel_MambaTokenModel_OpenSession_Good(t *testing.T) {
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
