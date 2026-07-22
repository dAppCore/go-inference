// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import "testing"

// mkRWKV7Model builds a synthetic nLayers-deep RWKV7Model for cfg/D/FF — the in-memory sibling of
// loader_test.go's mkCheckpoint (real tensor names, on-disk), used by tests that only need the loaded
// shape, not the loader path itself.
func mkRWKV7Model(cfg BlockConfig, D, FF, vocab, nLayers int) *RWKV7Model {
	layers := make([]RWKV7Layer, nLayers)
	for li := range layers {
		var preW, preB []float32
		if li == 0 {
			preW, preB = syn(D, li*97+1), syn(D, li*97+2)
		}
		layers[li] = RWKV7Layer{
			PreNormW: preW, PreNormB: preB,
			AttnNormW: syn(D, li*97+3), AttnNormB: syn(D, li*97+4),
			Attn:     mkTimeMixWeights(cfg, D, 2, 2, 2, 2, li*97+10, li),
			FfnNormW: syn(D, li*97+5), FfnNormB: syn(D, li*97+6),
			FFN: mkChannelMixWeights(D, FF, li*97+7),
		}
	}
	return &RWKV7Model{
		Embed: syn(vocab*D, 500), NormW: syn(D, 501), NormB: syn(D, 502), LMHead: nil,
		Layers: layers, Cfg: cfg, D: D, Vocab: vocab, FF: FF, Eps: 1e-5,
	}
}

func TestModel_NewSession_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 3)
	s := NewSession(m)
	if s.m != m {
		t.Fatal("NewSession did not retain the model reference")
	}
	if len(s.wkv) != 3 || len(s.shift1) != 3 || len(s.shift2) != 3 {
		t.Fatalf("state slot counts = wkv %d shift1 %d shift2 %d, want 3 each (one per layer)", len(s.wkv), len(s.shift1), len(s.shift2))
	}
}

func TestModel_RWKV7Session_Forward_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 2)
	h, err := NewSession(m).Forward([]int32{1, 2, 3})
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if len(h) != 3*m.D {
		t.Fatalf("hidden len %d, want %d", len(h), 3*m.D)
	}
}

func TestModel_RWKV7Session_Forward_Bad(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 1)
	if _, err := NewSession(m).Forward([]int32{999}); err == nil {
		t.Fatal("out-of-range token accepted")
	}
}

// TestModel_RWKV7Session_Forward_Ugly is the whole-model decode invariant: one prefill pass equals
// token-by-token decode carrying every layer's (WKV state, both shift registers) forward — lifting
// timemix.go/channelmix.go's own carry invariants to the full multi-layer stack.
func TestModel_RWKV7Session_Forward_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 2)
	tokens := []int32{3, 7, 1, 5}

	full, err := NewSession(m).Forward(tokens)
	if err != nil {
		t.Fatalf("full prefill: %v", err)
	}

	s := NewSession(m)
	var stepped []float32
	for _, tok := range tokens {
		h, serr := s.Forward([]int32{tok})
		if serr != nil {
			t.Fatalf("step: %v", serr)
		}
		stepped = append(stepped, h...)
	}
	for i := range full {
		if full[i] != stepped[i] {
			t.Fatalf("hidden[%d] = %v, want one-pass %v (decode carry diverged)", i, stepped[i], full[i])
		}
	}
}

func TestModel_RWKV7Session_headLogits_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 1)
	logits := NewSession(m).headLogits(syn(m.D, 7))
	if len(logits) != m.Vocab {
		t.Fatalf("logits len %d, want %d", len(logits), m.Vocab)
	}
}

// TestModel_RWKV7Session_headLogits_Ugly proves a nil LMHead ties the head to Embed (matNT(normed,
// Embed,...) rather than erroring or returning zeros.
func TestModel_RWKV7Session_headLogits_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 1)
	m.LMHead = nil
	tiedLogits := NewSession(m).headLogits(syn(m.D, 7))

	m2 := *m
	m2.LMHead = append([]float32(nil), m.Embed...)
	explicitLogits := NewSession(&m2).headLogits(syn(m.D, 7))
	for i := range tiedLogits {
		if tiedLogits[i] != explicitLogits[i] {
			t.Fatalf("tied-head logits[%d] = %v, want == explicit-tied %v", i, tiedLogits[i], explicitLogits[i])
		}
	}
}

func TestModel_RWKV7Session_Generate_Good(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 1)
	gen, err := NewSession(m).Generate([]int32{1, 2, 3}, 4, -1)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(gen) != 4 {
		t.Fatalf("generated %d tokens, want 4", len(gen))
	}
}

func TestModel_RWKV7Session_Generate_Bad(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 1)
	if _, err := NewSession(m).Generate(nil, 4, -1); err == nil {
		t.Fatal("empty prompt accepted")
	}
}

// TestModel_RWKV7Session_Generate_Ugly proves eosID stops generation early (short of maxNew) rather than
// always running the full budget.
func TestModel_RWKV7Session_Generate_Ugly(t *testing.T) {
	cfg := BlockConfig{NumHeads: 2, KeyDim: 4, ValueDim: 3}
	m := mkRWKV7Model(cfg, 8, 16, 32, 1)
	free, err := NewSession(m).Generate([]int32{1, 2}, 10, -1)
	if err != nil {
		t.Fatalf("free-running Generate: %v", err)
	}
	if len(free) == 0 {
		t.Fatal("free-running Generate produced nothing")
	}
	capped, err := NewSession(m).Generate([]int32{1, 2}, 10, int(free[0]))
	if err != nil {
		t.Fatalf("eos-capped Generate: %v", err)
	}
	if len(capped) != 1 {
		t.Fatalf("eos-capped Generate produced %d tokens, want 1 (stop on the very first token)", len(capped))
	}
}

func TestModel_argmaxF32_Good(t *testing.T) {
	if got := argmaxF32([]float32{1, 5, 3}); got != 1 {
		t.Fatalf("argmaxF32 = %d, want 1", got)
	}
}

// TestModel_argmaxF32_Ugly proves the FIRST occurrence wins on a tie, a well-defined (not arbitrary)
// tie-break.
func TestModel_argmaxF32_Ugly(t *testing.T) {
	if got := argmaxF32([]float32{2, 9, 9, 1}); got != 1 {
		t.Fatalf("argmaxF32 (tie) = %d, want 1 (first max)", got)
	}
}
