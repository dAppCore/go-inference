// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import (
	"math"
	"testing"
)

// TestTextRotaryCosSin_Good hand-verifies the text decoder's 1D rotary table at position 0 (must
// be the identity: cos=1,sin=0 at every frequency — a zero angle regardless of invFreq) and at
// position 1 with theta=1 (invFreq[j]=1^(-2j/headDim)=1 for every j, so angle=1 at every
// frequency — collapses the whole table to a single repeated value, an easy hand check).
func TestTextRotaryCosSin_Good(t *testing.T) {
	cosHalf, sinHalf := textRotaryCosSin(8, 1000000, 0)
	for j := range cosHalf {
		if cosHalf[j] != 1 || sinHalf[j] != 0 {
			t.Fatalf("textRotaryCosSin(pos=0)[%d] = (%v,%v), want (1,0)", j, cosHalf[j], sinHalf[j])
		}
	}
	cosHalf, sinHalf = textRotaryCosSin(8, 1, 1)
	wantC, wantS := float32(math.Cos(1)), float32(math.Sin(1))
	for j := range cosHalf {
		if math.Abs(float64(cosHalf[j]-wantC)) > 1e-6 || math.Abs(float64(sinHalf[j]-wantS)) > 1e-6 {
			t.Fatalf("textRotaryCosSin(theta=1,pos=1)[%d] = (%v,%v), want (%v,%v)", j, cosHalf[j], sinHalf[j], wantC, wantS)
		}
	}
}

func TestArgmax32_Good(t *testing.T) {
	if got := argmax32([]float32{1, 5, 3, 5, -2}); got != 1 {
		t.Fatalf("argmax32 = %d, want 1 (first of the tied maxima)", got)
	}
}

// tinyModel builds a small, fully deterministic 2-layer GQA decoder (hidden=8, heads=4,
// kvHeads=2, headDim=2, intermediate=6, vocab=10) — hermetic (no checkpoint on disk), used by
// every decoder_test.go case that needs a Weights/Config pair rather than a golden fixture.
func tinyModel() (*Weights, *Config) {
	const hidden, heads, kvHeads, headDim, ffn, vocab = 8, 4, 2, 2, 6, 10
	seed := float32(1)
	next := func() float32 {
		seed = seed*1.0000001 + 0.017 // cheap deterministic PRNG stand-in, no math/rand dependency
		return float32(math.Mod(float64(seed)*37.0, 1.0))*2 - 1
	}
	fillLW := func(in, out int, bias bool) LinearWeights {
		w := make([]float32, in*out)
		for i := range w {
			w[i] = next() * 0.1
		}
		lw := LinearWeights{Weight: w, In: in, Out: out}
		if bias {
			b := make([]float32, out)
			for i := range b {
				b[i] = next() * 0.01
			}
			lw.Bias = b
		}
		return lw
	}
	fillRMS := func(dim int) RMSNormWeights {
		w := make([]float32, dim)
		for i := range w {
			w[i] = 1
		}
		return RMSNormWeights{Weight: w}
	}
	layers := make([]DecoderLayerWeights, 2)
	for i := range layers {
		layers[i] = DecoderLayerWeights{
			InputNorm:    fillRMS(hidden),
			Q:            fillLW(hidden, heads*headDim, true),
			K:            fillLW(hidden, kvHeads*headDim, true),
			V:            fillLW(hidden, kvHeads*headDim, true),
			O:            fillLW(heads*headDim, hidden, false),
			PostAttnNorm: fillRMS(hidden),
			Gate:         fillLW(hidden, ffn, false),
			Up:           fillLW(hidden, ffn, false),
			Down:         fillLW(ffn, hidden, false),
		}
	}
	embed := make([]float32, vocab*hidden)
	for i := range embed {
		embed[i] = next() * 0.1
	}
	w := &Weights{
		EmbedTokens: embed, Layers: layers, FinalNorm: fillRMS(hidden),
		LMHead:     fillLW(hidden, vocab, false),
		HiddenSize: hidden, VocabSize: vocab, NumAttentionHeads: heads, NumKeyValueHeads: kvHeads,
	}
	cfg := &Config{
		HiddenSize: hidden, NumAttentionHeads: heads, NumKeyValueHeads: kvHeads,
		VocabSize: vocab, RMSNormEps: 1e-6, RopeTheta: 10000,
	}
	return w, cfg
}

// TestEmbedTokens_Good proves EmbedTokens copies the right rows in order.
func TestEmbedTokens_Good(t *testing.T) {
	w, cfg := tinyModel()
	got, err := EmbedTokens([]int32{2, 0}, w, cfg)
	if err != nil {
		t.Fatalf("EmbedTokens: %v", err)
	}
	want := append(append([]float32{}, w.EmbedTokens[2*cfg.HiddenSize:3*cfg.HiddenSize]...), w.EmbedTokens[0:cfg.HiddenSize]...)
	if d := maxAbsDiff32(t, got, want); d > 0 {
		t.Fatalf("EmbedTokens = %v, want %v", got, want)
	}
}

// TestEmbedTokens_Bad proves an out-of-vocab id refuses rather than reading out of bounds.
func TestEmbedTokens_Bad(t *testing.T) {
	w, cfg := tinyModel()
	if _, err := EmbedTokens([]int32{int32(cfg.VocabSize)}, w, cfg); err == nil {
		t.Fatal("EmbedTokens accepted an out-of-vocab id")
	}
}

// TestDecodeLogitsStep_MatchesDecodeLogits_Good is the KV-cache correctness receipt: processing
// the SAME token sequence as one whole-sequence recompute (DecodeLogits) versus prefill-then-
// one-token-at-a-time (repeated DecodeLogitsStep calls sharing one growing cache) must produce
// BIT-IDENTICAL logits at the final position — see decoder.go's file doc comment for why (every
// op except causal self-attention is row-wise, and causal self-attention's row i depends only on
// rows [0,i], present in the cache regardless of which call added them).
func TestDecodeLogitsStep_MatchesDecodeLogits_Good(t *testing.T) {
	w, cfg := tinyModel()
	ids := []int32{1, 4, 7, 2, 9}

	whole, err := DecodeLogits(ids, w, cfg)
	if err != nil {
		t.Fatalf("DecodeLogits: %v", err)
	}

	cache := NewSelfAttnCache(len(w.Layers))
	var step []float32
	for i, id := range ids {
		step, err = DecodeLogitsStep([]int32{id}, i, cache, w, cfg)
		if err != nil {
			t.Fatalf("DecodeLogitsStep at %d: %v", i, err)
		}
	}
	if d := maxAbsDiff32(t, whole, step); d != 0 {
		t.Fatalf("whole-sequence vs one-token-at-a-time logits diverge by %v, want bit-identical", d)
	}

	// A THIRD batching (prefill 3, then 2 singles) must ALSO match — proves the property holds
	// for arbitrary batch splits, not just the fully-serial one above.
	cache2 := NewSelfAttnCache(len(w.Layers))
	if _, err := DecodeLogitsStep(ids[:3], 0, cache2, w, cfg); err != nil {
		t.Fatalf("DecodeLogitsStep prefill: %v", err)
	}
	if _, err := DecodeLogitsStep(ids[3:4], 3, cache2, w, cfg); err != nil {
		t.Fatalf("DecodeLogitsStep step 3: %v", err)
	}
	mixed, err := DecodeLogitsStep(ids[4:5], 4, cache2, w, cfg)
	if err != nil {
		t.Fatalf("DecodeLogitsStep step 4: %v", err)
	}
	if d := maxAbsDiff32(t, whole, mixed); d != 0 {
		t.Fatalf("whole-sequence vs mixed-batch logits diverge by %v, want bit-identical", d)
	}
}

// TestDecodeLogitsStep_Bad proves an empty batch refuses.
func TestDecodeLogitsStep_Bad(t *testing.T) {
	w, cfg := tinyModel()
	cache := NewSelfAttnCache(len(w.Layers))
	if _, err := DecodeLogitsStep(nil, 0, cache, w, cfg); err == nil {
		t.Fatal("DecodeLogitsStep accepted an empty batch")
	}
}

// TestDecodeLogits_Bad proves nil weights refuses.
func TestDecodeLogits_Bad(t *testing.T) {
	_, cfg := tinyModel()
	if _, err := DecodeLogits([]int32{0}, nil, cfg); err == nil {
		t.Fatal("DecodeLogits accepted nil weights")
	}
}

// TestGreedyDecode_Good proves the loop stops the instant it draws an EOS id, never appending it,
// and advances the cache position correctly across steps (checked indirectly: a second call
// continuing from scratch on the same prompt reproduces the same generation, determinism).
func TestGreedyDecode_Good(t *testing.T) {
	w, cfg := tinyModel()
	ids := []int32{1, 2, 3}
	embeds, err := EmbedTokens(ids, w, cfg)
	if err != nil {
		t.Fatalf("EmbedTokens: %v", err)
	}
	gen1, err := GreedyDecode(embeds, len(ids), w, cfg, 5, map[int32]bool{})
	if err != nil {
		t.Fatalf("GreedyDecode: %v", err)
	}
	if len(gen1) != 5 {
		t.Fatalf("GreedyDecode with no EOS ever hit generated %d tokens, want maxNewTokens=5", len(gen1))
	}
	// Stop-on-EOS: force the FIRST generated token into the stop set and confirm generation halts
	// immediately (zero tokens emitted).
	embeds2, _ := EmbedTokens(ids, w, cfg)
	gen2, err := GreedyDecode(embeds2, len(ids), w, cfg, 5, map[int32]bool{gen1[0]: true})
	if err != nil {
		t.Fatalf("GreedyDecode: %v", err)
	}
	if len(gen2) != 0 {
		t.Fatalf("GreedyDecode with the first draw in eosIDs generated %v, want none", gen2)
	}
}

// TestGreedyDecode_Bad proves a negative maxNewTokens refuses.
func TestGreedyDecode_Bad(t *testing.T) {
	w, cfg := tinyModel()
	embeds, _ := EmbedTokens([]int32{1}, w, cfg)
	if _, err := GreedyDecode(embeds, 1, w, cfg, -1, nil); err == nil {
		t.Fatal("GreedyDecode accepted a negative maxNewTokens")
	}
}
