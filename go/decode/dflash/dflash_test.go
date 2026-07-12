// SPDX-Licence-Identifier: EUPL-1.2

package dflash_test

import (
	"math/rand"
	"testing"

	"dappco.re/go/inference/decode/dflash"
)

// eqInts reports whether two token slices are element-wise equal; a nil and an
// empty slice are treated as equal (both mean "no tokens").
func eqInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// windowOracle builds a deterministic greedy next-token oracle whose output
// depends on the last `win` tokens of the prefix — so a wrong draft token really
// does steer the continuation elsewhere, exercising divergence handling. It is a
// pure function of the prefix, so Autoregress and Generate see identical targets.
func windowOracle(vocab, salt, mult, win int) dflash.NextToken {
	return func(prefix []int) int {
		s := salt
		for j := 0; j < win && j < len(prefix); j++ {
			s += prefix[len(prefix)-1-j] * mult
		}
		return ((s % vocab) + vocab) % vocab
	}
}

// TestParseConfig_Good recognises a real DFlash checkpoint from its config.json
// marker and reads the block size, fused verifier layers, and target model.
func TestParseConfig_Good(t *testing.T) {
	data := []byte(`{
		"speculators_model_type": "dflash",
		"block_size": 8,
		"aux_hidden_state_layer_ids": [3, 13, 23, 32, 42],
		"speculators_config": {
			"speculative_tokens": 7,
			"verifier": {"name": "deepseek-ai/DeepSeek-V4-Flash"}
		}
	}`)
	cfg, ok := dflash.ParseConfig(data)
	if !ok {
		t.Fatal("a speculators_model_type=dflash config must be recognised")
	}
	if cfg.BlockSize != 8 {
		t.Fatalf("block_size: want 8, got %d", cfg.BlockSize)
	}
	if !eqInts(cfg.AuxHiddenLayerIDs, []int{3, 13, 23, 32, 42}) {
		t.Fatalf("aux_hidden_state_layer_ids: got %v", cfg.AuxHiddenLayerIDs)
	}
	if cfg.Verifier != "deepseek-ai/DeepSeek-V4-Flash" {
		t.Fatalf("verifier: got %q", cfg.Verifier)
	}
}

// TestParseConfig_Bad declines a non-DFlash checkpoint: a plain model or another
// speculator type is not recognised, so a caller falls through to the next kind.
func TestParseConfig_Bad(t *testing.T) {
	for _, data := range [][]byte{
		[]byte(`{"model_type": "gemma4_text"}`),
		[]byte(`{"speculators_model_type": "eagle3"}`),
		[]byte(`{"speculators_config": {"algorithm": "dflash"}}`), // marker absent
	} {
		if _, ok := dflash.ParseConfig(data); ok {
			t.Fatalf("non-dflash config must not be recognised: %s", data)
		}
	}
}

// TestParseConfig_Ugly covers the degenerate inputs: malformed JSON is declined
// (not a panic), and a dflash config missing block_size falls back to the
// speculators_config speculative_tokens, clamping to a usable ≥ 1.
func TestParseConfig_Ugly(t *testing.T) {
	if _, ok := dflash.ParseConfig([]byte(`{not json`)); ok {
		t.Fatal("malformed JSON must be declined")
	}
	if _, ok := dflash.ParseConfig(nil); ok {
		t.Fatal("nil data must be declined")
	}
	// block_size absent → fall back to speculative_tokens.
	cfg, ok := dflash.ParseConfig([]byte(`{"speculators_model_type":"dflash","speculators_config":{"speculative_tokens":5}}`))
	if !ok || cfg.BlockSize != 5 {
		t.Fatalf("block_size should fall back to speculative_tokens 5: ok=%v block=%d", ok, cfg.BlockSize)
	}
	// neither present → clamp to 1 rather than a dead 0-token drafter.
	cfg, ok = dflash.ParseConfig([]byte(`{"speculators_model_type":"dflash"}`))
	if !ok || cfg.BlockSize != 1 {
		t.Fatalf("absent block should clamp to 1: ok=%v block=%d", ok, cfg.BlockSize)
	}
}

// TestLookupProposer_Good is the canonical block-lookup case: the trailing token
// recurred earlier, so the drafter proposes the block of tokens that followed it.
func TestLookupProposer_Good(t *testing.T) {
	p := dflash.NewLookupProposer(dflash.Config{BlockSize: 3})
	// Trailing token 1 last occurred at index 0, followed by [2 3 4] → propose them.
	//                     0  1  2  3  4  5
	got := p.ProposeBlock([]int{1, 2, 3, 4, 5, 1})
	if want := []int{2, 3, 4}; !eqInts(got, want) {
		t.Fatalf("want %v, got %v", want, got)
	}
}

// TestLookupProposer_Bad covers the no-match arm: a trailing token with no earlier
// occurrence yields an empty block (the target then decodes normally).
func TestLookupProposer_Bad(t *testing.T) {
	p := dflash.NewLookupProposer(dflash.Config{BlockSize: 4})
	if got := p.ProposeBlock([]int{1, 2, 3, 4, 5}); len(got) != 0 {
		t.Fatalf("no earlier occurrence of trailing token → empty, got %v", got)
	}
}

// TestLookupProposer_Ugly covers the degenerate inputs and the cap: nil / short
// contexts never panic and propose nothing, and a match near the end clamps the
// block to the tokens that actually exist (BlockSize is only an upper bound).
func TestLookupProposer_Ugly(t *testing.T) {
	p := dflash.NewLookupProposer(dflash.Config{BlockSize: 8})
	if got := p.ProposeBlock(nil); len(got) != 0 {
		t.Fatalf("nil context → empty, got %v", got)
	}
	if got := p.ProposeBlock([]int{7}); len(got) != 0 {
		t.Fatalf("single-token context → empty, got %v", got)
	}
	// Trailing 9 recurs at index 0; only [9] follows before the end (the trailing
	// 9 itself), so the block clamps to that one token.
	if got := p.ProposeBlock([]int{9, 9}); !eqInts(got, []int{9}) {
		t.Fatalf("block must clamp to available tokens: want [9], got %v", got)
	}
	// A zero-value Config clamps BlockSize to 1: propose a single token on a hit.
	z := dflash.NewLookupProposer(dflash.Config{})
	if got := z.ProposeBlock([]int{4, 5, 6, 4}); !eqInts(got, []int{5}) {
		t.Fatalf("zero config should still propose one token: want [5], got %v", got)
	}
}

// TestAcceptBlock_FullAcceptGood covers a block the target fully accepts: every
// proposed token equals the target's own, so all are committed plus one free
// bonus token, and the accepted count is the whole block.
func TestAcceptBlock_FullAcceptGood(t *testing.T) {
	// Target that always emits len(prefix)%5 — deterministic, prefix-length driven.
	next := func(prefix []int) int { return len(prefix) % 5 }
	prefix := []int{9, 9} // len 2,3,4,... → next emits 2,3,4,0
	proposed := []int{2, 3}
	commit, accepted := dflash.AcceptBlock(prefix, proposed, next)
	if accepted != 2 {
		t.Fatalf("whole block should be accepted: want 2, got %d", accepted)
	}
	// commit = the two accepted (2,3) + bonus (len 4 → 4).
	if want := []int{2, 3, 4}; !eqInts(commit, want) {
		t.Fatalf("full accept commits block + bonus: want %v, got %v", want, commit)
	}
}

// TestAcceptBlock_DivergeGood covers the first-divergence rule: the matching
// prefix is accepted, the target's correction token is committed at the mismatch,
// and the rest of the block (conditioned on a wrong token) is discarded.
func TestAcceptBlock_DivergeGood(t *testing.T) {
	next := func(prefix []int) int { return len(prefix) % 5 }
	prefix := []int{9, 9}      // target continuation is 2,3,4,0,1,...
	proposed := []int{2, 9, 9} // matches at 0 (2), diverges at 1 (want 3, got 9)
	commit, accepted := dflash.AcceptBlock(prefix, proposed, next)
	if accepted != 1 {
		t.Fatalf("one token accepted before divergence: want 1, got %d", accepted)
	}
	if want := []int{2, 3}; !eqInts(commit, want) {
		t.Fatalf("commit is accepted prefix + correction, then stop: want %v, got %v", want, commit)
	}
}

// TestAcceptBlock_Ugly covers an empty proposed block: with nothing to verify the
// driver still commits one target token (the bonus arm), so a caller always makes
// progress — accepted is zero because no draft token was on offer.
func TestAcceptBlock_Ugly(t *testing.T) {
	next := func(prefix []int) int { return 7 }
	commit, accepted := dflash.AcceptBlock([]int{1}, nil, next)
	if accepted != 0 {
		t.Fatalf("empty block accepts no draft tokens: want 0, got %d", accepted)
	}
	if want := []int{7}; !eqInts(commit, want) {
		t.Fatalf("empty block still commits one target token: want %v, got %v", want, commit)
	}
}

// TestAutoregress_Good pins the speculation-OFF baseline: exactly maxTokens greedy
// tokens, each the target's argmax for the running prefix.
func TestAutoregress_Good(t *testing.T) {
	next := func(prefix []int) int { return len(prefix) } // 0,1,2,3 from empty prompt
	got := dflash.Autoregress(nil, 4, next)
	if want := []int{0, 1, 2, 3}; !eqInts(got, want) {
		t.Fatalf("want %v, got %v", want, got)
	}
	if got := dflash.Autoregress(nil, 0, next); len(got) != 0 {
		t.Fatalf("maxTokens 0 → empty, got %v", got)
	}
}

// TestGenerate_LosslessInvariantFuzz is the receipt: across 20 000 random targets
// and adversarial proposers (correct blocks, corrupted blocks, empty blocks,
// varying sizes), the speculative Generate must emit EXACTLY what plain greedy
// Autoregress emits — spec-on == spec-off, token-for-token. A single divergence
// fails the pass. This is DFlash's losslessness, proven independent of drafter
// quality: the drafter can be arbitrarily wrong and the output never changes.
func TestGenerate_LosslessInvariantFuzz(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for iter := 0; iter < 20000; iter++ {
		vocab := 2 + rng.Intn(6)
		next := windowOracle(vocab, rng.Intn(97), 1+rng.Intn(31), 1+rng.Intn(4))
		blockSize := 1 + rng.Intn(8)

		// Adversarial proposer: builds the TRUE continuation (peeking the oracle),
		// then sometimes corrupts a random position (→ divergence) and varies the
		// block length (including 0). Spans full accept + partial + empty.
		prop := dflash.ProposerFunc(func(ctx []int) []int {
			n := rng.Intn(blockSize + 1)
			b := make([]int, n)
			tmp := append([]int(nil), ctx...)
			for k := 0; k < n; k++ {
				tk := next(tmp)
				tmp = append(tmp, tk)
				b[k] = tk
			}
			if n > 0 && rng.Intn(3) == 0 { // corrupt → force a mismatch
				pos := rng.Intn(n)
				b[pos] = (b[pos] + 1 + rng.Intn(vocab)) % vocab
			}
			return b
		})

		promptLen := rng.Intn(5)
		prompt := make([]int, promptLen)
		for k := range prompt {
			prompt[k] = rng.Intn(vocab)
		}
		maxTok := 1 + rng.Intn(40)

		got, _ := dflash.Generate(prompt, maxTok, prop, next)
		want := dflash.Autoregress(prompt, maxTok, next)
		if !eqInts(got, want) {
			t.Fatalf("spec-on diverged from spec-off:\n iter=%d prompt=%v maxTok=%d\n got  %v\n want %v",
				iter, prompt, maxTok, got, want)
		}
	}
}

// TestGenerate_AcceptRateGood shows the drafter earning its keep: on a target with
// exploitable repetition the model-free LookupProposer lands a non-zero
// acceptance rate, and — the invariant again — the output still equals plain
// decode. Accept-rate speeds the SAME tokens up; it never changes them.
func TestGenerate_AcceptRateGood(t *testing.T) {
	// A cyclic target: next token is prefix-length mod 4, so the continuation is a
	// repeating 0,1,2,3,0,1,2,3… that a lookup drafter can predict once it recurs.
	next := func(prefix []int) int { return len(prefix) % 4 }
	p := dflash.NewLookupProposer(dflash.Config{BlockSize: 4})

	got, stats := dflash.Generate(nil, 64, p, next)
	want := dflash.Autoregress(nil, 64, next)
	if !eqInts(got, want) {
		t.Fatalf("lossless: Generate must equal Autoregress\n got  %v\n want %v", got, want)
	}
	if stats.AcceptRate() <= 0 {
		t.Fatalf("a lookup drafter on a repeating target should accept > 0: %+v", stats)
	}
	if stats.ProposedTokens == 0 {
		t.Fatal("expected the drafter to have proposed tokens")
	}
}

// TestGenerate_Ugly covers the degenerate calls: maxTokens ≤ 0 emits nothing, and
// a proposer that always returns empty still generates via the plain-step arm
// (the loop must make progress and terminate, not spin).
func TestGenerate_Ugly(t *testing.T) {
	next := func(prefix []int) int { return 3 }
	if got, _ := dflash.Generate(nil, 0, dflash.NewLookupProposer(dflash.Config{}), next); len(got) != 0 {
		t.Fatalf("maxTokens 0 → empty, got %v", got)
	}
	// Never-proposing drafter: Generate must still emit maxTokens via plain steps.
	empty := dflash.ProposerFunc(func([]int) []int { return nil })
	got, stats := dflash.Generate([]int{1}, 5, empty, next)
	if want := dflash.Autoregress([]int{1}, 5, next); !eqInts(got, want) {
		t.Fatalf("empty-drafter Generate must equal Autoregress: got %v want %v", got, want)
	}
	if stats.ProposedTokens != 0 || stats.AcceptedTokens != 0 {
		t.Fatalf("a never-proposing drafter proposes/accepts nothing: %+v", stats)
	}
}

// TestStats_AcceptRate pins the rate arithmetic and the no-speculation guard: the
// rate is accepted/proposed, and zero proposed reports 0 (not a divide-by-zero).
func TestStats_AcceptRate(t *testing.T) {
	if r := (dflash.Stats{ProposedTokens: 8, AcceptedTokens: 6}).AcceptRate(); r != 0.75 {
		t.Fatalf("6/8 should be 0.75, got %v", r)
	}
	if r := (dflash.Stats{}).AcceptRate(); r != 0 {
		t.Fatalf("no proposed tokens → 0, got %v", r)
	}
}
