// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/dflash"
	"dappco.re/go/inference/decode/tokenizer"
)

// assistant_dflash_zlab_acceptrate_test.go extends
// TestSpeculativeModel_DFlashZLab_RealPairedGenerate's single-prompt receipt
// with a per-round accept-rate SURVEY against the same real
// z-lab/Qwen3-4B-DFlash-b16 drafter + Qwen/Qwen3-4B target, driven over
// several prompts for enough rounds to read a distribution rather than one
// aggregate number. It reuses the exact propose/verify construction
// speculativeModel.generateDFlashZLab wires for live serving (same package,
// same unexported session primitives — embedID, headLogitsScratch,
// PrefillTokens, BoundaryLogits), with its own round loop substituted for
// dflash.Generate's so each round's accepted count is observable:
// inference.SpeculativeMetrics (what a plain paired Generate call returns)
// is aggregate-only and cannot report a distribution, only a total. Env-gated
// exactly as the paired-generate receipt; skips cleanly without both vars set
// (see that test's doc comment for the exact env/run recipe).
func TestSpeculativeModel_DFlashZLab_AcceptRateSurvey(t *testing.T) {
	draftDir := core.Getenv("LTHN_DFLASH_ZLAB_CKPT")
	targetDir := core.Getenv("LTHN_DFLASH_ZLAB_TARGET")
	if core.Trim(draftDir) == "" || core.Trim(targetDir) == "" {
		t.Skip("set LTHN_DFLASH_ZLAB_CKPT (drafter) and LTHN_DFLASH_ZLAB_TARGET (Qwen/Qwen3-4B target) to real local snapshots (see TestSpeculativeModel_DFlashZLab_RealPairedGenerate's doc comment)")
	}

	target, err := LoadDir(targetDir, 0)
	if err != nil {
		t.Fatalf("load target: %v", err)
	}
	defer func() { _ = target.Close() }()
	zlab, err := loadZLabDFlashDrafter(draftDir)
	if err != nil {
		t.Fatalf("load z-lab drafter: %v", err)
	}
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(targetDir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}

	// Eight prompts spanning prose completion, narrative, code, arithmetic and
	// a repetitive pangram — enough per-prompt tokens that, even at the low
	// accept-rate this architecture has shown on real weights, the combined
	// round count clears the ≥64 floor a meaningful distribution needs, and
	// enough variety that no single prompt's noise dominates the aggregate.
	const maxNewPerPrompt = 48
	prompts := []string{
		"The capital of France is",
		"Once upon a time, there was a",
		"def fibonacci(n):\n    if n <= 1:\n        return n\n    return",
		"The three primary colours are red, blue, and",
		"import numpy as np\n\ndef normalise(x):\n    return x /",
		"Water boils at 100 degrees Celsius, which is the same as",
		"1, 2, 3, 4,",
		"The quick brown fox jumps over the lazy",
	}

	var roundAccepts []int
	var totalProposed, totalAccepted, totalTokens int

	for _, prompt := range prompts {
		promptIDs := tok.Encode(prompt)
		if len(promptIDs) == 0 {
			t.Fatalf("empty encode for prompt %q", prompt)
		}
		maskEmbedRaw, err := target.embedID(zlab.MaskTokenID())
		if err != nil {
			t.Fatalf("prompt %q: embed mask token: %v", prompt, err)
		}
		maskEmbed := append([]byte(nil), maskEmbedRaw...) // embedID hands back shared scratch — pin it

		var runErr error
		proposer := &zLabDFlashProposer{
			drafter:   zlab,
			maskEmbed: maskEmbed,
			source: func(context []int) ([]float32, int, []byte, bool) {
				prefix := speculativeInt32s(context)
				raw, aerr := ExtractAuxHiddensAllRaw(target, prefix, zlab.AuxLayers())
				if aerr != nil {
					runErr = core.E("dflash-survey", "extract aux hiddens", aerr)
					return nil, 0, nil, false
				}
				anchor, eerr := target.embedID(prefix[len(prefix)-1])
				if eerr != nil {
					runErr = core.E("dflash-survey", "embed anchor", eerr)
					return nil, 0, nil, false
				}
				return raw, len(prefix), append([]byte(nil), anchor...), true
			},
			head: func(hidden []byte) (int32, error) {
				logits, herr := target.headLogitsScratch(hidden, false)
				if herr != nil {
					return 0, herr
				}
				return greedyBF16Suppressed(logits, target.arch.Vocab, nil)
			},
		}
		next := func(prefix []int) int {
			if runErr != nil {
				return 0
			}
			if perr := target.PrefillTokens(speculativeInt32s(prefix)); perr != nil {
				runErr = core.E("dflash-survey", "prefill verifier", perr)
				return 0
			}
			logits, lerr := target.BoundaryLogits()
			if lerr != nil {
				runErr = core.E("dflash-survey", "read verifier logits", lerr)
				return 0
			}
			id, serr := greedyBF16Suppressed(logits, target.arch.Vocab, nil)
			if serr != nil {
				runErr = core.E("dflash-survey", "select verifier token", serr)
				return 0
			}
			return int(id)
		}

		seq := speculativeInts(promptIDs)
		var out []int
		promptProposed, promptAccepted, promptRounds := 0, 0, 0
		for len(out) < maxNewPerPrompt {
			proposed := proposer.ProposeBlock(seq)
			if runErr != nil {
				t.Fatalf("prompt %q: %v", prompt, runErr)
			}
			if len(proposed) == 0 {
				tkID := next(seq)
				if runErr != nil {
					t.Fatalf("prompt %q: %v", prompt, runErr)
				}
				seq = append(seq, tkID)
				out = append(out, tkID)
				continue
			}
			commit, accepted := dflash.AcceptBlock(seq, proposed, next)
			if runErr != nil {
				t.Fatalf("prompt %q: %v", prompt, runErr)
			}
			roundAccepts = append(roundAccepts, accepted)
			promptProposed += len(proposed)
			promptAccepted += accepted
			promptRounds++
			for _, tkID := range commit {
				seq = append(seq, tkID)
				out = append(out, tkID)
				if len(out) >= maxNewPerPrompt {
					break
				}
			}
		}
		totalProposed += promptProposed
		totalAccepted += promptAccepted
		totalTokens += len(out)
		promptMean := 0.0
		if promptRounds > 0 {
			promptMean = float64(promptAccepted) / float64(promptRounds)
		}
		t.Logf("prompt %q: tokens=%d rounds=%d proposed=%d accepted=%d mean-accepted/round=%.3f completion=%q",
			prompt, len(out), promptRounds, promptProposed, promptAccepted, promptMean, tok.Decode(speculativeInt32s(out)))
	}

	totalRounds := len(roundAccepts)
	if totalRounds == 0 {
		t.Fatal("no rounds recorded — the drafter never engaged (every block was empty)")
	}
	dist := map[int]int{}
	for _, a := range roundAccepts {
		dist[a]++
	}
	for k := 0; k <= 15; k++ {
		if n := dist[k]; n > 0 {
			t.Logf("  accepted=%d: %d rounds (%.1f%%)", k, n, 100*float64(n)/float64(totalRounds))
		}
	}
	meanAccepted := float64(totalAccepted) / float64(totalRounds)
	acceptRate := 0.0
	if totalProposed > 0 {
		acceptRate = float64(totalAccepted) / float64(totalProposed)
	}
	tokensPerRound := float64(totalTokens) / float64(totalRounds)
	t.Logf("SURVEY TOTAL: rounds=%d proposed=%d accepted=%d mean-accepted/round=%.3f accept-rate=%.4f tokens/round=%.3f",
		totalRounds, totalProposed, totalAccepted, meanAccepted, acceptRate, tokensPerRound)
	if totalRounds < 64 {
		t.Logf("note: only %d rounds recorded (<64 target) — %d tokens/prompt needed fewer verify rounds than planned at this accept-rate; the mean-accepted/round figure above is still the honest per-round read", totalRounds, maxNewPerPrompt)
	}
}
