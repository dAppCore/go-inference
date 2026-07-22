// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
)

// TestSpeculativeModel_DFlashZLab_Round1Proposals is a diagnostic instrument,
// not a pin: for each survey prompt it reproduces the REFERENCE
// spec_generate's round-1 frame — context = prompt + the target's own greedy
// anchor (the reference samples that anchor during prefill, so its first
// draft round proposes with the anchor committed) — and logs our proposer's
// 15-token block alongside the anchor. Diffing this log against the
// reference's own round-1 `block_output_ids[:, 1:]` (the checkpoint's
// spec_generate, instrumented) splits the accept-rate gap cleanly: identical
// blocks put the fault in the verify/commit loop; divergent blocks put it in
// the drafter's inputs or forward. Env-gated exactly as the survey; skips
// without both vars.
func TestSpeculativeModel_DFlashZLab_Round1Proposals(t *testing.T) {
	draftDir := core.Getenv("LTHN_DFLASH_ZLAB_CKPT")
	targetDir := core.Getenv("LTHN_DFLASH_ZLAB_TARGET")
	if core.Trim(draftDir) == "" || core.Trim(targetDir) == "" {
		t.Skip("set LTHN_DFLASH_ZLAB_CKPT and LTHN_DFLASH_ZLAB_TARGET to real local snapshots")
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

	for _, prompt := range prompts {
		promptIDs := tok.Encode(prompt)
		if len(promptIDs) == 0 {
			t.Fatalf("empty encode for prompt %q", prompt)
		}
		// The reference's round-1 anchor: the target's own greedy pick after
		// the prompt (spec_generate samples it during prefill).
		if perr := target.PrefillTokens(promptIDs); perr != nil {
			t.Fatalf("prompt %q: prefill: %v", prompt, perr)
		}
		logits, lerr := target.BoundaryLogits()
		if lerr != nil {
			t.Fatalf("prompt %q: boundary logits: %v", prompt, lerr)
		}
		anchor, serr := greedyBF16Suppressed(logits, target.arch.Vocab, nil)
		if serr != nil {
			t.Fatalf("prompt %q: greedy anchor: %v", prompt, serr)
		}

		maskEmbedRaw, merr := target.embedID(zlab.MaskTokenID())
		if merr != nil {
			t.Fatalf("prompt %q: embed mask token: %v", prompt, merr)
		}
		maskEmbed := append([]byte(nil), maskEmbedRaw...)
		var runErr error
		proposer := &zLabDFlashProposer{
			drafter:   zlab,
			maskEmbed: maskEmbed,
			source: func(context []int) ([]float32, int, []byte, bool) {
				prefix := speculativeInt32s(context)
				raw, aerr := ExtractAuxHiddensAllRaw(target, prefix, zlab.AuxLayers())
				if aerr != nil {
					runErr = core.E("dflash-round1", "extract aux hiddens", aerr)
					return nil, 0, nil, false
				}
				anchorEmb, eerr := target.embedID(prefix[len(prefix)-1])
				if eerr != nil {
					runErr = core.E("dflash-round1", "embed anchor", eerr)
					return nil, 0, nil, false
				}
				return raw, len(prefix), append([]byte(nil), anchorEmb...), true
			},
			head: func(hidden []byte) (int32, error) {
				hl, herr := target.headLogitsScratch(hidden, false)
				if herr != nil {
					return 0, herr
				}
				return greedyBF16Suppressed(hl, target.arch.Vocab, nil)
			},
		}

		seq := append(speculativeInts(promptIDs), int(anchor))
		block := proposer.ProposeBlock(seq)
		if runErr != nil {
			t.Fatalf("prompt %q: %v", prompt, runErr)
		}
		t.Logf("ROUND1 anchor=%d proposals=%v prompt=%q", anchor, block, prompt[:min(len(prompt), 34)])
	}
}
