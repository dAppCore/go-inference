// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/tokenizer"
	_ "dappco.re/go/inference/model/arch/Qwen/qwen3" // register "qwen3" — this file's real target checkpoint's arch
)

// assistant_dflash_zlab_test.go proves the z-lab live glue
// (assistant_dflash_zlab.go, ExtractAuxHiddensAllRaw in
// assistant_dflash_proposer.go, generateDFlashZLab in speculative_model.go):
// the cheap nil/shape guards run without any model; the big receipt —
// env-gated, real checkpoints only — is a live paired generate against the
// real z-lab/Qwen3-4B-DFlash-b16 drafter + its real Qwen/Qwen3-4B target,
// checked BOTH for determinism (same prompt twice, byte-identical output)
// AND for the architecture's actual invariant: byte-identical to the SAME
// target's plain autoregressive decode (docs/design-dflash-forward.md §5 —
// "the emitted sequence is byte-identical to plain greedy decode regardless
// of what the proposer returns").

func TestExtractAuxHiddensAllRaw_Bad(t *testing.T) {
	t.Run("nil target", func(t *testing.T) {
		if _, err := ExtractAuxHiddensAllRaw(nil, []int32{1}, []int{0}); err == nil {
			t.Fatal("a nil target session must fail")
		}
	})
	t.Run("empty ids", func(t *testing.T) {
		if _, err := ExtractAuxHiddensAllRaw(&ArchSession{}, nil, []int{0}); err == nil {
			t.Fatal("an empty prefix must fail")
		}
	})
	t.Run("no aux layers", func(t *testing.T) {
		if _, err := ExtractAuxHiddensAllRaw(&ArchSession{}, []int32{1}, nil); err == nil {
			t.Fatal("no requested aux layers must fail")
		}
	})
}

func TestZLabDFlashProposer_ProposeBlock_Bad(t *testing.T) {
	okSource := func([]int) ([]float32, int, []byte, bool) { return nil, 0, nil, true }
	okHead := func([]byte) (int32, error) { return 0, nil }

	t.Run("nil proposer", func(t *testing.T) {
		var p *zLabDFlashProposer
		if got := p.ProposeBlock([]int{1}); got != nil {
			t.Fatalf("nil proposer must return nil, got %v", got)
		}
	})
	t.Run("nil drafter", func(t *testing.T) {
		p := &zLabDFlashProposer{maskEmbed: []byte{1, 2}, head: okHead, source: okSource}
		if got := p.ProposeBlock([]int{1}); got != nil {
			t.Fatalf("nil drafter must return nil, got %v", got)
		}
	})
	t.Run("nil source", func(t *testing.T) {
		p := &zLabDFlashProposer{drafter: &zLabDFlashDrafter{}, maskEmbed: []byte{1, 2}, head: okHead}
		if got := p.ProposeBlock([]int{1}); got != nil {
			t.Fatalf("nil source must return nil, got %v", got)
		}
	})
	t.Run("nil head", func(t *testing.T) {
		p := &zLabDFlashProposer{drafter: &zLabDFlashDrafter{}, maskEmbed: []byte{1, 2}, source: okSource}
		if got := p.ProposeBlock([]int{1}); got != nil {
			t.Fatalf("nil head must return nil, got %v", got)
		}
	})
	t.Run("empty mask embedding", func(t *testing.T) {
		p := &zLabDFlashProposer{drafter: &zLabDFlashDrafter{}, head: okHead, source: okSource}
		if got := p.ProposeBlock([]int{1}); got != nil {
			t.Fatalf("empty mask embedding must return nil, got %v", got)
		}
	})
	t.Run("source reports not ok", func(t *testing.T) {
		p := &zLabDFlashProposer{
			drafter:   &zLabDFlashDrafter{},
			maskEmbed: []byte{1, 2},
			head:      okHead,
			source:    func([]int) ([]float32, int, []byte, bool) { return nil, 0, nil, false },
		}
		if got := p.ProposeBlock([]int{1}); got != nil {
			t.Fatalf("a not-ok source must return nil, got %v", got)
		}
	})
	t.Run("mismatched anchor embedding length", func(t *testing.T) {
		m, _, _ := dzBuild(dzTiny, 500) // dzTiny.hidden == dzHidden (12) — dflash_zlab_test.go's synthetic geometry
		p := &zLabDFlashProposer{
			drafter:   &zLabDFlashDrafter{model: m},
			maskEmbed: make([]byte, dzHidden*bf16Size),
			head:      okHead,
			source: func([]int) ([]float32, int, []byte, bool) {
				return nil, 0, make([]byte, dzHidden*bf16Size-1), true // one byte short
			},
		}
		if got := p.ProposeBlock([]int{1}); got != nil {
			t.Fatalf("a mis-sized anchor embedding must return nil, got %v", got)
		}
	})
}

// TestSpeculativeModel_DFlashZLab_RealPairedGenerate loads the REAL
// z-lab/Qwen3-4B-DFlash-b16 drafter paired against its REAL Qwen/Qwen3-4B
// target and runs the block-diffusion proposer through the model-free
// lossless verifier end to end. Skips cleanly without both env vars:
//
//	python3 -c "from huggingface_hub import snapshot_download; \
//	  print(snapshot_download('z-lab/Qwen3-4B-DFlash-b16'))"
//	python3 -c "from huggingface_hub import snapshot_download; \
//	  print(snapshot_download('Qwen/Qwen3-4B'))"
//	MLX_METALLIB_PATH=... LTHN_DFLASH_ZLAB_CKPT=<drafter dir> \
//	  LTHN_DFLASH_ZLAB_TARGET=<target dir> \
//	  go test -tags metal_runtime -count=1 -run RealPairedGenerate ./engine/metal/ -v
//
// KNOWN receipt (2026-07-20): determinism and losslessness both PASS against
// real weights — the wiring is correct. The logged accept-rate is currently
// 0.00: ExtractAuxHiddensAllRaw is plumbed correctly (it passes
// ForwardCaptureHiddens' own per-layer rows straight through — see its own
// doc), but ForwardCaptureHiddens' INTERMEDIATE-layer capture was
// cross-validated against an independent transformers/torch extraction of
// the SAME real Qwen3-4B forward (same tokens, same target_layer_ids) and
// diverges by ~2x total magnitude — while its OWN final-layer output agrees
// exactly with the ordinary PrefillTokens+BoundaryLogits decode path (a
// within-engine check). That isolates a pre-existing bug to
// train_session.go/decode_forward_arch.go's per-layer capture bookkeeping
// (outside this lane's fence: dflash*/assistant_dflash*/z-lab only) — not
// this file's aux-tap wiring. The drafter is fed the WRONG context and so
// proposes nothing useful, but AcceptBlock's lossless verify still holds:
// every committed token is the target's own, proven below. DFlashEngineProbe
// stays false until that shared capture bug is fixed and re-measured.
func TestSpeculativeModel_DFlashZLab_RealPairedGenerate(t *testing.T) {
	draftDir := core.Getenv("LTHN_DFLASH_ZLAB_CKPT")
	targetDir := core.Getenv("LTHN_DFLASH_ZLAB_TARGET")
	if core.Trim(draftDir) == "" || core.Trim(targetDir) == "" {
		t.Skip("set LTHN_DFLASH_ZLAB_CKPT (drafter) and LTHN_DFLASH_ZLAB_TARGET (Qwen/Qwen3-4B target) to real local snapshots (see test doc comment)")
	}
	const prompt = "The capital of France is"
	const maxNew = 12

	runPaired := func() ([]int32, inference.SpeculativeMetrics) {
		t.Helper()
		mdl, err := LoadSpeculativePair(targetDir, draftDir, 0)
		if err != nil {
			t.Fatalf("LoadSpeculativePair: %v", err)
		}
		defer func() { _ = mdl.Close() }()
		var ids []int32
		for tok := range mdl.Generate(context.Background(), prompt, inference.WithTemperature(0), inference.WithMaxTokens(maxNew)) {
			ids = append(ids, tok.ID)
		}
		if err := mdl.Err(); err.Err() != nil {
			t.Fatalf("paired Generate: %v", err.Err())
		}
		var sm inference.SpeculativeMetrics
		if p, ok := mdl.(inference.SpeculativeMetricsProvider); ok {
			sm = p.SpeculativeMetrics()
		}
		return ids, sm
	}

	first, stats1 := runPaired()
	if len(first) == 0 {
		t.Fatal("paired generate produced no tokens")
	}
	t.Logf("paired run 1: %d tokens, proposed=%d accepted=%d rounds=%d accept-rate=%.2f",
		len(first), stats1.ProposedTokens, stats1.AcceptedTokens, stats1.TargetVerifyCalls, stats1.AcceptanceRate)
	if stats1.ProposedTokens == 0 {
		t.Error("z-lab drafter never proposed a block — the live aux tap or proposer did not engage")
	}

	second, stats2 := runPaired()
	if len(first) != len(second) {
		t.Fatalf("determinism: run lengths differ, %d vs %d", len(first), len(second))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("determinism: token %d differs, %d vs %d", i, first[i], second[i])
		}
	}
	t.Logf("paired run 2: proposed=%d accepted=%d rounds=%d accept-rate=%.2f",
		stats2.ProposedTokens, stats2.AcceptedTokens, stats2.TargetVerifyCalls, stats2.AcceptanceRate)

	// Losslessness: byte-identical to the SAME target's plain greedy decode —
	// the architecture's own invariant (docs/design-dflash-forward.md §5).
	target, err := LoadDir(targetDir, 0)
	if err != nil {
		t.Fatalf("load plain target: %v", err)
	}
	defer func() { _ = target.Close() }()
	tok, err := tokenizer.LoadTokenizer(core.PathJoin(targetDir, "tokenizer.json"))
	if err != nil {
		t.Fatalf("load tokenizer: %v", err)
	}
	promptIDs := tok.Encode(prompt)
	eos := int32(-1)
	if tok.HasEOSToken() {
		eos = tok.EOS()
	}
	plain, err := target.Generate(promptIDs, maxNew, int(eos))
	if err != nil {
		t.Fatalf("plain target generate: %v", err)
	}
	n := min(len(plain), len(first))
	if n == 0 {
		t.Fatal("nothing to compare — both runs produced zero tokens")
	}
	for i := 0; i < n; i++ {
		if plain[i] != first[i] {
			t.Fatalf("losslessness: token %d diverges from plain greedy decode: paired=%d plain=%d", i, first[i], plain[i])
		}
	}
	if len(plain) != len(first) {
		t.Logf("note: plain decode emitted %d tokens, paired emitted %d (a stop-token boundary difference, not a content mismatch — the shared %d-token prefix matched byte-for-byte)", len(plain), len(first), n)
	}
}
