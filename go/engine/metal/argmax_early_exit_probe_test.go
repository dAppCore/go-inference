// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64 && metal_runtime

package native

import (
	"os"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
)

// bf16ArgmaxIdx returns the argmax index of vocab bf16 logits bytes.
func bf16ArgmaxIdx(logits []byte) int {
	best, bestV := -1, float32(0)
	for i := 0; i*2+1 < len(logits); i++ {
		v := bf16ToF32(logits[i*2], logits[i*2+1])
		if best < 0 || v > bestV {
			best, bestV = i, v
		}
	}
	return best
}

// TestArgmaxDeterminationDepthRealE2B is the #368 ceiling probe: on real
// e2b-4bit greedy decode, project EVERY layer's output hidden through the
// (final-norm + LM-head) and ask at which layer the final argmax is already
// decided — the logit-lens curve. Early-exit can only ever skip the layers
// past the decision point, so this measures the exploitable ceiling BEFORE any
// exit machinery is designed (the #364 lesson: instrument the ceiling first,
// the detector second). Static-probe caveat: intermediate hiddens are not
// trained to be head-projectable, so this is the OPTIMISTIC bound for
// logit-lens-style detectors, not a promise.
//
//	LEM_REAL_E2B=1 MLX_METALLIB_PATH=... go test -run TestArgmaxDeterminationDepthRealE2B -v ./engine/metal/
func TestArgmaxDeterminationDepthRealE2B(t *testing.T) {
	if os.Getenv(MetallibPathEnv) == "" {
		t.Skip("metallib not set")
	}
	if os.Getenv("LEM_REAL_E2B") == "" {
		t.Skip("set LEM_REAL_E2B=1 to run the real e2b-4bit argmax-determination probe (loads ~2.7GB)")
	}
	dir := resolveE2B4bitDir(t)
	lm, dm, err := loadRegistered(dir)
	if err != nil {
		t.Fatalf("loadRegistered: %v", err)
	}
	defer func() { _ = dm.Close() }()
	sb, err := buildShardBuffers(dm)
	if err != nil {
		t.Fatalf("buildShardBuffers: %v", err)
	}
	defer func() { _ = sb.Close() }()
	qm, err := loadedToQuant(lm, lm.Embed.GroupSize, lm.Embed.Bits)
	if err != nil {
		t.Fatalf("loadedToQuant: %v", err)
	}
	// per-layer capture instruments stepToken only — build the session with the
	// ICB recorder off so every step re-encodes (byte-identical, not bypassed).
	prevICB := icbDisabledForTest
	icbDisabledForTest = true
	defer func() { icbDisabledForTest = prevICB }()
	sess, err := newArchQuantSessionShards(qm, lm.Arch, 2048, sb)
	if err != nil {
		t.Fatalf("session: %v", err)
	}

	tok, terr := tokenizer.LoadTokenizer(core.PathJoin(dir, "tokenizer.json"))
	if terr != nil {
		t.Fatalf("tokenizer: %v", terr)
	}
	prompt := tok.Encode("<start_of_turn>user\nExplain, step by step, why rivers meander across " +
		"flood plains instead of flowing straight, and what that teaches us about erosion " +
		"and deposition over long timescales.<end_of_turn>\n<start_of_turn>model\n")
	if len(prompt) < 8 {
		t.Fatalf("prompt tokenised to %d tokens", len(prompt))
	}

	if perr := sess.PrefillTokens(prompt[:len(prompt)-1]); perr != nil {
		t.Fatalf("prefill: %v", perr)
	}

	const N = 48 // decoded tokens probed
	nLayers := len(sess.state.specs)
	agree := make([]int, nLayers)   // per-layer argmax == final argmax
	decided := make([]int, nLayers) // histogram of decided-at layer (all layers >= L agree)

	prevFlag, prevCap := captureLayerHiddens, capturedLayerHiddens
	captureLayerHiddens = true
	defer func() { captureLayerHiddens = prevFlag; capturedLayerHiddens = prevCap }()

	id := prompt[len(prompt)-1]
	for step := 0; step < N; step++ {
		emb, eerr := sess.embed(id)
		if eerr != nil {
			t.Fatalf("embed: %v", eerr)
		}
		capturedLayerHiddens = nil
		hidden, serr := sess.StepWithID(id, emb)
		if serr != nil {
			t.Fatalf("step %d: %v", step, serr)
		}
		if len(capturedLayerHiddens) != nLayers {
			t.Fatalf("captured %d layer hiddens, want %d (capture path not engaged?)", len(capturedLayerHiddens), nLayers)
		}
		finalLogits, herr := sess.head(hidden, true)
		if herr != nil {
			t.Fatalf("head(final): %v", herr)
		}
		finalArg := bf16ArgmaxIdx(finalLogits)

		layerArg := make([]int, nLayers)
		for li := 0; li < nLayers; li++ {
			logits, lherr := sess.head(capturedLayerHiddens[li], true)
			if lherr != nil {
				t.Fatalf("head(layer %d): %v", li, lherr)
			}
			layerArg[li] = bf16ArgmaxIdx(logits)
			if layerArg[li] == finalArg {
				agree[li]++
			}
		}
		decidedAt := nLayers - 1
		for li := nLayers - 1; li >= 0 && layerArg[li] == finalArg; li-- {
			decidedAt = li
		}
		decided[decidedAt]++

		id = int32(finalArg) // greedy continuation
	}

	t.Logf("=== #368 argmax determination — real e2b-4bit, %d layers, %d greedy tokens ===", nLayers, N)
	for li := 0; li < nLayers; li++ {
		bar := ""
		for b := 0; b < agree[li]*40/N; b++ {
			bar += "#"
		}
		t.Logf("  layer %2d: agree %3.0f%%  decided-here %2d  %s", li, float64(agree[li])*100/float64(N), decided[li], bar)
	}
	cum, p50, p90 := 0, -1, -1
	meanDecided := 0.0
	for li := 0; li < nLayers; li++ {
		cum += decided[li]
		meanDecided += float64(li * decided[li])
		if p50 < 0 && cum*2 >= N {
			p50 = li
		}
		if p90 < 0 && cum*10 >= 9*N {
			p90 = li
		}
	}
	meanDecided /= float64(N)
	t.Logf("  decided-at: mean %.1f  p50 %d  p90 %d of %d layers -> skippable ceiling ~%.0f%% of the stack (optimistic, lens-style)",
		meanDecided, p50, p90, nLayers, (float64(nLayers-1)-meanDecided)*100/float64(nLayers))
}
