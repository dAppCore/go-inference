// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	core "dappco.re/go"
)

// capture_hidden_qwen3_oracle_test.go is the #66 ground-truth discriminator for
// ForwardCaptureHiddens' intermediate-layer taps, cross-validated against an
// independent transformers/torch extraction of the real Qwen/Qwen3-4B target.
// docs/design-dflash-forward.md §7b names the repro (prompt tokens
// [785,6722,315,9625,374] = "The capital of France is", target_layer_ids
// [1,9,17,25,33]) and reports a ~2x sumAbs divergence, attributed there to
// "ForwardCaptureHiddens' per-layer capture bookkeeping".
//
// FINDING (2026-07-20, this lane): that attribution is wrong. The tap is not
// the bug. Two proofs, both below:
//
//  1. Structural (read, not re-derived here): the plain per-token capture
//     (decode_forward_arch.go's captureLayerHiddens branches) stores `out`
//     strictly AFTER the attention half's residual add into hBuf and the MLP
//     half's residual add into out, before any PLE-gate/layer-scalar mutation
//     (both gemma4-only, inert for qwen3, which has neither) — exactly HF's
//     hidden_states[i] contract, the POST-block residual stream. The ICB
//     replay (decode_forward_arch_icb.go's stepBodyCapture) independently
//     carves the identical point via the #391-fixed recorded layerOpStarts
//     boundaries, and TestForwardCaptureHiddens2PassBoundaries already proves
//     both routes byte-identical across every layer on a synthetic
//     2-pass-triggering fixture. The "extract_context_feature" layer_id+1
//     indexing this file's oracle numbers use was verified against the REAL
//     z-lab/Qwen3-4B-DFlash-b16 checkpoint's own shipped utils.py
//     (extract_context_feature: `offset = 1; hidden_states[layer_id +
//     offset]`) — perLayerOut[l] (0-indexed decoder-layer output, no
//     embedding slot) already equals hidden_states[l+1] with NO adjustment,
//     so the existing consumer code (ExtractAuxHiddens/ExtractAuxHiddensAllRaw
//     in assistant_dflash_proposer.go, outside this lane's fence and
//     untouched) reads the right index too.
//
//  2. Empirical, on the real checkpoint (TestForwardCaptureHiddensQwen3FaithfulToOrdinaryDecode,
//     below): ForwardCaptureHiddens' final-layer row, read through this
//     session's own head, reproduces EXACTLY the next token an independent
//     ArchSession.Generate call on the identical prefix picks — proven with
//     ICB enabled (the session's default) AND with icbDisabledForTest forcing
//     the plain route. The tap is faithful to whatever the ordinary decode
//     forward computes, in BOTH routes.
//
// The real bug is upstream of both the tap and ordinary decode, entirely
// outside this lane's fence (train_session.go / decode_forward_arch.go capture
// branches only, per this lane's brief): model/arch/Qwen/qwen3/qwen3.go's
// Config.Arch() never sets Arch.EmbedScale, so embedScaleOf's documented
// fallback (decode_forward_arch.go: "0 = undeclared → backends fall back to
// √hidden") silently applies GEMMA's √hidden embedding scale (√2560 ≈ 50.6×)
// to qwen3's embeddings. Every OTHER non-gemma arch package under model/arch/
// (llama, qwen2, qwen35, qwenmoe, olmo, olmoe, mixtral, granite, gptoss,
// starcoder2, ...) explicitly sets `EmbedScale: 1`; qwen3.go is the one
// omission. That ~50x-too-large embedding feeds layer 0 and compounds through
// all 36 layers — worse at deeper layers (more compounding) and worst at the
// first token (this checkpoint's massive-activation/attention-sink dimensions
// onset around hidden_states[7] and dominate token 0's norm from there),
// exactly the divergence shape design-dflash-forward.md §7b reports. Ordinary
// decode is equally corrupted, independent of any capture: `lem generate`
// against this exact checkpoint currently degenerates on EVERY prompt tried
// ("The capital of France is" -> ",,illon...illonillonillonverevere,";
// "2 + 2 =" -> ".....ррroadcastroadcastroadcast..."; "Hello, my name is" ->
// blank lines then dots) — and this file's own faithful-to-ordinary-decode
// test proves the capture reproduces that SAME wrong forward, not a different
// one, so "fix the tap" cannot close this gap.
//
// A throwaway LOCAL `EmbedScale: 1` patch to qwen3.go (verified in this lane,
// NOT applied or committed — model/arch/Qwen/qwen3/qwen3.go is outside the
// capture-machinery fence) moved the grand-total sumAbs at the real
// target_layer_ids from 221,719 (oracleWantGrandTotalSumAbs below is
// 117,450.55 — the unfixed engine sits +88.8% over) to ~104,270 (-11.2%
// under) and changed the argmax read off the captured final hidden from 43614
// (a raw UTF-8 continuation byte, not a real token) to 220 (a plausible but
// still-wrong space token) against the oracle's 12095 (" Paris"). EmbedScale
// is the DOMINANT term, not the whole story — a smaller residual gap survives
// that fix alone, left for that fix's own lane to close and re-measure.
//
// Oracle: transformers 5.5.4 + torch 2.13.0, bfloat16, MPS, Qwen/Qwen3-4B,
// output_hidden_states=True over the 5 raw prompt tokens (no chat template).
// hidden_states[layer_id+1] per the verified offset=1 above. Not required at
// test time; the constants below are the readout of that one-time extraction.
//
//	MLX_METALLIB_PATH=... LTHN_DFLASH_ZLAB_TARGET=<Qwen/Qwen3-4B snapshot> \
//	  go test -tags metal_runtime -run CaptureHiddensQwen3 -v ./engine/metal/
const (
	// oraclePromptToken{0..4} = "The capital of France is" (Qwen3 BPE, no BOS —
	// the tokenizer does not prepend one for this checkpoint).
	oraclePromptTok0, oraclePromptTok1, oraclePromptTok2, oraclePromptTok3, oraclePromptTok4 = 785, 6722, 315, 9625, 374
	// oracleNextTokenWant is transformers' own greedy argmax for this prefix (" Paris") —
	// logged for context, NOT asserted: even the identified EmbedScale fix alone doesn't
	// reach it (see header), so asserting it here would gate on a fix this lane can't make.
	oracleNextTokenWant = 12095
	// oracleGrandTotalSumAbs is the sum, over all 5 aux layers x all 5 prompt tokens x 2560
	// hidden dims, of |hidden_states[layer_id+1]| for layer_id in {1,9,17,25,33}.
	oracleGrandTotalSumAbs = 117450.5478515625
	// oracleParityBandFrac is how far the engine's own grand-total sumAbs may sit from
	// oracleGrandTotalSumAbs (relative) and still count as "matches". 0.15 clearly fails
	// today's unfixed +88.8%-over state and sits just outside the identified EmbedScale
	// fix's own -11.2%-under residual (see header) — deliberately not wide enough to call
	// that partial fix a pass, since a further factor is known to remain.
	oracleParityBandFrac = 0.15
)

// oracleAuxLayerTotalSumAbs is the per-aux-layer total (all 5 tokens, all 2560 dims) from
// the same extraction, keyed by target_layer_ids value (the index ExtractAuxHiddens/
// ExtractAuxHiddensAllRaw already use directly into ForwardCaptureHiddens' perLayerOut).
var oracleAuxLayerTotalSumAbs = map[int]float64{
	1:  2387.3935546875,
	9:  16304.04296875,
	17: 16815.525390625,
	25: 26315.84375,
	33: 55627.7421875,
}

// TestForwardCaptureHiddensQwen3VsRealOracle captures the real Qwen/Qwen3-4B target's
// hidden states at DFlash's real target_layer_ids and compares the per-layer and
// grand-total sumAbs against the transformers/torch oracle above. Soft-gated (t.Errorf,
// not t.Fatalf): this metric is KNOWN not to hold today (see file header) — the root
// cause sits outside this lane's fence — so this assertion documents the gap precisely
// and becomes the acceptance receipt for that other lane's fix, without turning an
// unfixable-here condition into a landmine for anyone else who runs this suite.
func TestForwardCaptureHiddensQwen3VsRealOracle(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := core.Getenv("LTHN_DFLASH_ZLAB_TARGET")
	if core.Trim(targetDir) == "" {
		t.Skip("set LTHN_DFLASH_ZLAB_TARGET to a local Qwen/Qwen3-4B snapshot (see file doc comment)")
	}
	ids := []int32{oraclePromptTok0, oraclePromptTok1, oraclePromptTok2, oraclePromptTok3, oraclePromptTok4}

	target, err := LoadDir(targetDir, 0)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = target.Close() }()

	_, perLayer, err := target.ForwardCaptureHiddens(ids)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	rowBytes := target.arch.Hidden * bf16Size
	T := len(ids)

	sumAbs := func(row []byte) float64 {
		var s float64
		for _, v := range bf16ToF32Slice(row) {
			if v < 0 {
				v = -v
			}
			s += float64(v)
		}
		return s
	}

	grandTotal := 0.0
	for layerID, want := range oracleAuxLayerTotalSumAbs {
		if layerID < 0 || layerID >= len(perLayer) {
			t.Fatalf("target_layer_ids value %d out of range [0,%d) — checkpoint layer count changed?", layerID, len(perLayer))
		}
		row := perLayer[layerID]
		if len(row) != T*rowBytes {
			t.Fatalf("perLayer[%d] is %d bytes, want %d", layerID, len(row), T*rowBytes)
		}
		got := 0.0
		for tok := 0; tok < T; tok++ {
			got += sumAbs(row[tok*rowBytes : (tok+1)*rowBytes])
		}
		rel := (got - want) / want
		t.Logf("aux layer %d (hidden_states[%d]): got sumAbs=%.2f want=%.2f (rel=%+.1f%%)", layerID, layerID+1, got, want, rel*100)
		grandTotal += got
	}

	rel := (grandTotal - oracleGrandTotalSumAbs) / oracleGrandTotalSumAbs
	t.Logf("grand total: got sumAbs=%.2f want=%.2f (rel=%+.1f%%, band=±%.0f%%)", grandTotal, oracleGrandTotalSumAbs, rel*100, oracleParityBandFrac*100)
	if rel < -oracleParityBandFrac || rel > oracleParityBandFrac {
		t.Errorf("capture-vs-oracle grand total outside the %.0f%% band (rel=%+.1f%%) — KNOWN, root-caused OUTSIDE this lane's fence: "+
			"model/arch/Qwen/qwen3/qwen3.go's Config.Arch() never sets EmbedScale, so embedScaleOf's √hidden fallback "+
			"(decode_forward_arch.go) wrongly scales qwen3's embeddings by √2560≈50.6x (every sibling non-gemma arch package "+
			"sets EmbedScale:1; qwen3.go is the one omission). See this file's header for the full evidence trail and the "+
			"verified-but-unapplied one-line fix.", oracleParityBandFrac*100, rel*100)
	}

	// argmax off the captured final hidden — logged against the true oracle for context,
	// not asserted (see oracleNextTokenWant's own doc: even the identified fix alone
	// doesn't reach it).
	last := perLayer[len(perLayer)-1]
	finalHidden := last[(T-1)*rowBytes : T*rowBytes]
	if target.head != nil {
		logits, herr := target.head(finalHidden, true)
		if herr == nil {
			best, bestV := int32(-1), float32(-1e30)
			for i, v := range bf16ToF32Slice(logits) {
				if v > bestV {
					bestV, best = v, int32(i)
				}
			}
			t.Logf("argmax off captured final hidden = %d (oracle greedy next token = %d, \" Paris\")", best, oracleNextTokenWant)
		}
	}
}

// TestForwardCaptureHiddensQwen3FaithfulToOrdinaryDecode is the hard, TRUE-today
// assertion this lane actually owns: ForwardCaptureHiddens' final-layer row, read
// through the SAME session's head, must pick the identical next token an INDEPENDENT
// ArchSession.Generate call on the identical prefix picks — proven with ICB replay (the
// session's default) AND with icbDisabledForTest forcing the plain captureLayerHiddens
// route. This is what actually lives in this lane's fence (the tap), and it holds
// regardless of the qwen3 EmbedScale bug TestForwardCaptureHiddensQwen3VsRealOracle
// documents: both sides of this comparison run over the SAME (today, corrupted) forward,
// so this proves the tap is faithful to it, not that the forward itself is correct.
func TestForwardCaptureHiddensQwen3FaithfulToOrdinaryDecode(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := core.Getenv("LTHN_DFLASH_ZLAB_TARGET")
	if core.Trim(targetDir) == "" {
		t.Skip("set LTHN_DFLASH_ZLAB_TARGET to a local Qwen/Qwen3-4B snapshot (see capture_hidden_qwen3_oracle_test.go doc comment)")
	}
	ids := []int32{oraclePromptTok0, oraclePromptTok1, oraclePromptTok2, oraclePromptTok3, oraclePromptTok4}

	check := func(t *testing.T, label string) {
		t.Helper()
		captureSess, err := LoadDir(targetDir, 0)
		if err != nil {
			t.Fatalf("%s: LoadDir(capture): %v", label, err)
		}
		defer func() { _ = captureSess.Close() }()
		_, perLayer, err := captureSess.ForwardCaptureHiddens(ids)
		if err != nil {
			t.Fatalf("%s: ForwardCaptureHiddens: %v", label, err)
		}
		if captureSess.head == nil {
			t.Fatalf("%s: session has no head — cannot cross-check", label)
		}
		rowBytes := captureSess.arch.Hidden * bf16Size
		last := perLayer[len(perLayer)-1]
		finalHidden := last[(len(ids)-1)*rowBytes : len(ids)*rowBytes]
		logits, herr := captureSess.head(finalHidden, true)
		if herr != nil {
			t.Fatalf("%s: head: %v", label, herr)
		}
		capturedArgmax, bestV := int32(-1), float32(-1e30)
		for i, v := range bf16ToF32Slice(logits) {
			if v > bestV {
				bestV, capturedArgmax = v, int32(i)
			}
		}

		genSess, err := LoadDir(targetDir, 0)
		if err != nil {
			t.Fatalf("%s: LoadDir(generate): %v", label, err)
		}
		defer func() { _ = genSess.Close() }()
		next, err := genSess.Generate(ids, 1, -1) // eosID<0: no early stop, 1 token is all we want
		if err != nil {
			t.Fatalf("%s: Generate: %v", label, err)
		}
		if len(next) != 1 {
			t.Fatalf("%s: Generate returned %d tokens, want 1", label, len(next))
		}
		if capturedArgmax != next[0] {
			t.Fatalf("%s: capture's final-hidden argmax (%d) diverges from ordinary Generate's next token (%d) — a REAL tap regression", label, capturedArgmax, next[0])
		}
		t.Logf("%s: capture argmax == ordinary Generate next token == %d (both faithful to the SAME underlying forward)", label, next[0])
	}

	t.Run("icb-default", func(t *testing.T) { check(t, "icb-default") })
	t.Run("icb-disabled", func(t *testing.T) {
		prev := icbDisabledForTest
		icbDisabledForTest = true
		defer func() { icbDisabledForTest = prev }()
		check(t, "icb-disabled")
	})
}
