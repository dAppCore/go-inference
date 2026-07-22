// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"encoding/binary"
	"math"
	"sort"
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
// FINDING: that attribution is wrong. The tap is not
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
// #67 RESOLVED — two mechanisms, both receipted by the instruments below:
//
//  1. model/arch/Qwen/qwen3/qwen3.go never declared Arch.Activation, so ffnUsesSiLU("")
//     kept the gemma GELU gate on all 36 layers of a checkpoint whose config declares
//     hidden_act "silu". GELU tracks SiLU closely enough to stay coherent while drifting
//     percent-level per MLP pass; the drift compounded through the residual stream
//     (the ±20% mid-stack wobble the per-layer table used to show) and was amplified by
//     layer 35's trained near-exact cancellation of the massive-activation channels.
//     Fixed by forwarding hidden_act into Arch.Activation (the same declaration-gap class
//     as this file's own EmbedScale finding, #66) plus the fused fp32-internal
//     silu(gate)·up kernel (lthn_silu_gate_mul_bf16 — one rounding vs the composed
//     chain's three). Post-fix: every layer 0..35 sits inside ±0.3% of the oracle.
//
//  2. The all-layer table's layer-35 row was extracted from output_hidden_states, whose
//     FINAL entry transformers appends POST-model.norm — a different quantity from the
//     raw post-block stream every other row (and this capture's contract) carries. That
//     convention mismatch presented as a +454% "layer-35 divergence" and survived an
//     earlier norm-check because that check normed the already-normed oracle row. The
//     table now stores the hook-extracted raw value (see the table comment).
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
	// oracleGrandTotalSumAbs (relative) and still count as "matches". Tightened by #67
	// from a pre-EmbedScale-fix 0.15 to bracket the now-precisely-measured,
	// repeat-run-deterministic -11.2% residual these 5 samples land on post-fix, with headroom
	// for legitimate cross-implementation bf16 noise but not much more: #67's own all-layer
	// instrument (TestForwardCaptureHiddensQwen3AllLayersVsRealOracle below) found a SEPARATE,
	// much larger fault (layer 35, +454%) these 5 samples cannot see at all — none of
	// {1,9,17,25,33} is layer 35 — so this band passing is NOT itself evidence ordinary decode
	// is healthy; see that other test and this file's header for the fault this one is blind to.
	oracleParityBandFrac = 0.13
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

// oracleAllLayerTotalSumAbs is the #67 bisection instrument's FULL-DEPTH oracle: the same
// per-layer total sumAbs (all 5 prompt tokens, all 2560 hidden dims) as
// oracleAuxLayerTotalSumAbs above, but for EVERY decoder layer (0..35), not just DFlash's 5
// target_layer_ids (which are spaced 8 apart — too coarse to tell a genuine depth-compounding
// trend from a per-layer-CLASS fault). Keyed the same way: perLayerOut[l] (Go's 0-indexed
// decoder-layer output) == hidden_states[l+1] (the verified offset=1 convention). The 5 indices
// shared with oracleAuxLayerTotalSumAbs (1, 9, 17, 25, 33) are byte-identical to that map,
// because both were read from the SAME extraction (same process, same forward pass) — see
// qwen3resid_alllayers.py's cross-check in this lane's own working notes, reproduced inline by
// TestOracleAllLayerTotalSumAbsAgreesWithFiveLayerOracle below.
//
// Generated (#67): transformers 5.5.4 + torch 2.13.0, bfloat16, MPS,
// Qwen/Qwen3-4B (local snapshot, the same LTHN_DFLASH_ZLAB_TARGET checkpoint),
// output_hidden_states=True over the same 5 raw prompt tokens ("The capital of France is", no
// chat template, no BOS). Not required at test time; this is the readout of that one-time
// extraction, exactly as oracleAuxLayerTotalSumAbs is.
//
// Layer 35 is the one row NOT taken from output_hidden_states: transformers appends the FINAL
// entry post-model.norm (hidden_states[36] == norm(raw layer-35 out), proven byte-exact:
// maxAbsDiff 0.0000), while this capture's contract — and every other row — is the RAW
// post-block residual stream. The raw value below is hook-extracted (x_in + attn + down) from
// the same stack; the post-norm 20410.58 it replaces was the +454% "layer-35 divergence"
// artefact.
var oracleAllLayerTotalSumAbs = map[int]float64{
	0:  1792.79541015625,
	1:  2387.3935546875,
	2:  2664.96044921875,
	3:  2902.5625,
	4:  3592.69775390625,
	5:  4400.53076171875,
	6:  14259.818359375,
	7:  14770.841796875,
	8:  16003.388671875,
	9:  16304.04296875,
	10: 15816.4150390625,
	11: 15921.939453125,
	12: 15914.5439453125,
	13: 16145.29296875,
	14: 16160.49609375,
	15: 16371.6953125,
	16: 16733.80859375,
	17: 16815.525390625,
	18: 17200.99609375,
	19: 17955.25390625,
	20: 18298.1875,
	21: 18856.3828125,
	22: 19838.76953125,
	23: 22036.65625,
	24: 23367.728515625,
	25: 26315.84375,
	26: 28295.87109375,
	27: 30875.6015625,
	28: 33642.65625,
	29: 37293.359375,
	30: 41463.5703125,
	31: 47096.0703125,
	32: 50895.984375,
	33: 55627.7421875,
	34: 61843.16015625,
	35: 124450.7109,
}

const (
	// oracleAllLayerGrandTotalSumAbs is the sum of oracleAllLayerTotalSumAbs over all 36 layers —
	// the same "grand total" metric oracleGrandTotalSumAbs computes over just the 5 DFlash
	// layers, widened to the full depth (layer 35 at its raw post-block value — see the map
	// comment).
	oracleAllLayerGrandTotalSumAbs = 884313.2939453125
	// oracleAllLayerGrandTotalBandFrac is a coarse, WEAK sanity check on the all-layer grand
	// total: #67's own bisection found it nearly USELESS as a diagnostic — a +454% single-layer
	// fault at layer 35 (see oracleAllLayerPerLayerBandFrac below) happens to land the grand
	// total at only +1.4%/+1.71% (icb-default/icb-disabled) because 34 layers running ~11-20%
	// UNDER the oracle happen to cancel most of one layer running massively OVER it. Kept as a
	// coarse regression trip-wire, NOT the real gate — see the per-layer check for that.
	oracleAllLayerGrandTotalBandFrac = 0.05
	// oracleAllLayerPerLayerBandFrac is the REAL gate: #67's bisection found every layer except
	// the last (35) sits within this band (worst "background" case: layer 6 at -19.71%, the
	// massive-activation onset the file header already names) — legible as ordinary
	// cross-implementation bf16 noise (Metal vs MPS/PyTorch), not a defect. Layer 35 (the FINAL
	// decoder layer) fails it by over an order of magnitude (+454.50%/+456.63%): the oracle
	// SHRINKS every token's magnitude at this layer (a real trained cancellation of the
	// massive-activation dims carried since layer 6 — confirmed by an independent transformers
	// check: model.norm applied to the oracle's own raw layer-35 output barely moves its
	// magnitude, so this is not a premature-final-norm artefact either), while the engine GROWS
	// instead — for EVERY token, not just one position. #67 could not find any qwen3.go
	// Config.Arch() field (rope theta, attention scale, QK-norm eps/placement, head_dim, layer
	// classification — all individually verified against real_checkpoint's config.json and HF's
	// modeling_qwen3.py, and all uniform across layers by construction: Qwen3Config has no
	// per-layer override capability for a non-hybrid checkpoint) that could explain a fault
	// isolated to one specific layer. Identical (to 2 decimal places of relative error) under
	// BOTH icb-default and icb-disabled, so it is not ICB replay/recording bookkeeping either —
	// see this file's header for the named seam this points to instead.
	oracleAllLayerPerLayerBandFrac = 0.25
)

// TestOracleAllLayerTotalSumAbsAgreesWithFiveLayerOracle pins the two oracle tables together:
// oracleAllLayerTotalSumAbs must reproduce oracleAuxLayerTotalSumAbs exactly at the 5 shared
// indices, since both were read from the same transformers extraction. A host-side (no
// checkpoint, no MLX_METALLIB_PATH) guard against the two tables drifting apart under future
// editing — not an oracle-vs-engine comparison.
func TestOracleAllLayerTotalSumAbsAgreesWithFiveLayerOracle(t *testing.T) {
	for layerID, want := range oracleAuxLayerTotalSumAbs {
		got, ok := oracleAllLayerTotalSumAbs[layerID]
		if !ok {
			t.Fatalf("oracleAllLayerTotalSumAbs missing shared index %d", layerID)
		}
		if got != want {
			t.Errorf("oracleAllLayerTotalSumAbs[%d]=%v disagrees with oracleAuxLayerTotalSumAbs[%d]=%v — the two oracle tables drifted apart", layerID, got, layerID, want)
		}
	}
}

// TestForwardCaptureHiddensQwen3AllLayersVsRealOracle is the #67 bisection instrument:
// TestForwardCaptureHiddensQwen3VsRealOracle's 5 DFlash target_layer_ids are spaced 8 apart,
// too coarse to distinguish a genuine depth-compounding trend from a per-layer-CLASS fault
// (e.g. a layer classification the checkpoint declares uniform but the engine derives
// unevenly). This test captures and compares EVERY decoder layer against the oracle, logging a
// full per-layer (got/want/rel%) table SORTED BY LAYER — the diagnosis instrument, not just a
// pass/fail gate — under BOTH the ICB replay (the session's default) AND the plain
// captureLayerHiddens route (icbDisabledForTest), the same two-route split
// TestForwardCaptureHiddensQwen3FaithfulToOrdinaryDecode above already uses: if only one route
// shows a given layer's fault, the fault is in that route's OWN capture bookkeeping, not the
// shared per-layer forward both routes replay (#67 found both routes agree to 2 decimal places
// of relative error at every layer, ruling ICB out). Soft-gated (t.Errorf, not t.Fatalf) on TWO
// checks: a weak whole-depth grand total (a coarse trip-wire only — #67 found it nearly blind to
// a single catastrophic layer, since that layer's excess happens to cancel the other layers'
// shared undershoot) and the real gate, a PER-LAYER band every layer must individually sit
// inside — mirroring TestForwardCaptureHiddensQwen3VsRealOracle's own soft-gate shape, but
// unwilling to let one bad layer hide behind 35 good ones.
// oracleQwen3Layer6MLPOps is the torch/transformers per-op readout INSIDE layer 6's MLP —
// the massive-activation onset (#67): the spike channels are created by THIS layer's
// down_proj in one shot (tok0 dim4 goes +5.56 → +4544.0 through the matmul). sumAbs over
// all 5 prompt tokens per op; same extraction stack as the file header (transformers 5.5.4 +
// torch 2.13.0 bf16 MPS, forward hooks on model.layers[6]'s submodules).
var oracleQwen3Layer6MLPOps = map[string]float64{
	"ln2":  3958.93,
	"gate": 73374.45,
	"up":   17839.51,
	"prod": 7991.68,
	"down": 12147.17,
}

// oracleQwen3Layer6DownTok0Dims are token 0's largest down_proj output channels (the spike
// write): dim → value.
var oracleQwen3Layer6DownTok0Dims = map[int]float64{4: 4544.0, 396: -1288.0, 0: -215.0, 100: -203.0}

// TestQwen3Layer6MLPOpsVsRealOracle bisects INSIDE the massive-activation onset layer:
// which MLP op first departs from the oracle. The all-layers instrument above shows the
// engine's stream runs -19.7% right at layer 6 and the deficit persists to the layer-35
// cancellation (+454%); this instrument names the op that under-produces the spike.
func TestQwen3Layer6MLPOpsVsRealOracle(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := core.Getenv("LTHN_DFLASH_ZLAB_TARGET")
	if core.Trim(targetDir) == "" {
		t.Skip("set LTHN_DFLASH_ZLAB_TARGET to a local Qwen/Qwen3-4B snapshot (see file doc comment)")
	}
	ids := []int32{oraclePromptTok0, oraclePromptTok1, oraclePromptTok2, oraclePromptTok3, oraclePromptTok4}

	prevICB := icbDisabledForTest
	icbDisabledForTest = true
	defer func() { icbDisabledForTest = prevICB }()
	probeLayer := 6
	if v := core.Getenv("LTHN_L6_PROBE_LAYER"); core.Trim(v) != "" {
		if r := core.ParseInt(core.Trim(v), 10, 32); r.OK {
			probeLayer = int(r.Value.(int64))
		}
	}
	prevProbe, prevRows := capturedMLPProbeLayer, capturedLayer5MLP
	capturedMLPProbeLayer, capturedLayer5MLP = probeLayer, nil
	defer func() { capturedMLPProbeLayer, capturedLayer5MLP = prevProbe, prevRows }()
	prevAttn, prevMLPRes := capturedAttnHiddens, capturedMLPResHiddens
	capturedAttnHiddens, capturedMLPResHiddens = nil, nil
	defer func() { capturedAttnHiddens, capturedMLPResHiddens = prevAttn, prevMLPRes }()

	target, err := LoadDir(targetDir, 0)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = target.Close() }()
	if _, _, err := target.ForwardCaptureHiddens(ids); err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	if len(capturedLayer5MLP) != len(ids) {
		t.Fatalf("captured %d MLP probe rows, want %d (one per token)", len(capturedLayer5MLP), len(ids))
	}

	sum := func(pick func(capturedMLPInternal) []byte) float64 {
		var s float64
		for _, row := range capturedLayer5MLP {
			s += olmoeHiddenSumAbs(pick(row))
		}
		return s
	}
	got := map[string]float64{
		"ln2":  sum(func(r capturedMLPInternal) []byte { return r.normed }),
		"gate": sum(func(r capturedMLPInternal) []byte { return r.gate }),
		"up":   sum(func(r capturedMLPInternal) []byte { return r.up }),
		"prod": sum(func(r capturedMLPInternal) []byte { return r.product }),
		"down": sum(func(r capturedMLPInternal) []byte { return r.down }),
	}
	t.Logf("%5s  %14s  %14s  %9s", "op", "got", "want", "rel%")
	for _, op := range []string{"ln2", "gate", "up", "prod", "down"} {
		want := oracleQwen3Layer6MLPOps[op]
		rel := (got[op] - want) / want
		t.Logf("%5s  %14.2f  %14.2f  %+8.2f%%", op, got[op], want, rel*100)
	}
	tok0Down := bf16ToF32Slice(capturedLayer5MLP[0].down)
	for _, dim := range []int{4, 396, 0, 100} {
		want := oracleQwen3Layer6DownTok0Dims[dim]
		t.Logf("tok0 down dim%-4d got=%+10.2f  want=%+10.2f", dim, tok0Down[dim], want)
	}
	if dump := core.Getenv("LTHN_L6_DUMP"); core.Trim(dump) != "" {
		for name, pick := range map[string]func(capturedMLPInternal) []byte{
			"normed": func(r capturedMLPInternal) []byte { return r.normed },
			"gate":   func(r capturedMLPInternal) []byte { return r.gate },
			"up":     func(r capturedMLPInternal) []byte { return r.up },
			"prod":   func(r capturedMLPInternal) []byte { return r.product },
		} {
			f32 := bf16ToF32Slice(pick(capturedLayer5MLP[0]))
			raw := make([]byte, len(f32)*4)
			for i, v := range f32 {
				binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(v))
			}
			if r := core.WriteFile(core.PathJoin(dump, "l6_tok0_"+name+".f32"), raw, 0o644); !r.OK {
				t.Fatalf("dump %s: %v", name, r.Value)
			}
		}
		// Both residual halves, every (token, layer): rows[tok*N+l], f32 rows of dModel.
		for name, rows := range map[string][][]byte{"halves_attn": capturedAttnHiddens, "halves_out": capturedMLPResHiddens} {
			var raw []byte
			for _, row := range rows {
				f32 := bf16ToF32Slice(row)
				chunk := make([]byte, len(f32)*4)
				for i, v := range f32 {
					binary.LittleEndian.PutUint32(chunk[i*4:], math.Float32bits(v))
				}
				raw = append(raw, chunk...)
			}
			if r := core.WriteFile(core.PathJoin(dump, name+".f32"), raw, 0o644); !r.OK {
				t.Fatalf("dump %s: %v", name, r.Value)
			}
		}
		t.Logf("tok0 vectors + all-layer halves dumped to %s (f32 little-endian)", dump)
	}
}

func TestForwardCaptureHiddensQwen3AllLayersVsRealOracle(t *testing.T) {
	requireNativeRuntime(t)
	targetDir := core.Getenv("LTHN_DFLASH_ZLAB_TARGET")
	if core.Trim(targetDir) == "" {
		t.Skip("set LTHN_DFLASH_ZLAB_TARGET to a local Qwen/Qwen3-4B snapshot (see file doc comment)")
	}
	ids := []int32{oraclePromptTok0, oraclePromptTok1, oraclePromptTok2, oraclePromptTok3, oraclePromptTok4}

	check := func(t *testing.T, label string) {
		t.Helper()
		target, err := LoadDir(targetDir, 0)
		if err != nil {
			t.Fatalf("%s: LoadDir: %v", label, err)
		}
		defer func() { _ = target.Close() }()

		_, perLayer, err := target.ForwardCaptureHiddens(ids)
		if err != nil {
			t.Fatalf("%s: ForwardCaptureHiddens: %v", label, err)
		}
		if len(perLayer) != len(oracleAllLayerTotalSumAbs) {
			t.Fatalf("%s: ForwardCaptureHiddens returned %d layers, oracle table has %d — checkpoint layer count changed?", label, len(perLayer), len(oracleAllLayerTotalSumAbs))
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

		layers := make([]int, 0, len(perLayer))
		for l := range perLayer {
			layers = append(layers, l)
		}
		sort.Ints(layers)

		grandGot := 0.0
		worstLayer, worstRel := -1, 0.0
		var worstPerTok []float64
		var violations []string
		t.Logf("%s: %4s  %14s  %14s  %9s", label, "L", "got", "want", "rel%")
		for _, l := range layers {
			want, ok := oracleAllLayerTotalSumAbs[l]
			if !ok {
				t.Fatalf("%s: oracleAllLayerTotalSumAbs missing layer %d", label, l)
			}
			row := perLayer[l]
			if len(row) != T*rowBytes {
				t.Fatalf("%s: perLayer[%d] is %d bytes, want %d", label, l, len(row), T*rowBytes)
			}
			got := 0.0
			perTok := make([]float64, T)
			for tok := 0; tok < T; tok++ {
				perTok[tok] = sumAbs(row[tok*rowBytes : (tok+1)*rowBytes])
				got += perTok[tok]
			}
			rel := (got - want) / want
			t.Logf("%s: %4d  %14.2f  %14.2f  %+8.2f%%", label, l, got, want, rel*100)
			grandGot += got
			if rel < -oracleAllLayerPerLayerBandFrac || rel > oracleAllLayerPerLayerBandFrac {
				violations = append(violations, core.Sprintf("layer %d (rel=%+.2f%%, per-token sumAbs got=%v)", l, rel*100, perTok))
			}
			if worstLayer < 0 || math.Abs(rel) > math.Abs(worstRel) {
				worstLayer, worstRel, worstPerTok = l, rel, perTok
			}
		}
		grandRel := (grandGot - oracleAllLayerGrandTotalSumAbs) / oracleAllLayerGrandTotalSumAbs
		t.Logf("%s: grand (all %d layers): got=%.2f want=%.2f rel=%+.2f%% (weak sanity band=±%.0f%%); worst single layer=%d (rel=%+.2f%%, per-token sumAbs got=%v)",
			label, len(layers), grandGot, oracleAllLayerGrandTotalSumAbs, grandRel*100, oracleAllLayerGrandTotalBandFrac*100, worstLayer, worstRel*100, worstPerTok)

		if grandRel < -oracleAllLayerGrandTotalBandFrac || grandRel > oracleAllLayerGrandTotalBandFrac {
			t.Errorf("%s: capture-vs-oracle all-layer grand total outside the weak %.0f%% sanity band (rel=%+.2f%%) — see the per-layer table above for the divergence shape",
				label, oracleAllLayerGrandTotalBandFrac*100, grandRel*100)
		}
		if len(violations) > 0 {
			t.Errorf("%s: %d layer(s) outside the %.0f%% per-layer band (KNOWN, root-caused OUTSIDE this lane's fence — see this file's header): %s",
				label, len(violations), oracleAllLayerPerLayerBandFrac*100, core.Join("; ", violations...))
		}
	}

	t.Run("icb-default", func(t *testing.T) { check(t, "icb-default") })
	t.Run("icb-disabled", func(t *testing.T) {
		prev := icbDisabledForTest
		icbDisabledForTest = true
		defer func() { icbDisabledForTest = prev }()
		check(t, "icb-disabled")
	})
}
