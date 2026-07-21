// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"math"
	"sort"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/internal/enginegate"
	_ "dappco.re/go/inference/model/arch/allenai/olmoe"
)

// capture_hidden_olmoe_oracle_test.go is the (#65) per-layer oracle instrument for the OLMoE
// real-checkpoint parity gap named in real_checkpoint_olmoe_gpu_test.go /
// docs/zoo-moe-real-checkpoint-parity.md. It captures ForwardCaptureHiddens' per-layer post-block
// hidden sums at the receipt's own fixed prompt (olmoeCapitalPromptIDs, real_checkpoint_olmoe_gpu_test.go)
// and compares them against an independent mlx-lm extraction of the identical checkpoint at the
// identical ids — the same "which layer first disagrees" discriminator
// capture_hidden_qwen3_oracle_test.go established.
//
// Two independent, compounding defects fully explained the divergence this instrument originally
// found (layer 0 post-block +41%, growing with depth), both confirmed by reading the reference
// implementation and the engine source side by side:
//
// QK-NORM GRANULARITY: OLMoE's q_norm/k_norm (self_attn.{q,k}_norm.weight) are shape [2048] =
// [n_heads*head_dim] (hidden_size, NOT head_dim=128) — mlx_lm's own reference
// (site-packages/mlx_lm/models/olmoe.py, Attention): `self.q_norm = nn.RMSNorm(n_heads*head_dim,
// eps)`, applied on the FLAT [B,L,2048] projection BEFORE `.reshape(B,L,n_heads,-1)` — ONE RMSNorm
// reduction over all 2048 elements. gemma4's QK-norm is a DIFFERENT operation: one shared [headDim]
// weight RMS-normed independently per head. The engine applied the per-head kernel unconditionally,
// reading only the first headDim of 2048 weight elements and broadcasting that wrong slice across
// every head. Fixed by selecting granularity from the loaded weight's length at every consumer site
// (qkNormGranularity, qknorm_rope.go) and wrapping a whole-vector layer's projector
// (qkNormWideProjector) so the correct single-row RMSNorm runs immediately after the Q/K matmul —
// the choke point every consumer (plain/paged/shared-KV decode, the batched-dense prefill fold, the
// q8 KV landing, and the training forward/backward in train_real_layer.go/train_backward.go) already
// passes through, so none of those call sites needed their own changes.
//
// ROUTER COMBINE-WEIGHT ORDER: OLMoE's router (mlx_lm's OlmoeSparseMoeBlock) computes
// `mx.softmax(router_logits, axis=1)` over ALL 64 experts FIRST, then selects the top-8 indices and
// GATHERS their already-computed probabilities — renormalising to sum to 1 only if norm_topk_prob is
// set, which this checkpoint's config.json declares false, so the true combine weights are the raw
// post-softmax-over-64 values and do not sum to 1. engine/metal/router.go's shared device/host router
// always did the OPPOSITE order (top-k select, then softmax over just the K selected — which ALWAYS
// sums to 1), regardless of the arch's declared NormaliseMoETopK (model/arch.go — declared by every
// MoE arch's config parser, previously read nowhere). This order is correct for gemma4/mixtral
// (NormaliseMoETopK true: renormalising softmax-over-all's gathered top-K collapses to softmax over
// just that top-K — the same value, cheaper) and wrong for OLMoE. Fixed by wiring
// model.Arch.NormaliseMoETopK through MoELayerWeights.NormaliseTopK /
// MoEQuantLayerWeights.NormaliseTopK to the router: the top-K selection is unchanged (a monotonic
// transform of the scores), only the weight computation branches on the policy, with the device (GPU
// topK kernel) lane declining to the host path whenever the policy is false, since a fixed kernel can
// only implement the always-renormalise order.
//
// The fixes are proven at four independent resolutions: the production ArchSession's greedy
// generation matches the reference exactly (real_checkpoint_olmoe_gpu_test.go), every layer's
// post-block hidden sum tracks the oracle within band
// (TestForwardCaptureHiddensOLMoEAllLayersVsRealOracle, both the ICB-default and ICB-disabled
// routes), the embedding table output matches within band
// (TestForwardCaptureHiddensOLMoEEmbedVsRealOracle), and the attention half is isolated from the
// MoE half within band (TestForwardCaptureHiddensOLMoEAttnMLPBisectionVsRealOracle — the attn
// table holds the residual-inclusive x + Wo·attn the capture records; the mlp table equals the
// per-layer post-block values, so that side doubles as the capture-instrument identity check:
// capturedMLPResHiddens must track perLayerOut).
//
// Oracle: mlx_lm 0.31.3 (mlx 0.32.0), mlx-community/OLMoE-1B-7B-0125-Instruct-4bit (the identical
// checkpoint the receipt loads), olmoeCapitalPromptIDs fed directly as input_ids (no tokenizer
// round-trip — matching real_checkpoint_olmoe_gpu_test.go's own "ids hardcoded straight from mlx-lm"
// convention), one-shot causal forward (no kv cache, mlx_lm.models.base.create_attention_mask),
// float32 accumulation of |hidden| per mlx_lm's OWN Attention/OlmoeSparseMoeBlock/TransformerBlock
// forward read directly off site-packages (not re-derived). Cross-checked: the argmax off this
// extraction's own final hidden is 7785 (" Paris"), matching mlx-lm 0.31.3's reference generation
// exactly (real_checkpoint_olmoe_gpu_test.go's olmoeCapitalGenIDs[0]) — so the oracle extraction
// itself is trustworthy. Not required at test time; the constants below are the one-time readout.
//
//	MLX_METALLIB_PATH=... go test -tags metal_runtime -run CaptureHiddensOLMoE -v ./engine/metal/
const oracleOLMoEArgmaxWant = 7785 // " Paris" — mlx-lm 0.31.3 reference; this oracle's own argmax agrees

// oracleOLMoEEmbedSumAbs is embed_tokens' output sumAbs (all 5 prompt tokens, all 2048 dims) — layer
// 0's INPUT, before any attention/MoE runs. A sanity floor: olmoe/config.go declares EmbedScale: 1
// explicitly (unlike qwen3.go's #66 omission), so this should already track the oracle closely; if it
// didn't, the fault would be upstream of the per-layer block entirely.
const oracleOLMoEEmbedSumAbs = 31.53234100341797

// oracleOLMoELayerSumAbs is perLayerOut[l]'s (post-block, ForwardCaptureHiddens' own convention)
// sumAbs per layer, all 5 tokens summed, keyed by 0-indexed decoder layer (0..15 — OLMoE-1B-7B has
// 16 hidden layers, config.json's num_hidden_layers).
var oracleOLMoELayerSumAbs = map[int]float64{
	0:  46.016334533691406,
	1:  65.63995361328125,
	2:  160.38638305664062,
	3:  188.89402770996094,
	4:  201.52200317382812,
	5:  212.26454162597656,
	6:  227.75173950195312,
	7:  252.7152099609375,
	8:  288.267333984375,
	9:  343.22454833984375,
	10: 416.6954650878906,
	11: 483.67901611328125,
	12: 635.709716796875,
	13: 778.6072998046875,
	14: 931.8160400390625,
	15: 1127.736328125,
}

// oracleOLMoEAttnSumAbs / oracleOLMoEMLPSumAbs are the SAME extraction's residual-inclusive halves,
// matching what the capture variables record (decode_forward_arch.go, PLAIN icbDisabledForTest
// route only): attn = x + Wo·attn (before the MoE half runs), mlp = attn_res + FFN — the post-block
// value, so the mlp side equals oracleOLMoELayerSumAbs by construction and doubles as the
// capture-instrument identity check (capturedMLPResHiddens must track perLayerOut).
var oracleOLMoEAttnSumAbs = map[int]float64{
	0:  37.326515197754,
	1:  46.775234222412,
	2:  69.36083984375,
	3:  162.5450592041,
	4:  192.50723266602,
	5:  205.56518554688,
	6:  216.02139282227,
	7:  231.05601501465,
	8:  258.71295166016,
	9:  296.12786865234,
	10: 357.08334350586,
	11: 439.67974853516,
	12: 505.46264648438,
	13: 667.45446777344,
	14: 811.87286376953,
	15: 954.82830810547,
}

var oracleOLMoEMLPSumAbs = map[int]float64{
	0:  46.016334533691,
	1:  65.639953613281,
	2:  160.38638305664,
	3:  188.89402770996,
	4:  201.52200317383,
	5:  212.26454162598,
	6:  227.75173950195,
	7:  252.71520996094,
	8:  288.26733398438,
	9:  343.22454833984,
	10: 416.69546508789,
	11: 483.67901611328,
	12: 635.70971679688,
	13: 778.60729980469,
	14: 931.81604003906,
	15: 1127.736328125,
}

// oracleOLMoEPerLayerBandFrac is the per-layer relative-error band a healthy layer should sit
// inside — wide enough to absorb ordinary cross-implementation bf16/fp16 accumulation noise (the
// same order as capture_hidden_qwen3_oracle_test.go's ±20-25% all-layer band), not tuned to this
// (currently failing) checkpoint's actual numbers.
const oracleOLMoEPerLayerBandFrac = 0.25

// olmoeHiddenSumAbs sums |bf16(v)| over one token's dModel-wide bf16 row — the same metric
// capture_hidden_qwen3_oracle_test.go uses, factored out so all tests below share one definition.
func olmoeHiddenSumAbs(row []byte) float64 {
	var s float64
	for _, v := range bf16ToF32Slice(row) {
		if v < 0 {
			v = -v
		}
		s += float64(v)
	}
	return s
}

// TestForwardCaptureHiddensOLMoEAllLayersVsRealOracle captures OLMoE's per-layer post-block hidden
// (ForwardCaptureHiddens) at the real-checkpoint receipt's own fixed prompt and diffs the sumAbs
// against the mlx-lm oracle above, layer by layer, under BOTH the ICB replay (the session default)
// and the plain per-token route (icbDisabledForTest) — mirroring
// TestForwardCaptureHiddensQwen3AllLayersVsRealOracle's two-route split so an ICB-only bookkeeping
// fault (as opposed to the shared forward both routes replay) would show up as a route disagreement.
// Soft-gated (t.Errorf, not t.Fatalf): logs the full per-layer table (the diagnosis instrument) and
// documents the gap precisely rather than red-failing the whole package on an already-named,
// root-caused defect (see file header) outside this lane's provably-inert fix budget.
func TestForwardCaptureHiddensOLMoEAllLayersVsRealOracle(t *testing.T) {
	requireNativeRuntime(t)
	dir := enginegate.HFModelPath(t, "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit")

	check := func(t *testing.T, label string) {
		t.Helper()
		target, err := LoadDir(dir, 64)
		if err != nil {
			t.Fatalf("%s: LoadDir: %v", label, err)
		}
		defer func() { _ = target.Close() }()

		_, perLayer, err := target.ForwardCaptureHiddens(olmoeCapitalPromptIDs)
		if err != nil {
			t.Fatalf("%s: ForwardCaptureHiddens: %v", label, err)
		}
		if len(perLayer) != len(oracleOLMoELayerSumAbs) {
			t.Fatalf("%s: ForwardCaptureHiddens returned %d layers, oracle table has %d — checkpoint layer count changed?", label, len(perLayer), len(oracleOLMoELayerSumAbs))
		}
		rowBytes := target.arch.Hidden * bf16Size
		T := len(olmoeCapitalPromptIDs)

		layers := make([]int, 0, len(perLayer))
		for l := range perLayer {
			layers = append(layers, l)
		}
		sort.Ints(layers)

		worstLayer, worstRel := -1, 0.0
		var violations []string
		t.Logf("%s: %4s  %14s  %14s  %9s", label, "L", "got", "want", "rel%")
		for _, l := range layers {
			want, ok := oracleOLMoELayerSumAbs[l]
			if !ok {
				t.Fatalf("%s: oracleOLMoELayerSumAbs missing layer %d", label, l)
			}
			row := perLayer[l]
			if len(row) != T*rowBytes {
				t.Fatalf("%s: perLayer[%d] is %d bytes, want %d", label, l, len(row), T*rowBytes)
			}
			got := 0.0
			for tok := 0; tok < T; tok++ {
				got += olmoeHiddenSumAbs(row[tok*rowBytes : (tok+1)*rowBytes])
			}
			rel := (got - want) / want
			t.Logf("%s: %4d  %14.2f  %14.2f  %+8.2f%%", label, l, got, want, rel*100)
			if rel < -oracleOLMoEPerLayerBandFrac || rel > oracleOLMoEPerLayerBandFrac {
				violations = append(violations, core.Sprintf("layer %d (rel=%+.2f%%)", l, rel*100))
			}
			if worstLayer < 0 || math.Abs(rel) > math.Abs(worstRel) {
				worstLayer, worstRel = l, rel
			}
		}
		t.Logf("%s: worst layer=%d (rel=%+.2f%%)", label, worstLayer, worstRel*100)
		if target.head != nil {
			last := perLayer[len(perLayer)-1]
			finalHidden := last[(T-1)*rowBytes : T*rowBytes]
			if logits, herr := target.head(finalHidden, true); herr == nil {
				best, bestV := int32(-1), float32(-1e30)
				for i, v := range bf16ToF32Slice(logits) {
					if v > bestV {
						bestV, best = v, int32(i)
					}
				}
				t.Logf("%s: argmax off captured final hidden = %d (oracle greedy next token = %d, \" Paris\")", label, best, oracleOLMoEArgmaxWant)
			}
		}
		if len(violations) > 0 {
			t.Errorf("%s: %d layer(s) outside the %.0f%% per-layer band: %s",
				label, len(violations), oracleOLMoEPerLayerBandFrac*100, core.Join("; ", violations...))
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

// TestForwardCaptureHiddensOLMoEAttnMLPBisectionVsRealOracle isolates layer 0's fault to the
// attention half or the MLP(MoE) half — the SAME bisection technique
// capture_hidden_qwen3_oracle_test.go's file header describes for its own #67 layer-35 diagnosis,
// reading capturedAttnHiddens/capturedMLPResHiddens (decode_forward_arch.go), which only the plain
// per-token route populates (forced here via icbDisabledForTest — the ICB recorder carries no
// equivalent tap). Soft-gated per half, per layer: this is the instrument that names WHICH half is
// the first-diverging op, not a pass/fail gate on a defect already known to be present (see file
// header).
func TestForwardCaptureHiddensOLMoEAttnMLPBisectionVsRealOracle(t *testing.T) {
	requireNativeRuntime(t)
	dir := enginegate.HFModelPath(t, "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit")

	prevICB := icbDisabledForTest
	icbDisabledForTest = true
	defer func() { icbDisabledForTest = prevICB }()

	target, err := LoadDir(dir, 64)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = target.Close() }()

	prevAttn, prevMLP := capturedAttnHiddens, capturedMLPResHiddens
	capturedAttnHiddens, capturedMLPResHiddens = nil, nil
	defer func() { capturedAttnHiddens, capturedMLPResHiddens = prevAttn, prevMLP }()

	_, perLayer, err := target.ForwardCaptureHiddens(olmoeCapitalPromptIDs)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	N := len(perLayer)
	T := len(olmoeCapitalPromptIDs)

	if len(capturedAttnHiddens) != T*N || len(capturedMLPResHiddens) != T*N {
		t.Fatalf("capturedAttnHiddens/capturedMLPResHiddens length = %d/%d, want %d (T=%d * N=%d) — the plain route's per-layer capture is not wired the way this test assumes",
			len(capturedAttnHiddens), len(capturedMLPResHiddens), T*N, T, N)
	}

	perLayerHalfSumAbs := func(rows [][]byte, l int) float64 {
		got := 0.0
		for tok := 0; tok < T; tok++ {
			got += olmoeHiddenSumAbs(rows[tok*N+l])
		}
		return got
	}

	t.Logf("%4s  %14s  %14s  %9s  |  %14s  %14s  %9s", "L", "attn got", "attn want", "rel%", "mlp got", "mlp want", "rel%")
	for l := 0; l < N; l++ {
		attnWant, ok := oracleOLMoEAttnSumAbs[l]
		if !ok {
			t.Fatalf("oracleOLMoEAttnSumAbs missing layer %d", l)
		}
		mlpWant, ok := oracleOLMoEMLPSumAbs[l]
		if !ok {
			t.Fatalf("oracleOLMoEMLPSumAbs missing layer %d", l)
		}
		attnGot := perLayerHalfSumAbs(capturedAttnHiddens, l)
		mlpGot := perLayerHalfSumAbs(capturedMLPResHiddens, l)
		attnRel := (attnGot - attnWant) / attnWant
		mlpRel := (mlpGot - mlpWant) / mlpWant
		t.Logf("%4d  %14.2f  %14.2f  %+8.2f%%  |  %14.2f  %14.2f  %+8.2f%%", l, attnGot, attnWant, attnRel*100, mlpGot, mlpWant, mlpRel*100)
		if attnRel < -oracleOLMoEPerLayerBandFrac || attnRel > oracleOLMoEPerLayerBandFrac {
			t.Errorf("layer %d attention half outside the %.0f%% band (rel=%+.2f%%)", l, oracleOLMoEPerLayerBandFrac*100, attnRel*100)
		}
		if mlpRel < -oracleOLMoEPerLayerBandFrac || mlpRel > oracleOLMoEPerLayerBandFrac {
			t.Errorf("layer %d MLP/MoE half outside the %.0f%% band (rel=%+.2f%%)", l, oracleOLMoEPerLayerBandFrac*100, mlpRel*100)
		}
	}
}

// TestForwardCaptureHiddensOLMoEEmbedVsRealOracle is the sanity floor: layer 0's INPUT (embed_tokens'
// output, before any attention/MoE) must already track the oracle, or the fault would be upstream of
// the per-layer block entirely (e.g. a #66-shaped EmbedScale omission) rather than inside it.
// olmoe/config.go sets EmbedScale: 1 explicitly, so this is expected to pass; a failure here would
// redirect the whole investigation away from this file's attention finding.
func TestForwardCaptureHiddensOLMoEEmbedVsRealOracle(t *testing.T) {
	requireNativeRuntime(t)
	dir := enginegate.HFModelPath(t, "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit")

	target, err := LoadDir(dir, 64)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	defer func() { _ = target.Close() }()

	embeds, _, err := target.ForwardCaptureHiddens(olmoeCapitalPromptIDs)
	if err != nil {
		t.Fatalf("ForwardCaptureHiddens: %v", err)
	}
	got := 0.0
	for _, e := range embeds {
		got += olmoeHiddenSumAbs(e)
	}
	rel := (got - oracleOLMoEEmbedSumAbs) / oracleOLMoEEmbedSumAbs
	t.Logf("embed sumAbs: got=%.4f want=%.4f (rel=%+.2f%%)", got, oracleOLMoEEmbedSumAbs, rel*100)
	if rel < -oracleOLMoEPerLayerBandFrac || rel > oracleOLMoEPerLayerBandFrac {
		t.Errorf("embed sumAbs outside the %.0f%% band (rel=%+.2f%%) — the fault would be upstream of the per-layer block (EmbedScale or the embedding table), not this file's attention finding", oracleOLMoEPerLayerBandFrac*100, rel*100)
	}
}
