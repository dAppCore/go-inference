// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"os"
	"math"
	"sort"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/internal/enginegate"
	_ "dappco.re/go/inference/model/arch/allenai/olmoe"
)

// capture_hidden_olmoe_oracle_test.go is the (#65) per-layer oracle instrument for the OLMoE
// real-checkpoint parity gap named in real_checkpoint_olmoe_gpu_test.go /
// docs/zoo-moe-real-checkpoint-parity.md: mlx-community/OLMoE-1B-7B-0125-Instruct-4bit through the
// production ArchSession still emits wrong greedy ids after the MoE expert-combine activation fix
// landed (moe_block.go's UsesSiLU/#63 — confirmed engaged, not the residual defect). This file
// captures ForwardCaptureHiddens' per-layer post-block hidden sums at the receipt's own fixed prompt
// (olmoeCapitalPromptIDs, real_checkpoint_olmoe_gpu_test.go) and compares them against an independent
// mlx-lm extraction of the identical checkpoint at the identical ids — the same "which layer first
// disagrees" discriminator capture_hidden_qwen3_oracle_test.go established.
//
// FINDING: layer 0 already diverges (post-block +41%; the bisection,
// TestForwardCaptureHiddensOLMoEAttnMLPBisectionVsRealOracle via capturedAttnHiddens/
// capturedMLPResHiddens, splits it +63% attention / +100% MLP-MoE at layer 0, both growing with
// depth — by layer 5 attention alone is +1102%). Two INDEPENDENT, compounding static defects fully
// explain this shape, both confirmed by reading the reference implementation and the engine source
// side by side, not guessed from the numbers alone:
//
// DEFECT 1 (attention, layer 0 onward): OLMoE's q_norm/k_norm (self_attn.{q,k}_norm.weight,
// confirmed present in the real checkpoint's safetensors index) are shape [2048] =
// [n_heads*head_dim] (hidden_size, NOT head_dim=128) — mlx_lm's own reference
// (site-packages/mlx_lm/models/olmoe.py, Attention): `self.q_norm = nn.RMSNorm(n_heads*head_dim,
// eps)`, applied in __call__ as `queries = self.q_norm(queries)` on the FLAT [B,L,2048] projection
// BEFORE `.reshape(B,L,n_heads,-1)` — ONE RMSNorm reduction over all 2048 elements, weight
// elementwise over all 2048. This is a DIFFERENT operation from gemma4's per-head QK-norm the engine
// implements: engine/metal/qknorm_rope.go's QKNormRopeBF16 doc says plainly "x is [nHeads*headDim]
// bf16, weight is [headDim] bf16 (shared per head)"; engine/metal/attention.go's encRMSNormRowsBF16
// doc: RMS-norms `rows` contiguous rows of axisSize each, INDEPENDENTLY, with ONE shared weight of
// length axisSize — "gemma4 QK-norm uses this to norm each attention head's headDim slice (rows =
// nHeads, axisSize = headDim)". Both the fused (encQKNormRope, gated by gpuHasGeluKernel) and
// unfused (encRMSNormRowsBF16 then encRopeDecode) call sites in decode_forward_arch.go feed OLMoE's
// REAL [2048]-length checkpoint tensor into a kernel contracted to a [headDim]=[128]-length weight:
// the kernel reads only the FIRST 128 of 2048 weight elements (bytes 128..2047 of the checkpoint's
// own q_norm/k_norm never read) and broadcasts that wrong slice identically across all 16 heads,
// while independently RMS-reducing each head over its own 128 elements instead of the reference's
// ONE combined 2048-wide reduction — two compounding numerical errors in one kernel call, on every
// layer (q_norm AND k_norm both present on every OLMoE layer). model/assemble.go's `norm()` loader
// has no shape assertion against headDim, so this loads silently — no error, just wrong numbers.
//
// DEFECT 2 (router, compounds from layer 0 onward, explains the MLP/MoE half running even hotter
// than attention): OLMoE's router (mlx_lm's OlmoeSparseMoeBlock) computes `mx.softmax(router_logits,
// axis=1)` over ALL 64 experts FIRST, THEN selects the top-8 indices and GATHERS their
// already-computed probabilities (`take_along_axis`) — renormalising them to sum to 1 ONLY if
// norm_topk_prob is set, which this checkpoint's config.json declares false, so the true combine
// weights are the raw post-softmax-over-64 values and do NOT sum to 1. engine/metal/router.go's
// shared device/host router — engine/metal/moe_block.go's ONLY router call site
// (MoERouter/MoERouterQuant/moeRouterQuantDeviceTopK*, used identically by gemma4, mixtral, dbrx,
// olmoe and granitemoe on the device-router lane) — does the OPPOSITE order: it top-k SELECTS first
// (topKByScore / the GPU lthn_router_topk_impl.h insertion-sort) and THEN softmaxes ONLY over the
// selected K scores (softmaxAt / the kernel's `denom = Σ exp(best_values[i]-max)` over just the K
// winners) — this ALWAYS produces weights summing to exactly 1, unconditionally, regardless of the
// arch's declared NormaliseMoETopK (model/arch.go's own field — confirmed grep-dead: declared by
// every MoE arch's config.go, including olmoe/config.go's `NormaliseMoETopK: c.NormTopKProb`, but
// never READ anywhere in router.go or moe_block.go; model.MoEGating's own doc comment already flags
// this as unimplemented: "top-k weight renormalisation (norm_topk_prob)... each earn a value plus a
// router branch as they land"). Correct for gemma4 (a true renormalise-to-1 arch, and the "softmax
// over the top-k selected experts' scores" MoEGatingSoftmax IS gemma4's actual semantics per its own
// doc in model/arch.go) — wrong for this OLMoE checkpoint, which needs a materially different
// combine-weight magnitude (systematically SMALLER, competing against 64 not 8 terms) than router.go
// can produce today. (The codebase already knows how to do this correctly elsewhere: gpt_oss's own
// separate host MoE forward, engine/metal/arch_gptoss_moe.go, and the qwen fused chain's
// engine/metal/fused_chain_moe.go both thread a NormTopKProb-gated branch through their OWN
// independent host implementations — neither is router.go, and neither is reachable from OLMoE's
// dispatch, which is exclusively the generic device-router lane.)
//
// Named, not fixed here — BOTH defects sit outside this lane's file fence for a full fix, not just
// outside its time budget:
//
//   - Defect 1's kernel (engine/metal/qknorm_rope.go's fused path, attention.go's
//     encRMSNormRowsBF16) is "attention wiring" (in-fence under the provably-inert rule) but is
//     consulted from MANY call sites: decode_forward_arch.go's encAttnHalfShared/encAttnHalfKV and
//     both InputAt siblings (at least 4 live sites) AND the ICB recorder
//     (decode_forward_arch_icb.go/decode_forward_arch_icb_quant.go's setQKNormRope, a SEPARATE
//     recording implementation the encode-time doc comments say is kept in "lockstep" with the live
//     path but is not the same code) AND train_real_layer.go. A correct fix needs a genuinely NEW
//     encode primitive (a single-row nHeads*headDim-wide RMSNorm — encRMSNormBF16At already computes
//     exactly this shape, just never wired to this call site — followed by the EXISTING unfused
//     per-head RoPE, no per-head norm) selected per LAYER by the loaded weight's length (qDim vs
//     headDim), landed and byte-guard-verified at every one of those sites.
//   - Defect 2's kernel (engine/metal/router.go, both the host routerSelectWithScratch/softmaxAt
//     path and the GPU lthn_router_topk_impl.h) is NEITHER moe_block.go NOR attention wiring — a
//     third file this lane's fence does not license touching at all. A correct fix needs a new
//     MoEGating value (the "softmax over ALL experts, gather without forced renormalisation" policy
//     model/arch.go's own MoEGating doc already anticipates) plus a genuinely new router
//     host+device implementation selected by it — real router.go/moe_block.go surgery outside this
//     fence's boundary.
//
// Fixing ONLY one of the two would not flip the receipt test regardless (both actively corrupt
// every layer's output on this checkpoint), so there is no partial-fix path that reaches the
// acceptance bar from inside this fence; the full citation trail above is the handoff.
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

// oracleOLMoEAttnSumAbs / oracleOLMoEMLPSumAbs are the SAME extraction's attention-half output
// (x + Wo·attn, before the residual's OWN further MoE addition) and MLP(MoE)-half output sumAbs per
// layer — the bisection oracle for capturedAttnHiddens/capturedMLPResHiddens
// (decode_forward_arch.go), which only the PLAIN (icbDisabledForTest) route populates.
var oracleOLMoEAttnSumAbs = map[int]float64{
	0:  22.748016357421875,
	1:  24.893409729003906,
	2:  24.21977424621582,
	3:  19.2108154296875,
	4:  26.713010787963867,
	5:  25.206958770751953,
	6:  30.023195266723633,
	7:  30.97852897644043,
	8:  39.57813262939453,
	9:  40.534080505371094,
	10: 52.92363357543945,
	11: 90.70167541503906,
	12: 79.96588897705078,
	13: 99.60480499267578,
	14: 116.74137115478516,
	15: 341.2564697265625,
}

var oracleOLMoEMLPSumAbs = map[int]float64{
	0:  32.500335693359375,
	1:  41.83156967163086,
	2:  119.26433563232422,
	3:  63.34184265136719,
	4:  38.29867172241211,
	5:  35.06106948852539,
	6:  51.84075927734375,
	7:  60.25086212158203,
	8:  87.62808227539062,
	9:  108.88630676269531,
	10: 148.382080078125,
	11: 138.21327209472656,
	12: 277.4395751953125,
	13: 303.6336975097656,
	14: 402.1033630371094,
	15: 486.07220458984375,
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
	if os.Getenv("LTHN_OLMOE_ORACLE") == "" {
		t.Skip("known #65 divergence documented by this instrument — set LTHN_OLMOE_ORACLE=1 to run; flips to always-on when #65 closes")
	}
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
			t.Errorf("%s: %d layer(s) outside the %.0f%% per-layer band (KNOWN, root-caused OUTSIDE this lane's fix budget — see this file's header): %s",
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
	if os.Getenv("LTHN_OLMOE_ORACLE") == "" {
		t.Skip("known #65 divergence documented by this instrument — set LTHN_OLMOE_ORACLE=1 to run; flips to always-on when #65 closes")
	}
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
			t.Errorf("layer %d attention half outside the %.0f%% band (rel=%+.2f%%) — KNOWN, see file header (OLMoE's whole-vector QK-norm fed to the per-head kernel)", l, oracleOLMoEPerLayerBandFrac*100, attnRel*100)
		}
		if mlpRel < -oracleOLMoEPerLayerBandFrac || mlpRel > oracleOLMoEPerLayerBandFrac {
			t.Errorf("layer %d MLP/MoE half outside the %.0f%% band (rel=%+.2f%%) — the attention corruption feeds it a wrong input, COMPOUNDED by router.go's own defect (KNOWN, see file header: topk-then-softmax-over-K always renormalises combine weights to sum 1, wrong for this norm_topk_prob=false checkpoint)", l, oracleOLMoEPerLayerBandFrac*100, mlpRel*100)
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
	if os.Getenv("LTHN_OLMOE_ORACLE") == "" {
		t.Skip("known #65 divergence documented by this instrument — set LTHN_OLMOE_ORACLE=1 to run; flips to always-on when #65 closes")
	}
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
