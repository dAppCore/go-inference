// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/quant/mlxaffine"
	"dappco.re/go/inference/model/safetensors"
)

// moe_quant_test.go gates the PACKED MoE expert path: moeExpertFF and swigluExpertQuantInto (moe.go), and
// buildMoE's proj-based expert resolution (loader.go) — the fix for the bug where a quant checkpoint's
// routed + shared experts were widened to f32 at load regardless (buildMoE called f32() instead of proj()),
// an ~8x blow-up on a grouped checkpoint's dominant tensor class. Mirrors composed_quant_test.go's
// dense-projection quant tests one level down, at the MoE FFN.

// TestMoeExpertFF_Dense pins moeExpertFF's dense branch: len(Gate)/D, the same derivation
// swigluExpertInto's own FF local uses inline.
func TestMoeExpertFF_Dense(t *testing.T) {
	const D, FF = 8, 12
	e := MoEExpert{Gate: syn(FF*D, 1)}
	if got := moeExpertFF(&e, D); got != FF {
		t.Fatalf("moeExpertFF(dense) = %d, want %d", got, FF)
	}
}

// TestMoeExpertFF_Packed pins moeExpertFF's packed branch: GateQ.OutDim, NOT len(Gate)/D (Gate is nil for
// a packed expert, so that division would silently read 0).
func TestMoeExpertFF_Packed(t *testing.T) {
	const D, FF = 8, 12
	e := MoEExpert{GateQ: &model.QuantWeight{OutDim: FF, InDim: D}}
	if got := moeExpertFF(&e, D); got != FF {
		t.Fatalf("moeExpertFF(packed) = %d, want %d (OutDim, not len(Gate)/D)", got, FF)
	}
}

// TestSwigluExpertQuantInto_MatchesDequantised pins the packed expert kernel against the dense reference
// at the kernel level: for a synthetic quantised expert, swigluExpertQuantInto (matNTQuant's per-row host
// dequant, rounding to f32 at each of the three matvec boundaries) agrees with the SAME weights
// dequantised once and run through swigluExpertInto (which stays f64 from xt to out, never rounding
// mid-computation) to a tight relative tolerance. NOT bit-identical — see swigluExpertQuantInto's doc for
// why the two are a different rounding tier by design, not a bug.
func TestSwigluExpertQuantInto_MatchesDequantised(t *testing.T) {
	// gs must divide BOTH D (gate/up's inDim) and FF (down's inDim) — unlike
	// TestMatNTQuantHost_MatchesDequantMatNT's single-weight table, an expert quantises three
	// interdependent shapes off one group size.
	for _, tc := range []struct{ D, FF, bits, gs int }{
		{64, 24, 4, 8},
		{128, 40, 8, 8},
		{96, 16, 2, 16},
	} {
		gateQ, gateW := quantiseSynthetic(t, tc.FF, tc.D, tc.bits, tc.gs, 1)
		upQ, upW := quantiseSynthetic(t, tc.FF, tc.D, tc.bits, tc.gs, 2)
		downQ, downW := quantiseSynthetic(t, tc.D, tc.FF, tc.bits, tc.gs, 3)

		eQ := MoEExpert{GateQ: gateQ, UpQ: upQ, DownQ: downQ}
		eD := MoEExpert{Gate: gateW, Up: upW, Down: downW}

		xt := make([]float32, tc.D)
		for i := range xt {
			xt[i] = float32((i%7)-3) * 0.1
		}

		gotQ := make([]float32, tc.D)
		swigluExpertQuantInto(xt, eQ, tc.D, gotQ)
		wantD := swigluExpert(xt, eD, tc.D)

		maxRel := relError(t, "swigluExpertQuantInto", gotQ, wantD, 1e-4)
		t.Logf("D=%d FF=%d bits=%d gs=%d: packed vs dequantised-dense max relative error %.3e", tc.D, tc.FF, tc.bits, tc.gs, maxRel)
	}
}

// quantiseSynthetic quantises a syn(seed)-derived (outDim x inDim) weight and returns both the packed
// model.QuantWeight and its exact dequantised f32 values — the pair every packed-vs-dense parity test in
// this file compares.
func quantiseSynthetic(t testing.TB, outDim, inDim, bits, gs, seed int) (*model.QuantWeight, []float32) {
	t.Helper()
	w := syn(outDim*inDim, seed)
	packed, scales, biases, err := mlxaffine.QuantizeTensor(w, outDim, inDim, bits, gs)
	if err != nil {
		t.Fatalf("quantise: %v", err)
	}
	deq, err := mlxaffine.DequantizeTensor(packed, scales, biases, outDim, inDim, bits, gs)
	if err != nil {
		t.Fatalf("dequantise: %v", err)
	}
	return &model.QuantWeight{Packed: packed, Scales: scales, Biases: biases, Bits: bits, GroupSize: gs, OutDim: outDim, InDim: inDim}, deq
}

// relError asserts got and want agree to a relative tolerance (rel = |got-want| / (1+|want|), this file's
// established cross-implementation bar — see TestMoEFullMixture and neighbours), t.Errorf-ing every
// element that breaks it, and returns the max relative error observed for the caller to log.
func relError(t *testing.T, label string, got, want []float32, tol float64) float64 {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: length %d != %d", label, len(got), len(want))
	}
	var maxRel float64
	for i := range want {
		rel := math.Abs(float64(got[i]-want[i])) / (1 + math.Abs(float64(want[i])))
		if rel > maxRel {
			maxRel = rel
		}
		if rel > tol {
			t.Errorf("%s[%d] = %v, want %v (rel %v > tol %v)", label, i, got[i], want[i], rel, tol)
		}
	}
	return maxRel
}

// mkMoEExpertQuant builds one packed SwiGLU expert (Gate/Up [FF,D], Down [D,FF], quantised bits/gs) from
// syn(seed)-derived values, returning the packed MoEExpert plus a DENSE MoEExpert over the SAME
// (dequantised) values — the pair TestMoEMLP_Forward_PackedMatchesDequantised compares.
func mkMoEExpertQuant(t testing.TB, D, FF, bits, gs, seed int) (packed, dense MoEExpert) {
	t.Helper()
	gateQ, gateW := quantiseSynthetic(t, FF, D, bits, gs, seed+1)
	upQ, upW := quantiseSynthetic(t, FF, D, bits, gs, seed+2)
	downQ, downW := quantiseSynthetic(t, D, FF, bits, gs, seed+3)
	return MoEExpert{GateQ: gateQ, UpQ: upQ, DownQ: downQ}, MoEExpert{Gate: gateW, Up: upW, Down: downW}
}

// TestMoEMLP_Forward_PackedMatchesDequantised is the packed-MoE parity gate (done-gate #1): build a MoEMLP
// whose routed experts AND shared expert are ALL packed (grouped 4-bit; matNTQuant's per-row host dequant,
// since no engine backend is imported here), forward a multi-token input — top-2-of-6 routing, exercising
// different expert subsets per token — then build the SAME MoEMLP over the dequantised twin of every
// expert and forward again. The two must agree to a tight relative tolerance (see
// TestSwigluExpertQuantInto_MatchesDequantised for why it's a tolerance, not bit-identity).
func TestMoEMLP_Forward_PackedMatchesDequantised(t *testing.T) {
	const D, FF, nE, topK, bits, gs = 64, 96, 6, 2, 4, 32
	packedExperts := make([]MoEExpert, nE)
	denseExperts := make([]MoEExpert, nE)
	for e := range nE {
		packedExperts[e], denseExperts[e] = mkMoEExpertQuant(t, D, FF, bits, gs, e*10+1)
	}
	sharedPacked, sharedDense := mkMoEExpertQuant(t, D, FF, bits, gs, 900)
	router := syn(nE*D, 500)

	mQ := &MoEMLP{Router: router, Experts: packedExperts, Shared: &sharedPacked, TopK: topK, NormTopKProb: true}
	mD := &MoEMLP{Router: router, Experts: denseExperts, Shared: &sharedDense, TopK: topK, NormTopKProb: true}

	const L = 5
	x := syn(L*D, 777)
	gotQ := mQ.forward(x, L, D)
	wantD := mD.forward(x, L, D)

	maxRel := relError(t, "MoEMLP.forward(packed)", gotQ, wantD, 1e-4)
	t.Logf("packed MoE forward vs dequantised-dense: %d tokens x %d dims, top-%d of %d experts + shared, max relative error %.3e", L, D, topK, nE, maxRel)
}

// TestLoadComposedMoEQuantised loads a synthetic MoE checkpoint whose expert (routed + shared) tensors
// carry .scales/.biases siblings (quantiseInPlace) and asserts the loader keeps them PACKED — the
// regression guard for the bug this package fixes: buildMoE used to resolve every expert through f32(),
// which dequantises unconditionally, widening a grouped checkpoint's dominant tensor class at load
// regardless of whether it carries quant siblings. It then runs the same packed-forward vs
// dequantise-then-forward parity TestComposedQuantForwardMatchesDequantised uses for the dense
// projections, through the FULL model (embed → layer → head) via dequantiseInPlace's *MoEMLP case.
func TestLoadComposedMoEQuantised(t *testing.T) {
	const D, vocab = 8, 32
	const VH, HD, convDim, K, vDim = 4, 8, 64, 4, 32
	const moeFF, nE, sharedFF = 16, 6, 24
	const bits, gs = 4, 8
	ts := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": bf16T(syn(vocab*D, 1), vocab, D),
		"model.norm.weight":         bf16T(syn(D, 2), D),
		"lm_head.weight":            bf16T(syn(vocab*D, 3), vocab, D),
	}
	lp := "model.layers.0."
	ts[lp+"input_layernorm.weight"] = bf16T(syn(D, 1), D)
	ts[lp+"post_attention_layernorm.weight"] = bf16T(syn(D, 2), D)
	gp := lp + "linear_attn."
	ts[gp+"in_proj_qkv.weight"] = bf16T(syn(convDim*D, 20), convDim, D)
	ts[gp+"conv1d.weight"] = bf16T(syn(convDim*K, 21), convDim, 1, K)
	ts[gp+"conv1d.bias"] = bf16T(syn(convDim, 22), convDim)
	ts[gp+"in_proj_a.weight"] = bf16T(syn(VH*D, 23), VH, D)
	ts[gp+"A_log"] = bf16T(syn(VH, 24), VH)
	ts[gp+"dt_bias"] = bf16T(syn(VH, 25), VH)
	ts[gp+"in_proj_b.weight"] = bf16T(syn(VH*D, 26), VH, D)
	ts[gp+"in_proj_z.weight"] = bf16T(syn(vDim*D, 27), vDim, D)
	ts[gp+"norm.weight"] = bf16T(syn(HD, 28), HD)
	ts[gp+"out_proj.weight"] = bf16T(syn(D*vDim, 29), D, vDim)
	mp := lp + "mlp."
	ts[mp+"gate.weight"] = bf16T(syn(nE*D, 30), nE, D)
	for e := range nE {
		ep := mp + "experts." + itoa(e) + "."
		ts[ep+"gate_proj.weight"] = bf16T(syn(moeFF*D, e*5+40), moeFF, D)
		ts[ep+"up_proj.weight"] = bf16T(syn(moeFF*D, e*5+41), moeFF, D)
		ts[ep+"down_proj.weight"] = bf16T(syn(D*moeFF, e*5+42), D, moeFF)
		quantiseInPlace(t, ts, ep+"gate_proj.weight", bits, gs)
		quantiseInPlace(t, ts, ep+"up_proj.weight", bits, gs)
		quantiseInPlace(t, ts, ep+"down_proj.weight", bits, gs)
	}
	sp := mp + "shared_expert."
	ts[sp+"gate_proj.weight"] = bf16T(syn(sharedFF*D, 90), sharedFF, D)
	ts[sp+"up_proj.weight"] = bf16T(syn(sharedFF*D, 91), sharedFF, D)
	ts[sp+"down_proj.weight"] = bf16T(syn(D*sharedFF, 92), D, sharedFF)
	quantiseInPlace(t, ts, sp+"gate_proj.weight", bits, gs)
	quantiseInPlace(t, ts, sp+"up_proj.weight", bits, gs)
	quantiseInPlace(t, ts, sp+"down_proj.weight", bits, gs)

	config := []byte(`{"hidden_size":8,"num_hidden_layers":1,"intermediate_size":16,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,"num_experts_per_tok":2,"rope_theta":1000000,"partial_rotary_factor":0.5,"layer_types":["linear_attention"],"quantization":{"group_size":8,"bits":4}}`)

	m, err := LoadComposed(ts, config)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	if !m.Quantised {
		t.Fatal("Quantised flag not set")
	}
	moe, ok := m.Layers[0].MLP.(*MoEMLP)
	if !ok {
		t.Fatalf("layer 0 FFN is %T, want *MoEMLP", m.Layers[0].MLP)
	}
	if len(moe.Experts) != nE {
		t.Fatalf("experts = %d, want %d", len(moe.Experts), nE)
	}
	for e := range moe.Experts {
		if moe.Experts[e].GateQ == nil || moe.Experts[e].UpQ == nil || moe.Experts[e].DownQ == nil {
			t.Fatalf("expert %d: GateQ/UpQ/DownQ nil — buildMoE dequantised instead of keeping it packed", e)
		}
		if moe.Experts[e].Gate != nil || moe.Experts[e].Up != nil || moe.Experts[e].Down != nil {
			t.Fatalf("expert %d: dense Gate/Up/Down populated alongside the Q fields — proj must return exactly one representation", e)
		}
	}
	if moe.Shared == nil || moe.Shared.GateQ == nil {
		t.Fatal("shared expert not packed")
	}
	if len(moe.Router) != nE*D {
		t.Fatalf("router len = %d, want %d (the router stays dense f32 — it was never quantised)", len(moe.Router), nE*D)
	}

	tokens := []int32{1, 5, 3, 0, 7}
	sQ := NewSession(m)
	hQ, err := sQ.Forward(tokens)
	if err != nil {
		t.Fatalf("quant Forward: %v", err)
	}

	dequantiseInPlace(t, m)
	sD := NewSession(m)
	hD, err := sD.Forward(tokens)
	if err != nil {
		t.Fatalf("dense Forward: %v", err)
	}

	maxRel := relError(t, "hidden", hQ, hD, 1e-4)
	t.Logf("loaded quantised MoE checkpoint: %d experts packed (bits=%d gs=%d), full-model forward vs dequantised-dense max relative error %.3e", nE, bits, gs, maxRel)
}

// TestMoEExpertPackedVsDequantisedBytes is the memory receipt (done-gate #2): for one synthetic expert at
// a realistic MoE width and a common mlx group size (64), it measures the ACTUAL bytes a packed
// model.QuantWeight occupies (Packed + Scales + Biases) against what the SAME tensor would occupy
// dequantised to f32 (outDim·inDim·4), for all three of an expert's projections (gate/up/down) — the
// tensor class this package used to widen unconditionally at load. The ratio is a measured number, not a
// claim: it comes out below the oft-quoted "~8x" for 4-bit because that figure is the codes-only asymptote
// (group_size → ∞) — a real group size carries a small but non-zero scale+bias tax, included here. nE
// experts scale both totals linearly, so the ratio is expert-count-independent — one expert is
// representative of the whole MoE layer's win.
func TestMoEExpertPackedVsDequantisedBytes(t *testing.T) {
	const D, FF, bits, gs = 512, 1408, 4, 64
	shapes := [][2]int{{FF, D}, {FF, D}, {D, FF}} // gate_proj, up_proj, down_proj
	var packedBytes, dequantBytes int
	for i, sh := range shapes {
		outDim, inDim := sh[0], sh[1]
		w := syn(outDim*inDim, i+1)
		packed, scales, biases, err := mlxaffine.QuantizeTensor(w, outDim, inDim, bits, gs)
		if err != nil {
			t.Fatalf("quantise: %v", err)
		}
		packedBytes += len(packed) + len(scales) + len(biases)
		dequantBytes += outDim * inDim * 4 // f32
	}
	ratio := float64(dequantBytes) / float64(packedBytes)
	// Exact for 4-bit: dequant/packed = 4 / (0.5 + 4/groupSize) — packed codes are bits/8=0.5 bytes/elem;
	// scales+biases are 2 bf16 values (4 bytes) per group of groupSize elements, amortised per element as
	// 4/groupSize. Both D and FF here are exact multiples of gs, so this holds to float64 precision.
	wantRatio := 4.0 / (0.5 + 4.0/float64(gs))
	if math.Abs(ratio-wantRatio) > 0.01*wantRatio {
		t.Fatalf("measured ratio %.4fx too far from the formula's %.4fx (bits=%d gs=%d)", ratio, wantRatio, bits, gs)
	}
	if ratio < 5 {
		t.Fatalf("packed:dequantised ratio %.3fx too small to be the memory win this package exists for (want > 5x for 4-bit)", ratio)
	}
	t.Logf("one MoE expert's gate+up+down (D=%d FF=%d, %d-bit groups of %d): packed %d bytes, dequantised-f32 %d bytes, ratio %.3fx", D, FF, bits, gs, packedBytes, dequantBytes, ratio)
}
