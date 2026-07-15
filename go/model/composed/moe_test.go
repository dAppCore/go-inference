// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"
	"testing"

	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/arch/Qwen/qwen3"
	"dappco.re/go/inference/model/safetensors"
)

func TestMoESigmoidRouterDistributionReceipts(t *testing.T) {
	m := &MoEMLP{Router: []float32{1, 0, 0, 1, -1, -1}, Experts: []MoEExpert{{}, {}, {}}, TopK: 1, Gating: model.MoEGatingSigmoid}
	for _, receipt := range []struct {
		fill []float32
		want int
	}{{[]float32{3, 1}, 0}, {[]float32{1, 4}, 1}, {[]float32{-3, -2}, 2}} {
		probs, idx := make([]float64, 3), make([]int, 3)
		selected, denom := m.routeInto(receipt.fill, 2, probs, idx)
		if len(selected) != 1 || selected[0] != receipt.want {
			t.Fatalf("sigmoid router fill %v selected %v, want expert %d", receipt.fill, selected, receipt.want)
		}
		if denom != 1 || probs[selected[0]] <= 0 || probs[selected[0]] >= 1 {
			t.Fatalf("sigmoid receipt fill %v: score %v denominator %v", receipt.fill, probs[selected[0]], denom)
		}
	}
}

// TestMoEMLP_Forward_Golden pins the exact f32 bit-pattern of MoEMLP.forward over a fixed
// multi-token input with TopK=2 (selection + renormalise exercised), gating scratch-fusion
// refactors on bit-identical output — the mixture reference test above is 1e-4 tolerant.
func TestMoEMLP_Forward_Golden(t *testing.T) {
	const D, FF, nE, L = 8, 12, 6, 3
	m := mkMoEMLP(D, FF, nE, 2, 77)
	out := m.forward(syn(L*D, 1), L, D)
	wantOut := []uint32{0x40a18d25, 0xc0ddb35e, 0xc07b560b, 0xc0429d7a, 0xbf856967, 0x3f8221e8, 0x40810437, 0x40d86fb8, 0x4030bf64, 0xc07fb44e, 0xc00f4752, 0xbfe04404, 0xbf1c2100, 0x3f1306dc, 0x401c40fd, 0x40811fa2, 0x3fa03aa7, 0xbff74426, 0xbf877803, 0xbf58a28e, 0xbe9b0363, 0x3e898cde, 0x3fa28473, 0x4003ed40}
	if len(out) != len(wantOut) {
		t.Fatalf("out len %d, want %d", len(out), len(wantOut))
	}
	for i, v := range out {
		if b := math.Float32bits(v); b != wantOut[i] {
			t.Fatalf("out[%d] bits 0x%08x, want 0x%08x", i, b, wantOut[i])
		}
	}
}

func mkMoEMLP(D, FF, nE, topK, seed int) *MoEMLP {
	experts := make([]MoEExpert, nE)
	for e := range experts {
		experts[e] = MoEExpert{Gate: syn(FF*D, seed+e*10+1), Up: syn(FF*D, seed+e*10+2), Down: syn(D*FF, seed+e*10+3)}
	}
	return &MoEMLP{
		Router:       syn(nE*D, seed),
		Experts:      experts,
		Shared:       &MoEExpert{Gate: syn(FF*D, seed+500), Up: syn(FF*D, seed+501), Down: syn(D*FF, seed+502)},
		TopK:         topK,
		NormTopKProb: true, // the reference default; the golden + reference tests renormalise the selection
	}
}

// TestMoEFullMixture verifies the MoE forward against a reference with TopK = NumExperts (no truncation):
// out = Σ_e softmax(router·x)_e · SwiGLU_e(x) + SwiGLU_shared(x).
func TestMoEFullMixture(t *testing.T) {
	const D, FF, nE = 8, 12, 4
	m := mkMoEMLP(D, FF, nE, nE, 1)
	x := syn(D, 99)
	got := m.forward(x, 1, D)

	logits := make([]float64, nE)
	maxL := math.Inf(-1)
	for e := range nE {
		var acc float64
		for d := range D {
			acc += float64(x[d]) * float64(m.Router[e*D+d])
		}
		logits[e] = acc
		if acc > maxL {
			maxL = acc
		}
	}
	var sum float64
	for e := range logits {
		logits[e] = math.Exp(logits[e] - maxL)
		sum += logits[e]
	}
	want := make([]float64, D)
	for e := range nE {
		w := logits[e] / sum
		eo := swigluExpert(x, m.Experts[e], D)
		for d := range D {
			want[d] += w * float64(eo[d])
		}
	}
	so := swigluExpert(x, *m.Shared, D)
	for d := range D {
		want[d] += float64(so[d])
	}
	for d := range D {
		if math.Abs(float64(got[d])-want[d]) > 1e-4*(1+math.Abs(want[d])) {
			t.Errorf("out[%d] = %v, want %v (full softmax mixture + shared)", d, got[d], want[d])
		}
	}
	t.Log("MoE forward matches the reference: Σ softmax·SwiGLU_expert + SwiGLU_shared")
}

// TestTopKIndices pins the router's expert selection directly: the returned indices are
// exactly the k largest values (asserted as a set — the doc contract promises which experts,
// not an order), and k > len(v) clamps to the whole set. The mixture tests can't isolate
// this: a selection bug there is blended through softmax weights and expert outputs.
func TestTopKIndices(t *testing.T) {
	v := []float64{0.1, 0.9, 0.3, 0.7, 0.5}
	asSet := func(idx []int) map[int]bool {
		s := map[int]bool{}
		for _, i := range idx {
			s[i] = true
		}
		return s
	}
	got := asSet(topKIndices(v, 2))
	if len(got) != 2 || !got[1] || !got[3] {
		t.Fatalf("topKIndices(k=2) selected %v, want {1,3} (values 0.9, 0.7)", got)
	}
	got = asSet(topKIndices(v, 4))
	if len(got) != 4 || got[0] {
		t.Fatalf("topKIndices(k=4) selected %v, want everything but index 0 (the smallest, 0.1)", got)
	}
	if got := topKIndices(v, 10); len(got) != len(v) {
		t.Fatalf("topKIndices(k>n) returned %d indices, want the clamp to %d", len(got), len(v))
	}
}

// TestMoETruncatedMixture verifies the TopK < NumExperts path against an INDEPENDENT
// reference: the top-2-of-4 experts are found here by direct pairwise max-tracking (not
// topKIndices), their softmax weights renormalised over the pair, and only those two summed
// (+ shared). TestMoEFullMixture can't catch a truncation bug (TopK == NumExperts excludes
// nothing) and decode==prefill can't either (both paths would share it).
func TestMoETruncatedMixture(t *testing.T) {
	const D, FF, nE, topK = 8, 12, 4, 2
	m := mkMoEMLP(D, FF, nE, topK, 1)
	x := syn(D, 99)
	got := m.forward(x, 1, D)

	// independent router logits.
	logits := make([]float64, nE)
	for e := range nE {
		var acc float64
		for d := range D {
			acc += float64(x[d]) * float64(m.Router[e*D+d])
		}
		logits[e] = acc
	}
	// independent top-2: direct max-tracking, no shared selection code.
	best, second := 0, -1
	for e := 1; e < nE; e++ {
		switch {
		case logits[e] > logits[best]:
			second, best = best, e
		case second < 0 || logits[e] > logits[second]:
			second = e
		}
	}
	// precondition: the seeds must separate the 2nd- and 3rd-ranked experts, else a
	// wrong-selection bug could hide inside the tolerance.
	third := -1
	for e := range nE {
		if e != best && e != second && (third < 0 || logits[e] > logits[third]) {
			third = e
		}
	}
	if logits[second]-logits[third] < 1e-3 {
		t.Fatalf("test precondition: router logits %v too close between rank-2 and rank-3 — pick a different seed", logits)
	}

	maxL := logits[best]
	wb := math.Exp(logits[best] - maxL)
	ws := math.Exp(logits[second] - maxL)
	sum := wb + ws
	want := make([]float64, D)
	for _, sel := range []struct {
		e int
		w float64
	}{{best, wb / sum}, {second, ws / sum}} {
		eo := swigluExpert(x, m.Experts[sel.e], D)
		for d := range D {
			want[d] += sel.w * float64(eo[d])
		}
	}
	so := swigluExpert(x, *m.Shared, D)
	for d := range D {
		want[d] += float64(so[d])
	}
	for d := range D {
		if math.Abs(float64(got[d])-want[d]) > 1e-4*(1+math.Abs(want[d])) {
			t.Errorf("out[%d] = %v, want %v (top-2-of-4 renormalised mixture + shared)", d, got[d], want[d])
		}
	}
	t.Logf("MoE truncation verified: experts {%d,%d} of %d selected + renormalised, others excluded", best, second, nE)
}

// TestMoENormTopKProbFalse pins the norm_topk_prob=false routing against an INDEPENDENT reference: the
// top-k experts are combined by their RAW full-softmax weights (denominator over ALL experts), not
// renormalised over the selection. It also guards that the flag actually changes the output vs the
// renormalised path — a silent no-op would pass a self-referential test.
func TestMoENormTopKProbFalse(t *testing.T) {
	const D, FF, nE, topK = 8, 12, 5, 2
	experts := make([]MoEExpert, nE)
	for e := range experts {
		experts[e] = MoEExpert{Gate: syn(FF*D, 200+e*10+1), Up: syn(FF*D, 200+e*10+2), Down: syn(D*FF, 200+e*10+3)}
	}
	m := &MoEMLP{Router: syn(nE*D, 200), Experts: experts, TopK: topK, NormTopKProb: false}
	x := syn(D, 99)
	got := m.forward(x, 1, D)

	// independent reference: full softmax over ALL experts, top-k selected, summed WITHOUT renormalisation.
	logits := make([]float64, nE)
	maxL := math.Inf(-1)
	for e := range nE {
		var acc float64
		for d := range D {
			acc += float64(x[d]) * float64(m.Router[e*D+d])
		}
		logits[e] = acc
		if acc > maxL {
			maxL = acc
		}
	}
	probs := make([]float64, nE)
	var sumAll float64
	for e := range nE {
		probs[e] = math.Exp(logits[e] - maxL)
		sumAll += probs[e]
	}
	used := make([]bool, nE)
	sel := make([]int, 0, topK)
	for range topK {
		best := -1
		for e := range nE {
			if !used[e] && (best < 0 || probs[e] > probs[best]) {
				best = e
			}
		}
		used[best] = true
		sel = append(sel, best)
	}
	want := make([]float64, D)
	for _, e := range sel {
		w := probs[e] / sumAll // full-softmax weight — no top-k renormalisation
		eo := swigluExpert(x, m.Experts[e], D)
		for d := range D {
			want[d] += w * float64(eo[d])
		}
	}
	for d := range D {
		if math.Abs(float64(got[d])-want[d]) > 1e-4*(1+math.Abs(want[d])) {
			t.Errorf("out[%d] = %v, want %v (full-softmax weights, no top-k renorm)", d, got[d], want[d])
		}
	}
	mNorm := *m
	mNorm.NormTopKProb = true
	gotNorm := mNorm.forward(x, 1, D)
	for d := range D {
		if gotNorm[d] != got[d] {
			t.Logf("norm_topk_prob flips behaviour: renormalised[0]=%v raw[0]=%v", gotNorm[0], got[0])
			return
		}
	}
	t.Fatal("norm_topk_prob false vs true produced identical output — the flag is not wired")
}

// TestMoEMLP_OLMoERouterDistribution proves the production router used by
// MoEMLP.forward selects varied expert sets for signed, seeded OLMoE-style inputs.
func TestMoEMLP_OLMoERouterDistribution(t *testing.T) {
	const D, FF, nE, topK, tokens = 8, 12, 4, 2, 8
	m := mkMoEMLP(D, FF, nE, topK, 0x01e0e)
	m.Shared = nil
	m.SharedGate = nil
	m.NormTopKProb = false
	x := syn(tokens*D, 0x07e57)
	probs := make([]float64, nE)
	idx := make([]int, nE)
	distributions := map[[2]int]bool{}
	for token := range tokens {
		sel, _ := m.routeInto(x[token*D:(token+1)*D], D, probs, idx)
		distributions[[2]int{sel[0], sel[1]}] = true
	}
	if len(distributions) < 2 {
		t.Fatalf("production router selected %d distinct top-k distributions, want at least 2", len(distributions))
	}
	out := m.forward(x, tokens, D)
	if len(out) != tokens*D {
		t.Fatalf("MoE forward output length = %d, want %d", len(out), tokens*D)
	}
}

// TestMoESharedExpertGate pins the reference's sigmoid gate on the shared expert (σ(shared_expert_gate·x)):
// the gated and ungated variants differ by exactly (σ−1)·SwiGLU_shared(x), isolating the gate from the
// (identical) expert mixture. A precondition ensures σ is meaningfully below 1 so the gate is observable.
func TestMoESharedExpertGate(t *testing.T) {
	const D, FF, nE, sharedFF, topK = 8, 12, 4, 10, 2
	m := mkMoEMLP(D, FF, nE, topK, 33)
	m.Shared = &MoEExpert{Gate: syn(sharedFF*D, 700), Up: syn(sharedFF*D, 701), Down: syn(D*sharedFF, 702)}
	m.SharedGate = syn(D, 750)
	x := syn(D, 99)
	got := m.forward(x, 1, D)

	var dot float64
	for d := range D {
		dot += float64(x[d]) * float64(m.SharedGate[d])
	}
	g := 1.0 / (1.0 + math.Exp(-dot))
	if g >= 0.999 {
		t.Fatalf("test precondition: gate σ=%v too close to 1 to observe — pick a different seed", g)
	}
	so := swigluExpert(x, *m.Shared, D)
	mNo := *m
	mNo.SharedGate = nil
	gotUngated := mNo.forward(x, 1, D)
	for d := range D {
		delta := float64(got[d]) - float64(gotUngated[d]) // (σ−1)·SwiGLU_shared, the mixture cancels
		want := (g - 1) * float64(so[d])
		if math.Abs(delta-want) > 1e-4*(1+math.Abs(want)) {
			t.Errorf("shared-gate delta[%d] = %v, want %v (=(σ−1)·SwiGLU_shared)", d, delta, want)
		}
	}
	t.Logf("shared-expert sigmoid gate verified: σ(shared_expert_gate·x)=%v scales the shared contribution", g)
}

// TestLoadComposedMoEGatedShared loads a synthetic MoE checkpoint carrying mlp.shared_expert_gate.weight
// and a config with norm_topk_prob:false, and asserts BOTH flow into the built MoEMLP — the shared gate is
// bound and the routing flag is config-driven, not assumed.
func TestLoadComposedMoEGatedShared(t *testing.T) {
	const D, vocab = 8, 32
	const VH, HD, convDim, K, vDim = 4, 8, 64, 4, 32
	const moeFF, nE, sharedFF = 10, 6, 12
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
	}
	sp := mp + "shared_expert."
	ts[sp+"gate_proj.weight"] = bf16T(syn(sharedFF*D, 90), sharedFF, D)
	ts[sp+"up_proj.weight"] = bf16T(syn(sharedFF*D, 91), sharedFF, D)
	ts[sp+"down_proj.weight"] = bf16T(syn(D*sharedFF, 92), D, sharedFF)
	ts[mp+"shared_expert_gate.weight"] = bf16T(syn(D, 93), 1, D)

	config := []byte(`{"hidden_size":8,"num_hidden_layers":1,"intermediate_size":10,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,"num_experts_per_tok":2,"norm_topk_prob":false,"rope_theta":1000000,"partial_rotary_factor":0.5,"layer_types":["linear_attention"]}`)
	m, err := LoadComposed(ts, config)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	moe, ok := m.Layers[0].MLP.(*MoEMLP)
	if !ok {
		t.Fatalf("layer 0 FFN is %T, want *MoEMLP", m.Layers[0].MLP)
	}
	if len(moe.SharedGate) != D {
		t.Fatalf("SharedGate len = %d, want %d (shared_expert_gate.weight must bind)", len(moe.SharedGate), D)
	}
	if moe.NormTopKProb {
		t.Fatal("NormTopKProb = true, want false (config norm_topk_prob:false must flow through)")
	}
	if _, err := NewSession(m).Forward([]int32{1, 2, 3}); err != nil {
		t.Fatalf("forward: %v", err)
	}
	t.Log("MoE checkpoint with shared_expert_gate + norm_topk_prob:false loaded — both flow into MoEMLP")
}

// TestComposedMoEDecodeEqualsPrefill checks the orchestration with MoE FFN layers decodes bit-exact to
// prefill (the MoE is per-token stateless; the mixer state threads).
func TestComposedMoEDecodeEqualsPrefill(t *testing.T) {
	const D, vocab = 8, 32
	gd := qwen3.GatedDeltaConfig{KeyHeads: 2, ValueHeads: 4, HeadDim: 8, ConvKernel: 4, Eps: 1e-5}
	mk := func(li int) Layer {
		return Layer{InputNorm: syn(D, li*13+1), Mixer: mkGatedDeltaMixer(gd, D, li*13+20), PostAttnNorm: syn(D, li*13+2), MLP: mkMoEMLP(D, 12, 6, 2, li*13+100)}
	}
	m := &ComposedModel{Embed: syn(vocab*D, 100), NormF: syn(D, 101), D: D, Vocab: vocab, Eps: 1e-5, Layers: []Layer{mk(0), mk(1)}}
	tokens := []int32{1, 5, 9, 2, 7}
	prefill, err := NewSession(m).Forward(tokens)
	if err != nil {
		t.Fatalf("prefill: %v", err)
	}
	dec := NewSession(m)
	for t0, tok := range tokens {
		h, _ := dec.Forward([]int32{tok})
		for i := range D {
			if h[i] != prefill[t0*D+i] {
				t.Fatalf("token %d hidden[%d] = %v != prefill %v (MoE decode diverged)", t0, i, h[i], prefill[t0*D+i])
			}
		}
	}
	t.Log("composed decode == prefill bit-exact with MoE FFN layers")
}

// TestLoadComposedMoE loads a synthetic checkpoint whose MLPs are MoE (mlp.gate + experts + shared) and
// confirms the loader builds *MoEMLP FFNs and the model decodes.
func TestLoadComposedMoE(t *testing.T) {
	const D, vocab, nLayers = 8, 32, 2
	const VH, HD, convDim, K, vDim = 4, 8, 64, 4, 32
	const moeFF, nE, sharedFF = 10, 6, 12
	ts := map[string]safetensors.Tensor{
		"model.embed_tokens.weight": bf16T(syn(vocab*D, 1), vocab, D),
		"model.norm.weight":         bf16T(syn(D, 2), D),
		"lm_head.weight":            bf16T(syn(vocab*D, 3), vocab, D),
	}
	for i := range nLayers {
		lp := "model.layers." + itoa(i) + "."
		ts[lp+"input_layernorm.weight"] = bf16T(syn(D, i*200+1), D)
		ts[lp+"post_attention_layernorm.weight"] = bf16T(syn(D, i*200+2), D)
		// gated-delta mixer (all linear)
		gp := lp + "linear_attn."
		ts[gp+"in_proj_qkv.weight"] = bf16T(syn(convDim*D, i*200+20), convDim, D)
		ts[gp+"conv1d.weight"] = bf16T(syn(convDim*K, i*200+21), convDim, 1, K)
		ts[gp+"conv1d.bias"] = bf16T(syn(convDim, i*200+22), convDim)
		ts[gp+"in_proj_a.weight"] = bf16T(syn(VH*D, i*200+23), VH, D)
		ts[gp+"A_log"] = bf16T(syn(VH, i*200+24), VH)
		ts[gp+"dt_bias"] = bf16T(syn(VH, i*200+25), VH)
		ts[gp+"in_proj_b.weight"] = bf16T(syn(VH*D, i*200+26), VH, D)
		ts[gp+"in_proj_z.weight"] = bf16T(syn(vDim*D, i*200+27), vDim, D)
		ts[gp+"norm.weight"] = bf16T(syn(HD, i*200+28), HD)
		ts[gp+"out_proj.weight"] = bf16T(syn(D*vDim, i*200+29), D, vDim)
		// MoE MLP
		mp := lp + "mlp."
		ts[mp+"gate.weight"] = bf16T(syn(nE*D, i*200+30), nE, D)
		for e := range nE {
			ep := mp + "experts." + itoa(e) + "."
			ts[ep+"gate_proj.weight"] = bf16T(syn(moeFF*D, i*200+e*5+40), moeFF, D)
			ts[ep+"up_proj.weight"] = bf16T(syn(moeFF*D, i*200+e*5+41), moeFF, D)
			ts[ep+"down_proj.weight"] = bf16T(syn(D*moeFF, i*200+e*5+42), D, moeFF)
		}
		sp := mp + "shared_expert."
		ts[sp+"gate_proj.weight"] = bf16T(syn(sharedFF*D, i*200+90), sharedFF, D)
		ts[sp+"up_proj.weight"] = bf16T(syn(sharedFF*D, i*200+91), sharedFF, D)
		ts[sp+"down_proj.weight"] = bf16T(syn(D*sharedFF, i*200+92), D, sharedFF)
	}
	config := []byte(`{"hidden_size":8,"num_hidden_layers":2,"intermediate_size":10,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,"vocab_size":32,"rms_norm_eps":1e-5,"num_experts_per_tok":2,"rope_theta":1000000,"partial_rotary_factor":0.5,"full_attention_interval":0,"layer_types":["linear_attention","linear_attention"]}`)

	m, err := LoadComposed(ts, config)
	if err != nil {
		t.Fatalf("LoadComposed: %v", err)
	}
	for i, l := range m.Layers {
		moe, ok := l.MLP.(*MoEMLP)
		if !ok {
			t.Fatalf("layer %d FFN is %T, want *MoEMLP", i, l.MLP)
		}
		if len(moe.Experts) != nE || moe.Shared == nil || moe.TopK != 2 {
			t.Fatalf("layer %d MoE wrong: experts=%d shared=%v topK=%d", i, len(moe.Experts), moe.Shared != nil, moe.TopK)
		}
	}
	gen, err := NewSession(m).Generate([]int32{1, 2, 3}, 3, -1)
	if err != nil {
		t.Fatalf("generate: %v", err)
	}
	t.Logf("loaded MoE checkpoint: %d layers × %d experts (top-2) + shared, decodes → %v", nLayers, nE, gen)
}
