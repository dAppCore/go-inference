// SPDX-Licence-Identifier: EUPL-1.2

package composed

import (
	"math"
	"testing"

	"dappco.re/go/inference/model/qwen3"
	"dappco.re/go/inference/safetensors"
)

func mkMoEMLP(D, FF, nE, topK, seed int) *MoEMLP {
	experts := make([]MoEExpert, nE)
	for e := range experts {
		experts[e] = MoEExpert{Gate: syn(FF*D, seed+e*10+1), Up: syn(FF*D, seed+e*10+2), Down: syn(D*FF, seed+e*10+3)}
	}
	return &MoEMLP{
		Router:  syn(nE*D, seed),
		Experts: experts,
		Shared:  &MoEExpert{Gate: syn(FF*D, seed+500), Up: syn(FF*D, seed+501), Down: syn(D*FF, seed+502)},
		TopK:    topK,
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
	for e := 0; e < nE; e++ {
		var acc float64
		for d := 0; d < D; d++ {
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
	for e := 0; e < nE; e++ {
		w := logits[e] / sum
		eo := swigluExpert(x, m.Experts[e], D)
		for d := 0; d < D; d++ {
			want[d] += w * float64(eo[d])
		}
	}
	so := swigluExpert(x, *m.Shared, D)
	for d := 0; d < D; d++ {
		want[d] += float64(so[d])
	}
	for d := 0; d < D; d++ {
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
	for e := 0; e < nE; e++ {
		var acc float64
		for d := 0; d < D; d++ {
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
	for e := 0; e < nE; e++ {
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
		for d := 0; d < D; d++ {
			want[d] += sel.w * float64(eo[d])
		}
	}
	so := swigluExpert(x, *m.Shared, D)
	for d := 0; d < D; d++ {
		want[d] += float64(so[d])
	}
	for d := 0; d < D; d++ {
		if math.Abs(float64(got[d])-want[d]) > 1e-4*(1+math.Abs(want[d])) {
			t.Errorf("out[%d] = %v, want %v (top-2-of-4 renormalised mixture + shared)", d, got[d], want[d])
		}
	}
	t.Logf("MoE truncation verified: experts {%d,%d} of %d selected + renormalised, others excluded", best, second, nE)
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
		for i := 0; i < D; i++ {
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
	for i := 0; i < nLayers; i++ {
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
		for e := 0; e < nE; e++ {
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
