// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"math"
	"reflect"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestConfigArchDense fills an Arch from a dense (non-MoE) config and checks every
// neutral dim, the gemma4-specifics, and that the per-layer specs equal model.DeriveLayers
// with no MoE flag set.
func TestConfigArchDense(t *testing.T) {
	c := Config{
		HiddenSize: 256, NumHiddenLayers: 4, IntermediateSize: 512,
		NumAttentionHeads: 8, NumKeyValueHeads: 2, HeadDim: 64,
		VocabSize: 1000, RMSNormEps: 1e-5, RopeTheta: 10000,
		FinalLogitSoftcapping: 30, SlidingWindow: 128, NumKVSharedLayers: 1,
		LayerTypes:             []string{"full_attention", "sliding_attention", "full_attention", "sliding_attention"},
		VocabSizePerLayerInput: 500, HiddenSizePerLayerInput: 64, AttentionKEqV: true,
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	wantLayers := model.DeriveLayers(c.LayerTypes, 1)
	for i := range wantLayers {
		wantLayers[i].HeadDim, wantLayers[i].KVHeads = 64, 2 // uniform: no global_head_dim distinction
	}
	want := model.Arch{
		Hidden: 256, Heads: 8, KVHeads: 2, HeadDim: 64, GlobalHeadDim: 64, GlobalKVHeads: 2, FF: 512, Vocab: 1000,
		Experts: 0, TopK: 0, ExpertFF: 0,
		Eps: 1e-5, AttnScale: 1, EmbedScale: 16, RopeBase: 10000, RopeLocalBase: defaultRopeLocalTheta, RotaryDim: 64, RotaryDimLocal: 64, RopeScale: 1, SoftCap: 30, SlidingWindow: 128,
		PerLayerInputVocab: 500, PerLayerInputHidden: 64, AttentionKEqV: true, ValueNorm: true,
		Layer: wantLayers,
	}
	if !reflect.DeepEqual(a, want) {
		t.Fatalf("dense Arch mismatch:\n got %+v\nwant %+v", a, want)
	}
	for i, l := range a.Layer {
		if l.MoE {
			t.Fatalf("layer %d marked MoE in a dense config", i)
		}
	}
	t.Logf("dense Arch: all dims filled, %d layer specs ≡ model.DeriveLayers, no MoE", len(a.Layer))
}

// TestConfigArchMoE fills an Arch from a MoE config and checks the MoE dims plus that
// EVERY layer is marked MoE (gemma4 applies MoE uniformly, not interleaved).
func TestConfigArchMoE(t *testing.T) {
	c := Config{
		HiddenSize: 512, NumHiddenLayers: 3, IntermediateSize: 1024,
		NumAttentionHeads: 8, NumKeyValueHeads: 4, HeadDim: 64, VocabSize: 2000,
		LayerTypes:     []string{"full_attention", "full_attention", "sliding_attention"},
		EnableMoEBlock: true, NumExperts: 16, TopKExperts: 4, MoEIntermediateSize: 384,
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.Experts != 16 || a.TopK != 4 || a.ExpertFF != 384 {
		t.Fatalf("MoE dims: got Experts=%d TopK=%d ExpertFF=%d, want 16/4/384", a.Experts, a.TopK, a.ExpertFF)
	}
	if a.MoEGating != model.MoEGatingSoftmax {
		t.Fatalf("MoE gating: got %q, want the inferred %q", a.MoEGating, model.MoEGatingSoftmax)
	}
	wantLayers := model.DeriveLayers(c.LayerTypes, 0)
	for i := range wantLayers {
		wantLayers[i].MoE = true
		wantLayers[i].HeadDim, wantLayers[i].KVHeads = 64, 4 // uniform: no global_head_dim
	}
	if !reflect.DeepEqual(a.Layer, wantLayers) {
		t.Fatalf("MoE layer specs mismatch:\n got %+v\nwant %+v", a.Layer, wantLayers)
	}
	t.Logf("MoE Arch: Experts=%d TopK=%d ExpertFF=%d, all %d layers MoE", a.Experts, a.TopK, a.ExpertFF, len(a.Layer))
}

// TestConfigArchPerTypeHeadDim gates the real gemma4 geometry the uniform synthetic
// configs structurally couldn't reach: sliding layers use head_dim (256), full_attention
// layers use global_head_dim (512), and the full-attention rotaryDim is a fraction of
// GlobalHeadDim (512·0.25=128) not HeadDim. This is the e2b/12b/31b/26b shape — the gap
// that rejected real packs at the assembler.
func TestConfigArchPerTypeHeadDim(t *testing.T) {
	c := Config{
		HiddenSize: 1536, NumHiddenLayers: 4, IntermediateSize: 9216,
		NumAttentionHeads: 8, NumKeyValueHeads: 1, HeadDim: 256, GlobalHeadDim: 512,
		VocabSize: 1000, RMSNormEps: 1e-6, SlidingWindow: 512,
		LayerTypes: []string{"sliding_attention", "sliding_attention", "full_attention", "sliding_attention"},
		RopeParameters: map[string]RopeParam{
			"full_attention":    {RopeTheta: 1_000_000, PartialRotaryFactor: 0.25, RopeType: "proportional", Factor: 1},
			"sliding_attention": {RopeTheta: 10_000, PartialRotaryFactor: 1.0, RopeType: "default", Factor: 1},
		},
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.HeadDim != 256 || a.GlobalHeadDim != 512 {
		t.Fatalf("Arch HeadDim/GlobalHeadDim = %d/%d, want 256/512", a.HeadDim, a.GlobalHeadDim)
	}
	if a.MaxHeadDim() != 512 {
		t.Fatalf("MaxHeadDim = %d, want 512 (buffers must size to the larger head)", a.MaxHeadDim())
	}
	// full_attention rotaryDim from GlobalHeadDim (512·0.25=128); sliding from HeadDim (256·1.0).
	if a.RotaryDim != 128 || a.RotaryDimLocal != 256 {
		t.Fatalf("RotaryDim/Local = %d/%d, want 128/256", a.RotaryDim, a.RotaryDimLocal)
	}
	for i, l := range a.Layer {
		wantHD := 256
		if l.Attention == model.GlobalAttention {
			wantHD = 512
		}
		if l.HeadDim != wantHD {
			t.Fatalf("layer %d (attn=%d) HeadDim = %d, want %d", i, l.Attention, l.HeadDim, wantHD)
		}
		if l.KVHeads != 1 {
			t.Fatalf("layer %d KVHeads = %d, want 1", i, l.KVHeads)
		}
	}
	t.Logf("per-type head_dim resolved: sliding 256 / full(global) 512, rotaryDim 128/256 — the real gemma4 geometry")
}

// TestConfigArchDefaults checks the omitted-field defaults: head_dim ← hidden/heads,
// num_key_value_heads ← num_attention_heads, eps/rope ← gemma4 defaults, and absent
// layer_types ← all full_attention.
func TestConfigArchDefaults(t *testing.T) {
	c := Config{HiddenSize: 512, NumHiddenLayers: 2, IntermediateSize: 1024, NumAttentionHeads: 8, VocabSize: 100}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.HeadDim != 64 {
		t.Fatalf("HeadDim default: got %d, want 64 (512/8)", a.HeadDim)
	}
	if a.KVHeads != 8 {
		t.Fatalf("KVHeads default: got %d, want 8 (= heads)", a.KVHeads)
	}
	if a.Eps != defaultRMSNormEps || a.RopeBase != defaultRopeTheta || a.RopeScale != 1 {
		t.Fatalf("defaults: eps=%v rope=%v scale=%v", a.Eps, a.RopeBase, a.RopeScale)
	}
	if len(a.Layer) != 2 || a.Layer[0].Attention != model.GlobalAttention || a.Layer[1].Attention != model.GlobalAttention {
		t.Fatalf("absent layer_types should default to 2 global layers, got %+v", a.Layer)
	}
	t.Logf("defaults: HeadDim %d, KVHeads %d, eps %v, rope %v, %d global layers", a.HeadDim, a.KVHeads, a.Eps, a.RopeBase, len(a.Layer))
}

// TestConfigUnmarshal proves the json tags: a config.json-shaped document unmarshals
// (via core.JSONUnmarshal, the loader's path) into Config and fills the Arch.
func TestConfigUnmarshal(t *testing.T) {
	js := `{
		"hidden_size": 640, "num_hidden_layers": 2, "intermediate_size": 2048,
		"num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 256,
		"vocab_size": 262144, "rms_norm_eps": 1e-6, "rope_theta": 1000000,
		"sliding_window": 512, "num_kv_shared_layers": 1,
		"layer_types": ["sliding_attention", "full_attention"],
		"hidden_size_per_layer_input": 256, "vocab_size_per_layer_input": 262144,
		"enable_moe_block": true, "num_experts": 8, "top_k_experts": 2, "moe_intermediate_size": 1024
	}`
	var c Config
	if r := core.JSONUnmarshal([]byte(js), &c); !r.OK {
		t.Fatalf("JSONUnmarshal failed")
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.Hidden != 640 || a.Heads != 4 || a.KVHeads != 1 || a.HeadDim != 256 || a.FF != 2048 || a.Vocab != 262144 {
		t.Fatalf("unmarshalled dims wrong: %+v", a)
	}
	if a.Experts != 8 || a.TopK != 2 || a.ExpertFF != 1024 || !a.Layer[0].MoE {
		t.Fatalf("unmarshalled MoE wrong: Experts=%d TopK=%d ExpertFF=%d MoE0=%v", a.Experts, a.TopK, a.ExpertFF, a.Layer[0].MoE)
	}
	if a.SlidingWindow != 512 || a.PerLayerInputHidden != 256 || a.Layer[0].Attention != model.SlidingAttention {
		t.Fatalf("unmarshalled gemma4-specifics wrong: %+v", a)
	}
	t.Logf("json → Config → Arch: hidden %d, %d layers, MoE %dx top-%d, sliding %d", a.Hidden, len(a.Layer), a.Experts, a.TopK, a.SlidingWindow)
}

// TestConfigTextConfigWrapper gates the multimodal-wrapper nesting real gemma4 packs use: the
// text arch lives under text_config with quantization at the top level. Arch() must derive the
// same arch as the equivalent flat config, and ResolvedQuant must return the top-level quant.
func TestConfigTextConfigWrapper(t *testing.T) {
	flat := Config{
		HiddenSize: 256, NumHiddenLayers: 4, IntermediateSize: 512,
		NumAttentionHeads: 8, NumKeyValueHeads: 2, HeadDim: 64, VocabSize: 1000, RMSNormEps: 1e-5,
		SlidingWindow: 128, NumKVSharedLayers: 1,
		LayerTypes:     []string{"full_attention", "sliding_attention", "full_attention", "sliding_attention"},
		RopeParameters: map[string]RopeParam{"full_attention": {RopeTheta: 1000000, PartialRotaryFactor: 0.25, RopeType: "proportional"}},
	}
	wrapped := Config{TextConfig: &flat, Quantization: &model.QuantConfig{GroupSize: 64, Bits: 4}}

	fa, err := flat.Arch()
	if err != nil {
		t.Fatalf("flat Arch: %v", err)
	}
	wa, err := wrapped.Arch()
	if err != nil {
		t.Fatalf("wrapped Arch: %v", err)
	}
	if !reflect.DeepEqual(fa, wa) {
		t.Fatalf("wrapped Arch != flat Arch:\n got %+v\nwant %+v", wa, fa)
	}
	if q := wrapped.ResolvedQuant(); q == nil || q.GroupSize != 64 || q.Bits != 4 {
		t.Fatalf("ResolvedQuant should return the top-level quant, got %+v", q)
	}
	// json path: a nested document unmarshals + resolves (text_config arch, top-level quant).
	js := `{"model_type":"gemma4_text","quantization":{"group_size":64,"bits":4},
		"text_config":{"hidden_size":128,"num_hidden_layers":2,"num_attention_heads":2,"head_dim":64,"vocab_size":99}}`
	var c Config
	if r := core.JSONUnmarshal([]byte(js), &c); !r.OK {
		t.Fatal("nested config did not unmarshal")
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("nested Arch: %v", err)
	}
	if a.Hidden != 128 || a.Vocab != 99 || len(a.Layer) != 2 {
		t.Fatalf("nested arch came out wrong: %+v", a)
	}
	if q := c.ResolvedQuant(); q == nil || q.GroupSize != 64 {
		t.Fatalf("nested ResolvedQuant wrong: %+v", q)
	}
	t.Logf("text_config wrapper: nested arch ≡ flat arch, quantization resolved from the top level")
}

// TestConfigQuantOverrides gates mixed-precision quant parsing (gemma4 26B-A4B QAT): the
// scalar group_size/bits are the default, object-valued keys are per-module overrides (their
// language_model. prefix stripped), and "mode" (a scalar) is not an override.
func TestConfigQuantOverrides(t *testing.T) {
	js := `{"quantization":{"group_size":64,"bits":4,"mode":"affine",
		"language_model.model.layers.0.mlp.gate_proj":{"group_size":64,"bits":8},
		"language_model.model.layers.0.router.proj":{"group_size":32,"bits":8}},
		"text_config":{"hidden_size":128,"num_hidden_layers":2,"num_attention_heads":2,"head_dim":64,"vocab_size":99}}`
	var c Config
	if r := core.JSONUnmarshal([]byte(js), &c); !r.OK {
		t.Fatal("config did not unmarshal")
	}
	q := c.ResolvedQuant()
	if q == nil || q.GroupSize != 64 || q.Bits != 4 {
		t.Fatalf("default quant wrong: %+v", q)
	}
	if gs, b := q.For("model.layers.0.mlp.gate_proj"); gs != 64 || b != 8 {
		t.Fatalf("mlp override: gs%d b%d, want 64/8", gs, b)
	}
	if gs, b := q.For("model.layers.0.router.proj"); gs != 32 || b != 8 {
		t.Fatalf("router override: gs%d b%d, want 32/8", gs, b)
	}
	if gs, b := q.For("model.layers.0.self_attn.q_proj"); gs != 64 || b != 4 {
		t.Fatalf("default For: gs%d b%d, want 64/4", gs, b)
	}
	if _, ok := q.Overrides["mode"]; ok {
		t.Fatal(`"mode" should not be a module override`)
	}
	t.Logf("mixed-precision quant: default 64/4, mlp/router 8-bit overrides (prefix stripped), mode ignored")
}

// TestConfigArchErrors checks the load-bearing validations reject malformed configs.
func TestConfigArchErrors(t *testing.T) {
	cases := []struct {
		name string
		c    Config
	}{
		{"zero hidden", Config{HiddenSize: 0, NumHiddenLayers: 2, NumAttentionHeads: 8}},
		{"heads not multiple of kv", Config{HiddenSize: 256, NumHiddenLayers: 2, NumAttentionHeads: 8, NumKeyValueHeads: 3, HeadDim: 32}},
		{"layer_types length mismatch", Config{HiddenSize: 256, NumHiddenLayers: 4, NumAttentionHeads: 8, HeadDim: 32, LayerTypes: []string{"full_attention", "full_attention", "full_attention"}}},
		{"moe without experts", Config{HiddenSize: 256, NumHiddenLayers: 2, NumAttentionHeads: 8, HeadDim: 32, EnableMoEBlock: true}},
		{"topK exceeds experts", Config{HiddenSize: 256, NumHiddenLayers: 2, NumAttentionHeads: 8, HeadDim: 32, EnableMoEBlock: true, NumExperts: 4, TopKExperts: 8}},
		{"head_dim absent, indivisible", Config{HiddenSize: 100, NumHiddenLayers: 2, NumAttentionHeads: 8}},
	}
	for _, tc := range cases {
		if _, err := tc.c.Arch(); err == nil {
			t.Fatalf("%s: expected an error, got nil", tc.name)
		}
	}
	t.Logf("validation: all %d malformed configs rejected", len(cases))
}

// TestConfigRope checks per-attention-type RoPE: defaults (global 1e6 / sliding 1e4),
// top-level rope_theta sets the global, and rope_parameters overrides both.
func TestConfigRope(t *testing.T) {
	base := Config{HiddenSize: 128, NumHiddenLayers: 1, NumAttentionHeads: 2, HeadDim: 64, VocabSize: 10}
	a, err := base.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.RopeBase != defaultRopeTheta || a.RopeLocalBase != defaultRopeLocalTheta {
		t.Fatalf("defaults: RopeBase %v (want %v), RopeLocalBase %v (want %v)", a.RopeBase, defaultRopeTheta, a.RopeLocalBase, defaultRopeLocalTheta)
	}

	c := base
	c.RopeTheta = 500000
	a, _ = c.Arch()
	if a.RopeBase != 500000 || a.RopeLocalBase != defaultRopeLocalTheta {
		t.Fatalf("rope_theta: RopeBase %v (want 5e5), RopeLocalBase %v (want %v)", a.RopeBase, a.RopeLocalBase, defaultRopeLocalTheta)
	}

	c = base
	c.RopeParameters = map[string]RopeParam{
		"full_attention":    {RopeTheta: 2000000},
		"sliding_attention": {RopeTheta: 5000},
	}
	a, _ = c.Arch()
	if a.RopeBase != 2000000 || a.RopeLocalBase != 5000 {
		t.Fatalf("rope_parameters: RopeBase %v (want 2e6), RopeLocalBase %v (want 5e3)", a.RopeBase, a.RopeLocalBase)
	}

	// partial rotary: default is full (rotaryDim == headDim); a factor shrinks it.
	if a.RotaryDim != base.HeadDim || a.RotaryDimLocal != base.HeadDim {
		t.Fatalf("default rotary: got %d/%d, want full headDim %d", a.RotaryDim, a.RotaryDimLocal, base.HeadDim)
	}
	c = base
	c.RopeParameters = map[string]RopeParam{
		"full_attention":    {RopeTheta: 1000000, PartialRotaryFactor: 0.25},
		"sliding_attention": {RopeTheta: 10000}, // no factor → full rotary on sliding
	}
	a, _ = c.Arch()
	if a.RotaryDim != base.HeadDim/4 || a.RotaryDimLocal != base.HeadDim {
		t.Fatalf("partial rotary: got RotaryDim %d (want %d), RotaryDimLocal %d (want %d)", a.RotaryDim, base.HeadDim/4, a.RotaryDimLocal, base.HeadDim)
	}
	// proportional rope_type folds the base to base^(rotaryDim/headDim) for the partial full-attention.
	c = base
	c.RopeParameters = map[string]RopeParam{
		"full_attention": {RopeTheta: 1000000, PartialRotaryFactor: 0.25, RopeType: "proportional"},
	}
	a, _ = c.Arch()
	wantBase := float32(math.Pow(1000000, float64(base.HeadDim/4)/float64(base.HeadDim))) // 1e6^0.25
	if a.RotaryDim != base.HeadDim/4 {
		t.Fatalf("proportional rotaryDim %d, want %d", a.RotaryDim, base.HeadDim/4)
	}
	if math.Abs(float64(a.RopeBase-wantBase)) > 1e-2 {
		t.Fatalf("proportional base %v, want %v (1e6^0.25)", a.RopeBase, wantBase)
	}
	// "default" rope_type must NOT fold the base, even when partial.
	c.RopeParameters["full_attention"] = RopeParam{RopeTheta: 1000000, PartialRotaryFactor: 0.25, RopeType: "default"}
	a, _ = c.Arch()
	if a.RopeBase != 1000000 {
		t.Fatalf("default rope_type should leave the base unfolded, got %v", a.RopeBase)
	}
	t.Logf("rope: defaults 1e6/1e4 + full rotary; rope_theta sets global; partial_rotary_factor sets rotaryDim; proportional folds base→base^(rotaryDim/headDim) (%v), default leaves it", wantBase)
}

// TestConfigArchRealGemma4_26B_A4B_MoEGeometry pins Config.Arch() against the ACTUAL
// mlx-community/gemma-4-26B-A4B-it-qat-4bit config.json text_config: 30 layers, 2816
// hidden, 16 attention heads / 8 sliding KV heads / 2 global KV heads, head_dim 256 /
// global_head_dim 512, 128 experts routing top-8 at moe_intermediate_size 704 — the
// PER-EXPERT FFN width, which is NOT intermediate_size (2112, the dense/shared FFN size
// this MoE checkpoint's enable_moe_block=true means every layer never actually runs).
// A parser that swapped the two intermediate sizes would size the expert FFN 3x too wide
// and pass every synthetic test in this file (they use small, easily-confused round
// numbers) while silently corrupting a real 26B-A4B load.
func TestConfigArchRealGemma4_26B_A4B_MoEGeometry(t *testing.T) {
	lt := make([]string, 30)
	for i := range lt {
		if (i+1)%6 == 0 { // gemma4's real pattern: full_attention every 6th layer, first at index 5
			lt[i] = "full_attention"
		} else {
			lt[i] = "sliding_attention"
		}
	}
	c := Config{
		HiddenSize: 2816, NumHiddenLayers: 30, IntermediateSize: 2112,
		NumAttentionHeads: 16, NumKeyValueHeads: 8, HeadDim: 256, GlobalHeadDim: 512,
		NumGlobalKeyValueHeads: 2, VocabSize: 262144, SlidingWindow: 1024,
		LayerTypes: lt, AttentionKEqV: true,
		EnableMoEBlock: true, NumExperts: 128, TopKExperts: 8, MoEIntermediateSize: 704,
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.Experts != 128 || a.TopK != 8 {
		t.Fatalf("MoE routing = experts %d top-%d, want 128/8 (the real 26B-A4B config)", a.Experts, a.TopK)
	}
	if a.ExpertFF != 704 {
		t.Fatalf("ExpertFF = %d, want 704 (moe_intermediate_size — NOT intermediate_size 2112, the dense FFN size this MoE checkpoint doesn't use)", a.ExpertFF)
	}
	if a.FF != 2112 {
		t.Fatalf("FF (dense intermediate_size, carried for completeness) = %d, want 2112", a.FF)
	}
	if a.HeadDim != 256 || a.GlobalHeadDim != 512 {
		t.Fatalf("HeadDim/GlobalHeadDim = %d/%d, want 256/512", a.HeadDim, a.GlobalHeadDim)
	}
	if a.KVHeads != 8 || a.GlobalKVHeads != 2 {
		t.Fatalf("KVHeads/GlobalKVHeads = %d/%d, want 8/2 (sliding vs full KV-head split)", a.KVHeads, a.GlobalKVHeads)
	}
	for i, l := range a.Layer {
		if !l.MoE {
			t.Fatalf("layer %d not marked MoE — gemma4 applies MoE uniformly", i)
		}
	}
	if len(a.Layer) != 30 {
		t.Fatalf("layer count = %d, want 30", len(a.Layer))
	}
	t.Logf("real 26B-A4B geometry: 30 layers, hidden 2816, 16/8(sliding)/2(global) heads, 128 experts top-8 @ FF 704 (dense FF 2112 unused)")
}

// TestConfigArchRealGemma4_31B_DenseGeometry pins Config.Arch() against the ACTUAL
// mlx-community/gemma-4-31B-it-4bit config.json text_config: 60 layers (the deepest
// gemma4 release), 5376 hidden, 32 attention heads / 16 sliding KV heads / 4 global KV
// heads, head_dim 256 / global_head_dim 512, dense FFN 21504, no experts — the largest
// dense (non-MoE, non-unified) family member.
func TestConfigArchRealGemma4_31B_DenseGeometry(t *testing.T) {
	lt := make([]string, 60)
	for i := range lt {
		if (i+1)%6 == 0 {
			lt[i] = "full_attention"
		} else {
			lt[i] = "sliding_attention"
		}
	}
	c := Config{
		HiddenSize: 5376, NumHiddenLayers: 60, IntermediateSize: 21504,
		NumAttentionHeads: 32, NumKeyValueHeads: 16, HeadDim: 256, GlobalHeadDim: 512,
		NumGlobalKeyValueHeads: 4, VocabSize: 262144, SlidingWindow: 1024,
		LayerTypes: lt, AttentionKEqV: true,
	}
	a, err := c.Arch()
	if err != nil {
		t.Fatalf("Arch: %v", err)
	}
	if a.Experts != 0 || a.TopK != 0 || a.ExpertFF != 0 {
		t.Fatalf("31B is dense: Experts=%d TopK=%d ExpertFF=%d, want all 0", a.Experts, a.TopK, a.ExpertFF)
	}
	if a.MoEGating != "" {
		t.Fatalf("dense model declares no MoE gating: got %q, want empty", a.MoEGating)
	}
	if a.FF != 21504 {
		t.Fatalf("FF = %d, want 21504 (the real 31B dense intermediate_size)", a.FF)
	}
	if a.KVHeads != 16 || a.GlobalKVHeads != 4 {
		t.Fatalf("KVHeads/GlobalKVHeads = %d/%d, want 16/4", a.KVHeads, a.GlobalKVHeads)
	}
	if len(a.Layer) != 60 {
		t.Fatalf("layer count = %d, want 60 (the deepest gemma4 release)", len(a.Layer))
	}
	for i, l := range a.Layer {
		if l.MoE {
			t.Fatalf("layer %d marked MoE on a dense 31B config", i)
		}
	}
	t.Logf("real 31B geometry: 60 layers, hidden 5376, 32/16(sliding)/4(global) heads, dense FF 21504, no experts")
}
