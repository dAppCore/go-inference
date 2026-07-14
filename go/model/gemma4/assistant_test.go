// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference/model/mtp"
)

// realisticAssistantTextConfigJSON is a small but shape-faithful drafter text_config: a
// handful of layers on the same sliding/full schedule real gemma4 checkpoints use (a run of
// sliding_attention layers then one full_attention), MQA-style KV heads, and an explicit
// head_dim — every field buildAssistantConfig's validation requires declared, matching how a
// real gemma4-assistant checkpoint's config.json is shaped.
const realisticAssistantTextConfigJSON = `{
	"hidden_size": 512, "num_hidden_layers": 4, "intermediate_size": 2048,
	"num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 128,
	"vocab_size": 262144, "sliding_window": 512,
	"layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"]
}`

// TestParseAssistantConfigNested covers the standard shape real checkpoints ship: the
// drafter's own arch nested under text_config, backbone_hidden_size + the ordered-embedding
// declaration at the top level.
func TestParseAssistantConfigNested(t *testing.T) {
	js := `{"model_type":"gemma4_assistant","backbone_hidden_size":1536,"text_config":` + realisticAssistantTextConfigJSON + `}`
	cfg, err := ParseAssistantConfig([]byte(js))
	if err != nil {
		t.Fatalf("ParseAssistantConfig: %v", err)
	}
	if cfg.ModelType != "gemma4_assistant" || cfg.BackboneHidden != 1536 {
		t.Fatalf("cfg = %+v, want ModelType gemma4_assistant, BackboneHidden 1536", cfg)
	}
	if cfg.Arch.Hidden != 512 || len(cfg.Arch.Layer) != 4 {
		t.Fatalf("derived Arch = %+v, want Hidden 512, 4 layers", cfg.Arch)
	}
	if cfg.OrderedEmbeddings {
		t.Fatal("use_ordered_embeddings absent should default to false")
	}
	if cfg.Quant != nil {
		t.Fatalf("no quantization anywhere should resolve to nil Quant, got %+v", cfg.Quant)
	}
	if got := cfg.LayerType(3); got != "full_attention" {
		t.Fatalf("LayerType(3) = %q, want full_attention (the declared 4th-layer schedule)", got)
	}
}

// TestParseAssistantConfigFlatLegacy covers the early-export shape: no text_config wrapper,
// the drafter's arch fields flat at the top level alongside backbone_hidden_size — the
// pre-text_config format assistant.go's comment documents.
func TestParseAssistantConfigFlatLegacy(t *testing.T) {
	js := `{"backbone_hidden_size":1536,"hidden_size":512,"num_hidden_layers":4,"intermediate_size":2048,
		"num_attention_heads":4,"num_key_value_heads":1,"head_dim":128,"vocab_size":1000,
		"layer_types":["sliding_attention","sliding_attention","sliding_attention","full_attention"],"sliding_window":512}`
	cfg, err := ParseAssistantConfig([]byte(js))
	if err != nil {
		t.Fatalf("ParseAssistantConfig(flat legacy): %v", err)
	}
	// model_type absent entirely (not merely ""): ParseAssistantConfig defaults it.
	if cfg.ModelType != "gemma4_assistant" {
		t.Fatalf("ModelType = %q, want the gemma4_assistant default for an absent model_type", cfg.ModelType)
	}
	if cfg.Arch.Hidden != 512 || len(cfg.Arch.Layer) != 4 {
		t.Fatalf("flat-legacy derived Arch = %+v, want Hidden 512, 4 layers", cfg.Arch)
	}
}

// TestParseAssistantConfigQuantTopLevelOverride covers ResolvedQuant's convention mirrored
// here: a top-level quantization block outranks one nested under text_config.
func TestParseAssistantConfigQuantTopLevelOverride(t *testing.T) {
	js := `{"model_type":"gemma4_assistant","backbone_hidden_size":1536,
		"quantization":{"group_size":64,"bits":4},
		"text_config":` + realisticAssistantTextConfigJSON + `}`
	cfg, err := ParseAssistantConfig([]byte(js))
	if err != nil {
		t.Fatalf("ParseAssistantConfig: %v", err)
	}
	if cfg.Quant == nil || cfg.Quant.GroupSize != 64 || cfg.Quant.Bits != 4 {
		t.Fatalf("Quant = %+v, want the top-level group_size 64 / bits 4", cfg.Quant)
	}
}

// TestParseAssistantConfigQuantNestedFallback covers the fallback half: no top-level
// quantization block, so the nested text_config's own quantization resolves instead (Config's
// ResolvedQuant, reused here since assistant text arches ARE gemma4.Config).
func TestParseAssistantConfigQuantNestedFallback(t *testing.T) {
	js := `{"model_type":"gemma4_assistant","backbone_hidden_size":1536,"text_config":{
		"hidden_size": 512, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 1,
		"head_dim": 128, "vocab_size": 1000, "sliding_window": 512,
		"layer_types": ["full_attention", "full_attention"],
		"quantization": {"group_size": 32, "bits": 8}
	}}`
	cfg, err := ParseAssistantConfig([]byte(js))
	if err != nil {
		t.Fatalf("ParseAssistantConfig: %v", err)
	}
	if cfg.Quant == nil || cfg.Quant.GroupSize != 32 || cfg.Quant.Bits != 8 {
		t.Fatalf("Quant = %+v, want the nested text_config group_size 32 / bits 8", cfg.Quant)
	}
}

// TestParseAssistantConfigOrderedEmbeddings covers the ordered-embedding (centroid) head
// declaration: when use_ordered_embeddings is true, num_centroids/centroid_intermediate_top_k
// carry through; the companion validation failure (ordered set, centroids absent) is covered
// by TestBuildAssistantConfigValidation.
func TestParseAssistantConfigOrderedEmbeddings(t *testing.T) {
	js := `{"model_type":"gemma4_assistant","backbone_hidden_size":1536,
		"use_ordered_embeddings":true,"num_centroids":4096,"centroid_intermediate_top_k":8,
		"text_config":` + realisticAssistantTextConfigJSON + `}`
	cfg, err := ParseAssistantConfig([]byte(js))
	if err != nil {
		t.Fatalf("ParseAssistantConfig: %v", err)
	}
	if !cfg.OrderedEmbeddings || cfg.NumCentroids != 4096 || cfg.CentroidTopK != 8 {
		t.Fatalf("cfg = %+v, want OrderedEmbeddings true, NumCentroids 4096, CentroidTopK 8", cfg)
	}
}

// TestParseAssistantConfigUnsupportedModelType covers the model_type gate: a declared
// model_type that is neither gemma4_assistant nor gemma4_unified_assistant is rejected rather
// than silently accepted.
func TestParseAssistantConfigUnsupportedModelType(t *testing.T) {
	js := `{"model_type":"not-a-gemma4-assistant","backbone_hidden_size":1536,"text_config":` + realisticAssistantTextConfigJSON + `}`
	if _, err := ParseAssistantConfig([]byte(js)); err == nil {
		t.Fatal("expected an error for an unsupported assistant model_type")
	}
}

// TestParseAssistantConfigUnifiedModelType covers the second supported id
// (gemma4_unified_assistant, the 12B-unified family's drafter) end to end.
func TestParseAssistantConfigUnifiedModelType(t *testing.T) {
	js := `{"model_type":"gemma4_unified_assistant","backbone_hidden_size":3840,"text_config":` + realisticAssistantTextConfigJSON + `}`
	cfg, err := ParseAssistantConfig([]byte(js))
	if err != nil {
		t.Fatalf("ParseAssistantConfig(gemma4_unified_assistant): %v", err)
	}
	if cfg.ModelType != "gemma4_unified_assistant" || cfg.BackboneHidden != 3840 {
		t.Fatalf("cfg = %+v, want ModelType gemma4_unified_assistant, BackboneHidden 3840", cfg)
	}
}

// TestBuildAssistantConfigValidation table-drives every buildAssistantConfig rejection: an
// unsupported model_type, a non-positive backbone/hidden/layers/heads/head_dim, and ordered
// embeddings declared without centroids. Each case must fail — a defective validator that
// silently accepted any of these would load a drafter with a nonsensical shape.
func TestBuildAssistantConfigValidation(t *testing.T) {
	validText := Config{
		HiddenSize: 512, NumHiddenLayers: 2, NumAttentionHeads: 4, NumKeyValueHeads: 1, HeadDim: 128,
		VocabSize: 1000, LayerTypes: []string{"full_attention", "full_attention"},
	}
	cases := []struct {
		name                         string
		modelType                    string
		backbone, numCentroids, topK int
		ordered                      bool
		text                         Config
	}{
		{"unsupported model_type", "gemma4-not-a-real-assistant", 1536, 0, 0, false, validText},
		{"zero backbone", "gemma4_assistant", 0, 0, 0, false, validText},
		{"negative backbone", "gemma4_assistant", -1, 0, 0, false, validText},
		{"zero hidden_size", "gemma4_assistant", 1536, 0, 0, false, Config{NumHiddenLayers: 2, NumAttentionHeads: 4, HeadDim: 128}},
		{"zero num_hidden_layers", "gemma4_assistant", 1536, 0, 0, false, Config{HiddenSize: 512, NumAttentionHeads: 4, HeadDim: 128}},
		{"zero num_attention_heads", "gemma4_assistant", 1536, 0, 0, false, Config{HiddenSize: 512, NumHiddenLayers: 2, HeadDim: 128}},
		{"zero head_dim", "gemma4_assistant", 1536, 0, 0, false, Config{HiddenSize: 512, NumHiddenLayers: 2, NumAttentionHeads: 4}},
		{"ordered without centroids", "gemma4_assistant", 1536, 0, 0, true, validText},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := buildAssistantConfig(tc.modelType, tc.backbone, tc.numCentroids, tc.topK, tc.ordered, tc.text); err == nil {
				t.Fatalf("%s: expected an error, got nil", tc.name)
			}
		})
	}
}

// TestBuildAssistantConfigOrderedEmbeddingsGood covers the accept path companion to the
// "ordered without centroids" rejection above: ordered embeddings WITH a positive centroid
// count succeeds and carries the fields through untouched.
func TestBuildAssistantConfigOrderedEmbeddingsGood(t *testing.T) {
	text := Config{
		HiddenSize: 512, NumHiddenLayers: 2, NumAttentionHeads: 4, NumKeyValueHeads: 1, HeadDim: 128,
		VocabSize: 1000, LayerTypes: []string{"full_attention", "full_attention"},
	}
	cfg, err := buildAssistantConfig("gemma4_assistant", 1536, 4096, 8, true, text)
	if err != nil {
		t.Fatalf("buildAssistantConfig: %v", err)
	}
	if !cfg.OrderedEmbeddings || cfg.NumCentroids != 4096 || cfg.CentroidTopK != 8 || cfg.BackboneHidden != 1536 {
		t.Fatalf("cfg = %+v, want ordered/4096/8/1536", cfg)
	}
}

// gemma4AssistantGGUFMeta builds a realistic gemma4-assistant GGUF metadata map: 4 layers on
// a period-4 sliding/full schedule (matching the real family's period-5/6 shape at a testable
// scale), MQA-style KV heads, and a partial-rotary declaration (32-of-128 dims = the real
// gemma4 full_attention 0.25 factor) — every key AssistantConfigFromGGUF reads.
func gemma4AssistantGGUFMeta() map[string]any {
	return map[string]any{
		"general.architecture":                              "gemma4-assistant",
		"gemma4-assistant.block_count":                      uint32(4),
		"gemma4-assistant.embedding_length":                 uint32(512),
		"gemma4-assistant.attention.head_count":             uint32(4),
		"gemma4-assistant.attention.head_count_kv":          uint32(1),
		"gemma4-assistant.attention.key_length":             uint32(128),
		"gemma4-assistant.embedding_length_out":             uint32(1536),
		"gemma4-assistant.attention.sliding_window_pattern": uint32(4),
		"gemma4-assistant.attention.sliding_window":         uint32(512),
		"gemma4-assistant.feed_forward_length":              uint32(2048),
		"gemma4-assistant.vocab_size":                       uint32(1000),
		"gemma4-assistant.context_length":                   uint32(8192),
		"gemma4-assistant.rope.freq_base":                   float32(1000000),
		"gemma4-assistant.rope.freq_base_swa":               float32(10000),
		"gemma4-assistant.rope.dimension_count":             uint32(32),
		"gemma4-assistant.attention.layer_norm_rms_epsilon": float32(1e-6),
	}
}

// TestAssistantConfigFromGGUF covers the GGUF metadata path end to end against realistic
// values: the block_count/embedding_length/head_count/key_length required quartet, the
// sliding_window_pattern → layer_types synthesis (first full_attention at index pattern-1,
// matching how real gemma4 declares "every Nth is full"), and the partial-rotary factor
// derived from rope.dimension_count/head_dim (32/128 = 0.25, the real gemma4 full_attention
// factor).
func TestAssistantConfigFromGGUF(t *testing.T) {
	cfg, err := AssistantConfigFromGGUF(gemma4AssistantGGUFMeta(), 0)
	if err != nil {
		t.Fatalf("AssistantConfigFromGGUF: %v", err)
	}
	if cfg.ModelType != "gemma4_assistant" || cfg.BackboneHidden != 1536 {
		t.Fatalf("cfg = %+v, want ModelType gemma4_assistant, BackboneHidden 1536", cfg)
	}
	if cfg.Arch.Hidden != 512 || len(cfg.Arch.Layer) != 4 {
		t.Fatalf("Arch = %+v, want Hidden 512, 4 layers", cfg.Arch)
	}
	wantTypes := []string{"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"}
	for i, want := range wantTypes {
		if got := cfg.LayerType(i); got != want {
			t.Fatalf("LayerType(%d) = %q, want %q (pattern 4 → first full at index 3)", i, got, want)
		}
	}
	if cfg.Arch.RotaryDim != 32 {
		t.Fatalf("RotaryDim = %d, want 32 (rope.dimension_count 32 / head_dim 128 = 0.25 factor · 128)", cfg.Arch.RotaryDim)
	}
}

// TestAssistantConfigFromGGUFDefaults covers the GGUF path's own defaulting (independent of
// gemma4's config.json defaults): sliding_window_pattern absent → pattern 1 (every layer
// full_attention), rope.freq_base / freq_base_swa absent → the gemma4 1e6/1e4 constants,
// head_count_kv absent → MHA (falls back to head_count), vocab_size absent → the caller's
// vocabHint.
func TestAssistantConfigFromGGUFDefaults(t *testing.T) {
	meta := map[string]any{
		"general.architecture":                  "gemma4-assistant",
		"gemma4-assistant.block_count":          uint32(2),
		"gemma4-assistant.embedding_length":     uint32(256),
		"gemma4-assistant.attention.head_count": uint32(4),
		"gemma4-assistant.attention.key_length": uint32(64),
	}
	cfg, err := AssistantConfigFromGGUF(meta, 5000)
	if err != nil {
		t.Fatalf("AssistantConfigFromGGUF: %v", err)
	}
	// embedding_length_out absent → backbone falls back to hidden (embedding_length).
	if cfg.BackboneHidden != 256 {
		t.Fatalf("BackboneHidden = %d, want 256 (fallback to embedding_length)", cfg.BackboneHidden)
	}
	// pattern absent → 1 → every layer full_attention.
	for i := range 2 {
		if got := cfg.LayerType(i); got != "full_attention" {
			t.Fatalf("LayerType(%d) = %q, want full_attention (pattern defaults to 1)", i, got)
		}
	}
	if cfg.Arch.Vocab != 5000 {
		t.Fatalf("Vocab = %d, want the vocabHint 5000 (vocab_size absent from metadata)", cfg.Arch.Vocab)
	}
	if cfg.Arch.KVHeads != 4 {
		t.Fatalf("KVHeads = %d, want 4 (head_count_kv absent → falls back to head_count, MHA)", cfg.Arch.KVHeads)
	}
	if cfg.Arch.RopeBase != 1000000 || cfg.Arch.RopeLocalBase != 10000 {
		t.Fatalf("RopeBase/RopeLocalBase = %v/%v, want the gemma4 1e6/1e4 defaults", cfg.Arch.RopeBase, cfg.Arch.RopeLocalBase)
	}
}

// TestAssistantConfigFromGGUFWrongArchitecture covers the general.architecture gate: metadata
// whose declared architecture is not "gemma4-assistant" is rejected outright.
func TestAssistantConfigFromGGUFWrongArchitecture(t *testing.T) {
	meta := gemma4AssistantGGUFMeta()
	meta["general.architecture"] = "some-other-drafter"
	if _, err := AssistantConfigFromGGUF(meta, 0); err == nil {
		t.Fatal("expected an error when general.architecture is not gemma4-assistant")
	}
}

// TestAssistantConfigFromGGUFMissingRequiredFields covers the required-quartet gate: each of
// block_count / embedding_length / head_count / key_length missing (one at a time) from
// otherwise-complete metadata is rejected rather than silently zero-filled.
func TestAssistantConfigFromGGUFMissingRequiredFields(t *testing.T) {
	for _, key := range []string{
		"gemma4-assistant.block_count",
		"gemma4-assistant.embedding_length",
		"gemma4-assistant.attention.head_count",
		"gemma4-assistant.attention.key_length",
	} {
		t.Run(key, func(t *testing.T) {
			meta := gemma4AssistantGGUFMeta()
			delete(meta, key)
			if _, err := AssistantConfigFromGGUF(meta, 0); err == nil {
				t.Fatalf("expected an error with %q missing", key)
			}
		})
	}
}

// TestAssistantGGUFWeightName table-drives the GGUF→checkpoint tensor-name map: the four
// fixed top-level names, the full per-layer leaf set (norms/projections/gate-up-down/layer
// scalar), and the two rejection shapes (a malformed "blk." entry with no layer/leaf
// separator, and an unrecognised leaf) that must map to "" (not part of the format) rather
// than a wrong guess.
func TestAssistantGGUFWeightName(t *testing.T) {
	cases := []struct{ in, want string }{
		{"token_embd.weight", "model.embed_tokens.weight"},
		{"output_norm.weight", "model.norm.weight"},
		{"nextn.pre_projection.weight", "pre_projection.weight"},
		{"nextn.post_projection.weight", "post_projection.weight"},
		{"blk.0.attn_norm.weight", "model.layers.0.input_layernorm.weight"},
		{"blk.3.post_attention_norm.weight", "model.layers.3.post_attention_layernorm.weight"},
		{"blk.0.ffn_norm.weight", "model.layers.0.pre_feedforward_layernorm.weight"},
		{"blk.0.post_ffw_norm.weight", "model.layers.0.post_feedforward_layernorm.weight"},
		{"blk.0.attn_q.weight", "model.layers.0.self_attn.q_proj.weight"},
		{"blk.0.attn_q_norm.weight", "model.layers.0.self_attn.q_norm.weight"},
		{"blk.0.attn_output.weight", "model.layers.0.self_attn.o_proj.weight"},
		{"blk.0.ffn_gate.weight", "model.layers.0.mlp.gate_proj.weight"},
		{"blk.0.ffn_up.weight", "model.layers.0.mlp.up_proj.weight"},
		{"blk.0.ffn_down.weight", "model.layers.0.mlp.down_proj.weight"},
		{"blk.0.layer_output_scale.weight", "model.layers.0.layer_scalar.weight"},
		{"blk.malformed", ""},                  // no '.' after the "blk." prefix → no layer/leaf split
		{"blk.5.unrecognised_leaf.weight", ""}, // recognised layer, unmapped leaf
		{"vision_tower.some.weight", ""},       // no "blk." prefix at all — not this format
	}
	for _, c := range cases {
		if got := AssistantGGUFWeightName(c.in); got != c.want {
			t.Fatalf("AssistantGGUFWeightName(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestGgufMetaInt table-drives every numeric type the GGUF metadata decoder may hand back
// (uint32/int32/uint64/int64/int/float64) plus the miss case (key absent or wrong type) → 0.
func TestGgufMetaInt(t *testing.T) {
	cases := []struct {
		name string
		v    any
		want int
	}{
		{"uint32", uint32(7), 7},
		{"int32", int32(7), 7},
		{"uint64", uint64(7), 7},
		{"int64", int64(7), 7},
		{"int", int(7), 7},
		{"float64", float64(7), 7},
		{"wrong type (string)", "7", 0},
	}
	for _, c := range cases {
		meta := map[string]any{"k": c.v}
		if got := ggufMetaInt(meta, "k"); got != c.want {
			t.Fatalf("%s: ggufMetaInt = %d, want %d", c.name, got, c.want)
		}
	}
	if got := ggufMetaInt(map[string]any{}, "missing"); got != 0 {
		t.Fatalf("missing key: ggufMetaInt = %d, want 0", got)
	}
}

// TestGgufMetaFloat table-drives every numeric type the GGUF metadata decoder may hand back
// for a float field (float32/float64/uint32/int32) plus the miss case → 0.
func TestGgufMetaFloat(t *testing.T) {
	cases := []struct {
		name string
		v    any
		want float32
	}{
		{"float32", float32(1.5), 1.5},
		{"float64", float64(1.5), 1.5},
		{"uint32", uint32(2), 2},
		{"int32", int32(2), 2},
		{"wrong type (string)", "1.5", 0},
	}
	for _, c := range cases {
		meta := map[string]any{"k": c.v}
		if got := ggufMetaFloat(meta, "k"); got != c.want {
			t.Fatalf("%s: ggufMetaFloat = %v, want %v", c.name, got, c.want)
		}
	}
	if got := ggufMetaFloat(map[string]any{}, "missing"); got != 0 {
		t.Fatalf("missing key: ggufMetaFloat = %v, want 0", got)
	}
}

// TestAssistantRegistersInEngine pins that gemma4's assistant init() registered the config.json
// model_types, the legacy "" default, and the GGUF architecture id — the same reactive dispatch
// TestRegistersArch pins for the (non-assistant) arch registry.
func TestAssistantRegistersInEngine(t *testing.T) {
	for _, mt := range []string{"gemma4_assistant", "gemma4_unified_assistant", ""} {
		if _, ok := mtp.LookupAssistant(mt); !ok {
			t.Fatalf("gemma4's assistant init() should register an AssistantSpec for model_type %q", mt)
		}
	}
	if _, ok := mtp.LookupAssistantGGUF(assistantGGUFArch); !ok {
		t.Fatalf("gemma4's assistant init() should register for GGUF architecture %q", assistantGGUFArch)
	}
}
