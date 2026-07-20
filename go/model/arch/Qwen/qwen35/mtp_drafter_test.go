// SPDX-Licence-Identifier: EUPL-1.2

package qwen35

import (
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
	"dappco.re/go/inference/model/safetensors"
)

// qwen36_27B_MTP is the real mlx-community/Qwen3.6-27B-MTP-4bit config.json shape (snapshot
// 83795d546e9d328160e593fb0bf10b2bf2fe637e, read from the local Hugging Face cache — see
// mtp_drafter_real_test.go for the header-only tensor reconciliation against this same snapshot):
// mtp_num_hidden_layers/attn_output_gate/etc all live INSIDE text_config (the base's own text_config,
// reused verbatim for the drafter); block_size is top-level only.
func qwen36_27B_MTP() []byte {
	return []byte(core.Sprintf(`{
		"block_size": 3,
		"model_type": "qwen3_5_mtp",
		"quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
		"text_config": {
			"attn_output_gate": true,
			"full_attention_interval": 4,
			"head_dim": 256,
			"hidden_size": 5120,
			"intermediate_size": 17408,
			"layer_types": %s,
			"model_type": "qwen3_5_text",
			"mtp_num_hidden_layers": 1,
			"num_attention_heads": 24,
			"num_hidden_layers": 64,
			"num_key_value_heads": 4,
			"partial_rotary_factor": 0.25,
			"rms_norm_eps": 1e-06,
			"rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
			"tie_word_embeddings": false,
			"vocab_size": 248320
		},
		"tie_word_embeddings": false
	}`, hybridLayerTypesJSON(64, 4)))
}

// TestParseDrafterConfig_Good proves the real (nested text_config) drafter shape parses: the
// mtp_num_hidden_layers/block_size fields resolve from the right level (nested vs top-level
// respectively), and DrafterArch derives successfully.
func TestParseDrafterConfig_Good(t *testing.T) {
	cfg, err := ParseDrafterConfig(qwen36_27B_MTP())
	if err != nil {
		t.Fatalf("ParseDrafterConfig: %v", err)
	}
	if got := cfg.effective().MTPNumHiddenLayers; got != 1 {
		t.Errorf("effective().MTPNumHiddenLayers = %d, want 1 (nested under text_config on the real checkpoint)", got)
	}
	if cfg.BlockSize != 3 {
		t.Errorf("BlockSize = %d, want 3 (top-level only on the real checkpoint)", cfg.BlockSize)
	}
	if cfg.effective().HiddenSize != 5120 {
		t.Errorf("effective().HiddenSize = %d, want 5120 (the base's own hidden size, reused)", cfg.effective().HiddenSize)
	}
}

// TestParseDrafterConfig_Bad proves a config with no mtp_num_hidden_layers (a plain base config, or a
// malformed drafter config) refuses by name rather than silently deriving a bogus 0-layer head.
func TestParseDrafterConfig_Bad(t *testing.T) {
	if _, err := ParseDrafterConfig(qwen35_27B()); err == nil {
		t.Fatal("ParseDrafterConfig on a base (non-drafter) config succeeded, want an error")
	} else if !strings.Contains(err.Error(), "mtp_num_hidden_layers") {
		t.Errorf("error = %q, want it to name mtp_num_hidden_layers", err.Error())
	}
}

// TestParseDrafterConfig_Ugly proves an unparseable config.json refuses cleanly (never panics).
func TestParseDrafterConfig_Ugly(t *testing.T) {
	if _, err := ParseDrafterConfig([]byte("not json")); err == nil {
		t.Fatal("ParseDrafterConfig on invalid JSON succeeded, want an error")
	}
}

// TestConfigDrafterArch_Good pins the head's derived architecture against the REAL checkpoint's
// declared dims: one full-attention layer (not the base's 64-layer hybrid schedule), the base's own
// gated attention / head_dim / kv_heads / FF / rope carried through unchanged, dense (no MoE).
func TestConfigDrafterArch_Good(t *testing.T) {
	var cfg Config
	if r := core.JSONUnmarshal(qwen36_27B_MTP(), &cfg); !r.OK {
		t.Fatalf("config.json parse failed")
	}
	arch, err := cfg.DrafterArch()
	if err != nil {
		t.Fatalf("DrafterArch: %v", err)
	}
	if len(arch.Layer) != 1 {
		t.Fatalf("len(Layer) = %d, want 1 (mtp_num_hidden_layers, not the base's num_hidden_layers=64)", len(arch.Layer))
	}
	l := arch.Layer[0]
	if l.Mixer != model.MixerAttention {
		t.Errorf("Layer[0].Mixer = %v, want MixerAttention — the head is never gated-delta regardless of the base's schedule", l.Mixer)
	}
	if l.HeadDim != 256 || l.KVHeads != 4 {
		t.Errorf("Layer[0] headDim/kvHeads = %d/%d, want 256/4 (the base's own dims)", l.HeadDim, l.KVHeads)
	}
	if l.CacheIndex < 0 {
		t.Error("Layer[0] owns no KV cache slot — a full-attention layer must own one")
	}
	if arch.Heads != 24 {
		t.Errorf("Heads = %d, want 24", arch.Heads)
	}
	if arch.FF != 17408 {
		t.Errorf("FF = %d, want 17408 (intermediate_size, same width as the base)", arch.FF)
	}
	if !arch.AttnOutputGate {
		t.Error("AttnOutputGate = false, want true — inherited from the base's text_config (attn_output_gate)")
	}
	if arch.Activation != "silu" {
		t.Errorf("Activation = %q, want silu", arch.Activation)
	}
	if arch.RotaryDim != 64 {
		t.Errorf("RotaryDim = %d, want 64 (head_dim 256 * partial_rotary_factor 0.25)", arch.RotaryDim)
	}
	if arch.RopeBase != 1e7 {
		t.Errorf("RopeBase = %g, want 1e7", arch.RopeBase)
	}
	if arch.Experts != 0 || l.MoE {
		t.Error("a dense-base drafter derived a non-zero MoE arch")
	}
}

// TestConfigDrafterArch_Bad proves a config with mtp_num_hidden_layers <= 0 refuses by name.
func TestConfigDrafterArch_Bad(t *testing.T) {
	var cfg Config
	if r := core.JSONUnmarshal(qwen35_27B(), &cfg); !r.OK {
		t.Fatalf("config.json parse failed")
	}
	if _, err := cfg.DrafterArch(); err == nil {
		t.Fatal("DrafterArch on a base config (no mtp_num_hidden_layers) succeeded, want an error")
	} else if !strings.Contains(err.Error(), "mtp_num_hidden_layers") {
		t.Errorf("error = %q, want it to name mtp_num_hidden_layers", err.Error())
	}
}

// TestDrafterWeightNames pins the drafter's FLAT tensor layout (no "model." prefix, unlike the base)
// and the deliberate absence of Embed/LMHead — the mapping's central safety property.
func TestDrafterWeightNames(t *testing.T) {
	w := DrafterWeightNames()
	if w.LayerPrefix != "layers.%d" {
		t.Errorf("LayerPrefix = %q, want %q (flat, no model. prefix)", w.LayerPrefix, "layers.%d")
	}
	if w.FinalNorm != "norm.weight" {
		t.Errorf("FinalNorm = %q, want norm.weight (flat, no model. prefix)", w.FinalNorm)
	}
	if w.Embed != "" || w.LMHead != "" {
		t.Errorf("Embed/LMHead = %q/%q, want both empty — the drafter shares the base's embedding + LM head", w.Embed, w.LMHead)
	}
	// The per-layer suffixes are inherited from the base's own WeightNames() unchanged.
	base := WeightNames()
	if w.AttnNorm != base.AttnNorm || w.MLPNorm != base.MLPNorm || w.PostAttnNorm != base.PostAttnNorm {
		t.Errorf("per-layer norm suffixes diverged from the base's WeightNames(): got %+v", w)
	}
	if w.Q != base.Q || w.K != base.K || w.V != base.V || w.O != base.O {
		t.Errorf("attention suffixes diverged from the base's WeightNames(): got %+v", w)
	}
}

// TestDrafterTensorNames_Good proves the enumerated name list matches the real checkpoint's shape at
// nLayers=1 (4 top-level + 11 per-layer names) and contains every literal name confirmed against the
// real mlx-community/Qwen3.6-27B-MTP-4bit header (see mtp_drafter_real_test.go).
func TestDrafterTensorNames_Good(t *testing.T) {
	names := DrafterTensorNames(1)
	want := []string{
		"fc.weight", "pre_fc_norm_embedding.weight", "pre_fc_norm_hidden.weight", "norm.weight",
		"layers.0.input_layernorm.weight", "layers.0.post_attention_layernorm.weight",
		"layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
		"layers.0.self_attn.v_proj.weight", "layers.0.self_attn.o_proj.weight",
		"layers.0.self_attn.q_norm.weight", "layers.0.self_attn.k_norm.weight",
		"layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight", "layers.0.mlp.down_proj.weight",
	}
	if len(names) != len(want) {
		t.Fatalf("len(names) = %d, want %d: got %v", len(names), len(want), names)
	}
	got := map[string]bool{}
	for _, n := range names {
		got[n] = true
	}
	for _, w := range want {
		if !got[w] {
			t.Errorf("DrafterTensorNames(1) missing %q", w)
		}
	}
}

// TestDrafterTensorNames_Ugly proves the per-layer count scales linearly and never collides across
// layer indices (a naive string-concat bug could alias layer 1 onto layer 10, etc.).
func TestDrafterTensorNames_Ugly(t *testing.T) {
	names := DrafterTensorNames(3)
	if len(names) != 4+3*11 {
		t.Fatalf("len(names) = %d, want %d (4 top-level + 3*11 per-layer)", len(names), 4+3*11)
	}
	seen := map[string]bool{}
	for _, n := range names {
		if seen[n] {
			t.Fatalf("duplicate tensor name %q — layer indices collided", n)
		}
		seen[n] = true
	}
}

// --- synthetic Assemble round-trip -----------------------------------------------------------------

// draftMat/draftVec build BF16 zero-filled synthetic tensors — the same synthetic-weight pattern
// go/model/arch/openai/gptoss/weights_test.go uses for its model.Assemble round trip.
func draftMat(rows, cols int) safetensors.Tensor {
	return safetensors.Tensor{Shape: []int{rows, cols}, Data: make([]byte, rows*cols*2), Dtype: "BF16"}
}
func draftVec(n int) safetensors.Tensor {
	return safetensors.Tensor{Shape: []int{n}, Data: make([]byte, n*2), Dtype: "BF16"}
}

// draftSyntheticConfig is a small, flat (no text_config nesting — the real checkpoint's OTHER valid
// shape, per ParseDrafterConfig's effective()-based resolution) 2-layer, ungated, dense drafter: hidden
// 8, 2 heads, 1 kv head, head_dim 4 (2*4=8), FF 16 — hand-shapeable tensors, no attn_output_gate
// doubling, to keep the round-trip's shapes simple to reason about by hand.
func draftSyntheticConfig() []byte {
	return []byte(`{
		"model_type": "qwen3_5_mtp",
		"hidden_size": 8, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
		"intermediate_size": 16, "rms_norm_eps": 1e-6,
		"mtp_num_hidden_layers": 2, "block_size": 3
	}`)
}

// draftSyntheticTensors builds every tensor DrafterTensorNames(2) names, shaped for
// draftSyntheticConfig()'s dims (D=8, FF=16, heads=2, kvHeads=1, headDim=4, ungated).
func draftSyntheticTensors() map[string]safetensors.Tensor {
	const d, ff, heads, kvHeads, headDim = 8, 16, 2, 1, 4
	t := map[string]safetensors.Tensor{
		DrafterFCWeight:         draftMat(d, 2*d),
		DrafterEmbedNormWeight:  draftVec(d),
		DrafterHiddenNormWeight: draftVec(d),
		"norm.weight":           draftVec(d),
	}
	for i := 0; i < 2; i++ {
		p := core.Sprintf("layers.%d.", i)
		t[p+"input_layernorm.weight"] = draftVec(d)
		t[p+"post_attention_layernorm.weight"] = draftVec(d)
		t[p+"self_attn.q_proj.weight"] = draftMat(heads*headDim, d)
		t[p+"self_attn.k_proj.weight"] = draftMat(kvHeads*headDim, d)
		t[p+"self_attn.v_proj.weight"] = draftMat(kvHeads*headDim, d)
		t[p+"self_attn.o_proj.weight"] = draftMat(d, heads*headDim)
		t[p+"self_attn.q_norm.weight"] = draftVec(headDim)
		t[p+"self_attn.k_norm.weight"] = draftVec(headDim)
		t[p+"mlp.gate_proj.weight"] = draftMat(ff, d)
		t[p+"mlp.up_proj.weight"] = draftMat(ff, d)
		t[p+"mlp.down_proj.weight"] = draftMat(d, ff)
	}
	return t
}

// TestDrafterConfigAssemble_Good is the config/weight round-trip receipt: DrafterArch()'s derived Arch
// plus a LOCAL WeightNames copy (Embed/LMHead overridden to test-only placeholders — a real checkpoint
// never carries these; see TestDrafterAssembleWithoutEmbed_Bad for the realistic, embed-less shape)
// resolves every per-layer weight through model.Assemble against DrafterTensorNames()-shaped synthetic
// tensors: no unmatched name, no nil Q/K/V/O/Gate/Up/Down/norm.
func TestDrafterConfigAssemble_Good(t *testing.T) {
	var cfg Config
	if r := core.JSONUnmarshal(draftSyntheticConfig(), &cfg); !r.OK {
		t.Fatalf("config.json parse failed")
	}
	arch, err := cfg.DrafterArch()
	if err != nil {
		t.Fatalf("DrafterArch: %v", err)
	}
	if len(arch.Layer) != 2 {
		t.Fatalf("len(Layer) = %d, want 2", len(arch.Layer))
	}

	tensors := draftSyntheticTensors()
	const vocab = 8
	tensors["test.embed_tokens.weight"] = draftMat(vocab, 8) // test-only stand-in; a real checkpoint carries none

	names := DrafterWeightNames()
	names.Embed = "test.embed_tokens" // override ONLY for this test — see the doc comment above

	m, err := model.Assemble(tensors, arch, names)
	if err != nil {
		t.Fatalf("Assemble: %v", err)
	}
	if len(m.Layers) != 2 {
		t.Fatalf("len(Layers) = %d, want 2", len(m.Layers))
	}
	for i, L := range m.Layers {
		if L.Q == nil || L.K == nil || L.V == nil || L.O == nil {
			t.Errorf("layer %d: missing attention projection: %+v", i, L)
		}
		if L.QNorm == nil || L.KNorm == nil {
			t.Errorf("layer %d: missing q_norm/k_norm", i)
		}
		if L.Gate == nil || L.Up == nil || L.Down == nil {
			t.Errorf("layer %d: missing dense FFN projection (took the MoE branch unexpectedly?): %+v", i, L)
		}
		if len(L.AttnNorm) == 0 || len(L.MLPNorm) == 0 {
			t.Errorf("layer %d: missing input_layernorm/post_attention_layernorm", i)
		}
	}
	if m.Embed == nil {
		t.Error("Embed did not resolve against the test-only placeholder name")
	}
	if len(m.FinalNorm) == 0 {
		t.Error("FinalNorm (norm.weight) did not resolve")
	}
}

// TestDrafterAssembleWithoutEmbed_Bad is the refusal-contract receipt: DrafterWeightNames() UNCHANGED
// (Embed/LMHead empty, matching a REAL checkpoint's actual shape — no embed_tokens, no lm_head, shared
// from the base) against a realistic per-layer-only tensor set fails Assemble with a clean, named,
// non-crashing error — never a nil-pointer panic downstream. This is what a hypothetical standalone
// load attempt on a real drafter checkpoint would hit today (model.RegisterArch's own entry never
// reaches this far — it still refuses at Parse, unchanged — but nothing else in this codebase reaches
// Assemble on a drafter checkpoint either, so this pins the SAFETY NET behind that design choice).
func TestDrafterAssembleWithoutEmbed_Bad(t *testing.T) {
	var cfg Config
	if r := core.JSONUnmarshal(draftSyntheticConfig(), &cfg); !r.OK {
		t.Fatalf("config.json parse failed")
	}
	arch, err := cfg.DrafterArch()
	if err != nil {
		t.Fatalf("DrafterArch: %v", err)
	}
	tensors := draftSyntheticTensors() // no embed_tokens / lm_head — the real checkpoint's actual shape

	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("Assemble panicked instead of returning an error: %v", r)
		}
	}()
	m, err := model.Assemble(tensors, arch, DrafterWeightNames())
	if err == nil {
		t.Fatalf("Assemble succeeded without an embedding tensor, want a clean error; got %+v", m)
	}
	if !strings.Contains(err.Error(), "absent") {
		t.Errorf("error = %q, want it to name the absent tensor", err.Error())
	}
}
