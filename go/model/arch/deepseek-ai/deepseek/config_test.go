// SPDX-Licence-Identifier: EUPL-1.2

package deepseek

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
	coreio "dappco.re/go/io"
)

// Fixture source: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json
func TestConfig_Validate_Good(t *testing.T) {
	data, err := coreio.Local.Read(core.PathJoin("testdata", "deepseek-ai-deepseek-v2-lite-config.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cfg Config
	if r := core.JSONUnmarshal([]byte(data), &cfg); !r.OK {
		t.Fatalf("parse fixture: %s", r.Error())
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	if cfg.QHeadDim() != 192 || cfg.KVHeadDim() != 192 || cfg.ValueHeadDim != 128 {
		t.Fatalf("MLA heads = q %d kv %d v %d", cfg.QHeadDim(), cfg.KVHeadDim(), cfg.ValueHeadDim)
	}
	if cfg.KVLoRARank != 512 || cfg.NumRoutedExperts != 64 || cfg.NumExpertsPerTok != 6 {
		t.Fatalf("MLA/MoE geometry = %+v", cfg)
	}
}

func TestConfig_Validate_Bad(t *testing.T) {
	if err := (Config{}).Validate(); err == nil {
		t.Fatal("empty config accepted")
	}
}

func TestConfig_ValidateSparseGeometry_Bad(t *testing.T) {
	cfg := Config{HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32, KVLoRARank: 4, QKNoPEHeadDim: 2, QKRoPEHeadDim: 2, ValueHeadDim: 2}
	if err := cfg.Validate(); err == nil {
		t.Fatal("missing sparse-expert geometry accepted")
	}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("invalid MLA config accepted by Arch")
	}
}

// TestConfig_Validate_Ugly proves the top-k-equals-experts BOUNDARY is
// accepted: NumExpertsPerTok > NumRoutedExperts is rejected, but == must
// pass (every expert selected on every token is unusual but not invalid).
func TestConfig_Validate_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32, KVLoRARank: 4, QKNoPEHeadDim: 2, QKRoPEHeadDim: 2, ValueHeadDim: 2, NumRoutedExperts: 4, NumExpertsPerTok: 4, MoEIntermediateSize: 4}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("Validate rejected the top-k == experts boundary: %v", err)
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32, KVLoRARank: 4, QKNoPEHeadDim: 2, QKRoPEHeadDim: 2, ValueHeadDim: 2, NumRoutedExperts: 2, NumExpertsPerTok: 1, MoEIntermediateSize: 4}
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("MLA config incorrectly lowered to standard attention")
	}
}

// TestConfig_Arch_Good pins the documented "happy path" for an always-refuses
// arch: a realistic, Validate-passing DeepSeek-V2-Lite config still refuses,
// but with the MLA-specific message (not a Validate dimension error).
func TestConfig_Arch_Good(t *testing.T) {
	data, err := coreio.Local.Read(core.PathJoin("testdata", "deepseek-ai-deepseek-v2-lite-config.json"))
	if err != nil {
		t.Fatalf("read fixture: %v", err)
	}
	var cfg Config
	if r := core.JSONUnmarshal([]byte(data), &cfg); !r.OK {
		t.Fatalf("parse fixture: %s", r.Error())
	}
	_, err = cfg.Arch()
	if err == nil {
		t.Fatal("Arch: expected a clean MLA refusal, got a resolved architecture")
	}
	if !core.Contains(err.Error(), "MLA requires a separate attention implementation") {
		t.Fatalf("Arch refusal %q must name the MLA gap", err.Error())
	}
}

// TestConfig_Arch_Bad proves an invalid config fails at Validate — the error
// propagated is Validate's dimension error, NOT the fixed MLA-refusal
// message _Good and _Ugly both reach after Validate succeeds.
func TestConfig_Arch_Bad(t *testing.T) {
	_, err := (Config{}).Arch()
	if err == nil {
		t.Fatal("Arch accepted an empty config")
	}
	if core.Contains(err.Error(), "MLA requires a separate attention implementation") {
		t.Fatal("Arch: an invalid config must fail Validate, not reach the MLA refusal message")
	}
}

func TestConfig_QHeadDim_Good(t *testing.T) {
	cfg := Config{QKNoPEHeadDim: 128, QKRoPEHeadDim: 64}
	if got := cfg.QHeadDim(); got != 192 {
		t.Fatalf("QHeadDim = %d, want 192 (128 NoPE + 64 RoPE)", got)
	}
}

func TestConfig_QHeadDim_Bad(t *testing.T) {
	if got := (Config{}).QHeadDim(); got != 0 {
		t.Fatalf("QHeadDim = %d, want 0 for an unconfigured MLA split", got)
	}
}

// TestConfig_QHeadDim_Ugly proves QHeadDim sums whatever's given even with an
// asymmetric split (a zero RoPE component) — it doesn't assume both halves
// are non-zero.
func TestConfig_QHeadDim_Ugly(t *testing.T) {
	cfg := Config{QKNoPEHeadDim: 128, QKRoPEHeadDim: 0}
	if got := cfg.QHeadDim(); got != 128 {
		t.Fatalf("QHeadDim = %d, want 128 (RoPE component absent)", got)
	}
}

func TestConfig_KVHeadDim_Good(t *testing.T) {
	cfg := Config{QKNoPEHeadDim: 128, QKRoPEHeadDim: 64}
	if got := cfg.KVHeadDim(); got != 192 {
		t.Fatalf("KVHeadDim = %d, want 192 (matches QHeadDim — Q/K share width for the dot product)", got)
	}
}

func TestConfig_KVHeadDim_Bad(t *testing.T) {
	if got := (Config{}).KVHeadDim(); got != 0 {
		t.Fatalf("KVHeadDim = %d, want 0 for an unconfigured MLA split", got)
	}
}

// TestConfig_KVHeadDim_Ugly proves KVHeadDim tracks QHeadDim regardless of a
// DIFFERING ValueHeadDim — V's width is independent of Q/K's, unlike the
// zero-RoPE edge _Ugly on QHeadDim exercises.
func TestConfig_KVHeadDim_Ugly(t *testing.T) {
	cfg := Config{QKNoPEHeadDim: 128, QKRoPEHeadDim: 64, ValueHeadDim: 128}
	if got := cfg.KVHeadDim(); got != cfg.QHeadDim() {
		t.Fatalf("KVHeadDim = %d, want to equal QHeadDim (%d) regardless of ValueHeadDim (%d)", got, cfg.QHeadDim(), cfg.ValueHeadDim)
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(map[string]safetensors.Tensor{"ignored": {Shape: []int{1}}})
	if cfg.HiddenSize != 8 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}

func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if err := cfg.Validate(); err == nil {
		t.Fatal("empty config became valid after InferFromWeights")
	}
}

// TestConfig_InferFromWeights_Ugly proves the no-op does not paper over a
// missing sparse-expert geometry when the MLA dims ARE present — distinct
// from _Bad's totally-empty config.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, NumHiddenLayers: 1, NumAttentionHeads: 2, VocabSize: 32, KVLoRARank: 4, QKNoPEHeadDim: 2, QKRoPEHeadDim: 2, ValueHeadDim: 2}
	cfg.InferFromWeights(map[string]safetensors.Tensor{})
	if err := cfg.Validate(); err == nil {
		t.Fatal("missing sparse-expert geometry became valid after InferFromWeights")
	}
}
