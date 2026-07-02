// SPDX-Licence-Identifier: EUPL-1.2

// Tests for adapter_config.go — adapter_config.json parsing and the alias /
// default normalisation applied on top of it.

package lora

import "testing"

// TestParseAdapterConfig_Aliases_Good covers the PEFT alias chain: r→Rank,
// lora_alpha→Alpha (deriving Scale from Rank+Alpha), target_modules→TargetKeys.
// A missing-metadata config is also asserted here — ParseAdapterConfig must not
// fabricate rank/alpha/scale when the source config omits them.
func TestParseAdapterConfig_Aliases_Good(t *testing.T) {
	cfg, err := ParseAdapterConfig([]byte(`{"r":4,"lora_alpha":12,"target_modules":["q_proj","v_proj"]}`))
	if err != nil {
		t.Fatalf("ParseAdapterConfig() error = %v", err)
	}
	if cfg.Rank != 4 || cfg.Alpha != 12 || cfg.Scale != 3 {
		t.Fatalf("config rank/alpha/scale = %d/%f/%f, want 4/12/3", cfg.Rank, cfg.Alpha, cfg.Scale)
	}
	if !equalStringSlices(cfg.TargetKeys, []string{"q_proj", "v_proj"}) {
		t.Fatalf("TargetKeys = %v, want target_modules alias", cfg.TargetKeys)
	}

	missing, err := ParseAdapterConfig([]byte(`{}`))
	if err != nil {
		t.Fatalf("ParseAdapterConfig(missing) error = %v", err)
	}
	if missing.Rank != 0 || missing.Alpha != 0 || missing.Scale != 0 {
		t.Fatalf("missing rank/alpha/scale = %d/%f/%f, want zero metadata", missing.Rank, missing.Alpha, missing.Scale)
	}
}

// TestParseAdapterConfig_BadInvalidJSON covers the parse-failure path: bytes
// that are not valid JSON must surface an error rather than a zero-value
// AdapterConfig.
func TestParseAdapterConfig_BadInvalidJSON(t *testing.T) {
	if _, err := ParseAdapterConfig([]byte(`{broken`)); err == nil {
		t.Fatal("expected invalid JSON error")
	}
}

// TestParseAdapterConfig_TargetPrecedence_Good covers the TargetKeys
// fallback chain: an explicit target_keys always wins, then target_modules
// (PEFT), then lora_layers (mlx-lm) as the last resort.
func TestParseAdapterConfig_TargetPrecedence_Good(t *testing.T) {
	cfg, err := ParseAdapterConfig([]byte(`{
		"target_keys":["explicit"],
		"target_modules":["peft"],
		"lora_layers":["mlx-lm"]
	}`))
	if err != nil {
		t.Fatalf("ParseAdapterConfig() error = %v", err)
	}
	if !equalStringSlices(cfg.TargetKeys, []string{"explicit"}) {
		t.Fatalf("TargetKeys = %v, want explicit target_keys precedence", cfg.TargetKeys)
	}

	cfg, err = ParseAdapterConfig([]byte(`{
		"target_modules":["peft"],
		"lora_layers":["mlx-lm"]
	}`))
	if err != nil {
		t.Fatalf("ParseAdapterConfig(peft) error = %v", err)
	}
	if !equalStringSlices(cfg.TargetKeys, []string{"peft"}) {
		t.Fatalf("TargetKeys = %v, want PEFT target_modules before lora_layers", cfg.TargetKeys)
	}

	cfg, err = ParseAdapterConfig([]byte(`{"lora_layers":["mlx-lm"]}`))
	if err != nil {
		t.Fatalf("ParseAdapterConfig(mlx-lm) error = %v", err)
	}
	if !equalStringSlices(cfg.TargetKeys, []string{"mlx-lm"}) {
		t.Fatalf("TargetKeys = %v, want lora_layers fallback", cfg.TargetKeys)
	}
}

// TestNormalizeAdapterConfigForLoad_Defaults_Good covers the engine-load
// defaulting path: a fully empty config gets rank 8 / alpha 16 / scale 2, and
// a scale-only config derives alpha from the caller's scale rather than the
// generic 2x-rank fallback.
func TestNormalizeAdapterConfigForLoad_Defaults_Good(t *testing.T) {
	cfg := NormalizeAdapterConfigForLoad(AdapterConfig{})
	if cfg.Rank != 8 || cfg.Alpha != 16 || cfg.Scale != 2 {
		t.Fatalf("default rank/alpha/scale = %d/%f/%f, want 8/16/2", cfg.Rank, cfg.Alpha, cfg.Scale)
	}

	cfg = NormalizeAdapterConfigForLoad(AdapterConfig{Rank: 4, Scale: 1.5})
	if cfg.Alpha != 6 || cfg.Scale != 1.5 {
		t.Fatalf("scale-derived load alpha/scale = %f/%f, want 6/1.5", cfg.Alpha, cfg.Scale)
	}
}

// TestNormalizeAdapterConfigForLoad_ScaleOnly_Good covers the load-defaulting
// arm where an adapter supplies only scale (no rank, no alpha).
// NormalizeAdapterConfig cannot derive alpha from scale without a rank, so the
// load defaulting fills rank to 8 first and then back-fills alpha as
// scale*rank — the `case cfg.Scale != 0` branch.
func TestNormalizeAdapterConfigForLoad_ScaleOnly_Good(t *testing.T) {
	cfg := NormalizeAdapterConfigForLoad(AdapterConfig{Scale: 3})
	if cfg.Rank != 8 || cfg.Alpha != 24 || cfg.Scale != 3 {
		t.Fatalf("scale-only load rank/alpha/scale = %d/%f/%f, want 8/24/3", cfg.Rank, cfg.Alpha, cfg.Scale)
	}
}
