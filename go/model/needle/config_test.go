// SPDX-Licence-Identifier: EUPL-1.2

package needle

import "testing"

// TestConfig_DefaultConfig_Dims pins the published Needle geometry the reference
// depends on: 512-d, 8 query / 4 kv heads (head_dim 64), 12 encoder / 8 decoder
// layers, vocab 8192.
func TestConfig_DefaultConfig_Dims(t *testing.T) {
	c := DefaultConfig()
	if c.HeadDim() != 64 {
		t.Errorf("HeadDim = %d, want 64", c.HeadDim())
	}
	if c.KVDim() != 256 {
		t.Errorf("KVDim = %d, want 256", c.KVDim())
	}
	if c.NumEncoderLayers != 12 || c.NumDecoderLayers != 8 {
		t.Errorf("layers = %d/%d, want 12/8", c.NumEncoderLayers, c.NumDecoderLayers)
	}
	if c.EosTokenID != 1 || c.ToolsTokenID != 5 {
		t.Errorf("special ids eos=%d tools=%d, want 1/5", c.EosTokenID, c.ToolsTokenID)
	}
}

// TestConfig_LoadConfig_RealCheckpoint reads the real config.json and confirms it
// matches the published geometry (and that a present file overrides defaults).
func TestConfig_LoadConfig_RealCheckpoint(t *testing.T) {
	cfg, err := LoadConfig(snapshotDir)
	if err != nil {
		t.Fatalf("LoadConfig: %v", err)
	}
	if cfg.HiddenSize != 512 || cfg.VocabSize != 8192 {
		t.Errorf("hidden/vocab = %d/%d, want 512/8192", cfg.HiddenSize, cfg.VocabSize)
	}
	if cfg.NumKVHeads != 4 || cfg.NumHeads != 8 {
		t.Errorf("heads = %d/%d, want 8/4", cfg.NumHeads, cfg.NumKVHeads)
	}
}

// TestConfig_LoadConfig_MissingIsDefault confirms a directory without config.json
// yields DefaultConfig rather than an error — a bare weights+tokenizer pack runs.
func TestConfig_LoadConfig_MissingIsDefault(t *testing.T) {
	cfg, err := LoadConfig(t.TempDir())
	if err != nil {
		t.Fatalf("LoadConfig(empty dir): %v", err)
	}
	if cfg != DefaultConfig() {
		t.Errorf("missing config.json did not fall back to DefaultConfig")
	}
}
