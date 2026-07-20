// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// TestConfig_Arch_Good pins the documented "happy path" for an always-refuses arch: a
// realistic, dimensionally-valid published jetmoe-8b config still refuses, but with the
// MoA-specific message (not a dimension-guard error) — #59 item 6's restored named refusal.
func TestConfig_Arch_Good(t *testing.T) {
	data, err := coreio.Local.Read(core.PathJoin("testdata", "jetmoe-jetmoe-8b-config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg Config
	if r := core.JSONUnmarshal([]byte(data), &cfg); !r.OK {
		t.Fatal(r.Error())
	}
	_, err = cfg.Arch()
	if err == nil {
		t.Fatal("Arch: expected a clean MoA refusal, got a resolved architecture")
	}
	if !core.Contains(err.Error(), "Mixture-of-Attention (MoA) requires routed query/output attention projections with shared KV") {
		t.Fatalf("Arch refusal %q must name the MoA gap", err.Error())
	}
}

// TestConfig_Arch_Bad proves an incomplete config fails the dimension guard — the error
// propagated is the dimension-guard message, NOT the MoA refusal _Good and _Ugly both reach
// once dimensions validate.
func TestConfig_Arch_Bad(t *testing.T) {
	_, err := (Config{HiddenSize: 8}).Arch()
	if err == nil {
		t.Fatal("incomplete architecture accepted")
	}
	if core.Contains(err.Error(), "Mixture-of-Attention") {
		t.Fatal("Arch: an incomplete config must fail the dimension guard, not reach the MoA refusal")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, FFNHiddenSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, KVChannels: 4, MoENumExperts: 2, MoETopK: 3, VocabSize: 16}
	_, err := cfg.Arch()
	if err == nil {
		t.Fatal("top-k greater than expert count accepted")
	}
	if core.Contains(err.Error(), "Mixture-of-Attention") {
		t.Fatal("Arch: an invalid top-k must fail its own guard, not reach the MoA refusal")
	}
}

func TestConfig_InferFromWeights_Good(t *testing.T) {
	cfg := Config{HiddenSize: 8}
	cfg.InferFromWeights(nil)
	if cfg.HiddenSize != 8 {
		t.Fatalf("InferFromWeights changed config: %+v", cfg)
	}
}

func TestConfig_InferFromWeights_Bad(t *testing.T) {
	cfg := Config{}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("empty config became valid after InferFromWeights")
	}
}

// TestConfig_InferFromWeights_Ugly proves the no-op does not paper over the
// top-k-exceeds-experts guard — distinct from _Bad's all-zero rejection.
func TestConfig_InferFromWeights_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, FFNHiddenSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, KVChannels: 4, MoENumExperts: 2, MoETopK: 3, VocabSize: 16}
	cfg.InferFromWeights(nil)
	if _, err := cfg.Arch(); err == nil {
		t.Fatal("top-k greater than expert count became valid after InferFromWeights")
	}
}
