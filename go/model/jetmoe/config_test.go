// SPDX-Licence-Identifier: EUPL-1.2

package jetmoe

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func TestConfig_Arch_Good(t *testing.T) {
	data, err := coreio.Local.Read(core.PathJoin("testdata", "jetmoe-jetmoe-8b-config.json"))
	if err != nil {
		t.Fatal(err)
	}
	var cfg Config
	if r := core.JSONUnmarshal([]byte(data), &cfg); !r.OK {
		t.Fatal(r.Error())
	}
	arch, err := cfg.Arch()
	if err != nil {
		t.Fatal(err)
	}
	if arch.Hidden != 2048 || arch.Heads != 32 || arch.KVHeads != 16 || arch.HeadDim != 128 || arch.RotaryDim != 128 {
		t.Fatalf("attention geometry = hidden %d heads %d/%d head/rotary %d/%d", arch.Hidden, arch.Heads, arch.KVHeads, arch.HeadDim, arch.RotaryDim)
	}
	if arch.Experts != 8 || arch.TopK != 2 || arch.ExpertFF != 5632 || !arch.NormaliseMoETopK || arch.SharedExperts != 0 {
		t.Fatalf("MoE geometry = experts %d top-k %d expert FF %d normalise %v shared %d", arch.Experts, arch.TopK, arch.ExpertFF, arch.NormaliseMoETopK, arch.SharedExperts)
	}
}

func TestConfig_Arch_Bad(t *testing.T) {
	_, err := (Config{HiddenSize: 8}).Arch()
	if err == nil {
		t.Fatal("incomplete architecture accepted")
	}
}

func TestConfig_Arch_Ugly(t *testing.T) {
	cfg := Config{HiddenSize: 8, FFNHiddenSize: 4, NumHiddenLayers: 1, NumAttentionHeads: 2, NumKeyValueHeads: 1, KVChannels: 4, MoENumExperts: 2, MoETopK: 3, VocabSize: 16}
	_, err := cfg.Arch()
	if err == nil {
		t.Fatal("top-k greater than expert count accepted")
	}
}
