// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestGptOssRegistered pins the full registration through the registry surface (the exact path
// model.Load walks): gpt_oss is a REGISTERED model_type, its real config parses cleanly, its
// Weights mapping (sinks suffix included) is populated, and Arch resolves to a full decode Arch —
// the former sinks/bias/attention-factor refusal is lifted (#37, all three rungs consumed).
func TestGptOssRegistered(t *testing.T) {
	spec, ok := model.LookupArch("gpt_oss")
	if !ok {
		t.Fatal("model_type \"gpt_oss\" not registered — GPT-OSS must be a known arch, not fall to 'unknown model architecture'")
	}
	if spec.Parse == nil {
		t.Fatal("gpt_oss registered without a Parse func")
	}
	if spec.Weights.Embed == "" || spec.Weights.MoE.ExpGate == "" || spec.Weights.Sinks == "" {
		t.Fatalf("gpt_oss registered without a complete Weights mapping (sinks included): %+v", spec.Weights)
	}
	data := core.ReadFile(core.PathJoin("testdata", "openai-gpt-oss-20b-config.json"))
	if !data.OK {
		t.Fatal("read openai/gpt-oss-20b fixture")
	}
	ac, err := spec.Parse(data.Value.([]byte))
	if err != nil {
		t.Fatalf("Parse: %v (config.json should parse cleanly — gpt_oss IS recognised + configured)", err)
	}
	if ac == nil {
		t.Fatal("Parse returned a nil ArchConfig alongside a nil error")
	}
	a, err := ac.Arch()
	if err != nil {
		t.Fatalf("Arch: %v — the #37 boundary is lifted, the real config must resolve", err)
	}
	if len(a.Layer) != 24 || a.Experts != 32 {
		t.Fatalf("Arch via the registry = %d layers, %d experts — want 24/32", len(a.Layer), a.Experts)
	}
}

func TestRegister_Empty_Ugly(t *testing.T) {
	if _, ok := model.LookupArch(""); ok {
		t.Fatal("empty model type unexpectedly registered")
	}
}
