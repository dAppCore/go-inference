// SPDX-Licence-Identifier: EUPL-1.2

package gptoss

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestGptOssRefusal mirrors composed.TestComposedMTPRefusal: gpt_oss is a REGISTERED model_type
// (model.LookupArch succeeds — a user pointing lem at a GPT-OSS checkpoint gets direction, not "unknown
// model architecture"), its real config parses cleanly AND its Weights mapping is populated (recognised +
// configured + weight-mapped), but deriving a decode Arch from it still refuses — engine/metal has no
// consumer yet for attention sinks or the o_proj/router/expert additive biases, so no test here claims
// generation works.
func TestGptOssRefusal(t *testing.T) {
	spec, ok := model.LookupArch("gpt_oss")
	if !ok {
		t.Fatal("model_type \"gpt_oss\" not registered — GPT-OSS must be a known arch, not fall to 'unknown model architecture'")
	}
	if spec.Parse == nil {
		t.Fatal("gpt_oss registered without a Parse func")
	}
	if spec.Weights.Embed == "" || spec.Weights.MoE.ExpGate == "" {
		t.Fatalf("gpt_oss registered without a Weights mapping: %+v", spec.Weights)
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
	_, err = ac.Arch()
	if err == nil {
		t.Fatal("expected a clean refusal, got a decode Arch — gpt_oss serving is not yet validated, so Arch must not claim it")
	}
	if !core.Contains(err.Error(), "self_attn.sinks") {
		t.Fatalf("refusal message %q must name the attention-sinks gap", err.Error())
	}
	if !core.Contains(err.Error(), "geometry resolves cleanly") {
		t.Fatalf("refusal message %q must confirm the geometry itself resolved (this is the real checkpoint config)", err.Error())
	}
}

func TestRegister_Empty_Ugly(t *testing.T) {
	if _, ok := model.LookupArch(""); ok {
		t.Fatal("empty model type unexpectedly registered")
	}
}
