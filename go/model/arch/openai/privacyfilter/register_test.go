// SPDX-Licence-Identifier: EUPL-1.2

package privacyfilter

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestPrivacyFilterRefusal mirrors composed.TestComposedMTPRefusal: openai_privacy_filter is a REGISTERED
// model_type (model.LookupArch succeeds — a user pointing lem at this checkpoint gets direction, not
// "unknown model architecture"), and its real config parses cleanly, but deriving a decode Arch from it
// refuses with a message that says WHY — it is a PII token-classifier, not a generative arch.
func TestPrivacyFilterRefusal(t *testing.T) {
	spec, ok := model.LookupArch("openai_privacy_filter")
	if !ok {
		t.Fatal("model_type \"openai_privacy_filter\" not registered — an OpenAI PII classifier must be a known arch, not fall to 'unknown model architecture'")
	}
	if spec.Parse == nil {
		t.Fatal("openai_privacy_filter registered without a Parse func")
	}
	data := core.ReadFile(core.PathJoin("testdata", "openai-privacy-filter-config.json"))
	if !data.OK {
		t.Fatal("read openai/privacy-filter fixture")
	}
	ac, err := spec.Parse(data.Value.([]byte))
	if err != nil {
		t.Fatalf("Parse: %v (config.json should parse cleanly — the refusal belongs to Arch, not Parse)", err)
	}
	if ac == nil {
		t.Fatal("Parse returned a nil ArchConfig alongside a nil error")
	}
	_, err = ac.Arch()
	if err == nil {
		t.Fatal("expected a clean refusal, got a decode Arch")
	}
	if !core.Contains(err.Error(), "token-classification") {
		t.Fatalf("refusal message %q must identify openai_privacy_filter as a token-classification model", err.Error())
	}
	if !core.Contains(err.Error(), "not a generative arch") {
		t.Fatalf("refusal message %q must explain it is not a generative arch", err.Error())
	}
}

func TestRegister_Empty_Ugly(t *testing.T) {
	if _, ok := model.LookupArch(""); ok {
		t.Fatal("empty model type unexpectedly registered")
	}
}
