// SPDX-Licence-Identifier: EUPL-1.2

package whisper

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model"
)

// TestWhisperRefusal mirrors composed.TestComposedMTPRefusal: whisper is a REGISTERED model_type
// (model.LookupArch succeeds — a user pointing lem at a Whisper checkpoint gets direction, not "unknown
// model architecture"), and its real config parses cleanly, but deriving a decode Arch from it refuses with
// a message that says WHY — it is an ASR encoder-decoder, not a decoder-only causal-LM.
func TestWhisperRefusal(t *testing.T) {
	spec, ok := model.LookupArch("whisper")
	if !ok {
		t.Fatal("model_type \"whisper\" not registered — Whisper must be a known arch, not fall to 'unknown model architecture'")
	}
	if spec.Parse == nil {
		t.Fatal("whisper registered without a Parse func")
	}
	data := core.ReadFile(core.PathJoin("testdata", "openai-whisper-tiny-config.json"))
	if !data.OK {
		t.Fatal("read openai/whisper-tiny fixture")
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
	if !core.Contains(err.Error(), "encoder-decoder") {
		t.Fatalf("refusal message %q must identify whisper as an encoder-decoder ASR model", err.Error())
	}
	if !core.Contains(err.Error(), "not yet implemented") {
		t.Fatalf("refusal message %q must say the ASR forward is not yet implemented", err.Error())
	}
}

func TestRegister_Empty_Ugly(t *testing.T) {
	if _, ok := model.LookupArch(""); ok {
		t.Fatal("empty model type unexpectedly registered")
	}
}
