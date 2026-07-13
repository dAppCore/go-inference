// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	"testing"

	"dappco.re/go/inference/engine"
)

// TestGenerationConfigStops_Good pins the array form gemma4 ships:
// eos_token_id [1 <eos>, 106 <turn|>, 50 <|tool_response>].
func TestGenerationConfigStops_Good(t *testing.T) {
	ids := generationConfigStops([]byte(`{"eos_token_id":[1,106,50],"temperature":1.0}`))
	if len(ids) != 3 || ids[0] != 1 || ids[1] != 106 || ids[2] != 50 {
		t.Fatalf("stops = %v, want [1 106 50]", ids)
	}
}

// TestGenerationConfigStops_Bad pins the guard paths: malformed JSON, a
// missing field, and a non-numeric array all return nil (the engine's derived
// defaults still apply).
func TestGenerationConfigStops_Bad(t *testing.T) {
	for _, data := range []string{`not json`, `{}`, `{"eos_token_id":["x"]}`, `{"eos_token_id":null}`} {
		if got := generationConfigStops([]byte(data)); got != nil {
			t.Fatalf("stops(%q) = %v, want nil", data, got)
		}
	}
}

// TestGenerationConfigStops_Ugly pins the scalar form some checkpoints ship
// (a single integer rather than an array).
func TestGenerationConfigStops_Ugly(t *testing.T) {
	ids := generationConfigStops([]byte(`{"eos_token_id":1}`))
	if len(ids) != 1 || ids[0] != 1 {
		t.Fatalf("stops = %v, want [1]", ids)
	}
}

// TestLoadGenerationConfigStops pins the file path: an absent file returns
// nil rather than an error (soft-optional, like processor_config.json).
func TestLoadGenerationConfigStops(t *testing.T) {
	if got := loadGenerationConfigStops(t.TempDir()); got != nil {
		t.Fatalf("absent file stops = %v, want nil", got)
	}
}

// TestGenerationConfigSamplingDefaults_Good pins the shape every cached
// mlx-community gemma4 snapshot ships (e.g. gemma-4-E4B-it-qat-4bit): do_sample
// true, temperature 1.0, top_k 64, top_p 0.95, no suppress_tokens key at all.
func TestGenerationConfigSamplingDefaults_Good(t *testing.T) {
	got := generationConfigSamplingDefaults([]byte(`{
		"bos_token_id": 2,
		"do_sample": true,
		"eos_token_id": [1, 106, 50],
		"pad_token_id": 0,
		"temperature": 1.0,
		"top_k": 64,
		"top_p": 0.95,
		"transformers_version": "5.6.2"
	}`))
	if got.DoSample == nil || *got.DoSample != true {
		t.Fatalf("DoSample = %v, want true", got.DoSample)
	}
	if got.Temperature == nil || *got.Temperature != 1.0 {
		t.Fatalf("Temperature = %v, want 1.0", got.Temperature)
	}
	if got.TopK == nil || *got.TopK != 64 {
		t.Fatalf("TopK = %v, want 64", got.TopK)
	}
	if got.TopP == nil || *got.TopP != 0.95 {
		t.Fatalf("TopP = %v, want 0.95", got.TopP)
	}
	if got.SuppressTokens != nil {
		t.Fatalf("SuppressTokens = %v, want nil (key absent)", got.SuppressTokens)
	}
}

// TestGenerationConfigSamplingDefaults_Bad pins the guard paths: malformed
// JSON and a wholly empty object both return the zero value (every field
// absent), matching generationConfigStops' guard behaviour.
func TestGenerationConfigSamplingDefaults_Bad(t *testing.T) {
	for _, data := range []string{`not json`, `{}`} {
		got := generationConfigSamplingDefaults([]byte(data))
		if got.DoSample != nil || got.Temperature != nil || got.TopP != nil || got.TopK != nil || got.SuppressTokens != nil {
			t.Fatalf("sampling(%q) = %+v, want the zero value", data, got)
		}
	}
}

// TestGenerationConfigSamplingDefaults_Ugly pins the real shape a gemma4 large
// variant ships (gemma-4-12B-it-assistant-bf16): a SCALAR eos_token_id
// (unrelated to this parser, but present in the real file) alongside a
// suppress_tokens array — the one cached snapshot that actually declares it.
func TestGenerationConfigSamplingDefaults_Ugly(t *testing.T) {
	got := generationConfigSamplingDefaults([]byte(`{
		"bos_token_id": 2,
		"do_sample": true,
		"eos_token_id": 1,
		"pad_token_id": 0,
		"suppress_tokens": [258883, 258882],
		"temperature": 1.0,
		"top_k": 64,
		"top_p": 0.95,
		"transformers_version": "5.10.0.dev0"
	}`))
	if got.DoSample == nil || *got.DoSample != true {
		t.Fatalf("DoSample = %v, want true", got.DoSample)
	}
	if len(got.SuppressTokens) != 2 || got.SuppressTokens[0] != 258883 || got.SuppressTokens[1] != 258882 {
		t.Fatalf("SuppressTokens = %v, want [258883 258882]", got.SuppressTokens)
	}
}

// TestLoadGenerationConfigSamplingDefaults pins the file path: an absent file
// returns the zero value rather than an error (soft-optional, mirroring
// TestLoadGenerationConfigStops).
func TestLoadGenerationConfigSamplingDefaults(t *testing.T) {
	got := loadGenerationConfigSamplingDefaults(t.TempDir())
	if got.DoSample != nil || got.Temperature != nil || got.TopP != nil || got.TopK != nil || got.SuppressTokens != nil {
		t.Fatalf("absent file sampling = %+v, want the zero value", got)
	}
}

// TestNativeTokenModel_DeclaredSamplingDefaults pins the capability method
// (engine.SamplingDefaultsDeclarer): a nil receiver reports the zero value
// rather than panicking, and a loaded model passes its parsed field through
// unchanged.
func TestNativeTokenModel_DeclaredSamplingDefaults(t *testing.T) {
	var nilModel *NativeTokenModel
	if got := nilModel.DeclaredSamplingDefaults(); got.DoSample != nil || got.SuppressTokens != nil {
		t.Fatalf("nil receiver = %+v, want the zero value", got)
	}
	temp := float32(0.7)
	m := &NativeTokenModel{declaredSampling: engine.SamplingDefaults{Temperature: &temp, SuppressTokens: []int32{5}}}
	got := m.DeclaredSamplingDefaults()
	if got.Temperature == nil || *got.Temperature != 0.7 {
		t.Fatalf("Temperature = %v, want 0.7", got.Temperature)
	}
	if len(got.SuppressTokens) != 1 || got.SuppressTokens[0] != 5 {
		t.Fatalf("SuppressTokens = %v, want [5]", got.SuppressTokens)
	}
}
