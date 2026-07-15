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

// The byte-level parser table (Good / MinP / Bad / scalar-eos Ugly) now lives
// beside the shared engine.SamplingDefaults it produces —
// engine.ParseGenerationConfigSampling, tested in engine/sampling_defaults_test.go.
// engine/metal keeps the file-locating shell (loadGenerationConfigSamplingDefaults)
// and the capability method below, which are the metal-specific seams.

// TestLoadGenerationConfigSamplingDefaults pins the file path: an absent file
// returns the zero value rather than an error (soft-optional, mirroring
// TestLoadGenerationConfigStops).
func TestLoadGenerationConfigSamplingDefaults(t *testing.T) {
	got := loadGenerationConfigSamplingDefaults(t.TempDir())
	if got.DoSample != nil || got.Temperature != nil || got.TopP != nil || got.TopK != nil || got.MinP != nil || got.SuppressTokens != nil {
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
