// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import "testing"

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
