// SPDX-Licence-Identifier: EUPL-1.2

//go:build linux && amd64 && !rocm_legacy_server

package hip

import (
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/engine"
)

// writeHipGenerationConfig writes a generation_config.json fixture into dir and
// fails the test if the write does not land — the CPU-only seam that exercises
// hip's file-locating shell without a HIP device or a loaded model.
func writeHipGenerationConfig(t *testing.T, dir, body string) {
	t.Helper()
	write := core.WriteFile(core.PathJoin(dir, "generation_config.json"), []byte(body), 0o644)
	if !write.OK {
		t.Fatalf("write generation_config.json fixture: %v", write.Value)
	}
}

// TestLoadGenerationConfigSamplingDefaults_Good pins that hip reads the
// checkpoint's generation_config.json and delegates the parse to the shared
// engine parser: a gemma4-large fixture (scalar eos_token_id beside a
// suppress_tokens array) comes back with its declared sampling fields.
func TestLoadGenerationConfigSamplingDefaults_Good(t *testing.T) {
	dir := t.TempDir()
	writeHipGenerationConfig(t, dir, `{
		"bos_token_id": 2,
		"do_sample": true,
		"eos_token_id": 1,
		"suppress_tokens": [258883, 258882],
		"temperature": 1.0,
		"top_k": 64,
		"top_p": 0.95
	}`)
	got := loadGenerationConfigSamplingDefaults(dir)
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
	if len(got.SuppressTokens) != 2 || got.SuppressTokens[0] != 258883 || got.SuppressTokens[1] != 258882 {
		t.Fatalf("SuppressTokens = %v, want [258883 258882]", got.SuppressTokens)
	}
}

// TestLoadGenerationConfigSamplingDefaults_Bad pins the soft-optional path: an
// absent generation_config.json returns the zero value rather than an error, so
// a checkpoint without one keeps the engine's derived defaults.
func TestLoadGenerationConfigSamplingDefaults_Bad(t *testing.T) {
	got := loadGenerationConfigSamplingDefaults(t.TempDir())
	if got.DoSample != nil || got.Temperature != nil || got.TopP != nil || got.TopK != nil || got.MinP != nil || got.SuppressTokens != nil {
		t.Fatalf("absent file sampling = %+v, want the zero value", got)
	}
}

// TestHipTokenModel_DeclaredSamplingDefaults pins the capability method
// (engine.SamplingDefaultsDeclarer): a nil receiver reports the zero value
// rather than panicking, and a model passes its parsed field through unchanged.
func TestHipTokenModel_DeclaredSamplingDefaults(t *testing.T) {
	var nilModel *hipTokenModel
	if got := nilModel.DeclaredSamplingDefaults(); got.DoSample != nil || got.SuppressTokens != nil {
		t.Fatalf("nil receiver = %+v, want the zero value", got)
	}
	temp := float32(0.7)
	m := &hipTokenModel{declaredSampling: engine.SamplingDefaults{Temperature: &temp, SuppressTokens: []int32{5}}}
	got := m.DeclaredSamplingDefaults()
	if got.Temperature == nil || *got.Temperature != 0.7 {
		t.Fatalf("Temperature = %v, want 0.7", got.Temperature)
	}
	if len(got.SuppressTokens) != 1 || got.SuppressTokens[0] != 5 {
		t.Fatalf("SuppressTokens = %v, want [5]", got.SuppressTokens)
	}
}
