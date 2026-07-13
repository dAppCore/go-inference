// SPDX-Licence-Identifier: EUPL-1.2

package engine

import (
	"slices"
	"testing"

	"dappco.re/go/inference"
)

func ptrF32(v float32) *float32 { return &v }
func ptrInt(v int) *int         { return &v }
func ptrBool(v bool) *bool      { return &v }

// TestParseGenerationConfigSampling_Good pins the shape every cached
// mlx-community gemma4 snapshot ships (e.g. gemma-4-E4B-it-qat-4bit): do_sample
// true, temperature 1.0, top_k 64, top_p 0.95, no suppress_tokens or min_p key
// at all — those two come back nil ("the file said nothing").
func TestParseGenerationConfigSampling_Good(t *testing.T) {
	got := ParseGenerationConfigSampling([]byte(`{
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
	if got.MinP != nil {
		t.Fatalf("MinP = %v, want nil (key absent from gemma4 config)", got.MinP)
	}
}

// TestParseGenerationConfigSampling_MinP pins the shape the cached Qwen3.5
// snapshots ship (e.g. Qwen3.5-4B-OptiQ-4bit): min_p present as 0.0 — a declared
// key whose value is the zero, so the pointer is non-nil (declared) even though
// the value is disabled. This is the field that most exercises the
// declared-vs-zero distinction on the model side.
func TestParseGenerationConfigSampling_MinP(t *testing.T) {
	got := ParseGenerationConfigSampling([]byte(`{
		"do_sample": true,
		"temperature": 0.7,
		"top_p": 0.8,
		"top_k": 20,
		"min_p": 0.0,
		"repetition_penalty": 1.0,
		"presence_penalty": 1.5
	}`))
	if got.MinP == nil {
		t.Fatal("MinP = nil, want a non-nil pointer to 0.0 (the key is declared)")
	}
	if *got.MinP != 0.0 {
		t.Fatalf("MinP = %v, want 0.0", *got.MinP)
	}
	if got.Temperature == nil || *got.Temperature != 0.7 {
		t.Fatalf("Temperature = %v, want 0.7", got.Temperature)
	}
	if got.TopK == nil || *got.TopK != 20 {
		t.Fatalf("TopK = %v, want 20", got.TopK)
	}
}

// TestParseGenerationConfigSampling_Bad pins the guard paths: malformed JSON and
// a wholly empty object both return the zero value (every field absent).
func TestParseGenerationConfigSampling_Bad(t *testing.T) {
	for _, data := range []string{`not json`, `{}`} {
		got := ParseGenerationConfigSampling([]byte(data))
		if got.DoSample != nil || got.Temperature != nil || got.TopP != nil || got.TopK != nil || got.MinP != nil || got.SuppressTokens != nil {
			t.Fatalf("sampling(%q) = %+v, want the zero value", data, got)
		}
	}
}

// TestParseGenerationConfigSampling_Ugly pins the real shape a gemma4 large
// variant ships (gemma-4-12B-it-assistant-bf16): a SCALAR eos_token_id
// (unrelated to this parser, but present in the real file) alongside a
// suppress_tokens array — the one cached snapshot that actually declares it.
func TestParseGenerationConfigSampling_Ugly(t *testing.T) {
	got := ParseGenerationConfigSampling([]byte(`{
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

// gemma4Declared mirrors a gemma4 checkpoint's generation_config sampling
// block (temperature 1.0, top_k 64, top_p 0.95, do_sample true) — the fixture
// the precedence table folds against.
var gemma4Declared = SamplingDefaults{
	DoSample:    ptrBool(true),
	Temperature: ptrF32(1.0),
	TopP:        ptrF32(0.95),
	TopK:        ptrInt(64),
}

// TestSamplingDefaults_Apply_Good is the per-field precedence table (request-set
// > model-declared > engine fallback) for the shared fold both the plain and
// speculative decode paths call. Each scalar is read through its *Set flag, so
// an explicit zero (greedy/disabled) is honoured while an unset field takes the
// declared default.
func TestSamplingDefaults_Apply_Good(t *testing.T) {
	tests := []struct {
		name                         string
		declared                     SamplingDefaults
		in                           inference.GenerateConfig
		wantTemp, wantTopP, wantMinP float32
		wantTopK                     int
	}{
		{
			name:     "unset request takes every declared default",
			declared: gemma4Declared,
			in:       inference.GenerateConfig{},
			wantTemp: 1.0, wantTopP: 0.95, wantTopK: 64, wantMinP: 0,
		},
		{
			name:     "explicit non-zero request wins over declared",
			declared: gemma4Declared,
			in:       inference.GenerateConfig{Temperature: 0.3, TemperatureSet: true, TopK: 40, TopKSet: true, TopP: 0.8, TopPSet: true},
			wantTemp: 0.3, wantTopP: 0.8, wantTopK: 40, wantMinP: 0,
		},
		{
			name:     "explicit zero request stays greedy/disabled (the flag's whole point)",
			declared: gemma4Declared,
			in:       inference.GenerateConfig{TemperatureSet: true, TopKSet: true, TopPSet: true, MinPSet: true},
			wantTemp: 0, wantTopP: 0, wantTopK: 0, wantMinP: 0,
		},
		{
			name:     "declared min_p 0.0 folds as a no-op on an unset request",
			declared: SamplingDefaults{MinP: ptrF32(0)},
			in:       inference.GenerateConfig{},
			wantTemp: 0, wantTopP: 0, wantTopK: 0, wantMinP: 0,
		},
		{
			name:     "declared min_p > 0 folds onto an unset request",
			declared: SamplingDefaults{MinP: ptrF32(0.05)},
			in:       inference.GenerateConfig{},
			wantTemp: 0, wantTopP: 0, wantTopK: 0, wantMinP: 0.05,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.declared.Apply(tt.in)
			if got.Temperature != tt.wantTemp || got.TopP != tt.wantTopP || got.TopK != tt.wantTopK || got.MinP != tt.wantMinP {
				t.Fatalf("Apply(%+v) = temp %v topP %v topK %v minP %v, want %v/%v/%v/%v",
					tt.in, got.Temperature, got.TopP, got.TopK, got.MinP, tt.wantTemp, tt.wantTopP, tt.wantTopK, tt.wantMinP)
			}
		})
	}
}

// TestSamplingDefaults_Apply_Bad pins the empty-declarer path: a zero
// SamplingDefaults (no generation_config, or one without a sampling block)
// leaves the request untouched, byte-for-byte — the no-config behaviour pin.
func TestSamplingDefaults_Apply_Bad(t *testing.T) {
	in := inference.GenerateConfig{Temperature: 0.9, TemperatureSet: true, TopK: 40}
	got := (SamplingDefaults{}).Apply(in)
	if got.Temperature != 0.9 || got.TopK != 40 || !got.TemperatureSet {
		t.Fatalf("empty declarer Apply(%+v) = %+v, want unchanged", in, got)
	}
}

// TestSamplingDefaults_Apply_Ugly pins the two non-scalar precedence paths in
// one place: suppress_tokens folds only onto an empty request list, and
// do_sample is carried but never folded (it has no GenerateConfig counterpart).
func TestSamplingDefaults_Apply_Ugly(t *testing.T) {
	declared := SamplingDefaults{DoSample: ptrBool(false), SuppressTokens: []int32{7, 8}}

	// request with its own suppress list wins; do_sample is ignored throughout.
	reqSet := declared.Apply(inference.GenerateConfig{SuppressTokens: []int32{1}})
	if !slices.Equal(reqSet.SuppressTokens, []int32{1}) {
		t.Fatalf("request suppress = %v, want [1]", reqSet.SuppressTokens)
	}

	// empty request suppress list takes the declared list.
	unset := declared.Apply(inference.GenerateConfig{})
	if !slices.Equal(unset.SuppressTokens, []int32{7, 8}) {
		t.Fatalf("unset suppress = %v, want declared [7 8]", unset.SuppressTokens)
	}
	// do_sample false did NOT force Temperature or any scalar to change.
	if unset.Temperature != 0 || unset.TopK != 0 || unset.TopP != 0 || unset.MinP != 0 {
		t.Fatalf("do_sample fold leaked into scalars: %+v", unset)
	}
}
