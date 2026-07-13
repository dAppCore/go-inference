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
