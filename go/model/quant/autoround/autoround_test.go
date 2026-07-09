// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import (
	"encoding/binary"
	"math"
	"testing"
)

import core "dappco.re/go"

func TestAutoround_BuiltinProfiles_Good(t *testing.T) {
	profiles := BuiltinProfiles()
	if len(profiles) != 3 {
		t.Fatalf("BuiltinProfiles len = %d, want 3", len(profiles))
	}
	if profiles[0].ID != ProfileAutoRound || profiles[1].ID != ProfileAutoRoundBest || profiles[2].ID != ProfileAutoRoundLight {
		t.Fatalf("BuiltinProfiles ids = %q/%q/%q, want default/best/light order", profiles[0].ID, profiles[1].ID, profiles[2].ID)
	}
}

func TestAutoround_BuiltinProfiles_Bad(t *testing.T) {
	// BuiltinProfiles never errors, but it must return defensive copies: a
	// caller mutating the returned notes must not corrupt the builtin table.
	profiles := BuiltinProfiles()
	profiles[0].Notes[0] = "mutated"
	again := BuiltinProfiles()
	if again[0].Notes[0] == "mutated" {
		t.Fatal("BuiltinProfiles returned aliased notes; caller mutation leaked into the table")
	}
}

func TestAutoround_BuiltinProfiles_Ugly(t *testing.T) {
	// Degenerate read: every builtin profile must carry a resolvable scheme and
	// positive tuning knobs, so a downstream ConfigFromProfile can never divide
	// by a zero group size.
	for _, profile := range BuiltinProfiles() {
		if _, ok := ResolveScheme(profile.Scheme); !ok {
			t.Fatalf("BuiltinProfiles profile %q scheme %q does not resolve", profile.ID, profile.Scheme)
		}
		if profile.GroupSize <= 0 || profile.NSamples <= 0 || profile.SeqLen <= 0 {
			t.Fatalf("BuiltinProfiles profile %q has non-positive tuning knobs: %+v", profile.ID, profile)
		}
	}
}

func TestAutoround_LookupProfile_Good(t *testing.T) {
	profile, ok := LookupProfile(ProfileAutoRoundBest)
	if !ok {
		t.Fatal("LookupProfile(auto-round-best) ok = false")
	}
	if profile.Iters != 1000 || profile.NSamples != 512 || profile.SeqLen != 2048 {
		t.Fatalf("LookupProfile = %+v, want best tuning defaults", profile)
	}
}

func TestAutoround_LookupProfile_Bad(t *testing.T) {
	profile, ok := LookupProfile("missing")
	if ok {
		t.Fatal("LookupProfile(missing) ok = true, want false")
	}
	if profile.ID != "" {
		t.Fatalf("LookupProfile(missing) = %+v, want zero profile", profile)
	}
}

func TestAutoround_LookupProfile_Ugly(t *testing.T) {
	// Empty id is the degenerate lookup key: it must miss cleanly, never alias
	// the first builtin entry.
	if _, ok := LookupProfile(""); ok {
		t.Fatal("LookupProfile(\"\") ok = true, want clean miss")
	}
}

func TestAutoround_ConfigFromProfile_Good(t *testing.T) {
	profile, ok := LookupProfile(ProfileAutoRoundBest)
	if !ok {
		t.Fatal("LookupProfile(auto-round-best) ok = false")
	}
	cfg := ConfigFromProfile(profile)
	if cfg.Bits != 2 || cfg.GroupSize != 32 || !cfg.Symmetric {
		t.Fatalf("ConfigFromProfile = %+v, want W2A16 group 32 symmetric", cfg)
	}
	if cfg.Iters != 1000 || cfg.LearningRate != 5e-3 {
		t.Fatalf("ConfigFromProfile = %+v, want best iters and learning rate", cfg)
	}
}

func TestAutoround_ConfigFromProfile_Bad(t *testing.T) {
	// ConfigFromProfile derives bits from the resolved scheme, not from the
	// profile struct. A profile with an unknown scheme yields a zero-bits config
	// (GroupScheme returns the bare scheme), which must then be rejected by
	// QuantizeWeights' normalisation rather than silently quantising.
	cfg := ConfigFromProfile(Profile{Scheme: "bogus", GroupSize: 32})
	if cfg.Bits != 0 {
		t.Fatalf("ConfigFromProfile(unknown scheme) bits = %d, want 0", cfg.Bits)
	}
	if _, err := QuantizeWeights([]float32{1, 2}, cfg); err == nil {
		t.Fatal("QuantizeWeights(unknown-scheme config) err = nil, want unsupported scheme error")
	}
}

func TestAutoround_ConfigFromProfile_Ugly(t *testing.T) {
	// Zero-value profile: no scheme, no group size. GroupScheme cannot resolve,
	// so bits stays 0 and the learning rate is carried verbatim (0). The config
	// is intentionally non-quantisable until normalised.
	cfg := ConfigFromProfile(Profile{})
	if cfg.Bits != 0 || cfg.GroupSize != 0 {
		t.Fatalf("ConfigFromProfile(zero) = %+v, want zero bits and group size", cfg)
	}
}

func TestAutoround_Profile_GroupScheme_Good(t *testing.T) {
	profile := Profile{Scheme: SchemeW4A16, GroupSize: 64, Symmetric: true, ExportFormat: FormatAutoRound}
	info := profile.GroupScheme()
	if info.Scheme != SchemeW4A16 || info.Bits != 4 {
		t.Fatalf("GroupScheme = %+v, want W4A16 bits 4", info)
	}
	if info.GroupSize != 64 {
		t.Fatalf("GroupScheme group size = %d, want profile override 64", info.GroupSize)
	}
	if info.ExportFormat != FormatAutoRound {
		t.Fatalf("GroupScheme export format = %q, want auto_round", info.ExportFormat)
	}
}

func TestAutoround_Profile_GroupScheme_Bad(t *testing.T) {
	// An unresolvable scheme cannot produce bit-width metadata: GroupScheme
	// returns the bare scheme with zero bits rather than inventing a width.
	info := Profile{Scheme: "not-a-scheme"}.GroupScheme()
	if info.Scheme != "not-a-scheme" {
		t.Fatalf("GroupScheme(unknown) scheme = %q, want pass-through", info.Scheme)
	}
	if info.Bits != 0 {
		t.Fatalf("GroupScheme(unknown) bits = %d, want 0", info.Bits)
	}
}

func TestAutoround_Profile_GroupScheme_Ugly(t *testing.T) {
	// Profile with no group-size override must inherit the scheme's default
	// group size rather than collapsing to zero.
	info := Profile{Scheme: SchemeW4A16}.GroupScheme()
	if info.GroupSize != 128 {
		t.Fatalf("GroupScheme(no override) group size = %d, want scheme default 128", info.GroupSize)
	}
}

func TestAutoround_ResolveScheme_Good(t *testing.T) {
	w4, ok := ResolveScheme("w4a16")
	if !ok {
		t.Fatal("ResolveScheme(w4a16) ok = false")
	}
	if w4.Scheme != SchemeW4A16 || w4.Bits != 4 || w4.ActivationBits != 16 || w4.GroupSize != 128 {
		t.Fatalf("W4A16 info = %+v, want int4 weight-only defaults", w4)
	}
	gguf, ok := ResolveScheme("gguf:q4_k_m")
	if !ok {
		t.Fatal("ResolveScheme(gguf:q4_k_m) ok = false")
	}
	if gguf.Scheme != SchemeGGUFQ4KM || gguf.ExportFormat != FormatGGUFQ4KM || gguf.GroupSize != 256 {
		t.Fatalf("GGUF info = %+v, want Q4_K_M export defaults", gguf)
	}
}

func TestAutoround_ResolveScheme_Bad(t *testing.T) {
	info, ok := ResolveScheme("W3A16")
	if ok {
		t.Fatalf("ResolveScheme(W3A16) ok = true, want unsupported")
	}
	if info.Scheme != "" {
		t.Fatalf("ResolveScheme(W3A16) = %+v, want zero info", info)
	}
}

func TestAutoround_ResolveScheme_Ugly(t *testing.T) {
	// Mixed-case and whitespace-padded aliases must normalise to the same
	// canonical scheme — the loader sees these forms from third-party
	// quantisation configs.
	for _, alias := range []Scheme{"  w8a16  ", "W8A16", "w8a16"} {
		info, ok := ResolveScheme(alias)
		if !ok {
			t.Fatalf("ResolveScheme(%q) ok = false, want canonical W8A16", alias)
		}
		if info.Scheme != SchemeW8A16 || info.Bits != 8 {
			t.Fatalf("ResolveScheme(%q) = %+v, want W8A16 bits 8", alias, info)
		}
	}
	// The GGUF prefix has a dedicated normalisation branch that uppercases only
	// the segment after "gguf:" — a lower-case suffix must still resolve.
	info, ok := ResolveScheme("gguf:q4_k_m")
	if !ok || info.Scheme != SchemeGGUFQ4KM {
		t.Fatalf("ResolveScheme(gguf:q4_k_m) = %+v ok=%v, want GGUF Q4_K_M", info, ok)
	}
}

func TestAutoround_QuantizeWeights_Good(t *testing.T) {
	t.Run("RTN", func(t *testing.T) {
		weights := make([]float32, 32)
		for i := range weights {
			weights[i] = float32(i-16) / 8
		}
		got, err := QuantizeWeights(weights, QuantizeConfig{Scheme: SchemeW4A16, GroupSize: 32, Iters: 0})
		if err != nil {
			t.Fatalf("QuantizeWeights returned error: %v", err)
		}
		if got.Bits != 4 || got.GroupSize != 32 || got.Iters != 0 || len(got.QValues) != len(weights) || len(got.Scales) != 1 || len(got.Dequantized) != len(weights) {
			t.Fatalf("QuantizeWeights = %+v, want one W4A16 RTN group", got)
		}
	})
	t.Run("SignRound", func(t *testing.T) {
		weights := make([]float32, 32)
		weights[0] = 1.4
		weights[1] = 1.4
		weights[2] = 7
		gradients := make([]float32, len(weights))
		gradients[0] = 1
		gradients[1] = -1

		got, err := QuantizeWeights(weights, QuantizeConfig{
			Scheme:    SchemeW4A16,
			GroupSize: 32,
			Iters:     1,
			Gradients: gradients,
		})
		if err != nil {
			t.Fatalf("QuantizeWeights returned error: %v", err)
		}
		if got.QValues[0] != 1 || got.QValues[1] != 2 {
			t.Fatalf("qvalues[0:2] = %v, want sign-gradient floor/ceil split", got.QValues[:2])
		}
	})
}

func TestAutoround_QuantizeWeights_Bad(t *testing.T) {
	cases := []QuantizeConfig{
		{Bits: 5, GroupSize: 32},
		{Bits: 4, GroupSize: 16},
		{Scheme: "missing"},
		{Bits: 4, GroupSize: 32, Iters: -1},
		{Bits: 4, GroupSize: 32, Iters: 1, Gradients: []float32{1}},
	}
	for _, cfg := range cases {
		t.Run(string(cfg.Scheme), func(t *testing.T) {
			if _, err := QuantizeWeights([]float32{1, 2}, cfg); err == nil {
				t.Fatalf("QuantizeWeights(%+v) err = nil, want error", cfg)
			}
		})
	}
}

func TestAutoround_QuantizeWeights_Ugly(t *testing.T) {
	// Empty weight slice is the degenerate input: it must be rejected before any
	// group loop runs, never panic on weights[0].
	if _, err := QuantizeWeights(nil, QuantizeConfig{Scheme: SchemeW4A16, GroupSize: 32}); err == nil {
		t.Fatal("QuantizeWeights(nil) err = nil, want weights-required error")
	}
	// A single all-zero group has no dynamic range: the scale falls back to 1
	// and every value quantises to zero without dividing by zero.
	got, err := QuantizeWeights([]float32{0, 0, 0}, QuantizeConfig{Scheme: SchemeW4A16, GroupSize: 32})
	if err != nil {
		t.Fatalf("QuantizeWeights(zeros) error = %v", err)
	}
	if got.Scales[0] != 1 {
		t.Fatalf("QuantizeWeights(zeros) scale = %f, want fallback 1", got.Scales[0])
	}
	for i, q := range got.QValues {
		if q != 0 {
			t.Fatalf("QuantizeWeights(zeros) qvalue[%d] = %d, want 0", i, q)
		}
	}
}

// --- shared package-scoped test helpers (used across the autoround _test.go files) ---

func autoRoundTestProjection(name string, packed []byte, scales, zeroPoints, bias []float32) PackedProjection {
	tensor := PackTensor{
		Name:        name,
		Packed:      name + ".packed",
		Scales:      name + ".scales",
		ZeroPoints:  name + ".zeros",
		Shape:       []int32{1, 4},
		Bits:        2,
		GroupSize:   32,
		Symmetric:   true,
		PackedBytes: 1,
		Groups:      1,
		QMin:        -2,
		QMax:        1,
	}
	if len(bias) > 0 {
		tensor.Bias = name + ".bias"
	}
	return PackedProjection{
		Tensor: tensor,
		Weights: PackedWeights{
			Scheme:     SchemeW2A16,
			Format:     FormatAutoRound,
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
			Shape:      []int32{1, 4},
			Packed:     core.SliceClone(packed),
			Scales:     core.SliceClone(scales),
			ZeroPoints: core.SliceClone(zeroPoints),
			QMin:       -2,
			QMax:       1,
		},
		Bias: core.SliceClone(bias),
	}
}

type autoRoundSafetensorTensor struct {
	Name  string
	DType string
	Shape []int
	Raw   []byte
}

func autoRoundF32Tensor(name string, values []float32, shape ...int) autoRoundSafetensorTensor {
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	if len(shape) == 0 {
		shape = []int{len(values)}
	}
	return autoRoundSafetensorTensor{Name: name, DType: "F32", Shape: append([]int(nil), shape...), Raw: raw}
}

func writeAutoRoundSafetensors(t *testing.T, path string, tensors []autoRoundSafetensorTensor) {
	t.Helper()
	type entry struct {
		DType       string `json:"dtype"`
		Shape       []int  `json:"shape"`
		DataOffsets []int  `json:"data_offsets"`
	}
	header := map[string]entry{}
	var data []byte
	for _, tensor := range tensors {
		start := len(data)
		data = append(data, tensor.Raw...)
		header[tensor.Name] = entry{
			DType:       tensor.DType,
			Shape:       tensor.Shape,
			DataOffsets: []int{start, len(data)},
		}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal safetensors header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	out := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(headerBytes)))
	copy(out[8:], headerBytes)
	copy(out[8+len(headerBytes):], data)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		t.Fatalf("write safetensors: %v", result.Value)
	}
}

func assertAutoRoundFloat32SliceClose(t *testing.T, got, want []float32, epsilon float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len(got) = %d, want %d", len(got), len(want))
	}
	for i := range got {
		diff := got[i] - want[i]
		if diff < 0 {
			diff = -diff
		}
		if diff > epsilon {
			t.Fatalf("value[%d] = %f, want %f", i, got[i], want[i])
		}
	}
}
