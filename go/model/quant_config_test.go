// SPDX-Licence-Identifier: EUPL-1.2

package model

import "testing"

// TestNormalizeQuantizationMode covers the mode normaliser: mixed case + surrounding
// whitespace lowercase-trims to the declared mode, and an absent mode ("") defaults to
// "affine" — the mode every real gemma4 mlx-community checkpoint (26B-A4B, 31B) declares.
func TestNormalizeQuantizationMode(t *testing.T) {
	cases := []struct{ in, want string }{
		{"affine", "affine"},
		{"AFFINE", "affine"},
		{"  Affine  ", "affine"},
		{"mxfp4", "mxfp4"},
		{"MXFP8", "mxfp8"},
		{"nvfp4", "nvfp4"},
		{"", "affine"}, // absent mode → the affine default
	}
	for _, c := range cases {
		if got := NormalizeQuantizationMode(c.in); got != c.want {
			t.Fatalf("NormalizeQuantizationMode(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestQuantConfigValidateNil covers the nil-receiver case: a bf16 checkpoint (no
// quantization block at all) is valid — Validate must not panic or reject a nil *QuantConfig.
func TestQuantConfigValidateNil(t *testing.T) {
	var q *QuantConfig
	if err := q.Validate(); err != nil {
		t.Fatalf("nil QuantConfig.Validate() = %v, want nil (bf16 has no quant block)", err)
	}
}

// TestQuantConfigValidateAffine covers the affine mode's bit-width whitelist — the mode
// every gemma4 mlx-community checkpoint declares (26B-A4B and 31B both ship
// "mode":"affine"). 0 ("the model's config declares it") and every supported width pass;
// anything else is rejected.
func TestQuantConfigValidateAffine(t *testing.T) {
	for _, bits := range []int{0, 2, 3, 4, 5, 6, 8} {
		q := &QuantConfig{Mode: "affine", Bits: bits, GroupSize: 64}
		if err := q.Validate(); err != nil {
			t.Fatalf("affine bits=%d: Validate() = %v, want nil", bits, err)
		}
	}
	for _, bits := range []int{1, 7, 9, 16} {
		q := &QuantConfig{Mode: "affine", Bits: bits, GroupSize: 64}
		if err := q.Validate(); err == nil {
			t.Fatalf("affine bits=%d: expected an error, got nil", bits)
		}
	}
}

// TestQuantConfigValidateRealCheckpointGeometry pins that the actual mlx-community 26B-A4B
// and 31B checkpoints' declared quantization block (group_size 64, bits 4, mode affine)
// validates clean — the shape every gemma4 QAT-4bit release in the wild ships.
func TestQuantConfigValidateRealCheckpointGeometry(t *testing.T) {
	q := &QuantConfig{Mode: "affine", GroupSize: 64, Bits: 4}
	if err := q.Validate(); err != nil {
		t.Fatalf("real gemma4 26B-A4B/31B quant geometry (affine/64/4): Validate() = %v, want nil", err)
	}
}

// TestQuantConfigValidateEmptyModeDefaultsAffine covers Validate()'s effective default: a
// quant block with no "mode" field normalises to affine before the bits check runs (the
// zero-value Mode a JSON block omitting "mode" would leave).
func TestQuantConfigValidateEmptyModeDefaultsAffine(t *testing.T) {
	if err := (&QuantConfig{Bits: 4, GroupSize: 64}).Validate(); err != nil {
		t.Fatalf("empty mode defaults to affine: Validate() = %v, want nil", err)
	}
	if err := (&QuantConfig{Bits: 7, GroupSize: 64}).Validate(); err == nil {
		t.Fatal("empty mode defaults to affine, bits=7 unsupported: expected an error")
	}
}

// TestQuantConfigValidateMXFP4 covers the mxfp4 mode's fixed geometry: group_size must be 32
// (or undeclared/0), bits must be 4 (or undeclared/0).
func TestQuantConfigValidateMXFP4(t *testing.T) {
	good := []QuantConfig{
		{Mode: "mxfp4", GroupSize: 32, Bits: 4},
		{Mode: "mxfp4", GroupSize: 0, Bits: 0}, // undeclared geometry passes
	}
	for _, q := range good {
		if err := q.Validate(); err != nil {
			t.Fatalf("mxfp4 %+v: Validate() = %v, want nil", q, err)
		}
	}
	bad := []QuantConfig{
		{Mode: "mxfp4", GroupSize: 64, Bits: 4},
		{Mode: "mxfp4", GroupSize: 32, Bits: 8},
	}
	for _, q := range bad {
		if err := q.Validate(); err == nil {
			t.Fatalf("mxfp4 %+v: expected an error, got nil", q)
		}
	}
}

// TestQuantConfigValidateMXFP8 covers the mxfp8 mode's fixed geometry: group_size 32, bits 8.
func TestQuantConfigValidateMXFP8(t *testing.T) {
	if err := (&QuantConfig{Mode: "mxfp8", GroupSize: 32, Bits: 8}).Validate(); err != nil {
		t.Fatalf("mxfp8 valid geometry: Validate() = %v, want nil", err)
	}
	if err := (&QuantConfig{Mode: "mxfp8", GroupSize: 16, Bits: 8}).Validate(); err == nil {
		t.Fatal("mxfp8 wrong group_size: expected an error, got nil")
	}
	if err := (&QuantConfig{Mode: "mxfp8", GroupSize: 32, Bits: 4}).Validate(); err == nil {
		t.Fatal("mxfp8 wrong bits: expected an error, got nil")
	}
}

// TestQuantConfigValidateNVFP4 covers the nvfp4 mode's fixed geometry: group_size 16, bits 4.
func TestQuantConfigValidateNVFP4(t *testing.T) {
	if err := (&QuantConfig{Mode: "nvfp4", GroupSize: 16, Bits: 4}).Validate(); err != nil {
		t.Fatalf("nvfp4 valid geometry: Validate() = %v, want nil", err)
	}
	if err := (&QuantConfig{Mode: "nvfp4", GroupSize: 32, Bits: 4}).Validate(); err == nil {
		t.Fatal("nvfp4 wrong group_size: expected an error, got nil")
	}
	if err := (&QuantConfig{Mode: "nvfp4", GroupSize: 16, Bits: 8}).Validate(); err == nil {
		t.Fatal("nvfp4 wrong bits: expected an error, got nil")
	}
}

// TestQuantConfigValidateUnsupportedMode covers the default case: a mode string that is none
// of affine/mxfp4/mxfp8/nvfp4 is rejected rather than silently accepted.
func TestQuantConfigValidateUnsupportedMode(t *testing.T) {
	if err := (&QuantConfig{Mode: "int8_dynamic"}).Validate(); err == nil {
		t.Fatal("expected an error for an unsupported quant mode")
	}
}

// TestQuantConfigValidateNegative covers the two negative-value guards, which run before the
// mode switch (so they reject regardless of mode).
func TestQuantConfigValidateNegative(t *testing.T) {
	if err := (&QuantConfig{Mode: "affine", GroupSize: -1, Bits: 4}).Validate(); err == nil {
		t.Fatal("negative group_size: expected an error, got nil")
	}
	if err := (&QuantConfig{Mode: "affine", GroupSize: 64, Bits: -1}).Validate(); err == nil {
		t.Fatal("negative bits: expected an error, got nil")
	}
}
