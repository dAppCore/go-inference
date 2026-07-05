// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import "testing"

import core "dappco.re/go"

// TestPackCov_NilReceivers covers the nil-receiver guards on the unexported
// normalise methods. They are defensive (every production caller passes a
// non-nil value), so they are exercised directly from the package's own tests.
func TestPackCov_NilReceivers(t *testing.T) {
	var info *PackInfo
	info.normalise() // must be a no-op, not a nil-deref
	var tensor *PackTensor
	tensor.normalise(PackInfo{}) // likewise a no-op
}

// TestPackCov_ReadPackInfo_MalformedAutoRound covers the ReadPackInfo branch
// where the auto_round_config.json read returns a non-not-exist error (malformed
// JSON), which is surfaced rather than falling through to the quantization file.
func TestPackCov_ReadPackInfo_MalformedAutoRound(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, PackConfigFileAutoRound), []byte("{not json"), 0o644); !result.OK {
		t.Fatalf("seed malformed config: %v", result.Value)
	}
	if _, err := ReadPackInfo(dir); err == nil {
		t.Fatal("ReadPackInfo(malformed auto_round_config) error = nil, want JSON parse error")
	}
}

// TestPackCov_ReadPackInfo_MalformedQuantization covers the second-file branch:
// no auto_round_config.json, but a malformed quantization_config.json whose read
// returns a non-not-exist error.
func TestPackCov_ReadPackInfo_MalformedQuantization(t *testing.T) {
	dir := t.TempDir()
	if result := core.WriteFile(core.PathJoin(dir, PackConfigFileQuantization), []byte("{not json"), 0o644); !result.OK {
		t.Fatalf("seed malformed quantization config: %v", result.Value)
	}
	if _, err := ReadPackInfo(dir); err == nil {
		t.Fatal("ReadPackInfo(malformed quantization_config) error = nil, want JSON parse error")
	}
}

// TestPackCov_ReadPackInfoFile_MalformedJSON covers readPackInfoFile's
// JSONUnmarshal failure branch directly.
func TestPackCov_ReadPackInfoFile_MalformedJSON(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, PackConfigFileAutoRound)
	if result := core.WriteFile(path, []byte("not even close to json"), 0o644); !result.OK {
		t.Fatalf("seed malformed file: %v", result.Value)
	}
	if _, err := ReadPackInfoFile(path); err == nil {
		t.Fatal("ReadPackInfoFile(malformed JSON) error = nil, want parse error")
	}
}

// TestPackCov_PackInfo_Validate_BadGroupSize covers the group-size validation
// branch: a non-zero group size outside the allowed set is rejected.
func TestPackCov_PackInfo_Validate_BadGroupSize(t *testing.T) {
	info := PackInfo{QuantMethod: QuantMethodAutoRound, Bits: 4, GroupSize: 7}
	if err := info.Validate(); err == nil || !core.Contains(err.Error(), "group size") {
		t.Fatalf("Validate(group size 7) error = %v, want group size diagnostic", err)
	}
}

// TestPackCov_PackTensor_NormaliseInherits covers the tensor.normalise inherit
// branches for Bits and GroupSize: a tensor with zero bits/group size inherits
// them from the enclosing pack info.
func TestPackCov_PackTensor_NormaliseInherits(t *testing.T) {
	tensor := PackTensor{
		Name:       "weight",
		Packed:     "weight.packed",
		Scales:     "weight.scales",
		ZeroPoints: "weight.zeros",
		Shape:      []int32{4, 8},
		// Bits and GroupSize left at zero so they inherit from the info.
	}
	tensor.normalise(PackInfo{Bits: 4, GroupSize: 32, Symmetric: true})
	if tensor.Bits != 4 || tensor.GroupSize != 32 {
		t.Fatalf("normalise(zero bits/group) = %+v, want inherited Bits 4 / GroupSize 32", tensor)
	}
	if !tensor.Symmetric {
		t.Fatalf("normalise(false symmetry) Symmetric = false, want inherited true")
	}
}

// TestPackCov_PackTensor_Validate_BadBitsAndShape covers PackTensor.Validate's
// unsupported-bits branch, the packedShapeElements error branch (empty shape),
// and the group-count-mismatch branch.
func TestPackCov_PackTensor_Validate_BadBitsAndShape(t *testing.T) {
	t.Run("BadBits", func(t *testing.T) {
		tensor := PackTensor{Name: "weight", Packed: "p", Scales: "s", ZeroPoints: "z", Shape: []int32{4}, Bits: 5, GroupSize: 32}
		if err := tensor.Validate(); err == nil || !core.Contains(err.Error(), "tensor bits") {
			t.Fatalf("Validate(bits 5) error = %v, want tensor-bits diagnostic", err)
		}
	})
	t.Run("EmptyShape", func(t *testing.T) {
		tensor := PackTensor{Name: "weight", Packed: "p", Scales: "s", ZeroPoints: "z", Shape: nil, Bits: 4, GroupSize: 32}
		if err := tensor.Validate(); err == nil || !core.Contains(err.Error(), "shape is required") {
			t.Fatalf("Validate(empty shape) error = %v, want shape-required diagnostic", err)
		}
	})
	t.Run("GroupCountMismatch", func(t *testing.T) {
		// PackedBytes is correct for the shape/bits but Groups is wrong, so the
		// group-count branch fires after the packed-length check passes.
		tensor := PackTensor{Name: "weight", Packed: "p", Scales: "s", ZeroPoints: "z", Shape: []int32{4}, Bits: 4, GroupSize: 32, PackedBytes: 2, Groups: 5}
		if err := tensor.Validate(); err == nil || !core.Contains(err.Error(), "group count") {
			t.Fatalf("Validate(wrong group count) error = %v, want group-count diagnostic", err)
		}
	})
}

// TestPackCov_ValidateSafetensorsTensorMap_IndexFails covers the IndexFiles
// failure path inside ValidateSafetensorsTensorMap: a non-existent weight file
// cannot be indexed.
func TestPackCov_ValidateSafetensorsTensorMap_IndexFails(t *testing.T) {
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors: []PackTensor{{
			Name:       "weight",
			Packed:     "weight.packed",
			Scales:     "weight.scales",
			ZeroPoints: "weight.zeros",
			Shape:      []int32{1, 4},
			Bits:       2,
			GroupSize:  32,
			Symmetric:  true,
		}},
	}
	info.normalise()
	missing := core.PathJoin(t.TempDir(), "absent.safetensors")
	if err := ValidateSafetensorsTensorMap(info, []string{missing}); err == nil || !core.Contains(err.Error(), "index safetensors") {
		t.Fatalf("ValidateSafetensorsTensorMap(missing file) error = %v, want index diagnostic", err)
	}
}

// TestPackCov_ValidateSafetensorsTensorMap_MissingTensors covers the per-tensor
// validation failures inside ValidateSafetensorsTensorMap for the packed, scales,
// and bias tensors (the zeros case is already exercised elsewhere).
func TestPackCov_ValidateSafetensorsTensorMap_MissingTensors(t *testing.T) {
	withBias := func() PackInfo {
		info := PackInfo{
			Bits:          2,
			GroupSize:     32,
			Symmetric:     true,
			QuantMethod:   QuantMethodAutoRound,
			PackingFormat: string(FormatAutoRound),
			Tensors: []PackTensor{{
				Name:       "weight",
				Packed:     "weight.packed",
				Scales:     "weight.scales",
				ZeroPoints: "weight.zeros",
				Bias:       "weight.bias",
				Shape:      []int32{1, 4},
				Bits:       2,
				GroupSize:  32,
				Symmetric:  true,
			}},
		}
		info.normalise()
		return info
	}
	cases := []struct {
		name    string
		tensors []autoRoundSafetensorTensor
	}{
		{"Packed", []autoRoundSafetensorTensor{
			autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
			autoRoundF32Tensor("weight.bias", []float32{0.25}, 1),
		}},
		{"Scales", []autoRoundSafetensorTensor{
			{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
			autoRoundF32Tensor("weight.bias", []float32{0.25}, 1),
		}},
		{"Bias", []autoRoundSafetensorTensor{
			{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
			autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := core.PathJoin(t.TempDir(), "model.safetensors")
			writeAutoRoundSafetensors(t, path, tc.tensors)
			if err := ValidateSafetensorsTensorMap(withBias(), []string{path}); err == nil || !core.Contains(err.Error(), "missing") {
				t.Fatalf("ValidateSafetensorsTensorMap(missing %s) error = %v, want missing-tensor diagnostic", tc.name, err)
			}
		})
	}
}

// TestPackCov_ValidateSafetensorsTensor_DTypeAndElements covers
// validateSafetensorsTensor's dtype-mismatch and element-count-mismatch branches
// directly via ValidateSafetensorsTensorMap.
func TestPackCov_ValidateSafetensorsTensor_DTypeAndElements(t *testing.T) {
	info := func() PackInfo {
		i := PackInfo{
			Bits:          2,
			GroupSize:     32,
			Symmetric:     true,
			QuantMethod:   QuantMethodAutoRound,
			PackingFormat: string(FormatAutoRound),
			Tensors: []PackTensor{{
				Name:       "weight",
				Packed:     "weight.packed",
				Scales:     "weight.scales",
				ZeroPoints: "weight.zeros",
				Shape:      []int32{1, 4},
				Bits:       2,
				GroupSize:  32,
				Symmetric:  true,
			}},
		}
		i.normalise()
		return i
	}
	t.Run("WrongDType", func(t *testing.T) {
		path := core.PathJoin(t.TempDir(), "model.safetensors")
		// packed declared as F32 instead of U8.
		writeAutoRoundSafetensors(t, path, []autoRoundSafetensorTensor{
			autoRoundF32Tensor("weight.packed", []float32{0}, 1),
			autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
		})
		if err := ValidateSafetensorsTensorMap(info(), []string{path}); err == nil || !core.Contains(err.Error(), "dtype") {
			t.Fatalf("ValidateSafetensorsTensorMap(wrong packed dtype) error = %v, want dtype diagnostic", err)
		}
	})
	t.Run("WrongElements", func(t *testing.T) {
		path := core.PathJoin(t.TempDir(), "model.safetensors")
		// packed declared with the wrong byte count (2 instead of 1).
		writeAutoRoundSafetensors(t, path, []autoRoundSafetensorTensor{
			{Name: "weight.packed", DType: "U8", Shape: []int{2}, Raw: []byte{0, 0}},
			autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
		})
		if err := ValidateSafetensorsTensorMap(info(), []string{path}); err == nil || !core.Contains(err.Error(), "elements") {
			t.Fatalf("ValidateSafetensorsTensorMap(wrong packed elements) error = %v, want elements diagnostic", err)
		}
	})
}

// TestPackCov_InferScheme covers the Bits==8 branches (FP8 vs W8A16) and the
// default empty-scheme branch of inferScheme, reached through normalise when the
// pack info carries no explicit scheme.
func TestPackCov_InferScheme(t *testing.T) {
	cases := []struct {
		name     string
		bits     int
		dataType string
		want     Scheme
	}{
		{"FP8", 8, "fp8", SchemeFP8Static},
		{"Float8", 8, "float8", SchemeFP8Static},
		{"W8A16", 8, "int", SchemeW8A16},
		{"Default", 5, "int", ""}, // bits outside the inference table → empty scheme
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			info := PackInfo{Bits: tc.bits, DataType: tc.dataType, PackingFormat: string(FormatAutoRound)}
			info.normalise() // Scheme == "" → inferScheme runs
			if info.Scheme != tc.want {
				t.Fatalf("inferScheme(bits=%d type=%q) = %q, want %q", tc.bits, tc.dataType, info.Scheme, tc.want)
			}
		})
	}
}
