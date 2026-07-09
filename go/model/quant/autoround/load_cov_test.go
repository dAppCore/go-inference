// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import (
	"encoding/binary"
	"testing"
)

import core "dappco.re/go"

// loadCovInfo builds a native W2 1x4 pack info with a single tensor (optionally
// declaring a bias), ready for LoadPackedProjectionFromSafetensors.
func loadCovInfo(t *testing.T, withBias bool) PackInfo {
	t.Helper()
	tensor := PackTensor{
		Name:       "weight",
		Packed:     "weight.packed",
		Scales:     "weight.scales",
		ZeroPoints: "weight.zeros",
		Shape:      []int32{1, 4},
		Bits:       2,
		GroupSize:  32,
		Symmetric:  true,
	}
	if withBias {
		tensor.Bias = "weight.bias"
	}
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors:       []PackTensor{tensor},
	}
	info.normalise()
	return info
}

// writeTruncatedSafetensors writes a safetensors file whose header declares the
// given tensors but whose payload is cut short by trimBytes — so any reader that
// honours the declared data offsets fails with a truncated-payload error.
func writeTruncatedSafetensors(t *testing.T, path string, tensors []autoRoundSafetensorTensor, trimBytes int) {
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
		header[tensor.Name] = entry{DType: tensor.DType, Shape: tensor.Shape, DataOffsets: []int{start, len(data)}}
	}
	encoded := core.JSONMarshal(header)
	if !encoded.OK {
		t.Fatalf("marshal truncated header: %v", encoded.Value)
	}
	headerBytes := encoded.Value.([]byte)
	full := make([]byte, 8+len(headerBytes)+len(data))
	binary.LittleEndian.PutUint64(full[:8], uint64(len(headerBytes)))
	copy(full[8:], headerBytes)
	copy(full[8+len(headerBytes):], data)
	if trimBytes > 0 && trimBytes < len(data) {
		full = full[:len(full)-trimBytes]
	}
	if result := core.WriteFile(path, full, 0o644); !result.OK {
		t.Fatalf("write truncated safetensors: %v", result.Value)
	}
}

// TestLoadCov_TensorValidateFails covers the tensor.Validate() rejection after a
// successful name lookup: the stored tensor omits a required companion name, so
// it cannot be loaded even though the name resolves.
func TestLoadCov_TensorValidateFails(t *testing.T) {
	tensor := PackTensor{
		Name:   "weight",
		Packed: "weight.packed",
		// Scales / ZeroPoints intentionally absent.
		Shape:     []int32{1, 4},
		Bits:      2,
		GroupSize: 32,
		Symmetric: true,
	}
	info := PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors:       []PackTensor{tensor},
	}
	info.normalise()
	if _, err := LoadPackedProjectionFromSafetensors(info, []string{"unused.safetensors"}, "weight"); err == nil || !core.Contains(err.Error(), "packed, scales, and zero_points") {
		t.Fatalf("Load(tensor missing companions) error = %v, want companion diagnostic", err)
	}
}

// TestLoadCov_IndexFilesFails covers the safetensors.IndexFiles error path: a
// weight file that does not exist cannot be indexed.
func TestLoadCov_IndexFilesFails(t *testing.T) {
	info := loadCovInfo(t, false)
	missing := core.PathJoin(t.TempDir(), "absent.safetensors")
	if _, err := LoadPackedProjectionFromSafetensors(info, []string{missing}, "weight"); err == nil || !core.Contains(err.Error(), "index safetensors") {
		t.Fatalf("Load(missing weight file) error = %v, want index diagnostic", err)
	}
}

// TestLoadCov_MissingRefs covers the three tensor-ref lookup failures (packed,
// scales, zero-points) by omitting each in turn from the safetensors file.
func TestLoadCov_MissingRefs(t *testing.T) {
	info := loadCovInfo(t, false)
	cases := []struct {
		name    string
		tensors []autoRoundSafetensorTensor
	}{
		{"Packed", []autoRoundSafetensorTensor{
			autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
		}},
		{"Scales", []autoRoundSafetensorTensor{
			{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
		}},
		{"ZeroPoints", []autoRoundSafetensorTensor{
			{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
			autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := core.PathJoin(t.TempDir(), "model.safetensors")
			writeAutoRoundSafetensors(t, path, tc.tensors)
			if _, err := LoadPackedProjectionFromSafetensors(info, []string{path}, "weight"); err == nil || !core.Contains(err.Error(), "missing") {
				t.Fatalf("Load(missing %s) error = %v, want missing-tensor diagnostic", tc.name, err)
			}
		})
	}
}

// TestLoadCov_ReadFailures covers the payload-read failures for the packed,
// scales, and zero-point tensors. Each file is truncated so the declared data
// offsets run past the end of the file.
func TestLoadCov_ReadFailures(t *testing.T) {
	info := loadCovInfo(t, false)
	// Order tensors so the truncation lands on the named tensor's payload: the
	// last tensor in the file is the one whose tail bytes are cut.
	cases := []struct {
		name    string
		tensors []autoRoundSafetensorTensor
		trim    int
	}{
		{"ZeroPoints", []autoRoundSafetensorTensor{
			{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
			autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
		}, 4},
		{"Scales", []autoRoundSafetensorTensor{
			{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
			autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
		}, 4},
		{"Packed", []autoRoundSafetensorTensor{
			autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
			// Packed last; its single byte is trimmed.
			{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
		}, 1},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := core.PathJoin(t.TempDir(), "model.safetensors")
			writeTruncatedSafetensors(t, path, tc.tensors, tc.trim)
			if _, err := LoadPackedProjectionFromSafetensors(info, []string{path}, "weight"); err == nil {
				t.Fatalf("Load(truncated %s) error = nil, want read failure", tc.name)
			}
		})
	}
}

// TestLoadCov_BiasRefMissing covers the bias-ref lookup failure: the tensor
// declares a bias but the safetensors file omits it.
func TestLoadCov_BiasRefMissing(t *testing.T) {
	info := loadCovInfo(t, true)
	path := core.PathJoin(t.TempDir(), "model.safetensors")
	writeAutoRoundSafetensors(t, path, []autoRoundSafetensorTensor{
		{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
		autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
		autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
		// weight.bias omitted.
	})
	if _, err := LoadPackedProjectionFromSafetensors(info, []string{path}, "weight"); err == nil || !core.Contains(err.Error(), "missing") {
		t.Fatalf("Load(missing bias) error = %v, want missing-tensor diagnostic", err)
	}
}

// TestLoadCov_BiasReadFails covers the bias payload-read failure: the bias
// tensor is present but its declared payload is truncated.
func TestLoadCov_BiasReadFails(t *testing.T) {
	info := loadCovInfo(t, true)
	path := core.PathJoin(t.TempDir(), "model.safetensors")
	writeTruncatedSafetensors(t, path, []autoRoundSafetensorTensor{
		{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
		autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
		autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
		// Bias last so its tail is the trimmed region.
		autoRoundF32Tensor("weight.bias", []float32{0.25}, 1),
	}, 4)
	if _, err := LoadPackedProjectionFromSafetensors(info, []string{path}, "weight"); err == nil {
		t.Fatal("Load(truncated bias) error = nil, want bias read failure")
	}
}

// TestLoadCov_LookupTensorRef_DTypeAndElements covers lookupAutoRoundTensorRef's
// dtype-mismatch and element-count-mismatch branches directly: the scales tensor
// is written with the wrong dtype (dtype branch) and the packed tensor with the
// wrong element count (elements branch).
func TestLoadCov_LookupTensorRef_DTypeAndElements(t *testing.T) {
	info := loadCovInfo(t, false)
	t.Run("WrongDType", func(t *testing.T) {
		path := core.PathJoin(t.TempDir(), "model.safetensors")
		// scales declared as U8 instead of F32.
		writeAutoRoundSafetensors(t, path, []autoRoundSafetensorTensor{
			{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
			{Name: "weight.scales", DType: "U8", Shape: []int{4}, Raw: []byte{0, 0, 0, 0}},
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
		})
		if _, err := LoadPackedProjectionFromSafetensors(info, []string{path}, "weight"); err == nil || !core.Contains(err.Error(), "dtype") {
			t.Fatalf("Load(wrong scales dtype) error = %v, want dtype diagnostic", err)
		}
	})
	t.Run("WrongElements", func(t *testing.T) {
		path := core.PathJoin(t.TempDir(), "model.safetensors")
		// packed declared with 2 bytes (elements 2) but the tensor expects 1.
		writeAutoRoundSafetensors(t, path, []autoRoundSafetensorTensor{
			{Name: "weight.packed", DType: "U8", Shape: []int{2}, Raw: []byte{0, 0}},
			autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
			autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
		})
		if _, err := LoadPackedProjectionFromSafetensors(info, []string{path}, "weight"); err == nil || !core.Contains(err.Error(), "elements") {
			t.Fatalf("Load(wrong packed elements) error = %v, want elements diagnostic", err)
		}
	})
}
