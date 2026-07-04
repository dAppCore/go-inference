// SPDX-Licence-Identifier: EUPL-1.2

package autoround

import "testing"

import core "dappco.re/go"

func TestPack_ReadPackInfo_Good(t *testing.T) {
	t.Run("Quantization", func(t *testing.T) {
		dir := t.TempDir()
		path := core.PathJoin(dir, PackConfigFileQuantization)
		if result := core.WriteFile(path, []byte(`{
			"bits": 4,
			"group_size": 128,
			"sym": true,
			"data_type": "int",
			"iters": 1000,
			"nsamples": 512,
			"seqlen": 2048,
			"autoround_version": "0.13.0",
			"quant_method": "auto-round",
			"packing_format": "auto_round:auto_gptq",
			"tensors": [
				{
					"name": "model.layers.0.self_attn.q_proj.weight",
					"packed": "model.layers.0.self_attn.q_proj.weight.packed",
					"scales": "model.layers.0.self_attn.q_proj.weight.scales",
					"zero_points": "model.layers.0.self_attn.q_proj.weight.zeros",
					"shape": [4, 8],
					"bits": 4,
					"group_size": 32,
					"sym": true
				}
			],
			"extra_config": {
				"model.layers.0.self_attn.q_proj": {"bits": 16}
			}
		}`), 0o644); !result.OK {
			t.Fatalf("WriteFile() error = %v", result.Value)
		}

		info, err := ReadPackInfo(dir)
		if err != nil {
			t.Fatalf("ReadPackInfo() error = %v", err)
		}
		if info == nil || info.Bits != 4 || info.GroupSize != 128 || info.Scheme != SchemeW4A16 || info.ExportFormat != FormatAutoRound || info.LayerOverrideN != 1 {
			t.Fatalf("ReadPackInfo() = %+v, want W4A16 AutoRound sidecar", info)
		}
		if info.TensorCount != 1 || !info.NativeTensorMap() {
			t.Fatalf("tensor metadata count=%d native=%v, want one native tensor mapping", info.TensorCount, info.NativeTensorMap())
		}
	})
	t.Run("GGUF", func(t *testing.T) {
		dir := t.TempDir()
		if result := core.WriteFile(core.PathJoin(dir, PackConfigFileAutoRound), []byte(`{
			"bits": 4,
			"group_size": 256,
			"sym": true,
			"quant_method": "autoround",
			"packing_format": "gguf:q4_k_m"
		}`), 0o644); !result.OK {
			t.Fatalf("WriteFile() error = %v", result.Value)
		}

		info, err := ReadPackInfo(dir)
		if err != nil {
			t.Fatalf("ReadPackInfo() error = %v", err)
		}
		if info == nil || info.Scheme != SchemeGGUFQ4KM || info.ExportFormat != FormatGGUFQ4KM || !info.GGUFExport() {
			t.Fatalf("ReadPackInfo() = %+v, want GGUF Q4_K_M AutoRound export", info)
		}
	})
}

func TestPack_ReadPackInfo_Bad(t *testing.T) {
	t.Run("IgnoresOtherQuantization", func(t *testing.T) {
		dir := t.TempDir()
		if result := core.WriteFile(core.PathJoin(dir, PackConfigFileQuantization), []byte(`{
			"bits": 4,
			"group_size": 128,
			"quant_method": "gptq"
		}`), 0o644); !result.OK {
			t.Fatalf("WriteFile() error = %v", result.Value)
		}

		info, err := ReadPackInfo(dir)
		if err != nil {
			t.Fatalf("ReadPackInfo(non-AutoRound quantization_config) error = %v", err)
		}
		if info != nil {
			t.Fatalf("ReadPackInfo(non-AutoRound quantization_config) = %+v, want nil", info)
		}
	})
	t.Run("RejectsInvalidTensorMap", func(t *testing.T) {
		dir := t.TempDir()
		if result := core.WriteFile(core.PathJoin(dir, PackConfigFileAutoRound), []byte(`{
			"bits": 4,
			"group_size": 32,
			"sym": true,
			"quant_method": "auto-round",
			"packing_format": "auto_round",
			"tensors": [
				{
					"name": "model.layers.0.mlp.gate_proj.weight",
					"packed": "model.layers.0.mlp.gate_proj.weight.packed",
					"scales": "model.layers.0.mlp.gate_proj.weight.scales",
					"zero_points": "model.layers.0.mlp.gate_proj.weight.zeros",
					"shape": [3],
					"bits": 4,
					"group_size": 32,
					"packed_bytes": 1
				}
			]
		}`), 0o644); !result.OK {
			t.Fatalf("WriteFile() error = %v", result.Value)
		}

		_, err := ReadPackInfo(dir)
		if err == nil || !core.Contains(err.Error(), "packed length") {
			t.Fatalf("ReadPackInfo(invalid tensor map) error = %v, want packed length diagnostic", err)
		}
	})
}

func TestPack_ReadPackInfo_Ugly(t *testing.T) {
	// A directory with neither sidecar present is the degenerate read: it
	// returns (nil, nil) rather than erroring on a missing file.
	info, err := ReadPackInfo(t.TempDir())
	if err != nil {
		t.Fatalf("ReadPackInfo(no sidecar) error = %v, want nil", err)
	}
	if info != nil {
		t.Fatalf("ReadPackInfo(no sidecar) = %+v, want nil", info)
	}
}

func TestPack_ReadPackInfoFile_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, PackConfigFileAutoRound)
	if result := core.WriteFile(path, []byte(`{
		"bits": 4,
		"group_size": 128,
		"sym": true,
		"quant_method": "auto-round",
		"packing_format": "auto_round"
	}`), 0o644); !result.OK {
		t.Fatalf("WriteFile() error = %v", result.Value)
	}
	info, err := ReadPackInfoFile(path)
	if err != nil {
		t.Fatalf("ReadPackInfoFile() error = %v", err)
	}
	if info == nil || info.Bits != 4 || info.Scheme != SchemeW4A16 {
		t.Fatalf("ReadPackInfoFile() = %+v, want W4A16 AutoRound sidecar", info)
	}
}

func TestPack_ReadPackInfoFile_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, PackConfigFileQuantization)
	if result := core.WriteFile(path, []byte(`{
		"bits": 4,
		"group_size": 128,
		"quant_method": "gptq"
	}`), 0o644); !result.OK {
		t.Fatalf("WriteFile() error = %v", result.Value)
	}
	// Unlike ReadPackInfo, the direct-file reader is strict: a non-AutoRound
	// quant method is an error, not a silent nil.
	if _, err := ReadPackInfoFile(path); err == nil {
		t.Fatal("ReadPackInfoFile(non-AutoRound) error = nil, want strict direct-file error")
	}
}

func TestPack_ReadPackInfoFile_Ugly(t *testing.T) {
	// A path that does not exist surfaces the underlying not-exist error rather
	// than a nil info; the reader does not swallow filesystem errors on a direct
	// file read.
	_, err := ReadPackInfoFile(core.PathJoin(t.TempDir(), "absent.json"))
	if err == nil || !core.IsNotExist(err) {
		t.Fatalf("ReadPackInfoFile(missing) error = %v, want not-exist error", err)
	}
}

func TestPack_ClonePackInfo_Good(t *testing.T) {
	original := &PackInfo{
		Bits:          4,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Tensors: []PackTensor{{
			Name:       "weight",
			Packed:     "weight.packed",
			Scales:     "weight.scales",
			ZeroPoints: "weight.zeros",
			Shape:      []int32{1, 2},
			Bits:       4,
			GroupSize:  32,
			Symmetric:  true,
		}},
		LayerOverrides: map[string]LayerConfig{"weight": {Bits: 16}},
	}
	original.normalise()

	cloned := ClonePackInfo(original)
	if cloned == nil || cloned.Bits != 4 || len(cloned.Tensors) != 1 || cloned.LayerOverrideN != 1 {
		t.Fatalf("ClonePackInfo = %+v, want a faithful copy", cloned)
	}
	// Mutating the clone's nested slices and map must not touch the original.
	cloned.Tensors[0].Shape[0] = 99
	cloned.LayerOverrides["weight"] = LayerConfig{Bits: 2}
	if original.Tensors[0].Shape[0] == 99 {
		t.Fatal("ClonePackInfo aliased the tensor shape slice")
	}
	if original.LayerOverrides["weight"].Bits != 16 {
		t.Fatal("ClonePackInfo aliased the layer-override map")
	}
}

func TestPack_ClonePackInfo_Bad(t *testing.T) {
	// A nil input is the degenerate clone: it returns nil rather than
	// dereferencing a nil pointer.
	if cloned := ClonePackInfo(nil); cloned != nil {
		t.Fatalf("ClonePackInfo(nil) = %+v, want nil", cloned)
	}
}

func TestPack_ClonePackInfo_Ugly(t *testing.T) {
	// An info with no tensors and no overrides clones to an independent copy
	// with nil nested fields — the clone must still be a distinct pointer.
	original := &PackInfo{Bits: 4, GroupSize: 32, QuantMethod: QuantMethodAutoRound}
	cloned := ClonePackInfo(original)
	if cloned == original {
		t.Fatal("ClonePackInfo(no nested) returned the same pointer")
	}
	if cloned.Tensors != nil || cloned.LayerOverrides != nil {
		t.Fatalf("ClonePackInfo(no nested) = %+v, want nil nested fields", cloned)
	}
}

func TestPack_PackInfo_Validate_Good(t *testing.T) {
	info := PackInfo{
		Bits:          4,
		GroupSize:     128,
		Symmetric:     true,
		QuantMethod:   QuantMethodAutoRound,
		PackingFormat: string(FormatAutoRound),
		Scheme:        SchemeW4A16,
	}
	info.normalise()
	if err := info.Validate(); err != nil {
		t.Fatalf("Validate() error = %v, want nil for a well-formed W4A16 pack", err)
	}
}

func TestPack_PackInfo_Validate_Bad(t *testing.T) {
	// Wrong quant method is the headline rejection — Validate gates everything
	// behind the AutoRound method.
	if err := (PackInfo{QuantMethod: "gptq", Bits: 4, GroupSize: 128}).Validate(); err == nil || !core.Contains(err.Error(), "quant_method") {
		t.Fatalf("Validate(non-AutoRound) error = %v, want quant_method diagnostic", err)
	}
	// An unsupported bit width is rejected.
	if err := (PackInfo{QuantMethod: QuantMethodAutoRound, Bits: 5, GroupSize: 128}).Validate(); err == nil || !core.Contains(err.Error(), "bits") {
		t.Fatalf("Validate(bad bits) error = %v, want bits diagnostic", err)
	}
}

func TestPack_PackInfo_Validate_Ugly(t *testing.T) {
	// Negative tuning knobs are degenerate: each is rejected with its own
	// diagnostic rather than flowing into downstream arithmetic.
	cases := []struct {
		name string
		info PackInfo
		want string
	}{
		{"iters", PackInfo{QuantMethod: QuantMethodAutoRound, Bits: 4, GroupSize: 128, Iters: -1}, "iters"},
		{"nsamples", PackInfo{QuantMethod: QuantMethodAutoRound, Bits: 4, GroupSize: 128, NSamples: -1}, "nsamples"},
		{"seqlen", PackInfo{QuantMethod: QuantMethodAutoRound, Bits: 4, GroupSize: 128, SeqLen: -1}, "seqlen"},
		{"scheme", PackInfo{QuantMethod: QuantMethodAutoRound, Bits: 4, GroupSize: 128, Scheme: "W5A16"}, "unsupported scheme"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.info.Validate()
			if err == nil || !core.Contains(err.Error(), tc.want) {
				t.Fatalf("Validate(%s) error = %v, want %q diagnostic", tc.name, err, tc.want)
			}
		})
	}
}

func TestPack_PackInfo_NativeFormat_Good(t *testing.T) {
	if !(PackInfo{PackingFormat: "auto_round"}).NativeFormat() {
		t.Fatal("NativeFormat(auto_round) = false, want true")
	}
	if !(PackInfo{PackingFormat: "auto_round:auto_gptq"}).NativeFormat() {
		t.Fatal("NativeFormat(auto_round:auto_gptq) = false, want true for the prefixed native form")
	}
}

func TestPack_PackInfo_NativeFormat_Bad(t *testing.T) {
	if (PackInfo{PackingFormat: "gguf:q4_k_m"}).NativeFormat() {
		t.Fatal("NativeFormat(gguf:q4_k_m) = true, want false for a GGUF export")
	}
}

func TestPack_PackInfo_NativeFormat_Ugly(t *testing.T) {
	// Whitespace-padded and mixed-case forms normalise inside NativeFormat, so
	// they still count as native; an empty format does not.
	if !(PackInfo{PackingFormat: "  AUTO_ROUND  "}).NativeFormat() {
		t.Fatal("NativeFormat(padded mixed-case) = false, want true after normalisation")
	}
	if (PackInfo{PackingFormat: ""}).NativeFormat() {
		t.Fatal("NativeFormat(empty) = true, want false")
	}
}

func TestPack_PackInfo_GGUFExport_Good(t *testing.T) {
	if !(PackInfo{ExportFormat: FormatGGUFQ4KM}).GGUFExport() {
		t.Fatal("GGUFExport(format gguf:q4_k_m) = false, want true")
	}
	if !(PackInfo{PackingFormat: "gguf:q4_k_m"}).GGUFExport() {
		t.Fatal("GGUFExport(packing gguf) = false, want true via the packing format")
	}
}

func TestPack_PackInfo_GGUFExport_Bad(t *testing.T) {
	if (PackInfo{ExportFormat: FormatAutoRound, PackingFormat: "auto_round"}).GGUFExport() {
		t.Fatal("GGUFExport(native auto_round) = true, want false")
	}
}

func TestPack_PackInfo_GGUFExport_Ugly(t *testing.T) {
	// A zero-value info exports nothing — neither format field mentions GGUF.
	if (PackInfo{}).GGUFExport() {
		t.Fatal("GGUFExport(zero) = true, want false")
	}
}

func TestPack_PackInfo_NativeTensorMap_Good(t *testing.T) {
	info := PackInfo{
		PackingFormat: "auto_round",
		Tensors: []PackTensor{{
			Name:       "weight",
			Packed:     "weight.packed",
			Scales:     "weight.scales",
			ZeroPoints: "weight.zeros",
			Shape:      []int32{1, 2},
			Bits:       4,
			GroupSize:  32,
		}},
	}
	if !info.NativeTensorMap() {
		t.Fatal("NativeTensorMap(native + tensors) = false, want true")
	}
}

func TestPack_PackInfo_NativeTensorMap_Bad(t *testing.T) {
	// Native format but no tensors: there is no map to consume.
	if (PackInfo{PackingFormat: "auto_round"}).NativeTensorMap() {
		t.Fatal("NativeTensorMap(no tensors) = true, want false")
	}
}

func TestPack_PackInfo_NativeTensorMap_Ugly(t *testing.T) {
	// Tensors present but a GGUF (non-native) packing format: the map is not a
	// native tensor map.
	info := PackInfo{
		PackingFormat: "gguf:q4_k_m",
		Tensors: []PackTensor{{
			Name:       "weight",
			Packed:     "weight.packed",
			Scales:     "weight.scales",
			ZeroPoints: "weight.zeros",
			Shape:      []int32{1, 2},
			Bits:       4,
			GroupSize:  32,
		}},
	}
	if info.NativeTensorMap() {
		t.Fatal("NativeTensorMap(gguf + tensors) = true, want false for a non-native format")
	}
}

func TestPack_PackTensor_Validate_Good(t *testing.T) {
	tensor := PackTensor{
		Name:       "weight",
		Packed:     "weight.packed",
		Scales:     "weight.scales",
		ZeroPoints: "weight.zeros",
		Shape:      []int32{4, 8},
		Bits:       4,
		GroupSize:  32,
	}
	tensor.normalise(PackInfo{Bits: 4, GroupSize: 32})
	if err := tensor.Validate(); err != nil {
		t.Fatalf("Validate() error = %v, want nil for a consistent tensor", err)
	}
}

func TestPack_PackTensor_Validate_Bad(t *testing.T) {
	// A missing name is the headline rejection.
	if err := (PackTensor{Packed: "p", Scales: "s", ZeroPoints: "z", Shape: []int32{4}, Bits: 4, GroupSize: 32}).Validate(); err == nil || !core.Contains(err.Error(), "name") {
		t.Fatalf("Validate(no name) error = %v, want name diagnostic", err)
	}
	// A packed-byte count that disagrees with shape/bits is rejected.
	bad := PackTensor{Name: "weight", Packed: "p", Scales: "s", ZeroPoints: "z", Shape: []int32{4}, Bits: 4, GroupSize: 32, PackedBytes: 1, Groups: 1}
	if err := bad.Validate(); err == nil || !core.Contains(err.Error(), "packed length") {
		t.Fatalf("Validate(wrong packed bytes) error = %v, want packed length diagnostic", err)
	}
}

func TestPack_PackTensor_Validate_Ugly(t *testing.T) {
	// Missing companion tensor names is the degenerate map entry: a packed
	// tensor without scales/zero_points cannot be dequantised, so Validate
	// rejects it before any byte arithmetic.
	tensor := PackTensor{Name: "weight", Packed: "weight.packed", Shape: []int32{4}, Bits: 4, GroupSize: 32}
	if err := tensor.Validate(); err == nil || !core.Contains(err.Error(), "packed, scales, and zero_points") {
		t.Fatalf("Validate(missing companions) error = %v, want companion-tensor diagnostic", err)
	}
	// A non-positive group size cannot define groups.
	zeroGroup := PackTensor{Name: "weight", Packed: "p", Scales: "s", ZeroPoints: "z", Shape: []int32{4}, Bits: 4, GroupSize: 0}
	if err := zeroGroup.Validate(); err == nil || !core.Contains(err.Error(), "group size") {
		t.Fatalf("Validate(zero group size) error = %v, want group size diagnostic", err)
	}
}

func TestPack_ValidateSafetensorsTensorMap_Good(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.safetensors")
	writeAutoRoundSafetensors(t, path, []autoRoundSafetensorTensor{
		{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0b11100100}},
		autoRoundF32Tensor("weight.scales", []float32{0.5}, 1),
		autoRoundF32Tensor("weight.zeros", []float32{0}, 1),
	})
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
	if err := ValidateSafetensorsTensorMap(info, []string{path}); err != nil {
		t.Fatalf("ValidateSafetensorsTensorMap() error = %v, want nil", err)
	}
}

func TestPack_ValidateSafetensorsTensorMap_Bad(t *testing.T) {
	dir := t.TempDir()
	path := core.PathJoin(dir, "model.safetensors")
	writeAutoRoundSafetensors(t, path, []autoRoundSafetensorTensor{
		{Name: "weight.packed", DType: "U8", Shape: []int{1}, Raw: []byte{0}},
		autoRoundF32Tensor("weight.scales", []float32{1}, 1),
	})
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
	if err := ValidateSafetensorsTensorMap(info, []string{path}); err == nil || !core.Contains(err.Error(), "missing") {
		t.Fatalf("ValidateSafetensorsTensorMap(missing zeros) error = %v, want missing tensor diagnostic", err)
	}
}

func TestPack_ValidateSafetensorsTensorMap_Ugly(t *testing.T) {
	// A non-native pack info has no tensor map to validate: the function is a
	// no-op and returns nil rather than indexing safetensors.
	bare := PackInfo{Bits: 4, GroupSize: 32, QuantMethod: QuantMethodAutoRound, PackingFormat: "gguf:q4_k_m"}
	if err := ValidateSafetensorsTensorMap(bare, []string{"unused.safetensors"}); err != nil {
		t.Fatalf("ValidateSafetensorsTensorMap(non-native) error = %v, want nil no-op", err)
	}
}
