// SPDX-Licence-Identifier: EUPL-1.2

package autoround_test

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/model/quant/autoround"
)

// ExampleReadPackInfo reads an AutoRound sidecar from a model directory,
// inferring the scheme and export format from the recorded bit width.
func ExampleReadPackInfo() {
	dir := core.MkdirTemp("", "autoround-readinfo-example-*").Value.(string)
	path := core.PathJoin(dir, autoround.PackConfigFileAutoRound)
	core.WriteFile(path, []byte(`{
		"bits": 4,
		"group_size": 128,
		"sym": true,
		"quant_method": "auto-round",
		"packing_format": "auto_round"
	}`), 0o644)
	info, err := autoround.ReadPackInfo(dir)
	if err != nil || info == nil {
		core.Println("read failed")
		return
	}
	core.Println(info.Bits, info.Scheme, info.ExportFormat)
	// Output: 4 W4A16 auto_round
}

// ExampleReadPackInfoFile reads a sidecar by explicit path, applying the strict
// rule that a non-AutoRound config is an error.
func ExampleReadPackInfoFile() {
	dir := core.MkdirTemp("", "autoround-readfile-example-*").Value.(string)
	path := core.PathJoin(dir, autoround.PackConfigFileAutoRound)
	core.WriteFile(path, []byte(`{
		"bits": 2,
		"group_size": 32,
		"sym": true,
		"quant_method": "auto-round",
		"packing_format": "auto_round"
	}`), 0o644)
	info, err := autoround.ReadPackInfoFile(path)
	if err != nil || info == nil {
		core.Println("read failed")
		return
	}
	core.Println(info.Bits, info.Scheme)
	// Output: 2 W2A16
}

// ExampleClonePackInfo deep-copies a pack info so the clone's nested tensor
// slice can be mutated without disturbing the original.
func ExampleClonePackInfo() {
	original := &autoround.PackInfo{
		Bits:          4,
		GroupSize:     32,
		QuantMethod:   autoround.QuantMethodAutoRound,
		PackingFormat: string(autoround.FormatAutoRound),
		LayerOverrides: map[string]autoround.LayerConfig{
			"weight": {Bits: 16},
		},
	}
	cloned := autoround.ClonePackInfo(original)
	cloned.Bits = 8
	core.Println(original.Bits, cloned.Bits, len(cloned.LayerOverrides))
	// Output: 4 8 1
}

// ExamplePackInfo_Validate accepts a well-formed AutoRound pack and reports the
// nil error as success.
func ExamplePackInfo_Validate() {
	info := autoround.PackInfo{
		Bits:          4,
		GroupSize:     128,
		Symmetric:     true,
		QuantMethod:   autoround.QuantMethodAutoRound,
		PackingFormat: string(autoround.FormatAutoRound),
		Scheme:        autoround.SchemeW4A16,
	}
	core.Println(info.Validate() == nil)
	// Output: true
}

// ExamplePackInfo_NativeFormat reports whether the packing format is the native
// AutoRound layout (true) versus a GGUF export (false).
func ExamplePackInfo_NativeFormat() {
	core.Println(autoround.PackInfo{PackingFormat: "auto_round"}.NativeFormat())
	core.Println(autoround.PackInfo{PackingFormat: "gguf:q4_k_m"}.NativeFormat())
	// Output:
	// true
	// false
}

// ExamplePackInfo_GGUFExport reports whether the pack targets a GGUF export
// path, detected from either the export format or the packing format.
func ExamplePackInfo_GGUFExport() {
	core.Println(autoround.PackInfo{ExportFormat: autoround.FormatGGUFQ4KM}.GGUFExport())
	core.Println(autoround.PackInfo{PackingFormat: "auto_round"}.GGUFExport())
	// Output:
	// true
	// false
}

// ExamplePackInfo_NativeTensorMap reports whether the info carries a native
// per-tensor map ready for by-name projection loading.
func ExamplePackInfo_NativeTensorMap() {
	withMap := autoround.PackInfo{
		PackingFormat: "auto_round",
		Tensors: []autoround.PackTensor{{
			Name:       "weight",
			Packed:     "weight.packed",
			Scales:     "weight.scales",
			ZeroPoints: "weight.zeros",
			Shape:      []int32{1, 2},
			Bits:       4,
			GroupSize:  32,
		}},
	}
	core.Println(withMap.NativeTensorMap())
	core.Println(autoround.PackInfo{PackingFormat: "auto_round"}.NativeTensorMap())
	// Output:
	// true
	// false
}

// ExamplePackTensor_Validate checks a tensor-map entry for internal
// consistency between its shape, bit width and derived packed-byte count.
func ExamplePackTensor_Validate() {
	tensor := autoround.PackTensor{
		Name:        "weight",
		Packed:      "weight.packed",
		Scales:      "weight.scales",
		ZeroPoints:  "weight.zeros",
		Shape:       []int32{4, 8},
		Bits:        4,
		GroupSize:   32,
		PackedBytes: 16,
		Groups:      1,
	}
	core.Println(tensor.Validate() == nil)
	// Output: true
}

// ExampleValidateSafetensorsTensorMap confirms that every tensor named in a
// native pack's map is present in the safetensors payload with the right dtype
// and element count.
func ExampleValidateSafetensorsTensorMap() {
	dir := core.MkdirTemp("", "autoround-validatemap-example-*").Value.(string)
	path := core.PathJoin(dir, "model.safetensors")
	projection := exampleProjection("weight", 0b11100100, 0.5, nil)
	info := autoround.PackInfo{
		Bits:          2,
		GroupSize:     32,
		Symmetric:     true,
		QuantMethod:   autoround.QuantMethodAutoRound,
		PackingFormat: string(autoround.FormatAutoRound),
		Tensors:       []autoround.PackTensor{projection.Tensor},
	}
	if _, err := autoround.WriteNativePack(context.Background(), dir, info, []autoround.PackedProjection{projection}); err != nil {
		core.Println(err.Error())
		return
	}
	got, err := autoround.ReadPackInfo(dir)
	if err != nil || got == nil {
		core.Println("read failed")
		return
	}
	core.Println(autoround.ValidateSafetensorsTensorMap(*got, []string{path}) == nil)
	// Output: true
}
