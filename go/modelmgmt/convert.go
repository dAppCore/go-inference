package modelmgmt

import (
	"maps"
	"math"
	"regexp"
	"slices"
	"strconv"

	"dappco.re/go"
	"dappco.re/go/inference/safetensors"
	coreio "dappco.re/go/io"
)

var (
	layerRe  = regexp.MustCompile(`layers\.(\d+)`)
	moduleRe = regexp.MustCompile(`model\.layers\.\d+\.(.*?)\.lora_[ab]$`)
)

// RenameMLXKey converts an MLX tensor key to PEFT format. The lora_a/lora_b
// suffixes are anchored literals, so a HasSuffix check replaces the regex
// engine (which dominated allocations on this per-tensor path).
func RenameMLXKey(mlxKey string) string {
	if core.HasSuffix(mlxKey, ".lora_a") {
		return "base_model.model." + mlxKey[:len(mlxKey)-len(".lora_a")] + ".lora_A.default.weight"
	}
	if core.HasSuffix(mlxKey, ".lora_b") {
		return "base_model.model." + mlxKey[:len(mlxKey)-len(".lora_b")] + ".lora_B.default.weight"
	}
	return "base_model.model." + mlxKey
}

// SafetensorsHeader re-exports safetensors.SafetensorsHeader so existing ml
// consumers keep compiling — the codec itself lives in the safetensors
// package (the format leaf), not here.
type SafetensorsHeader = safetensors.SafetensorsHeader

// SafetensorsTensorInfo re-exports safetensors.SafetensorsTensorInfo.
type SafetensorsTensorInfo = safetensors.SafetensorsTensorInfo

// SafetensorsData re-exports safetensors.SafetensorsData.
type SafetensorsData = safetensors.SafetensorsData

// ReadSafetensors reads a safetensors file and returns tensor info and raw
// data. Delegates to the safetensors format package.
//
//	r := modelmgmt.ReadSafetensors(path)
//	if !r.OK { return r }
//	data := r.Value.(modelmgmt.SafetensorsData)
func ReadSafetensors(path string) core.Result {
	return safetensors.ReadSafetensors(path)
}

// GetTensorData extracts raw bytes for a tensor from the data section.
// Delegates to the safetensors format package.
func GetTensorData(info SafetensorsTensorInfo, allData []byte) []byte {
	return safetensors.GetTensorData(info, allData)
}

// TransposeFloat32 transposes a (rows, cols) float32 matrix to (cols, rows).
func TransposeFloat32(data []byte, rows, cols int) []byte {
	if len(data) != rows*cols*4 {
		return data
	}
	result := make([]byte, len(data))
	for r := range rows {
		for c := range cols {
			srcOff := (r*cols + c) * 4
			dstOff := (c*rows + r) * 4
			copy(result[dstOff:dstOff+4], data[srcOff:srcOff+4])
		}
	}
	return result
}

// TransposeFloat16 transposes a (rows, cols) float16 matrix to (cols, rows).
func TransposeFloat16(data []byte, rows, cols int) []byte {
	if len(data) != rows*cols*2 {
		return data
	}
	result := make([]byte, len(data))
	for r := range rows {
		for c := range cols {
			srcOff := (r*cols + c) * 2
			dstOff := (c*rows + r) * 2
			copy(result[dstOff:dstOff+2], data[srcOff:srcOff+2])
		}
	}
	return result
}

// TransposeBFloat16 transposes a (rows, cols) bfloat16 matrix to (cols, rows).
func TransposeBFloat16(data []byte, rows, cols int) []byte {
	return TransposeFloat16(data, rows, cols)
}

// WriteSafetensors writes tensors to a safetensors file. Delegates to the
// safetensors format package.
//
//	r := modelmgmt.WriteSafetensors(path, tensors, tensorData)
//	if !r.OK { return r }
func WriteSafetensors(path string, tensors map[string]SafetensorsTensorInfo, tensorData map[string][]byte) core.Result {
	return safetensors.WriteSafetensors(path, tensors, tensorData)
}

// ConvertMLXtoPEFT converts an MLX LoRA adapter to HuggingFace PEFT format.
func ConvertMLXtoPEFT(safetensorsPath, configPath, outputDir, baseModelName string) core.Result {
	if err := coreio.Local.EnsureDir(outputDir); err != nil {
		return core.Fail(core.E("modelmgmt.ConvertMLXtoPEFT", "create output dir", err))
	}

	safetensorsResult := ReadSafetensors(safetensorsPath)
	if !safetensorsResult.OK {
		return core.Fail(core.E("modelmgmt.ConvertMLXtoPEFT", "read safetensors", safetensorsResult.Value.(error)))
	}
	loaded := safetensorsResult.Value.(SafetensorsData)
	tensors := loaded.Tensors
	tensorData := loaded.Data
	core.Print(nil, "loaded %d tensors from %s", len(tensors), safetensorsPath)

	peftTensors := make(map[string]SafetensorsTensorInfo, len(tensors))
	peftData := make(map[string][]byte, len(tensors))

	for mlxKey, info := range tensors {
		peftKey := RenameMLXKey(mlxKey)
		data := GetTensorData(info, tensorData)

		if len(info.Shape) == 2 {
			rows, cols := info.Shape[0], info.Shape[1]
			switch info.Dtype {
			case "F32":
				data = TransposeFloat32(data, rows, cols)
			case "F16":
				data = TransposeFloat16(data, rows, cols)
			case "BF16":
				data = TransposeBFloat16(data, rows, cols)
			}
			info.Shape = []int{cols, rows}
		}

		peftTensors[peftKey] = info
		peftData[peftKey] = data
	}

	outSafetensors := core.JoinPath(outputDir, "adapter_model.safetensors")
	if result := WriteSafetensors(outSafetensors, peftTensors, peftData); !result.OK {
		return core.Fail(core.E("modelmgmt.ConvertMLXtoPEFT", "write safetensors", result.Value.(error)))
	}

	cfgData, err := coreio.Local.Read(configPath)
	if err != nil {
		return core.Fail(core.E("modelmgmt.ConvertMLXtoPEFT", "read config", err))
	}

	var mlxConfig struct {
		LoraParameters struct {
			Rank    int     `json:"rank"`
			Scale   float64 `json:"scale"`
			Dropout float64 `json:"dropout"`
		} `json:"lora_parameters"`
	}
	if r := core.JSONUnmarshalString(cfgData, &mlxConfig); !r.OK {
		return core.Fail(core.E("modelmgmt.ConvertMLXtoPEFT", "parse config", r.Value.(error)))
	}

	rank := mlxConfig.LoraParameters.Rank
	if rank == 0 {
		rank = 8
	}
	scale := mlxConfig.LoraParameters.Scale
	if scale == 0 {
		scale = 20.0
	}

	modules := make(map[string]bool)
	layers := make(map[int]bool)
	for k := range tensors {
		if m := moduleRe.FindStringSubmatch(k); m != nil {
			parts := core.Split(m[1], ".")
			modules[parts[len(parts)-1]] = true
		}
		if m := layerRe.FindStringSubmatch(k); m != nil {
			n, _ := strconv.Atoi(m[1])
			layers[n] = true
		}
	}

	sortedModules := slices.Sorted(maps.Keys(modules))
	sortedLayers := slices.Sorted(maps.Keys(layers))

	peftConfig := map[string]any{
		"auto_mapping":            nil,
		"base_model_name_or_path": baseModelName,
		"bias":                    "none",
		"fan_in_fan_out":          false,
		"inference_mode":          true,
		"init_lora_weights":       true,
		"layers_pattern":          nil,
		"layers_to_transform":     sortedLayers,
		"lora_alpha":              math.Round(scale * float64(rank)),
		"lora_dropout":            mlxConfig.LoraParameters.Dropout,
		"modules_to_save":         nil,
		"peft_type":               "LORA",
		"r":                       rank,
		"revision":                nil,
		"target_modules":          sortedModules,
		"task_type":               "CAUSAL_LM",
	}

	if err := coreio.Local.Write(core.JoinPath(outputDir, "adapter_config.json"), core.JSONMarshalString(peftConfig)); err != nil {
		return core.Fail(core.E("modelmgmt.ConvertMLXtoPEFT", "write adapter_config.json", err))
	}

	core.Print(nil, "converted %d tensors, %d layers, target modules: %v",
		len(peftTensors), len(sortedLayers), sortedModules)

	return core.Ok(nil)
}
