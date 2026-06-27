package modelmgmt

import (
	"encoding/binary"
	"maps"
	"math"
	"regexp"
	"slices"
	"strconv"

	"dappco.re/go"
	coreio "dappco.re/go/io"
)

var (
	loraARe  = regexp.MustCompile(`\.lora_a$`)
	loraBRe  = regexp.MustCompile(`\.lora_b$`)
	layerRe  = regexp.MustCompile(`layers\.(\d+)`)
	moduleRe = regexp.MustCompile(`model\.layers\.\d+\.(.*?)\.lora_[ab]$`)
)

// RenameMLXKey converts an MLX tensor key to PEFT format.
func RenameMLXKey(mlxKey string) string {
	key := mlxKey
	key = loraARe.ReplaceAllString(key, ".lora_A.default.weight")
	key = loraBRe.ReplaceAllString(key, ".lora_B.default.weight")
	key = "base_model.model." + key
	return key
}

// SafetensorsHeader represents the header of a safetensors file.
type SafetensorsHeader struct {
	Metadata map[string]string                `json:"__metadata__,omitempty"`
	Tensors  map[string]SafetensorsTensorInfo `json:"-"`
}

// SafetensorsTensorInfo describes a tensor's dtype, shape, and data location.
type SafetensorsTensorInfo struct {
	Dtype       string `json:"dtype"`
	Shape       []int  `json:"shape"`
	DataOffsets [2]int `json:"data_offsets"`
}

// SafetensorsData carries parsed safetensors metadata and tensor bytes.
type SafetensorsData struct {
	Tensors map[string]SafetensorsTensorInfo
	Data    []byte
}

// ReadSafetensors reads a safetensors file and returns tensor info and raw data.
func ReadSafetensors(path string) core.Result {
	raw, err := coreio.Local.Read(path)
	if err != nil {
		return core.Fail(core.E("ml.ReadSafetensors", "read file", err))
	}
	data := []byte(raw)

	if len(data) < 8 {
		return core.Fail(core.E("ml.ReadSafetensors", "file too small", nil))
	}

	headerSize := int(binary.LittleEndian.Uint64(data[:8]))
	if 8+headerSize > len(data) {
		return core.Fail(core.E("ml.ReadSafetensors", core.Sprintf("invalid header size %d", headerSize), nil))
	}

	headerJSON := data[8 : 8+headerSize]
	tensorData := data[8+headerSize:]

	var rawHeader map[string]SafetensorsTensorInfo
	if r := core.JSONUnmarshalString(string(headerJSON), &rawHeader); !r.OK {
		return core.Fail(core.E("ml.ReadSafetensors", "parse header", r.Value.(error)))
	}
	delete(rawHeader, "__metadata__")

	tensors := make(map[string]SafetensorsTensorInfo)
	for key, raw := range rawHeader {
		tensors[key] = raw
	}

	return core.Ok(SafetensorsData{Tensors: tensors, Data: tensorData})
}

// GetTensorData extracts raw bytes for a tensor from the data section.
func GetTensorData(info SafetensorsTensorInfo, allData []byte) []byte {
	return allData[info.DataOffsets[0]:info.DataOffsets[1]]
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

// WriteSafetensors writes tensors to a safetensors file.
func WriteSafetensors(path string, tensors map[string]SafetensorsTensorInfo, tensorData map[string][]byte) core.Result {
	keys := slices.Sorted(maps.Keys(tensors))

	offset := 0
	updatedTensors := make(map[string]SafetensorsTensorInfo)
	for _, k := range keys {
		info := tensors[k]
		data := tensorData[k]
		info.DataOffsets = [2]int{offset, offset + len(data)}
		updatedTensors[k] = info
		offset += len(data)
	}

	headerMap := make(map[string]any)
	for k, info := range updatedTensors {
		headerMap[k] = info
	}

	headerJSON := []byte(core.JSONMarshalString(headerMap))

	f, err := coreio.Local.Create(path)
	if err != nil {
		return core.Fail(core.E("ml.WriteSafetensors", core.Sprintf("create %s", path), err))
	}
	defer f.Close()

	headerSizeBuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(headerSizeBuf, uint64(len(headerJSON)))

	if _, err := f.Write(headerSizeBuf); err != nil {
		return core.Fail(err)
	}
	if _, err := f.Write(headerJSON); err != nil {
		return core.Fail(err)
	}

	for _, k := range keys {
		if _, err := f.Write(tensorData[k]); err != nil {
			return core.Fail(err)
		}
	}

	return core.Ok(nil)
}

// ConvertMLXtoPEFT converts an MLX LoRA adapter to HuggingFace PEFT format.
func ConvertMLXtoPEFT(safetensorsPath, configPath, outputDir, baseModelName string) core.Result {
	if err := coreio.Local.EnsureDir(outputDir); err != nil {
		return core.Fail(core.E("ml.ConvertMLXtoPEFT", "create output dir", err))
	}

	safetensorsResult := ReadSafetensors(safetensorsPath)
	if !safetensorsResult.OK {
		return core.Fail(core.E("ml.ConvertMLXtoPEFT", "read safetensors", safetensorsResult.Value.(error)))
	}
	loaded := safetensorsResult.Value.(SafetensorsData)
	tensors := loaded.Tensors
	tensorData := loaded.Data
	core.Print(nil, "loaded %d tensors from %s", len(tensors), safetensorsPath)

	peftTensors := make(map[string]SafetensorsTensorInfo)
	peftData := make(map[string][]byte)

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
		return core.Fail(core.E("ml.ConvertMLXtoPEFT", "write safetensors", result.Value.(error)))
	}

	cfgData, err := coreio.Local.Read(configPath)
	if err != nil {
		return core.Fail(core.E("ml.ConvertMLXtoPEFT", "read config", err))
	}

	var mlxConfig struct {
		LoraParameters struct {
			Rank    int     `json:"rank"`
			Scale   float64 `json:"scale"`
			Dropout float64 `json:"dropout"`
		} `json:"lora_parameters"`
	}
	if r := core.JSONUnmarshalString(cfgData, &mlxConfig); !r.OK {
		return core.Fail(core.E("ml.ConvertMLXtoPEFT", "parse config", r.Value.(error)))
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
		return core.Fail(core.E("ml.ConvertMLXtoPEFT", "write adapter_config.json", err))
	}

	core.Print(nil, "converted %d tensors, %d layers, target modules: %v",
		len(peftTensors), len(sortedLayers), sortedModules)

	return core.Ok(nil)
}
