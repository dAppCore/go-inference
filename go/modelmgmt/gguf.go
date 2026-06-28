package modelmgmt

import (
	"cmp"
	"encoding/binary"
	"io"
	"math"
	"regexp"
	"slices"
	"strconv"

	"dappco.re/go"
	"dappco.re/go/inference"
	coreio "dappco.re/go/io"
)

// GGUFInfo re-exports inference.GGUFInfo so ml consumers can inspect GGUF
// metadata without importing a concrete runtime package.
type GGUFInfo = inference.GGUFInfo

// DiscoveredModel re-exports inference.DiscoveredModel for discovery results.
type DiscoveredModel = inference.DiscoveredModel

// ReadGGUFInfo reads GGUF header metadata from a model file or directory.
//
//	r := modelmgmt.ReadGGUFInfo("/models/gemma-3-1b.gguf")
//	if !r.OK { return r }
//	info := r.Value.(modelmgmt.GGUFInfo)
func ReadGGUFInfo(modelPath string) core.Result {
	return core.ResultOf(inference.ReadGGUFInfo(modelPath))
}

// DiscoverModels walks a directory and returns all loadable models found
// (both safetensors directories and standalone GGUF files).
//
//	models := modelmgmt.DiscoverModels("/Volumes/Data/lem/models")
//	// []DiscoveredModel{{Path, ModelType, QuantBits, Format}, ...}
func DiscoverModels(basePath string) []DiscoveredModel {
	return inference.DiscoverModels(basePath)
}

// GGUF format constants.
const (
	ggufMagic     = 0x46554747 // "GGUF" little-endian
	ggufVersion   = 3
	ggufAlignment = 32
)

// GGUF metadata value types.
const (
	ggufTypeUint32  = 4
	ggufTypeFloat32 = 6
	ggufTypeString  = 8
)

// GGML tensor data types.
const (
	ggmlTypeF32  = 0
	ggmlTypeF16  = 1
	ggmlTypeBF16 = 30
)

// ggufMetadata is a key-value pair in the GGUF header.
type ggufMetadata struct {
	key       string
	valueType uint32
	value     any // string, uint32, or float32
}

// ggufTensor describes a tensor in the GGUF file.
type ggufTensor struct {
	name  string
	dims  []uint64
	dtype uint32 // ggmlType*
	data  []byte
}

// gemma3ModuleMap maps HuggingFace module names to GGUF tensor names.
var gemma3ModuleMap = map[string]string{
	"self_attn.q_proj": "attn_q",
	"self_attn.k_proj": "attn_k",
	"self_attn.v_proj": "attn_v",
	"self_attn.o_proj": "attn_output",
	"mlp.gate_proj":    "ffn_gate",
	"mlp.up_proj":      "ffn_up",
	"mlp.down_proj":    "ffn_down",
}

var mlxLoraKeyRe = regexp.MustCompile(`^model\.layers\.(\d+)\.(.*?)\.(lora_[ab])$`)

// blkLayerRe extracts the layer number from a GGUF tensor name (e.g. "blk.5.").
// Compiled once at package init; recompiling per call dominates allocations.
var blkLayerRe = regexp.MustCompile(`blk\.(\d+)\.`)

// MLXTensorToGGUF converts an MLX LoRA tensor name to GGUF LoRA tensor name.
// Input:  "model.layers.0.self_attn.q_proj.lora_a"
// Output: "blk.0.attn_q.weight.lora_a"
func MLXTensorToGGUF(mlxName string) core.Result {
	m := mlxLoraKeyRe.FindStringSubmatch(mlxName)
	if m == nil {
		return core.Fail(core.E("modelmgmt.MLXTensorToGGUF", core.Sprintf("unrecognised MLX LoRA key: %s", mlxName), nil))
	}

	layerNum := m[1]
	module := m[2]
	loraSuffix := m[3]

	ggufModule, ok := gemma3ModuleMap[module]
	if !ok {
		return core.Fail(core.E("modelmgmt.MLXTensorToGGUF", core.Sprintf("unknown module %q in %s", module, mlxName), nil))
	}

	return core.Ok("blk." + layerNum + "." + ggufModule + ".weight." + loraSuffix)
}

// SafetensorsDtypeToGGML maps safetensors dtype strings to GGML types.
func SafetensorsDtypeToGGML(dtype string) core.Result {
	switch dtype {
	case "F32":
		return core.Ok(uint32(ggmlTypeF32))
	case "F16":
		return core.Ok(uint32(ggmlTypeF16))
	case "BF16":
		return core.Ok(uint32(ggmlTypeBF16))
	default:
		return core.Fail(core.E("modelmgmt.SafetensorsDtypeToGGML", core.Sprintf("unsupported dtype %q for GGUF", dtype), nil))
	}
}

// ConvertMLXtoGGUFLoRA converts an MLX LoRA adapter to GGUF LoRA format.
func ConvertMLXtoGGUFLoRA(safetensorsPath, configPath, outputPath, architecture string) core.Result {
	cfgData, err := coreio.Local.Read(configPath)
	if err != nil {
		return core.Fail(core.E("modelmgmt.ConvertMLXtoGGUFLoRA", "read config", err))
	}

	var mlxConfig struct {
		LoraParameters struct {
			Rank  int     `json:"rank"`
			Scale float64 `json:"scale"`
		} `json:"lora_parameters"`
	}
	if r := core.JSONUnmarshalString(cfgData, &mlxConfig); !r.OK {
		return core.Fail(core.E("modelmgmt.ConvertMLXtoGGUFLoRA", "parse config", r.Value.(error)))
	}

	rank := mlxConfig.LoraParameters.Rank
	if rank == 0 {
		rank = 8
	}
	scale := mlxConfig.LoraParameters.Scale
	if scale == 0 {
		scale = 20.0
	}
	loraAlpha := float32(math.Round(scale * float64(rank)))

	safetensorsResult := ReadSafetensors(safetensorsPath)
	if !safetensorsResult.OK {
		return core.Fail(core.E("modelmgmt.ConvertMLXtoGGUFLoRA", "read safetensors", safetensorsResult.Value.(error)))
	}
	loaded := safetensorsResult.Value.(SafetensorsData)
	tensors := loaded.Tensors
	tensorData := loaded.Data
	core.Print(nil, "loaded %d tensors from %s", len(tensors), safetensorsPath)

	ggufTensors := make([]ggufTensor, 0, len(tensors))
	for mlxKey, info := range tensors {
		ggufNameResult := MLXTensorToGGUF(mlxKey)
		if !ggufNameResult.OK {
			return ggufNameResult
		}
		ggufName := ggufNameResult.Value.(string)

		ggmlTypeResult := SafetensorsDtypeToGGML(info.Dtype)
		if !ggmlTypeResult.OK {
			return core.Fail(core.E("modelmgmt.ConvertMLXtoGGUFLoRA", core.Sprintf("tensor %s", mlxKey), ggmlTypeResult.Value.(error)))
		}
		ggmlType := ggmlTypeResult.Value.(uint32)

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
			ggufTensors = append(ggufTensors, ggufTensor{
				name:  ggufName,
				dims:  []uint64{uint64(rows), uint64(cols)},
				dtype: ggmlType,
				data:  data,
			})
		} else {
			dims := make([]uint64, len(info.Shape))
			for i, s := range info.Shape {
				dims[i] = uint64(s)
			}
			ggufTensors = append(ggufTensors, ggufTensor{
				name:  ggufName,
				dims:  dims,
				dtype: ggmlType,
				data:  data,
			})
		}
	}

	slices.SortFunc(ggufTensors, func(a, b ggufTensor) int {
		return cmp.Compare(a.name, b.name)
	})

	metadata := []ggufMetadata{
		{key: "general.type", valueType: ggufTypeString, value: "adapter"},
		{key: "general.architecture", valueType: ggufTypeString, value: architecture},
		{key: "adapter.type", valueType: ggufTypeString, value: "lora"},
		{key: "adapter.lora.alpha", valueType: ggufTypeFloat32, value: loraAlpha},
	}

	if result := writeGGUF(outputPath, metadata, ggufTensors); !result.OK {
		return core.Fail(core.E("modelmgmt.ConvertMLXtoGGUFLoRA", "write GGUF", result.Value.(error)))
	}

	core.Print(nil, "wrote GGUF LoRA: %s (%d tensors, alpha=%.0f)", outputPath, len(ggufTensors), loraAlpha)
	return core.Ok(nil)
}

// writeGGUF writes a GGUF v3 file.
func writeGGUF(path string, metadata []ggufMetadata, tensors []ggufTensor) core.Result {
	f, err := coreio.Local.Create(path)
	if err != nil {
		return core.Fail(err)
	}
	defer f.Close()

	w := &ggufWriter{f: f}

	w.writeUint32(ggufMagic)
	w.writeUint32(ggufVersion)
	w.writeUint64(uint64(len(tensors)))
	w.writeUint64(uint64(len(metadata)))

	for _, kv := range metadata {
		w.writeString(kv.key)
		w.writeUint32(kv.valueType)
		switch kv.valueType {
		case ggufTypeString:
			w.writeString(kv.value.(string))
		case ggufTypeUint32:
			w.writeUint32(kv.value.(uint32))
		case ggufTypeFloat32:
			w.writeFloat32(kv.value.(float32))
		}
	}

	dataOffset := uint64(0)
	for _, t := range tensors {
		w.writeString(t.name)
		w.writeUint32(uint32(len(t.dims)))
		for _, d := range t.dims {
			w.writeUint64(d)
		}
		w.writeUint32(t.dtype)
		w.writeUint64(dataOffset)

		dataOffset += uint64(len(t.data))
		if rem := dataOffset % ggufAlignment; rem != 0 {
			dataOffset += ggufAlignment - rem
		}
	}

	pos := w.pos
	if rem := pos % ggufAlignment; rem != 0 {
		pad := ggufAlignment - rem
		w.writeBytes(make([]byte, pad))
	}

	for _, t := range tensors {
		w.writeBytes(t.data)
		if rem := uint64(len(t.data)) % ggufAlignment; rem != 0 {
			w.writeBytes(make([]byte, ggufAlignment-rem))
		}
	}

	return core.ResultOf(nil, w.err)
}

// ggufWriter tracks position and accumulates errors.
type ggufWriter struct {
	f   io.WriteCloser
	pos uint64
	err error
}

func (w *ggufWriter) writeBytes(b []byte) {
	if w.err != nil {
		return
	}
	n, err := w.f.Write(b)
	w.pos += uint64(n)
	if err != nil {
		w.err = err
	}
}

func (w *ggufWriter) writeUint32(v uint32) {
	b := make([]byte, 4)
	binary.LittleEndian.PutUint32(b, v)
	w.writeBytes(b)
}

func (w *ggufWriter) writeUint64(v uint64) {
	b := make([]byte, 8)
	binary.LittleEndian.PutUint64(b, v)
	w.writeBytes(b)
}

func (w *ggufWriter) writeFloat32(v float32) {
	w.writeUint32(math.Float32bits(v))
}

func (w *ggufWriter) writeString(s string) {
	w.writeUint64(uint64(len(s)))
	w.writeBytes([]byte(s))
}

// DetectArchFromConfig tries to infer the model architecture from adapter_config.json.
func DetectArchFromConfig(configPath string) string {
	data, err := coreio.Local.Read(configPath)
	if err != nil {
		return "gemma3"
	}
	var cfg struct {
		LoraParameters struct {
			Rank int `json:"rank"`
		} `json:"lora_parameters"`
	}
	core.JSONUnmarshalString(data, &cfg)
	return "gemma3"
}

// ArchitectureGGUFMap maps model tags to GGUF architecture names.
var ArchitectureGGUFMap = map[string]string{
	"gemma-3-1b":  "gemma3",
	"gemma-3-4b":  "gemma3",
	"gemma-3-12b": "gemma3",
	"gemma-3-27b": "gemma3",
}

// ModelTagToGGUFArch returns the GGUF architecture for a model tag.
func ModelTagToGGUFArch(modelTag string) string {
	if arch, ok := ArchitectureGGUFMap[modelTag]; ok {
		return arch
	}
	return "gemma3"
}

// GGUFModelBlobPath returns the path to the GGUF model blob in Ollama's store.
func GGUFModelBlobPath(ollamaModelsDir, modelName string) core.Result {
	parts := core.SplitN(modelName, ":", 2)
	family := parts[0]
	tag := "latest"
	if len(parts) > 1 {
		tag = parts[1]
	}

	manifestPath := core.Sprintf("%s/manifests/registry.ollama.ai/library/%s/%s", ollamaModelsDir, family, tag)
	data, err := coreio.Local.Read(manifestPath)
	if err != nil {
		return core.Fail(core.E("modelmgmt.GGUFModelBlobPath", core.Sprintf("read manifest %s", manifestPath), err))
	}

	var manifest struct {
		Layers []struct {
			MediaType string `json:"mediaType"`
			Digest    string `json:"digest"`
		} `json:"layers"`
	}
	if r := core.JSONUnmarshalString(data, &manifest); !r.OK {
		return core.Fail(core.E("modelmgmt.GGUFModelBlobPath", "parse manifest", r.Value.(error)))
	}

	for _, layer := range manifest.Layers {
		if layer.MediaType == "application/vnd.ollama.image.model" {
			blobName := core.Replace(layer.Digest, ":", "-")
			return core.Ok(core.Sprintf("%s/blobs/%s", ollamaModelsDir, blobName))
		}
	}

	return core.Fail(core.E("modelmgmt.GGUFModelBlobPath", core.Sprintf("no model layer found in manifest for %s", modelName), nil))
}

// ParseLayerFromTensorName extracts the layer number from a GGUF tensor name.
func ParseLayerFromTensorName(name string) core.Result {
	m := blkLayerRe.FindStringSubmatch(name)
	if m == nil {
		return core.Fail(core.E("modelmgmt.ParseLayerFromTensorName", core.Sprintf("no layer number in %s", name), nil))
	}
	return core.ResultOf(strconv.Atoi(m[1]))
}
