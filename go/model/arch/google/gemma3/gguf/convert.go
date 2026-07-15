// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

// gemma3ModelTypes are the config.json model_type values that select the
// gemma-3 conversion lane. HF spells the text model "gemma3_text"; "gemma3" is
// accepted for a checkpoint that names the family directly.
var gemma3ModelTypes = map[string]bool{"gemma3_text": true, "gemma3": true}

// gemma3ConfigModelType decodes just the top-level model_type from a config.json
// so the converter can detect the gemma-3 lane without parsing the whole file.
type gemma3ConfigModelType struct {
	ModelType string `json:"model_type"`
}

// isGemma3Config reports whether a config.json describes a gemma-3 text model.
func isGemma3Config(configJSON []byte) bool {
	var probe gemma3ConfigModelType
	if r := core.JSONUnmarshal(configJSON, &probe); !r.OK {
		return false
	}
	return gemma3ModelTypes[probe.ModelType]
}

// isGemma3SupportedQuantizeFormat reports whether format has a gemma-3 export
// path. The K-quant "_M" recipes and q8_0 are supported; every tensor whose
// inner row length is not a 256-multiple falls back to a 32-block quant
// (gemma3TensorType), so the file loads regardless of hidden size. The
// data-dependent / non-GGUF formats (GPTQ/AWQ/fp8/NF4/MX) are out of scope for
// this lane.
func isGemma3SupportedQuantizeFormat(format basegguf.QuantizeFormat) bool {
	switch format {
	case basegguf.QuantizeQ4_K_M, basegguf.QuantizeQ8_0, basegguf.QuantizeQ6_K,
		basegguf.QuantizeQ5_K_M, basegguf.QuantizeQ3_K_M, basegguf.QuantizeQ2_K_M:
		return true
	default:
		return false
	}
}

// quantizeGemma3ModelPack converts a gemma-3 dense safetensors checkpoint into
// the tensor and metadata records for a text GGUF llama.cpp loads: canonical
// tensor names, reversed GGUF shapes, the divisibility-aware per-tensor type
// policy (gemma3TensorType), the gemma "(1 + weight)" RMS-norm fold baked into
// every norm weight, and the full gemma3.* + tokenizer.ggml.* header. format
// must be one of isGemma3SupportedQuantizeFormat's set — the caller
// (basegguf.QuantizeModelPack, via the registered QuantizeLane) gates this
// before calling in.
func quantizeGemma3ModelPack(source basegguf.Source, configJSON []byte, tensors []basegguf.DenseSafetensor, format basegguf.QuantizeFormat) ([]basegguf.Tensor, []basegguf.MetadataEntry, error) {
	quantized := make([]basegguf.Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		canonical, err := gemma3CanonicalTensorName(tensor.Name)
		if err != nil {
			return nil, nil, err
		}
		tensorType := gemma3TensorType(format, canonical, gemma3TensorRowLength(tensor.Shape))
		values := tensor.Data
		if core.HasSuffix(canonical, "norm.weight") {
			// Gemma3RMSNorm computes output * (1 + weight); llama.cpp's gemma3
			// graph reads the weight directly, so the +1 is folded into the
			// stored weight at export (convert_hf_to_gguf's norm_shift). Fold on
			// a copy — the source slice is shared with no other consumer here,
			// but a copy keeps the transform obviously side-effect-free.
			values = gemma3AddScalar(tensor.Data, 1.0)
		}
		data, err := encodeGemma3TensorData(values, tensorType)
		if err != nil {
			return nil, nil, core.E("quantizeGemma3ModelPack", "encode "+canonical, err)
		}
		quantized = append(quantized, basegguf.Tensor{
			Name:  canonical,
			Type:  tensorType,
			Shape: gemma3GGUFShape(tensor.Shape),
			Data:  data,
		})
	}
	if len(quantized) == 0 {
		return nil, nil, core.NewError("gguf: no gemma3 tensors found in source")
	}

	metadata, err := gemma3Metadata(configJSON, gemma3FileType(format), gemma3ModelName(source.Root))
	if err != nil {
		return nil, nil, err
	}
	tokenizerEntries, err := gemma3Tokenizer(source.Root)
	if err != nil {
		return nil, nil, err
	}
	metadata = append(metadata, tokenizerEntries...)

	// tokenizer.chat_template, when the checkpoint ships one, lets llama.cpp
	// apply the gemma chat format in -cnv mode. Best-effort: the file is
	// optional and does not gate loading.
	if tmpl := core.ReadFile(core.PathJoin(source.Root, "chat_template.jinja")); tmpl.OK {
		metadata = append(metadata, basegguf.MetadataEntry{
			Key:       "tokenizer.chat_template",
			ValueType: basegguf.ValueTypeString,
			Value:     core.AsString(tmpl.Value.([]byte)),
		})
	}
	return quantized, metadata, nil
}

// gemma3ModelName derives a general.name from the checkpoint directory basename,
// dropping the -bf16/-f32 dense-precision suffix. An empty result simply omits
// general.name.
func gemma3ModelName(root string) string {
	base := core.PathBase(core.TrimSuffix(root, "/"))
	base = core.TrimSuffix(base, "-bf16")
	base = core.TrimSuffix(base, "-f32")
	return base
}

// gemma3AddScalar returns a copy of data with delta added to every element —
// used to fold the gemma "(1 + weight)" RMS-norm bias into a norm tensor without
// mutating the caller's slice.
func gemma3AddScalar(data []float32, delta float32) []float32 {
	out := make([]float32, len(data))
	for i, v := range data {
		out[i] = v + delta
	}
	return out
}

// encodeGemma3TensorData encodes dense float32 tensor values into the GGUF data
// bytes for tensorType: raw little-endian F32 for norms, the Q8_0 32-element
// block stream, or a 256-element-superblock K-quant stream. A block-quantised
// tensor whose element count is not a whole number of blocks is an error rather
// than a silently mis-encoded tensor. The block-quantised cases route through
// basegguf.Quantize, the format package's shared quantise entry.
func encodeGemma3TensorData(data []float32, tensorType uint32) ([]byte, error) {
	switch tensorType {
	case basegguf.TensorTypeF32:
		return gemma3EncodeF32(data), nil
	case basegguf.TensorTypeQ8_0:
		return basegguf.Quantize(basegguf.QuantizeQ8_0, data)
	case basegguf.TensorTypeQ2K:
		return basegguf.Quantize(basegguf.QuantizeQ2_K, data)
	case basegguf.TensorTypeQ3K:
		return basegguf.Quantize(basegguf.QuantizeQ3_K, data)
	case basegguf.TensorTypeQ4K:
		return basegguf.Quantize(basegguf.QuantizeQ4_K, data)
	case basegguf.TensorTypeQ5K:
		return basegguf.Quantize(basegguf.QuantizeQ5_K, data)
	case basegguf.TensorTypeQ6K:
		return basegguf.Quantize(basegguf.QuantizeQ6_K, data)
	default:
		return nil, core.Errorf("gguf: gemma3 has no encoder for tensor type %d", tensorType)
	}
}

// gemma3EncodeF32 packs float32 values as little-endian F32 bytes.
func gemma3EncodeF32(data []float32) []byte {
	out := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}
