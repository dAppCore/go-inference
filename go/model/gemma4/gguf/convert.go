// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"
	"strconv"

	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

// gemma4ModelType is the config.json model_type that selects the gemma-4
// conversion lane.
const gemma4ModelType = "gemma4"

// gemma4ConfigModelType decodes just the top-level model_type from a config.json
// so the converter can detect the gemma-4 lane without parsing the whole file.
type gemma4ConfigModelType struct {
	ModelType string `json:"model_type"`
}

// isGemma4Config reports whether a config.json describes a gemma-4 model.
func isGemma4Config(configJSON []byte) bool {
	var probe gemma4ConfigModelType
	if r := core.JSONUnmarshal(configJSON, &probe); !r.OK {
		return false
	}
	return probe.ModelType == gemma4ModelType
}

// isGemma4SupportedQuantizeFormat reports whether format has an oracle-grade
// gemma-4 export receipt: #28 established q4_k_m against the unsloth
// gemma-4-E2B-it-Q4_K_M oracle; #53 added q8_0 (oracle-confirmed against the
// unsloth gemma-4-E2B-it-Q8_0.gguf), q6_k, q5_k_m, and q3_k_m (llama.cpp's
// own quantize source as the reference — no oracle on disk for those three).
// q2_k uses llama.cpp's conventional mixed Q2_K recipe. GPTQ/AWQ/fp8/NF4/MX
// remain gated on an operator decision (docs/design-quant-formats.md).
func isGemma4SupportedQuantizeFormat(format basegguf.QuantizeFormat) bool {
	switch format {
	case basegguf.QuantizeQ4_K_M, basegguf.QuantizeQ8_0, basegguf.QuantizeQ6_K, basegguf.QuantizeQ5_K_M, basegguf.QuantizeQ3_K_M, basegguf.QuantizeQ2_K_M:
		return true
	default:
		return false
	}
}

// gemma4CanonicalLayerIndex returns the block index of a canonical per-layer
// tensor (blk.<N>.*), or -1 for a whole-model tensor.
func gemma4CanonicalLayerIndex(canonical string) int {
	rest, ok := core.CutPrefix(canonical, "blk.")
	if !ok {
		return -1
	}
	index, _, found := core.Cut(rest, ".")
	if !found {
		return -1
	}
	n, err := strconv.Atoi(index)
	if err != nil {
		return -1
	}
	return n
}

// quantizeGemma4ModelPack quantises a gemma-4 dense safetensors checkpoint into
// the tensor and metadata records for a text GGUF llama.cpp loads: canonical
// tensor names, reversed GGUF shapes, format's per-tensor type policy
// (gemma4TensorType), and the full gemma4.* + tokenizer.ggml.* header. The
// multimodal towers are excluded (a text GGUF carries the language model
// only). format must be one of isGemma4SupportedQuantizeFormat's set — the
// caller (basegguf.QuantizeModelPack, via the registered QuantizeLane) gates
// this before calling in.
func quantizeGemma4ModelPack(source basegguf.Source, configJSON []byte, tensors []basegguf.DenseSafetensor, format basegguf.QuantizeFormat) ([]basegguf.Tensor, []basegguf.MetadataEntry, error) {
	var config gemma4Config
	if r := core.JSONUnmarshal(configJSON, &config); !r.OK {
		return nil, nil, core.E("quantizeGemma4ModelPack", "parse config.json", r.Err())
	}
	layerCount := config.TextConfig.NumHiddenLayers
	if layerCount <= 0 {
		return nil, nil, core.NewError("gguf: gemma4 config.json has no text_config.num_hidden_layers")
	}

	// Per-layer MLP width, filled from each layer's ffn_gate tensor. gemma-4's
	// MatFormer double-wide MLP makes the deeper layers 2x wide, so this is read
	// from the tensors (ground truth) rather than broadcast from config.
	feedForward := make([]int32, layerCount)

	quantized := make([]basegguf.Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		if basegguf.IsMultimodalTowerTensor(tensor.Name) {
			continue
		}
		canonical, err := gemma4CanonicalTensorName(tensor.Name)
		if err != nil {
			return nil, nil, err
		}
		layerIndex := gemma4CanonicalLayerIndex(canonical)
		if layerIndex >= 0 && layerIndex < layerCount && core.HasSuffix(canonical, ".ffn_gate.weight") && len(tensor.Shape) > 0 {
			// The source ffn_gate shape is [out_width, hidden]; the out width is
			// the per-layer feed_forward_length.
			feedForward[layerIndex] = int32(tensor.Shape[0])
		}
		tensorType := gemma4TensorType(format, canonical, layerIndex, layerCount)
		data, err := encodeGemma4TensorData(tensor.Data, tensorType)
		if err != nil {
			return nil, nil, core.E("quantizeGemma4ModelPack", "encode "+canonical, err)
		}
		quantized = append(quantized, basegguf.Tensor{
			Name:  canonical,
			Type:  tensorType,
			Shape: gemma4GGUFShape(tensor.Shape),
			Data:  data,
		})
	}
	if len(quantized) == 0 {
		return nil, nil, core.NewError("gguf: no gemma4 text-stack tensors found in source")
	}
	for layer, width := range feedForward {
		if width <= 0 {
			return nil, nil, core.Errorf("gguf: gemma4 layer %d has no ffn_gate tensor to size feed_forward_length", layer)
		}
	}

	// rope_freqs.weight is a computed tensor (no source counterpart) llama.cpp
	// requires for gemma-4: the partial-rotary frequency mask for the
	// full-attention layers.
	ropeFreqs, err := gemma4RopeFreqsTensor(config.TextConfig.GlobalHeadDim, config.TextConfig.RopeParameters.FullAttention.PartialRotaryFactor)
	if err != nil {
		return nil, nil, err
	}
	quantized = append(quantized, ropeFreqs)

	metadata, err := gemma4FullMetadata(source.Root, configJSON, feedForward, format)
	if err != nil {
		return nil, nil, err
	}
	return quantized, metadata, nil
}

// gemma4FileType returns the GGUF general.file_type value llama.cpp's own
// LlamaFileType enum assigns to format — the LLAMA_FTYPE_MOSTLY_* constant a
// real llama-quantize run records for this recipe. Cross-checked against the
// general.file_type key actually stored in the on-disk unsloth
// gemma-4-E2B-it-Q4_K_M.gguf (15) and gemma-4-E2B-it-Q8_0.gguf (7) oracles,
// and against gguf-py's LlamaFileType table (pip gguf==0.17.1) for q6_k/
// q5_k_m/q3_k_m, which have no oracle on disk to read the key from directly.
func gemma4FileType(format basegguf.QuantizeFormat) uint32 {
	switch format {
	case basegguf.QuantizeQ8_0:
		return 7
	case basegguf.QuantizeQ3_K_M:
		return 12
	case basegguf.QuantizeQ2_K_M:
		return 10
	case basegguf.QuantizeQ5_K_M:
		return 17
	case basegguf.QuantizeQ6_K:
		return 18
	default: // basegguf.QuantizeQ4_K_M
		return 15
	}
}

// gemma4FullMetadata assembles the complete gemma-4 header: the architecture
// hyperparameters from config.json (with per-layer feed_forward_length taken
// from the tensor widths) and the tokenizer block from tokenizer.json (read
// from the checkpoint root). general.file_type follows format (gemma4FileType).
func gemma4FullMetadata(root string, configJSON []byte, feedForward []int32, format basegguf.QuantizeFormat) ([]basegguf.MetadataEntry, error) {
	metadata, err := gemma4Metadata(configJSON, feedForward, gemma4FileType(format), gemma4ModelName(root))
	if err != nil {
		return nil, err
	}
	tokenizerRead := core.ReadFile(core.PathJoin(root, "tokenizer.json"))
	if !tokenizerRead.OK {
		return nil, core.E("gemma4FullMetadata", "read tokenizer.json", tokenizerRead.Err())
	}
	tokenizerEntries, err := gemma4Tokenizer(tokenizerRead.Bytes())
	if err != nil {
		return nil, err
	}
	metadata = append(metadata, tokenizerEntries...)

	// tokenizer.chat_template, when the checkpoint ships one, lets llama.cpp
	// apply the gemma chat format (the instruction-tuned model expects it;
	// without it a raw prompt yields an immediate end-of-turn). Best-effort:
	// the file is optional and its absence just omits the key.
	if tmpl := core.ReadFile(core.PathJoin(root, "chat_template.jinja")); tmpl.OK {
		metadata = append(metadata, basegguf.MetadataEntry{
			Key:       "tokenizer.chat_template",
			ValueType: basegguf.ValueTypeString,
			Value:     core.AsString(tmpl.Bytes()),
		})
	}
	return metadata, nil
}

// gemma4ModelName derives a general.name from the checkpoint directory basename,
// dropping the -bf16/-f32 dense-precision suffix (gemma-4-E2B-it-bf16 ->
// gemma-4-E2B-it). An empty result (unusual root) simply omits general.name.
func gemma4ModelName(root string) string {
	base := core.PathBase(core.TrimSuffix(root, "/"))
	base = core.TrimSuffix(base, "-bf16")
	base = core.TrimSuffix(base, "-f32")
	return base
}

// gemma4RopeFreqsDisabled is the frequency-scale sentinel llama.cpp reads as
// "do not rotate this dimension pair": a huge divisor drives the effective
// rotary frequency to ~0. gemma-4's oracle uses 1e30 for the non-rotated tail.
const gemma4RopeFreqsDisabled = 1e30

// gemma4RopeFreqsTensor builds the computed rope_freqs.weight tensor llama.cpp
// requires for gemma-4's partial-rotary full-attention rope. The tensor has
// dimensionCount/2 float32 entries: the first (dimensionCount *
// partialRotaryFactor)/2 frequency pairs rotate (scale 1.0) and the remainder
// are disabled (scale 1e30). For gemma-4-E2B (dimensionCount 512,
// partial_rotary_factor 0.25) this yields 64 ones followed by 192 sentinels,
// matching the oracle exactly. Derived from config; no source tensor exists.
func gemma4RopeFreqsTensor(dimensionCount int, partialRotaryFactor float32) (basegguf.Tensor, error) {
	if dimensionCount <= 0 || dimensionCount%2 != 0 {
		return basegguf.Tensor{}, core.Errorf("gguf: gemma4 rope dimension_count %d must be a positive even number", dimensionCount)
	}
	if partialRotaryFactor <= 0 || partialRotaryFactor > 1 {
		return basegguf.Tensor{}, core.Errorf("gguf: gemma4 partial_rotary_factor %g must be in (0,1]", partialRotaryFactor)
	}
	length := dimensionCount / 2
	rotaryPairs := int(float64(dimensionCount)*float64(partialRotaryFactor)) / 2
	freqs := make([]float32, length)
	for i := range freqs {
		if i < rotaryPairs {
			freqs[i] = 1.0
		} else {
			freqs[i] = gemma4RopeFreqsDisabled
		}
	}
	return basegguf.Tensor{
		Name:  "rope_freqs.weight",
		Type:  basegguf.TensorTypeF32,
		Shape: []uint64{uint64(length)},
		Data:  encodeGemma4F32(freqs),
	}, nil
}

// encodeGemma4TensorData encodes dense float32 tensor values into the GGUF data
// bytes for tensorType: raw little-endian F32, truncated BF16, the Q8_0
// 32-element block stream, or a 256-element-superblock K-quant stream. A
// block-quantised tensor whose element count is not a whole number of blocks
// is an error rather than a silently mis-encoded tensor. The block-quantised
// cases route through basegguf.Quantize — this package's own copy of the
// nine quantise kernels retired with the relocation (#59); Quantize is
// byte-identical (same kernel, same pre-sized capacity) and is the format
// package's intended public entry, not a private implementation detail.
func encodeGemma4TensorData(data []float32, tensorType uint32) ([]byte, error) {
	switch tensorType {
	case basegguf.TensorTypeF32:
		return encodeGemma4F32(data), nil
	case basegguf.TensorTypeBF16:
		return encodeGemma4BF16(data), nil
	case basegguf.TensorTypeQ8_0:
		if len(data)%32 != 0 {
			return nil, core.Errorf("gguf: Q8_0 tensor has %d elements, not a multiple of 32", len(data))
		}
		return basegguf.Quantize(basegguf.QuantizeQ8_0, data)
	case basegguf.TensorTypeQ2K:
		if len(data)%256 != 0 {
			return nil, core.Errorf("gguf: Q2_K tensor has %d elements, not a multiple of 256", len(data))
		}
		return basegguf.Quantize(basegguf.QuantizeQ2_K, data)
	case basegguf.TensorTypeQ3K:
		if len(data)%256 != 0 {
			return nil, core.Errorf("gguf: Q3_K tensor has %d elements, not a multiple of 256", len(data))
		}
		return basegguf.Quantize(basegguf.QuantizeQ3_K, data)
	case basegguf.TensorTypeQ4K:
		if len(data)%256 != 0 {
			return nil, core.Errorf("gguf: Q4_K tensor has %d elements, not a multiple of 256", len(data))
		}
		return basegguf.Quantize(basegguf.QuantizeQ4_K, data)
	case basegguf.TensorTypeQ5K:
		if len(data)%256 != 0 {
			return nil, core.Errorf("gguf: Q5_K tensor has %d elements, not a multiple of 256", len(data))
		}
		return basegguf.Quantize(basegguf.QuantizeQ5_K, data)
	case basegguf.TensorTypeQ6K:
		if len(data)%256 != 0 {
			return nil, core.Errorf("gguf: Q6_K tensor has %d elements, not a multiple of 256", len(data))
		}
		return basegguf.Quantize(basegguf.QuantizeQ6_K, data)
	default:
		return nil, core.Errorf("gguf: gemma4 has no encoder for tensor type %d", tensorType)
	}
}

// encodeGemma4F32 packs float32 values as little-endian F32 bytes.
func encodeGemma4F32(data []float32) []byte {
	out := make([]byte, len(data)*4)
	for i, v := range data {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(v))
	}
	return out
}

// encodeGemma4BF16 packs float32 values as little-endian BF16 (bfloat16) bytes
// using round-to-nearest-even. gemma-4's per_layer_model_proj is BF16 in both
// the source checkpoint and the oracle; since the source is already BF16 the
// float32 round-trip is exact, but rounding is applied so the encoder is
// correct for any float32 input.
func encodeGemma4BF16(data []float32) []byte {
	out := make([]byte, len(data)*2)
	for i, v := range data {
		binary.LittleEndian.PutUint16(out[i*2:], float32ToBF16(v))
	}
	return out
}

// float32ToBF16 truncates a float32 to bfloat16 (the high 16 bits) with
// round-to-nearest-even. NaN payloads are preserved as a quiet NaN.
func float32ToBF16(f float32) uint16 {
	bits := math.Float32bits(f)
	if bits&0x7FFFFFFF > 0x7F800000 {
		// NaN — keep it a NaN after truncation.
		return uint16(bits>>16) | 0x0040
	}
	rounded := bits + 0x7FFF + ((bits >> 16) & 1)
	return uint16(rounded >> 16)
}
