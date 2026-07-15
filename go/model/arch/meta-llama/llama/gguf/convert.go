// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"encoding/binary"
	"math"

	core "dappco.re/go"
	basegguf "dappco.re/go/inference/model/gguf"
)

func isLlamaConfig(configJSON []byte) bool {
	var probe struct {
		ModelType string `json:"model_type"`
	}
	return core.JSONUnmarshal(configJSON, &probe).OK && probe.ModelType == llamaArch
}

func isLlamaSupportedQuantizeFormat(format basegguf.QuantizeFormat) bool {
	return format == basegguf.QuantizeQ4_K_M || format == basegguf.QuantizeQ8_0
}

func quantizeLlamaModelPack(source basegguf.Source, configJSON []byte, tensors []basegguf.DenseSafetensor, format basegguf.QuantizeFormat) ([]basegguf.Tensor, []basegguf.MetadataEntry, error) {
	config, err := parseLlamaConfig(configJSON)
	if err != nil {
		return nil, nil, err
	}
	quantized := make([]basegguf.Tensor, 0, len(tensors))
	for _, tensor := range tensors {
		canonical, nameErr := llamaCanonicalTensorName(tensor.Name)
		if nameErr != nil {
			return nil, nil, nameErr
		}
		tensorType := llamaTensorType(format, canonical, llamaCanonicalLayerIndex(canonical), config.NumHiddenLayers, config.TieWordEmbeddings)
		data, encodeErr := encodeLlamaTensorData(tensor.Data, tensorType)
		if encodeErr != nil {
			return nil, nil, core.E("quantizeLlamaModelPack", "encode "+canonical, encodeErr)
		}
		quantized = append(quantized, basegguf.Tensor{Name: canonical, Type: tensorType, Shape: llamaGGUFShape(tensor.Shape), Data: data})
	}
	if len(quantized) == 0 {
		return nil, nil, core.NewError("gguf: no llama tensors found in source")
	}
	metadata, err := llamaMetadata(configJSON, llamaFileType(format), llamaModelName(source.Root))
	if err != nil {
		return nil, nil, err
	}
	tokenizerRead := core.ReadFile(core.PathJoin(source.Root, "tokenizer.json"))
	if !tokenizerRead.OK {
		return nil, nil, core.E("quantizeLlamaModelPack", "read tokenizer.json", tokenizerRead.Err())
	}
	tokenizerMetadata, err := llamaTokenizer(tokenizerRead.Value.([]byte), config)
	if err != nil {
		return nil, nil, err
	}
	metadata = append(metadata, tokenizerMetadata...)
	if template := core.ReadFile(core.PathJoin(source.Root, "chat_template.jinja")); template.OK {
		metadata = append(metadata, basegguf.MetadataEntry{Key: "tokenizer.chat_template", ValueType: basegguf.ValueTypeString, Value: core.AsString(template.Value.([]byte))})
	}
	return quantized, metadata, nil
}

func llamaModelName(root string) string {
	return core.TrimSuffix(core.TrimSuffix(core.PathBase(core.TrimSuffix(root, "/")), "-bf16"), "-f32")
}

func encodeLlamaTensorData(values []float32, tensorType uint32) ([]byte, error) {
	switch tensorType {
	case basegguf.TensorTypeF32:
		encoded := make([]byte, len(values)*4)
		for i, value := range values {
			binary.LittleEndian.PutUint32(encoded[i*4:], math.Float32bits(value))
		}
		return encoded, nil
	case basegguf.TensorTypeQ8_0:
		if len(values)%32 != 0 {
			return nil, core.Errorf("gguf: Q8_0 tensor has %d elements, not a multiple of 32", len(values))
		}
		return basegguf.Quantize(basegguf.QuantizeQ8_0, values)
	case basegguf.TensorTypeQ4K:
		if len(values)%256 != 0 {
			return nil, core.Errorf("gguf: Q4_K tensor has %d elements, not a multiple of 256", len(values))
		}
		return basegguf.Quantize(basegguf.QuantizeQ4_K, values)
	case basegguf.TensorTypeQ5K:
		if len(values)%256 != 0 {
			return nil, core.Errorf("gguf: Q5_K tensor has %d elements, not a multiple of 256", len(values))
		}
		return basegguf.Quantize(basegguf.QuantizeQ5_K, values)
	case basegguf.TensorTypeQ6K:
		if len(values)%256 != 0 {
			return nil, core.Errorf("gguf: Q6_K tensor has %d elements, not a multiple of 256", len(values))
		}
		return basegguf.Quantize(basegguf.QuantizeQ6_K, values)
	default:
		return nil, core.Errorf("gguf: unsupported llama tensor type %d", tensorType)
	}
}
