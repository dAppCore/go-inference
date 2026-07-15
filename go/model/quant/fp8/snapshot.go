// SPDX-Licence-Identifier: EUPL-1.2

package fp8

import (
	"context"
	"encoding/binary"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

type Result struct {
	OutputDir, WeightFile, ConfigFile               string
	TensorCount, QuantizedWeights, PassthroughCount int
	SourceBytes, OutputBytes                        int64
}

func ConvertSnapshot(ctx context.Context, srcDir, outDir string, progress func(string, bool, int, int)) (*Result, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if core.Trim(srcDir) == "" || core.Trim(outDir) == "" {
		return nil, core.NewError("fp8: source and output directories are required")
	}
	shards := core.PathGlob(core.PathJoin(srcDir, "*.safetensors"))
	core.SliceSort(shards)
	if len(shards) == 0 {
		return nil, core.NewError("fp8: source has no safetensors shards")
	}
	idx, err := safetensors.IndexFiles(shards)
	if err != nil {
		return nil, core.E("fp8.ConvertSnapshot", "index source shards", err)
	}
	names := core.SliceClone(idx.Names)
	core.SliceSort(names)
	cache := safetensors.NewShardCache()
	defer cache.Close()
	tensors := make(map[string]safetensors.Tensor)
	result := &Result{OutputDir: outDir}
	for i, name := range names {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		ref := idx.Tensors[name]
		eligible := core.HasSuffix(name, ".weight") && len(ref.Shape) == 2 && isFloat(ref.DType) && isFP8LinearWeight(name)
		if progress != nil {
			progress(name, eligible, i+1, len(names))
		}
		raw, readErr := cache.ReadRefRaw(ref)
		if readErr != nil {
			return nil, core.E("fp8.ConvertSnapshot", "read "+name, readErr)
		}
		result.SourceBytes += ref.ByteLen
		shape := ints(ref.Shape)
		if !eligible {
			tensors[name] = safetensors.Tensor{Dtype: ref.DType, Shape: shape, Data: raw}
			result.PassthroughCount++
			continue
		}
		values, decodeErr := safetensors.DecodeFloat32(ref.DType, raw, product(shape))
		if decodeErr != nil {
			return nil, decodeErr
		}
		quantized, quantErr := Quantize(values)
		if quantErr != nil {
			return nil, quantErr
		}
		tensors[name] = safetensors.Tensor{Dtype: "F8_E4M3", Shape: shape, Data: quantized.Data}
		var scale [4]byte
		binary.LittleEndian.PutUint32(scale[:], math.Float32bits(quantized.Scale))
		tensors[core.TrimSuffix(name, ".weight")+".weight_scale"] = safetensors.Tensor{Dtype: "F32", Shape: []int{}, Data: scale[:]}
		result.QuantizedWeights++
	}
	result.TensorCount = len(tensors)
	blob, err := safetensors.Encode(tensors)
	if err != nil {
		return nil, err
	}
	if r := core.MkdirAll(outDir, 0o755); !r.OK {
		return nil, r.Err()
	}
	result.WeightFile = core.PathJoin(outDir, "model.safetensors")
	result.ConfigFile = core.PathJoin(outDir, "config.json")
	result.OutputBytes = int64(len(blob))
	if r := core.WriteFile(result.WeightFile, blob, 0o644); !r.OK {
		return nil, r.Err()
	}
	if err := copySidecars(srcDir, outDir); err != nil {
		return nil, err
	}
	// Schema reference: https://huggingface.co/nm-testing/Meta-Llama-3.1-8B-Instruct-FP8-hf
	quantConfig := map[string]any{"config_groups": map[string]any{"group_0": map[string]any{"targets": []string{"Linear"}, "weights": map[string]any{"num_bits": 8, "type": "float", "symmetric": true, "strategy": "tensor", "dynamic": false}}}, "format": "naive-quantized", "quant_method": "compressed-tensors", "quantization_status": "compressed"}
	if err := writeModelConfig(result.ConfigFile, quantConfig); err != nil {
		return nil, err
	}
	return result, nil
}

func isFloat(dtype string) bool { return dtype == "F32" || dtype == "F16" || dtype == "BF16" }
func ints(values []uint64) []int {
	out := make([]int, len(values))
	for i, value := range values {
		out[i] = int(value)
	}
	return out
}
func product(values []int) int {
	out := 1
	for _, value := range values {
		out *= value
	}
	return out
}
func copySidecars(src, out string) error {
	for _, path := range core.PathGlob(core.PathJoin(src, "*")) {
		name := core.PathBase(path)
		if core.HasSuffix(name, ".safetensors") || core.HasSuffix(name, ".safetensors.index.json") || name == "quantization_config.json" {
			continue
		}
		read := core.ReadFile(path)
		if read.OK {
			if write := core.WriteFile(core.PathJoin(out, name), read.Value.([]byte), 0o644); !write.OK {
				return write.Err()
			}
		}
	}
	return nil
}

func writeModelConfig(path string, quantConfig map[string]any) error {
	config := make(map[string]any)
	read := core.ReadFile(path)
	if read.OK {
		if decoded := core.JSONUnmarshal(read.Value.([]byte), &config); !decoded.OK {
			return decoded.Err()
		}
	}
	config["quantization_config"] = quantConfig
	encoded := core.JSONMarshalIndent(config, "", "  ")
	if !encoded.OK {
		return encoded.Err()
	}
	write := core.WriteFile(path, encoded.Value.([]byte), 0o644)
	if !write.OK {
		return write.Err()
	}
	return nil
}

// isFP8LinearWeight reports whether a weight is a transformer linear the
// compressed-tensors fp8 convention quantises. Embeddings and the LM head are
// the standard ignore set (vLLM-consumed checkpoints list them under
// quantization_config.ignore) — casting them costs quality for no serving win.
func isFP8LinearWeight(name string) bool {
	lower := core.Lower(name)
	return !core.Contains(lower, "embed") && !core.Contains(lower, "lm_head")
}
