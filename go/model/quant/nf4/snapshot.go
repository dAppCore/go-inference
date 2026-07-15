// SPDX-Licence-Identifier: EUPL-1.2

package nf4

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
		return nil, core.NewError("nf4: source and output directories are required")
	}
	shards := core.PathGlob(core.PathJoin(srcDir, "*.safetensors"))
	core.SliceSort(shards)
	if len(shards) == 0 {
		return nil, core.NewError("nf4: source has no safetensors shards")
	}
	idx, err := safetensors.IndexFiles(shards)
	if err != nil {
		return nil, core.E("nf4.ConvertSnapshot", "index source shards", err)
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
		eligible := core.HasSuffix(name, ".weight") && len(ref.Shape) == 2 && isFloat(ref.DType) && isNF4LinearWeight(name)
		if progress != nil {
			progress(name, eligible, i+1, len(names))
		}
		raw, readErr := cache.ReadRefRaw(ref)
		if readErr != nil {
			return nil, readErr
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
		quantized, quantErr := Quantize(values, shape)
		if quantErr != nil {
			return nil, quantErr
		}
		// bitsandbytes Params4bit serialises the packed byte stream as [bytes, 1].
		tensors[name] = safetensors.Tensor{Dtype: "U8", Shape: []int{len(quantized.Data), 1}, Data: quantized.Data}
		base := name + "."
		tensors[base+"absmax"] = safetensors.Tensor{Dtype: "F32", Shape: []int{len(quantized.Absmax)}, Data: encodeF32(quantized.Absmax)}
		tensors[base+"quant_map"] = safetensors.Tensor{Dtype: "F32", Shape: []int{16}, Data: encodeF32(Codebook[:])}
		state := core.Sprintf(`{"quant_type":"nf4","blocksize":64,"dtype":"float32","shape":[%d,%d]}`, shape[0], shape[1])
		tensors[base+"quant_state.bitsandbytes__nf4"] = safetensors.Tensor{Dtype: "U8", Shape: []int{len(state)}, Data: []byte(state)}
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
	// Schema reference: https://huggingface.co/axolotl-ai-co/TinyLlama_v1.1-bnb-nf4-bf16
	quantConfig := map[string]any{"quant_method": "bitsandbytes", "load_in_4bit": true, "load_in_8bit": false, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "float32", "bnb_4bit_quant_storage": "uint8", "bnb_4bit_use_double_quant": false}
	if err := writeModelConfig(result.ConfigFile, quantConfig); err != nil {
		return nil, err
	}
	return result, nil
}

func encodeF32(values []float32) []byte {
	out := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(out[i*4:], math.Float32bits(value))
	}
	return out
}
func ints(values []uint64) []int {
	out := make([]int, len(values))
	for i, value := range values {
		out[i] = int(value)
	}
	return out
}
func isFloat(dtype string) bool { return dtype == "F32" || dtype == "F16" || dtype == "BF16" }
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

// isNF4LinearWeight reports whether a weight is a transformer linear
// bitsandbytes quantises. Embeddings and the LM head stay passthrough —
// bnb's Linear4bit replaces nn.Linear modules only, and its default skip set
// (llm_int8_skip_modules) carries lm_head; a checkpoint with NF4 embeddings
// has no loader on the consumer side.
func isNF4LinearWeight(name string) bool {
	lower := core.Lower(name)
	return !core.Contains(lower, "embed") && !core.Contains(lower, "lm_head")
}
