// SPDX-Licence-Identifier: EUPL-1.2

package quant_test

import (
	"testing"

	"dappco.re/go/inference/quant"
)

// Realistic HuggingFace quantization_config payloads, one per method — the same
// shapes the loader meets in the wild. Bare objects exercise the top-level path
// (absent nested key → hasQuantSignal → fromConfig); the nested fixture wraps one
// in a full config.json so the quantization_config-extraction path is measured.

const benchGPTQBare = `{
	"quant_method": "gptq",
	"bits": 4,
	"group_size": 128,
	"desc_act": true,
	"sym": true,
	"damp_percent": 0.1,
	"checkpoint_format": "gptq",
	"lm_head": false
}`

const benchGPTQNested = `{
	"architectures": ["LlamaForCausalLM"],
	"hidden_size": 4096,
	"num_hidden_layers": 32,
	"torch_dtype": "float16",
	"quantization_config": {
		"quant_method": "gptq",
		"bits": 8,
		"group_size": 32,
		"desc_act": false,
		"sym": false
	}
}`

const benchAWQBare = `{
	"quant_method": "awq",
	"w_bit": 4,
	"q_group_size": 64,
	"zero_point": false,
	"version": "marlin"
}`

const benchCompressedW4A16 = `{
	"quant_method": "compressed-tensors",
	"format": "pack-quantized",
	"config_groups": {
		"group_0": {
			"targets": ["Linear"],
			"weights": {
				"num_bits": 4,
				"type": "int",
				"symmetric": true,
				"strategy": "group",
				"group_size": 128,
				"actorder": "weight"
			},
			"input_activations": null
		}
	}
}`

const benchFP8Bare = `{
	"quant_method": "fp8",
	"activation_scheme": "dynamic",
	"weight_block_size": [128, 128],
	"fmt": "e4m3"
}`

const benchBitsAndBytesBare = `{
	"quant_method": "bitsandbytes",
	"load_in_4bit": true,
	"load_in_8bit": false,
	"bnb_4bit_quant_type": "nf4",
	"bnb_4bit_compute_dtype": "bfloat16",
	"bnb_4bit_use_double_quant": true
}`

// A full config.json with no quantization_config — the unquantised-model path
// (no nested key, hasQuantSignal false → MethodNone).
const benchNoQuant = `{
	"architectures": ["Gemma3ForCausalLM"],
	"hidden_size": 2304,
	"num_hidden_layers": 26,
	"torch_dtype": "bfloat16",
	"vocab_size": 262144
}`

// Package-level sinks so the compiler cannot eliminate the work under test.
var (
	sinkInfo quant.QuantInfo
	sinkErr  error
	sinkStr  string
)

// --- Parse: one per format (each exercises its per-method read* helper) ---

func BenchmarkParse_GPTQ(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkInfo, sinkErr = quant.Parse(benchGPTQBare)
	}
}

func BenchmarkParse_GPTQNested(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkInfo, sinkErr = quant.Parse(benchGPTQNested)
	}
}

func BenchmarkParse_AWQ(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkInfo, sinkErr = quant.Parse(benchAWQBare)
	}
}

func BenchmarkParse_CompressedTensors(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkInfo, sinkErr = quant.Parse(benchCompressedW4A16)
	}
}

func BenchmarkParse_FP8(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkInfo, sinkErr = quant.Parse(benchFP8Bare)
	}
}

func BenchmarkParse_BitsAndBytes(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkInfo, sinkErr = quant.Parse(benchBitsAndBytesBare)
	}
}

func BenchmarkParse_NoQuant(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkInfo, sinkErr = quant.Parse(benchNoQuant)
	}
}

func BenchmarkParse_Empty(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkInfo, sinkErr = quant.Parse("")
	}
}

// --- String: the compact log/registry rendering, across its branches ---

func BenchmarkString_GPTQ(b *testing.B) {
	qi, err := quant.Parse(benchGPTQBare)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkStr = qi.String()
	}
}

func BenchmarkString_CompressedTensors(b *testing.B) {
	qi, err := quant.Parse(benchCompressedW4A16)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkStr = qi.String()
	}
}

func BenchmarkString_None(b *testing.B) {
	qi := quant.QuantInfo{Method: quant.MethodNone}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkStr = qi.String()
	}
}
