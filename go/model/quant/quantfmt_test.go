// SPDX-Licence-Identifier: EUPL-1.2

package quant

import "testing"

// fixtures — realistic HuggingFace quantization_config payloads, one per method.
// Each is the bare quantization_config object; the "nested" fixtures wrap one in
// a full config.json so both entry points are exercised.

const gptqBare = `{
	"quant_method": "gptq",
	"bits": 4,
	"group_size": 128,
	"desc_act": true,
	"sym": true,
	"damp_percent": 0.1,
	"checkpoint_format": "gptq",
	"lm_head": false
}`

const gptqNested = `{
	"architectures": ["LlamaForCausalLM"],
	"hidden_size": 4096,
	"quantization_config": {
		"quant_method": "gptq",
		"bits": 8,
		"group_size": 32,
		"desc_act": false,
		"sym": false
	}
}`

// AWQ in the wild uses both the canonical (bits/group_size) and the AutoAWQ
// (w_bit/q_group_size) spellings; the parser must accept either.
const awqBare = `{
	"quant_method": "awq",
	"bits": 4,
	"group_size": 128,
	"zero_point": true,
	"version": "gemm"
}`

const awqAutoSpelling = `{
	"quant_method": "awq",
	"w_bit": 4,
	"q_group_size": 64,
	"zero_point": false,
	"version": "marlin"
}`

// compressed-tensors carries a config_groups map; the weight scheme lives in
// each group's "weights" object. No input_activations → 16-bit activations,
// so this derives to W4A16.
const compressedW4A16 = `{
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

// compressed-tensors W8A8: both weight and activation quantised to 8 bits.
const compressedW8A8 = `{
	"quant_method": "compressed-tensors",
	"format": "int-quantized",
	"config_groups": {
		"group_0": {
			"targets": ["Linear"],
			"weights": {
				"num_bits": 8,
				"type": "int",
				"symmetric": true,
				"strategy": "channel"
			},
			"input_activations": {
				"num_bits": 8,
				"type": "int",
				"symmetric": false,
				"strategy": "token"
			}
		}
	}
}`

// fp8 block-wise dynamic checkpoint.
const fp8Bare = `{
	"quant_method": "fp8",
	"activation_scheme": "dynamic",
	"weight_block_size": [128, 128],
	"fmt": "e4m3"
}`

// AMD Quark's native "quark" quant_method, trimmed from a real checkpoint's
// quantization_config: FP8-KV, per-tensor E4M3
// (amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test/blob/main/config.json).
const quarkFP8Bare = `{
	"quant_method": "quark",
	"quant_mode": "eager_mode",
	"global_quant_config": {
		"weight": {
			"dtype": "fp8_e4m3",
			"qscheme": "per_tensor",
			"group_size": null,
			"is_dynamic": false,
			"symmetric": null
		},
		"input_tensors": {
			"dtype": "fp8_e4m3",
			"qscheme": "per_tensor",
			"is_dynamic": false
		},
		"output_tensors": null
	}
}`

// AMD Quark's native "quark" quant_method, trimmed from a real checkpoint's
// quantization_config: the OCP-microscaled MXFP4/MX lane (dtype fp4,
// per_group, group_size 32, scale_format e8m0)
// (amd/Qwen3.5-397B-A17B-MXFP4/blob/main/config.json).
const quarkMXFP4Bare = `{
	"quant_method": "quark",
	"quant_mode": "eager_mode",
	"global_quant_config": {
		"weight": {
			"dtype": "fp4",
			"qscheme": "per_group",
			"group_size": 32,
			"is_dynamic": false,
			"scale_format": "e8m0",
			"symmetric": null
		},
		"input_tensors": {
			"dtype": "fp4",
			"qscheme": "per_group",
			"group_size": 32,
			"is_dynamic": true,
			"scale_format": "e8m0"
		},
		"output_tensors": null
	}
}`

// quark nested in a full config.json, using int8 per-channel weights with
// dynamic per-token activations — exercises the Symmetric/Activation paths
// the two bare fixtures above leave untouched.
const quarkNested = `{
	"architectures": ["LlamaForCausalLM"],
	"hidden_size": 4096,
	"quantization_config": {
		"quant_method": "quark",
		"global_quant_config": {
			"weight": {"dtype": "int8", "qscheme": "per_channel", "symmetric": true},
			"input_tensors": {"dtype": "int8", "is_dynamic": true}
		}
	}
}`

const bitsandbytesBare = `{
	"quant_method": "bitsandbytes",
	"load_in_4bit": true,
	"load_in_8bit": false,
	"bnb_4bit_quant_type": "nf4",
	"bnb_4bit_compute_dtype": "bfloat16",
	"bnb_4bit_use_double_quant": true
}`

const bitsandbytes8bit = `{
	"quant_method": "bitsandbytes",
	"load_in_4bit": false,
	"load_in_8bit": true
}`

// A full config.json with no quantization_config at all — an unquantised model.
const noQuantConfig = `{
	"architectures": ["Gemma3ForCausalLM"],
	"hidden_size": 2304,
	"torch_dtype": "bfloat16"
}`

// TestQuantFmt_Parse_Good — every supported method parses its fields, from both
// a bare quantization_config object and a nested config.json.
func TestQuantFmt_Parse_Good(t *testing.T) {
	t.Run("gptq", func(t *testing.T) {
		qi, err := Parse(gptqBare)
		if err != nil {
			t.Fatalf("parse gptq: %v", err)
		}
		if qi.Method != MethodGPTQ {
			t.Errorf("method: got %q, want %q", qi.Method, MethodGPTQ)
		}
		if qi.Bits != 4 {
			t.Errorf("bits: got %d, want 4", qi.Bits)
		}
		if qi.GroupSize != 128 {
			t.Errorf("group_size: got %d, want 128", qi.GroupSize)
		}
		if !qi.Symmetric {
			t.Errorf("symmetric: got false, want true")
		}
		if !qi.DescAct {
			t.Errorf("desc_act: got false, want true")
		}
		// Raw retains a method-specific field the struct does not model.
		if qi.Raw["damp_percent"] == nil {
			t.Errorf("raw should retain damp_percent, got %v", qi.Raw)
		}
	})

	t.Run("gptq nested in config.json", func(t *testing.T) {
		qi, err := Parse(gptqNested)
		if err != nil {
			t.Fatalf("parse nested gptq: %v", err)
		}
		if qi.Method != MethodGPTQ {
			t.Errorf("method: got %q, want %q", qi.Method, MethodGPTQ)
		}
		if qi.Bits != 8 {
			t.Errorf("bits: got %d, want 8", qi.Bits)
		}
		if qi.GroupSize != 32 {
			t.Errorf("group_size: got %d, want 32", qi.GroupSize)
		}
		if qi.Symmetric {
			t.Errorf("symmetric: got true, want false")
		}
		if qi.DescAct {
			t.Errorf("desc_act: got true, want false")
		}
	})

	t.Run("awq canonical", func(t *testing.T) {
		qi, err := Parse(awqBare)
		if err != nil {
			t.Fatalf("parse awq: %v", err)
		}
		if qi.Method != MethodAWQ {
			t.Errorf("method: got %q, want %q", qi.Method, MethodAWQ)
		}
		if qi.Bits != 4 {
			t.Errorf("bits: got %d, want 4", qi.Bits)
		}
		if qi.GroupSize != 128 {
			t.Errorf("group_size: got %d, want 128", qi.GroupSize)
		}
		// AWQ zero_point → asymmetric.
		if qi.Symmetric {
			t.Errorf("symmetric: got true, want false (zero_point set)")
		}
	})

	t.Run("awq AutoAWQ spelling", func(t *testing.T) {
		qi, err := Parse(awqAutoSpelling)
		if err != nil {
			t.Fatalf("parse awq auto: %v", err)
		}
		if qi.Bits != 4 {
			t.Errorf("bits (w_bit): got %d, want 4", qi.Bits)
		}
		if qi.GroupSize != 64 {
			t.Errorf("group_size (q_group_size): got %d, want 64", qi.GroupSize)
		}
		// zero_point=false → symmetric.
		if !qi.Symmetric {
			t.Errorf("symmetric: got false, want true (zero_point false)")
		}
	})

	t.Run("compressed-tensors w4a16", func(t *testing.T) {
		qi, err := Parse(compressedW4A16)
		if err != nil {
			t.Fatalf("parse compressed: %v", err)
		}
		if qi.Method != MethodCompressedTensors {
			t.Errorf("method: got %q, want %q", qi.Method, MethodCompressedTensors)
		}
		if qi.Bits != 4 {
			t.Errorf("bits: got %d, want 4", qi.Bits)
		}
		if qi.GroupSize != 128 {
			t.Errorf("group_size: got %d, want 128", qi.GroupSize)
		}
		if !qi.Symmetric {
			t.Errorf("symmetric: got false, want true")
		}
		if !qi.DescAct {
			t.Errorf("actorder weight should map to desc_act true")
		}
	})

	t.Run("fp8", func(t *testing.T) {
		qi, err := Parse(fp8Bare)
		if err != nil {
			t.Fatalf("parse fp8: %v", err)
		}
		if qi.Method != MethodFP8 {
			t.Errorf("method: got %q, want %q", qi.Method, MethodFP8)
		}
		if qi.Bits != 8 {
			t.Errorf("bits: got %d, want 8", qi.Bits)
		}
		if qi.Activation != "dynamic" {
			t.Errorf("activation: got %q, want dynamic", qi.Activation)
		}
	})

	t.Run("bitsandbytes 4bit", func(t *testing.T) {
		qi, err := Parse(bitsandbytesBare)
		if err != nil {
			t.Fatalf("parse bnb: %v", err)
		}
		if qi.Method != MethodBitsAndBytes {
			t.Errorf("method: got %q, want %q", qi.Method, MethodBitsAndBytes)
		}
		if qi.Bits != 4 {
			t.Errorf("bits: got %d, want 4", qi.Bits)
		}
		if qi.Scheme != "nf4" {
			t.Errorf("scheme: got %q, want nf4", qi.Scheme)
		}
	})

	t.Run("bitsandbytes 8bit", func(t *testing.T) {
		qi, err := Parse(bitsandbytes8bit)
		if err != nil {
			t.Fatalf("parse bnb8: %v", err)
		}
		if qi.Bits != 8 {
			t.Errorf("bits: got %d, want 8", qi.Bits)
		}
	})

	t.Run("quark fp8", func(t *testing.T) {
		qi, err := Parse(quarkFP8Bare)
		if err != nil {
			t.Fatalf("parse quark fp8: %v", err)
		}
		if qi.Method != MethodQuark {
			t.Errorf("method: got %q, want %q", qi.Method, MethodQuark)
		}
		if qi.Bits != 8 {
			t.Errorf("bits: got %d, want 8", qi.Bits)
		}
		if qi.Scheme != "fp8_e4m3" {
			t.Errorf("scheme: got %q, want fp8_e4m3", qi.Scheme)
		}
		if qi.Activation != "static" {
			t.Errorf("activation: got %q, want static", qi.Activation)
		}
		if qi.GroupSize != 0 {
			t.Errorf("group_size: got %d, want 0 (per-tensor)", qi.GroupSize)
		}
	})

	t.Run("quark mxfp4", func(t *testing.T) {
		qi, err := Parse(quarkMXFP4Bare)
		if err != nil {
			t.Fatalf("parse quark mxfp4: %v", err)
		}
		if qi.Method != MethodQuark {
			t.Errorf("method: got %q, want %q", qi.Method, MethodQuark)
		}
		if qi.Bits != 4 {
			t.Errorf("bits: got %d, want 4", qi.Bits)
		}
		if qi.GroupSize != 32 {
			t.Errorf("group_size: got %d, want 32", qi.GroupSize)
		}
		if qi.Scheme != "fp4" {
			t.Errorf("scheme: got %q, want fp4", qi.Scheme)
		}
		if qi.Activation != "dynamic" {
			t.Errorf("activation: got %q, want dynamic", qi.Activation)
		}
	})

	t.Run("quark nested in config.json", func(t *testing.T) {
		qi, err := Parse(quarkNested)
		if err != nil {
			t.Fatalf("parse quark nested: %v", err)
		}
		if qi.Bits != 8 {
			t.Errorf("bits: got %d, want 8", qi.Bits)
		}
		if !qi.Symmetric {
			t.Errorf("symmetric: got false, want true")
		}
		if qi.Activation != "dynamic" {
			t.Errorf("activation: got %q, want dynamic", qi.Activation)
		}
	})
}

// TestQuantFmt_Parse_Bad — absent config is not an error (it means "none"); a
// malformed JSON document is a typed error.
func TestQuantFmt_Parse_Bad(t *testing.T) {
	t.Run("no quantization_config → none", func(t *testing.T) {
		qi, err := Parse(noQuantConfig)
		if err != nil {
			t.Fatalf("absent config should not error: %v", err)
		}
		if qi.Method != MethodNone {
			t.Errorf("method: got %q, want %q", qi.Method, MethodNone)
		}
	})

	t.Run("empty string → none", func(t *testing.T) {
		qi, err := Parse("")
		if err != nil {
			t.Fatalf("empty should not error: %v", err)
		}
		if qi.Method != MethodNone {
			t.Errorf("method: got %q, want %q", qi.Method, MethodNone)
		}
	})

	t.Run("whitespace only → none", func(t *testing.T) {
		qi, err := Parse("   \n\t  ")
		if err != nil {
			t.Fatalf("whitespace should not error: %v", err)
		}
		if qi.Method != MethodNone {
			t.Errorf("method: got %q, want %q", qi.Method, MethodNone)
		}
	})

	t.Run("malformed JSON → error", func(t *testing.T) {
		_, err := Parse(`{"quant_method": "gptq", "bits": }`)
		if err == nil {
			t.Fatalf("malformed JSON should error")
		}
	})

	t.Run("malformed nested quantization_config → error", func(t *testing.T) {
		_, err := Parse(`{"hidden_size": 10, "quantization_config": {"bits": 4,}}`)
		if err == nil {
			t.Fatalf("malformed nested object should error")
		}
	})
}

// TestQuantFmt_Parse_Ugly — odd-but-valid inputs: a quant_method we do not know,
// a config_groups with no usable weight scheme, and a JSON value that is not an
// object (a bare array / scalar).
func TestQuantFmt_Parse_Ugly(t *testing.T) {
	t.Run("unknown method", func(t *testing.T) {
		qi, err := Parse(`{"quant_method": "exllama-v9000", "bits": 3}`)
		if err != nil {
			t.Fatalf("unknown method should not error: %v", err)
		}
		if qi.Method != MethodUnknown {
			t.Errorf("method: got %q, want %q", qi.Method, MethodUnknown)
		}
		// Even unknown, generic top-level fields are still harvested.
		if qi.Bits != 3 {
			t.Errorf("bits: got %d, want 3 (generic fallthrough)", qi.Bits)
		}
	})

	t.Run("quant_method missing but bits present", func(t *testing.T) {
		qi, err := Parse(`{"bits": 4, "group_size": 128}`)
		if err != nil {
			t.Fatalf("should not error: %v", err)
		}
		// No quant_method and not recognisably any method → unknown, fields kept.
		if qi.Method != MethodUnknown {
			t.Errorf("method: got %q, want %q", qi.Method, MethodUnknown)
		}
		if qi.Bits != 4 {
			t.Errorf("bits: got %d, want 4", qi.Bits)
		}
	})

	t.Run("compressed-tensors with empty config_groups", func(t *testing.T) {
		qi, err := Parse(`{"quant_method": "compressed-tensors", "format": "dense", "config_groups": {}}`)
		if err != nil {
			t.Fatalf("should not error: %v", err)
		}
		if qi.Method != MethodCompressedTensors {
			t.Errorf("method: got %q, want %q", qi.Method, MethodCompressedTensors)
		}
		// No groups → no derivable bits, but method is still recognised.
		if qi.Bits != 0 {
			t.Errorf("bits: got %d, want 0", qi.Bits)
		}
	})

	t.Run("compressed-tensors with malformed weights value", func(t *testing.T) {
		// weights is the wrong shape (a string, not an object) — tolerated, no
		// scheme derived, method still recognised.
		qi, err := Parse(`{"quant_method": "compressed-tensors", "config_groups": {"g": {"weights": "oops"}}}`)
		if err != nil {
			t.Fatalf("should not error: %v", err)
		}
		if qi.Method != MethodCompressedTensors {
			t.Errorf("method: got %q, want %q", qi.Method, MethodCompressedTensors)
		}
		if qi.Bits != 0 {
			t.Errorf("bits: got %d, want 0", qi.Bits)
		}
	})

	t.Run("JSON array is not an object → none", func(t *testing.T) {
		// A top-level array unmarshals into map[string]any with an error; Parse
		// surfaces that as a typed error.
		_, err := Parse(`[1, 2, 3]`)
		if err == nil {
			t.Fatalf("non-object JSON should error")
		}
	})

	t.Run("nested quantization_config wrong type → none", func(t *testing.T) {
		// quantization_config present but not an object — treated as absent.
		qi, err := Parse(`{"quantization_config": "not-an-object"}`)
		if err != nil {
			t.Fatalf("should not error: %v", err)
		}
		if qi.Method != MethodNone {
			t.Errorf("method: got %q, want %q", qi.Method, MethodNone)
		}
	})

	t.Run("bits as float (JSON number) is coerced", func(t *testing.T) {
		// JSON numbers decode to float64; the reader must coerce to int.
		qi, err := Parse(`{"quant_method": "gptq", "bits": 4.0, "group_size": 128.0}`)
		if err != nil {
			t.Fatalf("should not error: %v", err)
		}
		if qi.Bits != 4 {
			t.Errorf("bits: got %d, want 4", qi.Bits)
		}
		if qi.GroupSize != 128 {
			t.Errorf("group_size: got %d, want 128", qi.GroupSize)
		}
	})
}

// TestQuantFmt_Scheme_Good — the derived W{w}A{a} scheme string.
func TestQuantFmt_Scheme_Good(t *testing.T) {
	t.Run("w4a16 from no input_activations", func(t *testing.T) {
		qi, err := Parse(compressedW4A16)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if qi.Scheme != "W4A16" {
			t.Errorf("scheme: got %q, want W4A16", qi.Scheme)
		}
	})

	t.Run("w8a8 from 8-bit input_activations", func(t *testing.T) {
		qi, err := Parse(compressedW8A8)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if qi.Scheme != "W8A8" {
			t.Errorf("scheme: got %q, want W8A8", qi.Scheme)
		}
		if qi.Bits != 8 {
			t.Errorf("bits: got %d, want 8", qi.Bits)
		}
		// Weight symmetric=true is what QuantInfo.Symmetric tracks.
		if !qi.Symmetric {
			t.Errorf("symmetric: got false, want true (weight symmetric)")
		}
		// Activation strategy "token" → dynamic activation.
		if qi.Activation != "token" {
			t.Errorf("activation: got %q, want token", qi.Activation)
		}
	})

	t.Run("gptq derives w4a16 by convention", func(t *testing.T) {
		qi, err := Parse(gptqBare)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		// GPTQ/AWQ are weight-only → A16 activation by convention.
		if qi.Scheme != "W4A16" {
			t.Errorf("scheme: got %q, want W4A16", qi.Scheme)
		}
	})
}

// TestQuantFmt_Scheme_Bad — schemes that cannot be derived stay empty rather
// than producing a bogus "W0A16".
func TestQuantFmt_Scheme_Bad(t *testing.T) {
	t.Run("no bits → empty scheme", func(t *testing.T) {
		qi, err := Parse(`{"quant_method": "compressed-tensors", "config_groups": {}}`)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if qi.Scheme != "" {
			t.Errorf("scheme: got %q, want empty", qi.Scheme)
		}
	})

	t.Run("fp8 has no W/A scheme", func(t *testing.T) {
		qi, err := Parse(fp8Bare)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		// fp8 is block-wise, not a W/A integer scheme — Scheme stays empty.
		if qi.Scheme != "" {
			t.Errorf("scheme: got %q, want empty for fp8", qi.Scheme)
		}
	})

	t.Run("none has no scheme", func(t *testing.T) {
		qi, err := Parse(noQuantConfig)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if qi.Scheme != "" {
			t.Errorf("scheme: got %q, want empty", qi.Scheme)
		}
	})
}

// TestQuantFmt_Scheme_Ugly — desc_act / symmetric flag derivation under odd
// compressed-tensors inputs.
func TestQuantFmt_Scheme_Ugly(t *testing.T) {
	t.Run("asymmetric weight + no actorder", func(t *testing.T) {
		qi, err := Parse(`{
			"quant_method": "compressed-tensors",
			"format": "pack-quantized",
			"config_groups": {"g": {"weights": {"num_bits": 4, "symmetric": false, "strategy": "group", "group_size": 64}}}
		}`)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if qi.Symmetric {
			t.Errorf("symmetric: got true, want false")
		}
		if qi.DescAct {
			t.Errorf("desc_act: got true, want false (no actorder)")
		}
		if qi.Scheme != "W4A16" {
			t.Errorf("scheme: got %q, want W4A16", qi.Scheme)
		}
		if qi.GroupSize != 64 {
			t.Errorf("group_size: got %d, want 64", qi.GroupSize)
		}
	})

	t.Run("actorder group is also desc_act", func(t *testing.T) {
		qi, err := Parse(`{
			"quant_method": "compressed-tensors",
			"config_groups": {"g": {"weights": {"num_bits": 4, "symmetric": true, "actorder": "group"}}}
		}`)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if !qi.DescAct {
			t.Errorf("actorder group should map to desc_act true")
		}
	})

	t.Run("actorder false-y string stays desc_act false", func(t *testing.T) {
		qi, err := Parse(`{
			"quant_method": "compressed-tensors",
			"config_groups": {"g": {"weights": {"num_bits": 8, "symmetric": true, "actorder": "none"}}}
		}`)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if qi.DescAct {
			t.Errorf("actorder none should map to desc_act false")
		}
	})
}

// TestQuantFmt_String_Good — the compact log/registry rendering.
func TestQuantFmt_String_Good(t *testing.T) {
	t.Run("gptq with group size", func(t *testing.T) {
		qi, err := Parse(gptqBare)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		got := qi.String()
		if got != "gptq:4bit:g128" {
			t.Errorf("string: got %q, want gptq:4bit:g128", got)
		}
	})

	t.Run("awq with group size", func(t *testing.T) {
		qi, err := Parse(awqAutoSpelling)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		got := qi.String()
		if got != "awq:4bit:g64" {
			t.Errorf("string: got %q, want awq:4bit:g64", got)
		}
	})

	t.Run("compressed-tensors includes scheme", func(t *testing.T) {
		qi, err := Parse(compressedW4A16)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		got := qi.String()
		// scheme present → appended.
		if got != "compressed-tensors:4bit:g128:W4A16" {
			t.Errorf("string: got %q, want compressed-tensors:4bit:g128:W4A16", got)
		}
	})

	t.Run("quark mxfp4 includes group size and dtype scheme", func(t *testing.T) {
		qi, err := Parse(quarkMXFP4Bare)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		got := qi.String()
		if got != "quark:4bit:g32:fp4" {
			t.Errorf("string: got %q, want quark:4bit:g32:fp4", got)
		}
	})

	t.Run("quark fp8 has no group size", func(t *testing.T) {
		qi, err := Parse(quarkFP8Bare)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		got := qi.String()
		if got != "quark:8bit:fp8_e4m3" {
			t.Errorf("string: got %q, want quark:8bit:fp8_e4m3", got)
		}
	})
}

// TestQuantFmt_String_Bad — degenerate infos render without panicking and omit
// the parts they do not have.
func TestQuantFmt_String_Bad(t *testing.T) {
	t.Run("none renders as none", func(t *testing.T) {
		qi := QuantInfo{Method: MethodNone}
		if qi.String() != "none" {
			t.Errorf("string: got %q, want none", qi.String())
		}
	})

	t.Run("zero value renders as none", func(t *testing.T) {
		var qi QuantInfo
		if qi.String() != "none" {
			t.Errorf("zero-value string: got %q, want none", qi.String())
		}
	})

	t.Run("method only, no bits", func(t *testing.T) {
		qi := QuantInfo{Method: MethodFP8}
		// No bits and no group size → just the method.
		if qi.String() != "fp8" {
			t.Errorf("string: got %q, want fp8", qi.String())
		}
	})

	t.Run("bits but no group size omits g part", func(t *testing.T) {
		qi := QuantInfo{Method: MethodGPTQ, Bits: 4}
		if qi.String() != "gptq:4bit" {
			t.Errorf("string: got %q, want gptq:4bit", qi.String())
		}
	})
}

// TestQuantFmt_String_Ugly — fp8 dynamic and unknown method rendering.
func TestQuantFmt_String_Ugly(t *testing.T) {
	t.Run("fp8 dynamic", func(t *testing.T) {
		qi, err := Parse(fp8Bare)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		got := qi.String()
		// fp8: bits with no group size, no W/A scheme.
		if got != "fp8:8bit" {
			t.Errorf("string: got %q, want fp8:8bit", got)
		}
	})

	t.Run("unknown method renders its bits", func(t *testing.T) {
		qi, err := Parse(`{"quant_method": "exllama-v9000", "bits": 3}`)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if qi.String() != "unknown:3bit" {
			t.Errorf("string: got %q, want unknown:3bit", qi.String())
		}
	})

	t.Run("bitsandbytes renders scheme as suffix", func(t *testing.T) {
		qi, err := Parse(bitsandbytesBare)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		// bnb has no group size; scheme nf4 is appended.
		if qi.String() != "bitsandbytes:4bit:nf4" {
			t.Errorf("string: got %q, want bitsandbytes:4bit:nf4", qi.String())
		}
	})
}

// TestQuantFmt_BitsAndBytes_Default — a bitsandbytes config with neither
// load_in_4bit nor load_in_8bit defaults to 4-bit loading.
func TestQuantFmt_BitsAndBytes_Default(t *testing.T) {
	qi, err := Parse(`{"quant_method": "bitsandbytes", "bnb_4bit_quant_type": "fp4"}`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if qi.Bits != 4 {
		t.Errorf("bits: got %d, want 4 (default)", qi.Bits)
	}
	if qi.Scheme != "fp4" {
		t.Errorf("scheme: got %q, want fp4", qi.Scheme)
	}
}

// TestQuantFmt_CompressedTensors_Edges — the tolerant branches of the
// config_groups walk: a non-object config_groups, a non-object group, a group
// whose weights have no num_bits, all skipped without error.
func TestQuantFmt_CompressedTensors_Edges(t *testing.T) {
	t.Run("config_groups is not an object", func(t *testing.T) {
		qi, err := Parse(`{"quant_method": "compressed-tensors", "config_groups": "nope"}`)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if qi.Method != MethodCompressedTensors || qi.Bits != 0 {
			t.Errorf("got %+v, want method compressed-tensors, bits 0", qi)
		}
	})

	t.Run("a group is not an object", func(t *testing.T) {
		qi, err := Parse(`{"quant_method": "compressed-tensors", "config_groups": {"g": 7}}`)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if qi.Bits != 0 {
			t.Errorf("bits: got %d, want 0", qi.Bits)
		}
	})

	t.Run("weights present but num_bits missing is skipped", func(t *testing.T) {
		qi, err := Parse(`{"quant_method": "compressed-tensors", "config_groups": {"g": {"weights": {"type": "int"}}}}`)
		if err != nil {
			t.Fatalf("parse: %v", err)
		}
		if qi.Bits != 0 {
			t.Errorf("bits: got %d, want 0 (num_bits missing)", qi.Bits)
		}
	})
}

// TestQuantFmt_GPTQ_NoBits_EmptyScheme — a GPTQ config without bits derives no
// scheme (weightOnlyScheme guards against a bogus W0A16).
func TestQuantFmt_GPTQ_NoBits_EmptyScheme(t *testing.T) {
	qi, err := Parse(`{"quant_method": "gptq", "group_size": 128}`)
	if err != nil {
		t.Fatalf("parse: %v", err)
	}
	if qi.Scheme != "" {
		t.Errorf("scheme: got %q, want empty (no bits)", qi.Scheme)
	}
}

// TestQuarkDtypeBits — every documented AMD Quark dtype string maps to its
// own sign+exponent+mantissa (or integer) bit width; an unrecognised or bare
// "OCP MX" tag stays unmapped (0) rather than guessed.
func TestQuarkDtypeBits(t *testing.T) {
	cases := map[string]int{
		"int2": 2, "int4": 4, "fp4": 4,
		"fp6_e3m2": 6, "fp6_e2m3": 6, "MX6": 6,
		"int8": 8, "fp8_e4m3": 8, "fp8_e5m2": 8,
		"MX9":      9,
		"bfloat16": 16, "float16": 16, "bfp16": 16,
		"OCP MX": 0, "": 0, "nf4": 0,
	}
	for dtype, want := range cases {
		if got := quarkDtypeBits(dtype); got != want {
			t.Errorf("quarkDtypeBits(%q): got %d, want %d", dtype, got, want)
		}
	}
}

// TestQuantFmt_Helpers_Internal — the small typed accessors have defensive
// branches that JSON decoding alone never reaches (a genuine int in the map, a
// non-error error payload). They are exercised directly here, as the package's
// own white-box tests.
func TestQuantFmt_Helpers_Internal(t *testing.T) {
	t.Run("num honours a real int", func(t *testing.T) {
		// JSON decodes numbers to float64; a map built in Go may hold an int.
		if got := num(map[string]any{"bits": 4}, "bits"); got != 4 {
			t.Errorf("num int: got %d, want 4", got)
		}
	})

	t.Run("num returns 0 for wrong type", func(t *testing.T) {
		if got := num(map[string]any{"bits": "four"}, "bits"); got != 0 {
			t.Errorf("num string: got %d, want 0", got)
		}
	})

	t.Run("waScheme guards zero weight bits", func(t *testing.T) {
		if got := waScheme(0, 16); got != "" {
			t.Errorf("waScheme(0,16): got %q, want empty", got)
		}
	})

	t.Run("asError passes a non-error value through as nil", func(t *testing.T) {
		// core.Result may, in principle, carry a non-error Value on failure;
		// asError must not panic and yields nil.
		if got := asError("not an error"); got != nil {
			t.Errorf("asError(string): got %v, want nil", got)
		}
		if got := asError(nil); got != nil {
			t.Errorf("asError(nil): got %v, want nil", got)
		}
	})
}
