// SPDX-Licence-Identifier: EUPL-1.2

// Package quantfmt reads a HuggingFace quantisation_config into a normalised,
// method-agnostic struct so the go-mlx loader knows how a model is quantised
// before it dequantises. It is pure-Go (CGO-free) and depends only on the core
// framework.
//
//	qi, err := quantfmt.Parse(configJSON)   // full config.json or bare object
//	if err != nil { return core.E("loader.Quant", "bad quant config", err) }
//	if qi.Method == quantfmt.MethodGPTQ { ... }
//	core.Print(qi.String())                 // "gptq:4bit:g128"
//
// The reader understands the field names each method writes: GPTQ (bits,
// group_size, desc_act, sym), AWQ (w_bit/bits, q_group_size/group_size,
// zero_point, version), compressed-tensors (config_groups → weight num_bits,
// type, symmetric, strategy, actorder; derives a W{w}A{a} Scheme), fp8
// (activation_scheme, weight_block_size), bitsandbytes (load_in_4bit /
// load_in_8bit, bnb_4bit_quant_type) and AMD Quark's native "quark" shape
// (global_quant_config.weight.{dtype,group_size,symmetric},
// global_quant_config.input_tensors.is_dynamic — see MethodQuark).
package quant

import (
	"strconv"

	core "dappco.re/go"
)

// Method is the quantisation method a model was produced with. The string
// values match the HuggingFace "quant_method" field verbatim.
//
//	if qi.Method == quantfmt.MethodAWQ { useAWQDequant() }
type Method string

const (
	// MethodCompressedTensors is the vLLM/llm-compressor "compressed-tensors"
	// format, with a config_groups map of per-target weight schemes.
	MethodCompressedTensors Method = "compressed-tensors"
	// MethodAWQ is Activation-aware Weight Quantisation (AutoAWQ).
	MethodAWQ Method = "awq"
	// MethodGPTQ is GPTQ / AutoGPTQ weight-only quantisation.
	MethodGPTQ Method = "gptq"
	// MethodFP8 is 8-bit floating-point (e4m3/e5m2) weight quantisation.
	MethodFP8 Method = "fp8"
	// MethodBitsAndBytes is the bitsandbytes load_in_4bit / load_in_8bit format.
	MethodBitsAndBytes Method = "bitsandbytes"
	// MethodQuark is AMD Quark's native quantisation_config serialization — a
	// richer nested shape (global_quant_config / layer_quant_config /
	// algo_config) than the other methods' flat bits/group_size fields; see
	// readQuark. Quark covers FP8, INT8/4/2, FP6, and the OCP-microscaled
	// MXFP4/MX lane (global_quant_config.weight: dtype "fp4", qscheme
	// "per_group", group_size 32, scale_format "e8m0" — AMD's MI350/CDNA4
	// low-precision format; see model/quant/mxfp4 for the element codec).
	//
	// Confirmed against two real checkpoints — quant_method "quark" appears
	// verbatim in both:
	//   - huggingface.co/amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test/blob/main/config.json
	//     (global_quant_config.weight.dtype = "fp8_e4m3", qscheme "per_tensor")
	//   - huggingface.co/amd/Qwen3.5-397B-A17B-MXFP4/blob/main/config.json
	//     (global_quant_config.weight.dtype = "fp4", qscheme "per_group",
	//     group_size 32, scale_format "e8m0")
	// and by Quark's own HF integration doc
	// (huggingface.co/docs/transformers/en/quantization/quark): "Public models
	// using Quark native serialization can be found at
	// huggingface.co/models?other=quark". The same doc notes Quark can ALSO
	// re-emit AutoAWQ-/native-fp8-compatible configs tagged "awq"/"fp8"
	// instead — those already dispatch to MethodAWQ/MethodFP8, not this one.
	MethodQuark Method = "quark"
	// MethodNone means the model carries no quantisation_config — full precision.
	MethodNone Method = "none"
	// MethodUnknown means a quantisation_config is present but its method is
	// unrecognised; generic fields are still harvested where possible.
	MethodUnknown Method = "unknown"
)

// QuantInfo is the normalised view of a model's quantisation. Fields that a
// given method does not express stay at their zero value; method-specific keys
// the struct does not model are preserved in Raw.
//
// QuantInfo answers a different question from model.QuantConfig
// (model/quant_config.go), and the two are intentionally NOT unified: this
// type detects HOW AN EXTERNAL TOOL QUANTISED A CHECKPOINT — it reads the
// HuggingFace "quantization_config" key (gptq/awq/compressed-tensors/fp8/
// bitsandbytes/quark provenance and their method-specific parameters), which
// is import/interop metadata no in-tree engine currently loads from. Once a
// model is converted into this engine's own on-disk shape, the ENGINE'S OWN
// runtime dequant instructions — group_size/bits/mode plus per-module
// overrides — live in the different, MLX-native "quantization" block that
// model.QuantConfig parses; that is what the assembler actually calls at
// load time. Use QuantInfo when detecting/importing an externally-produced
// checkpoint; use model.QuantConfig when driving this engine's own
// dequantisation. See docs/design-rocm.md §B.3 (dispatch item 11) for the
// full reasoning.
//
//	qi := quantfmt.QuantInfo{Method: quantfmt.MethodGPTQ, Bits: 4, GroupSize: 128}
type QuantInfo struct {
	Method     Method         // the quantisation method (see Method consts)
	Bits       int            // weight bit-width (e.g. 4, 8)
	GroupSize  int            // weight quantisation group size (-1 / 0 = per-channel / unset)
	Scheme     string         // derived W{w}A{a} scheme (compressed-tensors / weight-only) or method tag (bnb)
	Symmetric  bool           // symmetric weight quantisation (no zero-point)
	DescAct    bool           // activation-order reordering (GPTQ desc_act / CT actorder)
	Activation string         // activation scheme: "static"/"dynamic" (fp8) or strategy (compressed-tensors)
	Raw        map[string]any // the original quantisation_config, for anything not modelled above
}

// quantConfigKey is the field that nests a quantisation_config inside a full
// config.json.
const quantConfigKey = "quantization_config"

// Parse reads a quantisation_config into a QuantInfo. It accepts either a full
// config.json (it looks for the nested "quantization_config") or a bare
// quantisation_config object. An absent config is not an error — it yields
// MethodNone. Malformed JSON yields a typed core.E error.
//
//	qi, err := quantfmt.Parse(`{"quant_method":"gptq","bits":4,"group_size":128}`)
//	qi, err := quantfmt.Parse(fullConfigJSON) // pulls .quantization_config out
func Parse(configJSON string) (QuantInfo, error) {
	if core.Trim(configJSON) == "" {
		return QuantInfo{Method: MethodNone}, nil
	}

	var top map[string]any
	r := core.JSONUnmarshalString(configJSON, &top)
	if !r.OK {
		return QuantInfo{}, core.E("quantfmt.Parse", "malformed quantisation config JSON", asError(r.Value))
	}

	// Prefer the nested quantisation_config when present and object-shaped.
	if nested, ok := top[quantConfigKey]; ok {
		obj, isObj := nested.(map[string]any)
		if !isObj {
			// quantization_config present but not an object → no usable config.
			return QuantInfo{Method: MethodNone}, nil
		}
		return fromConfig(obj), nil
	}

	// No nested key: the document is either a bare quantisation_config or a full
	// config.json that simply has no quantisation. Distinguish by whether any
	// quant-shaped signal is present; if none, the model is full precision.
	if !hasQuantSignal(top) {
		return QuantInfo{Method: MethodNone}, nil
	}
	return fromConfig(top), nil
}

// hasQuantSignal reports whether a top-level object looks like a
// quantisation_config — it carries a quant_method or one of the per-method
// shape fields. Used to tell a bare config from an unquantised config.json.
//
//	hasQuantSignal(map[string]any{"bits": 4.0}) == true
func hasQuantSignal(m map[string]any) bool {
	signals := []string{
		"quant_method",  // every method
		"bits", "w_bit", // gptq / awq
		"config_groups",                // compressed-tensors
		"load_in_4bit", "load_in_8bit", // bitsandbytes
		"activation_scheme", "weight_block_size", // fp8
	}
	for _, k := range signals {
		if _, ok := m[k]; ok {
			return true
		}
	}
	return false
}

// fromConfig maps a single quantisation_config object to a QuantInfo, dispatching
// on its quant_method.
//
//	qi := fromConfig(map[string]any{"quant_method": "gptq", "bits": 4.0})
func fromConfig(cfg map[string]any) QuantInfo {
	qi := QuantInfo{Raw: cfg}
	method := normaliseMethod(str(cfg, "quant_method"))
	qi.Method = method

	switch method {
	case MethodGPTQ:
		readGPTQ(&qi, cfg)
	case MethodAWQ:
		readAWQ(&qi, cfg)
	case MethodCompressedTensors:
		readCompressedTensors(&qi, cfg)
	case MethodFP8:
		readFP8(&qi, cfg)
	case MethodBitsAndBytes:
		readBitsAndBytes(&qi, cfg)
	case MethodQuark:
		readQuark(&qi, cfg)
	default:
		// Unknown method, or none given but other quant fields present — harvest
		// the generic bits/group_size so callers still learn the width.
		readGeneric(&qi, cfg)
	}
	return qi
}

// normaliseMethod maps the raw quant_method string to a Method. An empty value
// becomes MethodUnknown (a config object with no method but quant-shaped fields),
// not MethodNone — absence of the whole config is handled in Parse.
//
//	normaliseMethod("GPTQ") == MethodGPTQ
func normaliseMethod(raw string) Method {
	switch core.Lower(core.Trim(raw)) {
	case "compressed-tensors", "compressed_tensors":
		return MethodCompressedTensors
	case "awq":
		return MethodAWQ
	case "gptq":
		return MethodGPTQ
	case "fp8":
		return MethodFP8
	case "bitsandbytes", "bnb":
		return MethodBitsAndBytes
	case "quark":
		return MethodQuark
	default:
		return MethodUnknown
	}
}

// readGPTQ fills GPTQ fields: bits, group_size, desc_act, sym.
//
//	readGPTQ(&qi, map[string]any{"bits": 4.0, "group_size": 128.0, "sym": true})
func readGPTQ(qi *QuantInfo, cfg map[string]any) {
	qi.Bits = num(cfg, "bits")
	qi.GroupSize = num(cfg, "group_size")
	qi.Symmetric = boolOf(cfg, "sym")
	qi.DescAct = boolOf(cfg, "desc_act")
	qi.Scheme = weightOnlyScheme(qi.Bits)
}

// readAWQ fills AWQ fields, accepting both the canonical (bits/group_size) and
// AutoAWQ (w_bit/q_group_size) spellings. zero_point=true means asymmetric.
//
//	readAWQ(&qi, map[string]any{"w_bit": 4.0, "q_group_size": 64.0, "zero_point": false})
func readAWQ(qi *QuantInfo, cfg map[string]any) {
	qi.Bits = numAny(cfg, "bits", "w_bit")
	qi.GroupSize = numAny(cfg, "group_size", "q_group_size")
	// zero_point present and true → asymmetric; default (absent) → symmetric.
	qi.Symmetric = !boolOr(cfg, "zero_point", false)
	qi.Scheme = weightOnlyScheme(qi.Bits)
}

// readFP8 fills fp8 fields. fp8 is 8-bit by definition; the activation scheme is
// static or dynamic, and weight_block_size signals block-wise quantisation.
//
//	readFP8(&qi, map[string]any{"activation_scheme": "dynamic", "weight_block_size": []any{128.0, 128.0}})
func readFP8(qi *QuantInfo, cfg map[string]any) {
	qi.Bits = 8
	qi.Activation = str(cfg, "activation_scheme")
	// fp8 has no integer W/A scheme; leave Scheme empty for callers/logging.
}

// readBitsAndBytes fills bitsandbytes fields. load_in_4bit / load_in_8bit pick
// the width; bnb_4bit_quant_type (fp4/nf4) becomes the Scheme tag.
//
//	readBitsAndBytes(&qi, map[string]any{"load_in_4bit": true, "bnb_4bit_quant_type": "nf4"})
func readBitsAndBytes(qi *QuantInfo, cfg map[string]any) {
	switch {
	case boolOf(cfg, "load_in_8bit"):
		qi.Bits = 8
	case boolOf(cfg, "load_in_4bit"):
		qi.Bits = 4
		qi.Scheme = str(cfg, "bnb_4bit_quant_type")
	default:
		// Neither flag set — bitsandbytes defaults to 4-bit loading.
		qi.Bits = 4
		qi.Scheme = str(cfg, "bnb_4bit_quant_type")
	}
}

// readQuark fills fields from AMD Quark's native "quark" config shape: bits
// and the Scheme tag come from global_quant_config.weight.dtype
// (quarkDtypeBits), GroupSize and Symmetric come from the same weight block,
// and Activation mirrors fp8's static/dynamic reading, derived from
// global_quant_config.input_tensors.is_dynamic. A "per_group" qscheme with an
// "e8m0" scale_format and group_size 32 is AMD's OCP-microscaled MXFP4/MX lane
// (model/quant/mxfp4 packs/unpacks that element format); a "per_tensor"
// qscheme with no group_size is the plain per-tensor FP8/INT8 lane. Confirmed
// against two real checkpoints — see MethodQuark's doc comment for both
// config.json URLs.
//
//	readQuark(&qi, map[string]any{"global_quant_config": map[string]any{"weight": map[string]any{"dtype": "fp4", "group_size": 32.0}}})
func readQuark(qi *QuantInfo, cfg map[string]any) {
	gqc, ok := cfg["global_quant_config"].(map[string]any)
	if !ok {
		return
	}
	if weight, ok := gqc["weight"].(map[string]any); ok {
		dtype := str(weight, "dtype")
		qi.Bits = quarkDtypeBits(dtype)
		qi.GroupSize = num(weight, "group_size")
		qi.Symmetric = boolOf(weight, "symmetric")
		qi.Scheme = dtype
	}
	if input, ok := gqc["input_tensors"].(map[string]any); ok {
		if boolOf(input, "is_dynamic") {
			qi.Activation = "dynamic"
		} else {
			qi.Activation = "static"
		}
	}
}

// quarkDtypeBits maps AMD Quark's documented weight/activation dtype strings
// to a nominal bit-width. The dtype set is Quark's own documented "Data
// types" support matrix
// (huggingface.co/docs/transformers/en/quantization/quark): int8, int4, int2,
// bfloat16, float16, fp8_e5m2, fp8_e4m3, fp6_e3m2, fp6_e2m3, fp4, OCP MX, MX6,
// MX9, bfp16. Each bit count is that format's own sign+exponent+mantissa (or
// integer) width; the bare "OCP MX" tag names no fixed element width so it
// stays unmapped (0) rather than guessed. Verified against two real
// checkpoints: amd/Llama-3.1-8B-Instruct-FP8-KV-Quark-test ("fp8_e4m3") and
// amd/Qwen3.5-397B-A17B-MXFP4 ("fp4" — the OCP-microscaled MXFP4 element
// model/quant/mxfp4 packs/unpacks).
//
//	quarkDtypeBits("fp4") == 4
func quarkDtypeBits(dtype string) int {
	switch core.Lower(core.Trim(dtype)) {
	case "int2":
		return 2
	case "int4", "fp4":
		return 4
	case "fp6_e3m2", "fp6_e2m3", "mx6":
		return 6
	case "int8", "fp8_e4m3", "fp8_e5m2":
		return 8
	case "mx9":
		return 9
	case "bfloat16", "float16", "bfp16":
		return 16
	default:
		return 0
	}
}

// readGeneric harvests top-level bits/group_size for unknown or method-less
// configs so the width is still reported.
//
//	readGeneric(&qi, map[string]any{"bits": 3.0})
func readGeneric(qi *QuantInfo, cfg map[string]any) {
	qi.Bits = numAny(cfg, "bits", "w_bit")
	qi.GroupSize = numAny(cfg, "group_size", "q_group_size")
	qi.Symmetric = boolOf(cfg, "sym")
	qi.DescAct = boolOf(cfg, "desc_act")
}

// readCompressedTensors derives the weight scheme from the first usable
// config_groups entry: weight num_bits / symmetric / strategy / group_size /
// actorder, and the input_activations num_bits (absent → 16-bit) to build the
// W{w}A{a} Scheme.
//
//	readCompressedTensors(&qi, cfg) // cfg has "config_groups": {"group_0": {"weights": {...}}}
func readCompressedTensors(qi *QuantInfo, cfg map[string]any) {
	groups, ok := cfg["config_groups"].(map[string]any)
	if !ok {
		return
	}
	for _, g := range groups {
		grp, ok := g.(map[string]any)
		if !ok {
			continue
		}
		weights, ok := grp["weights"].(map[string]any)
		if !ok {
			continue
		}
		wbits := num(weights, "num_bits")
		if wbits == 0 {
			continue
		}
		qi.Bits = wbits
		qi.GroupSize = num(weights, "group_size")
		qi.Symmetric = boolOf(weights, "symmetric")
		qi.DescAct = actOrder(str(weights, "actorder"))

		// Activation bits: from input_activations if quantised, else 16-bit.
		abits := 16
		if act, ok := grp["input_activations"].(map[string]any); ok {
			if n := num(act, "num_bits"); n != 0 {
				abits = n
			}
			qi.Activation = str(act, "strategy")
		}
		qi.Scheme = waScheme(wbits, abits)
		return // first usable group wins
	}
}

// actOrder reports whether a compressed-tensors weight "actorder" value implies
// activation-order reordering (desc_act). "weight" and "group" do; "none"/""
// do not.
//
//	actOrder("group") == true
func actOrder(v string) bool {
	switch core.Lower(core.Trim(v)) {
	case "weight", "group", "true":
		return true
	default:
		return false
	}
}

// weightOnlyScheme renders the conventional W{bits}A16 scheme for weight-only
// methods (GPTQ / AWQ). Zero bits → empty (nothing derivable).
//
//	weightOnlyScheme(4) == "W4A16"
func weightOnlyScheme(bits int) string {
	if bits == 0 {
		return ""
	}
	return waScheme(bits, 16)
}

// waScheme renders a W{w}A{a} scheme string. Zero weight bits → empty.
//
//	waScheme(4, 16) == "W4A16"
func waScheme(wbits, abits int) string {
	if wbits == 0 {
		return ""
	}
	return "W" + core.Itoa(wbits) + "A" + core.Itoa(abits)
}

// intLen returns the count of decimal digits needed to render n. n is always
// non-negative at its call sites (bit-width and group size), so it is used to
// size the String() builder exactly — one allocation, no slack bytes.
//
//	intLen(128) == 3
func intLen(n int) int {
	l := 1
	for n >= 10 {
		n /= 10
		l++
	}
	return l
}

// String renders a compact tag for logs and the registry: method, bit-width,
// group size, and scheme — each part omitted when absent.
//
//	quantfmt.QuantInfo{Method: quantfmt.MethodGPTQ, Bits: 4, GroupSize: 128}.String() == "gptq:4bit:g128"
//	quantfmt.QuantInfo{Method: quantfmt.MethodNone}.String() == "none"
func (qi QuantInfo) String() string {
	if qi.Method == "" || qi.Method == MethodNone {
		return string(MethodNone)
	}
	// The scheme suffix carries non-redundant information only for
	// compressed-tensors (the W/A split) and bitsandbytes (fp4/nf4). For the
	// weight-only methods (GPTQ / AWQ) the W{bits}A16 is implied by bits, so it
	// is omitted to keep the tag tight.
	showScheme := qi.Scheme != "" && qi.Method != MethodGPTQ && qi.Method != MethodAWQ

	// Sum the exact length first, then write once. The earlier `out += ...`
	// chain allocated a fresh string for each appended part (3-4 allocs/call);
	// a single exact-sized Builder produces the tag in one allocation — the
	// inherent floor (the returned string itself) — with no slack bytes and no
	// throwaway strconv temporaries.
	n := len(qi.Method)
	if qi.Bits != 0 {
		n += 4 + intLen(qi.Bits) // ':' + digits + "bit"
	}
	if qi.GroupSize > 0 {
		n += 2 + intLen(qi.GroupSize) // ":g" + digits
	}
	if showScheme {
		n += 1 + len(qi.Scheme) // ':' + scheme
	}

	var b core.Builder
	b.Grow(n)
	b.WriteString(string(qi.Method))
	var scratch [20]byte // stack — holds the digits of an int64
	if qi.Bits != 0 {
		b.WriteByte(':')
		b.Write(strconv.AppendInt(scratch[:0], int64(qi.Bits), 10))
		b.WriteString("bit")
	}
	if qi.GroupSize > 0 {
		b.WriteString(":g")
		b.Write(strconv.AppendInt(scratch[:0], int64(qi.GroupSize), 10))
	}
	if showScheme {
		b.WriteByte(':')
		b.WriteString(qi.Scheme)
	}
	return b.String()
}

// --- small typed accessors over the decoded map[string]any ---

// str reads a string field, returning "" when absent or wrong-typed.
//
//	str(cfg, "quant_method")
func str(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return v
	}
	return ""
}

// num reads a numeric field as int. JSON numbers decode to float64, so a float
// is coerced; an int (rare, from non-JSON callers) is honoured too.
//
//	num(cfg, "bits") // 4.0 → 4
func num(m map[string]any, key string) int {
	switch v := m[key].(type) {
	case float64:
		return int(v)
	case int:
		return v
	default:
		return 0
	}
}

// numAny reads the first present numeric field among keys (e.g. AWQ's
// bits/w_bit aliases).
//
//	numAny(cfg, "bits", "w_bit")
func numAny(m map[string]any, keys ...string) int {
	for _, k := range keys {
		if _, ok := m[k]; ok {
			if n := num(m, k); n != 0 {
				return n
			}
		}
	}
	return 0
}

// boolOf reads a bool field, defaulting to false when absent or wrong-typed.
//
//	boolOf(cfg, "sym")
func boolOf(m map[string]any, key string) bool {
	return boolOr(m, key, false)
}

// boolOr reads a bool field with an explicit default for the absent/wrong-typed
// case — lets AWQ treat "no zero_point key" as symmetric.
//
//	boolOr(cfg, "zero_point", false)
func boolOr(m map[string]any, key string, def bool) bool {
	if v, ok := m[key].(bool); ok {
		return v
	}
	return def
}

// asError adapts a core.Result error payload back to an error for core.E's cause
// slot, tolerating a nil/non-error value.
//
//	asError(r.Value)
func asError(v any) error {
	if err, ok := v.(error); ok {
		return err
	}
	return nil
}
