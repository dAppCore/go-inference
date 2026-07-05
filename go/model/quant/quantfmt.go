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
// (activation_scheme, weight_block_size) and bitsandbytes (load_in_4bit /
// load_in_8bit, bnb_4bit_quant_type).
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
