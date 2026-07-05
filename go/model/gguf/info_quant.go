// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import (
	"slices"
	"sort"
	"strconv"

	core "dappco.re/go"
)

func ggufTensorBits(tensorType uint32) int {
	details := ggufTensorTypeDetails(tensorType)
	if !details.Known || !details.Quantized {
		return 0
	}
	return details.Bits
}

type ggufTensorTypeDetailsInfo struct {
	Name      string
	DType     string
	Bits      int
	BlockSize int
	Quantized bool
	Known     bool
}

// ggufTensorTypeDetailsTable — direct lookup by tensorType id, replaces the
// 35-case switch in the per-tensor hot path. IDs are bounded 0..40 with
// gaps (4, 5, 36, 37, 38 unused in current GGML — the last three are the
// removed IQ4_NL_4_4/4_8/8_8 slots, which is why MXFP4/NVFP4 land at 39/40
// here rather than the 38/39 the separate llama_ftype file-type enum uses
// for the same quant families); unused entries default to the zero
// ggufTensorTypeDetailsInfo (Known=false, treated as unknown).
var ggufTensorTypeDetailsTable = [41]ggufTensorTypeDetailsInfo{
	ggufTensorTypeF32:      {Name: "f32", DType: "float32", Bits: 32, Known: true},
	ggufTensorTypeF16:      {Name: "f16", DType: "float16", Bits: 16, Known: true},
	TensorTypeQ4_0:         {Name: "q4_0", DType: "ggml_q4_0", Bits: 4, BlockSize: 32, Quantized: true, Known: true},
	ggufTensorTypeQ4_1:     {Name: "q4_1", DType: "ggml_q4_1", Bits: 4, BlockSize: 32, Quantized: true, Known: true},
	ggufTensorTypeQ5_0:     {Name: "q5_0", DType: "ggml_q5_0", Bits: 5, BlockSize: 32, Quantized: true, Known: true},
	ggufTensorTypeQ5_1:     {Name: "q5_1", DType: "ggml_q5_1", Bits: 5, BlockSize: 32, Quantized: true, Known: true},
	TensorTypeQ8_0:         {Name: "q8_0", DType: "ggml_q8_0", Bits: 8, BlockSize: 32, Quantized: true, Known: true},
	ggufTensorTypeQ8_1:     {Name: "q8_1", DType: "ggml_q8_1", Bits: 8, BlockSize: 32, Quantized: true, Known: true},
	ggufTensorTypeQ2K:      {Name: "q2_k", DType: "ggml_q2_k", Bits: 2, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeQ3K:      {Name: "q3_k", DType: "ggml_q3_k", Bits: 3, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeQ4K:      {Name: "q4_k", DType: "ggml_q4_k", Bits: 4, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeQ5K:      {Name: "q5_k", DType: "ggml_q5_k", Bits: 5, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeQ6K:      {Name: "q6_k", DType: "ggml_q6_k", Bits: 6, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeQ8K:      {Name: "q8_k", DType: "ggml_q8_k", Bits: 8, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeIQ2XXS:   {Name: "iq2_xxs", DType: "ggml_iq2_xxs", Bits: 2, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeIQ2XS:    {Name: "iq2_xs", DType: "ggml_iq2_xs", Bits: 2, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeIQ3XXS:   {Name: "iq3_xxs", DType: "ggml_iq3_xxs", Bits: 3, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeIQ1S:     {Name: "iq1_s", DType: "ggml_iq1_s", Bits: 1, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeIQ4NL:    {Name: "iq4_nl", DType: "ggml_iq4_nl", Bits: 4, BlockSize: 32, Quantized: true, Known: true},
	ggufTensorTypeIQ3S:     {Name: "iq3_s", DType: "ggml_iq3_s", Bits: 3, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeIQ2S:     {Name: "iq2_s", DType: "ggml_iq2_s", Bits: 2, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeIQ4XS:    {Name: "iq4_xs", DType: "ggml_iq4_xs", Bits: 4, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeI8:       {Name: "i8", DType: "int8", Bits: 8, Known: true},
	ggufTensorTypeI16:      {Name: "i16", DType: "int16", Bits: 16, Known: true},
	ggufTensorTypeI32:      {Name: "i32", DType: "int32", Bits: 32, Known: true},
	ggufTensorTypeI64:      {Name: "i64", DType: "int64", Bits: 64, Known: true},
	ggufTensorTypeF64:      {Name: "f64", DType: "float64", Bits: 64, Known: true},
	ggufTensorTypeIQ1M:     {Name: "iq1_m", DType: "ggml_iq1_m", Bits: 1, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeBF16:     {Name: "bf16", DType: "bfloat16", Bits: 16, Known: true},
	ggufTensorTypeQ4_0_4_4: {Name: "q4_0_4_4", DType: "ggml_q4_0_4_4", Bits: 4, BlockSize: 32, Quantized: true, Known: true},
	ggufTensorTypeQ4_0_4_8: {Name: "q4_0_4_8", DType: "ggml_q4_0_4_8", Bits: 4, BlockSize: 32, Quantized: true, Known: true},
	ggufTensorTypeQ4_0_8_8: {Name: "q4_0_8_8", DType: "ggml_q4_0_8_8", Bits: 4, BlockSize: 32, Quantized: true, Known: true},
	ggufTensorTypeTQ1_0:    {Name: "tq1_0", DType: "ggml_tq1_0", Bits: 1, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeTQ2_0:    {Name: "tq2_0", DType: "ggml_tq2_0", Bits: 2, BlockSize: 256, Quantized: true, Known: true},
	ggufTensorTypeMXFP4:    {Name: "mxfp4", DType: "ggml_mxfp4", Bits: 4, BlockSize: 32, Quantized: true, Known: true},
	ggufTensorTypeNVFP4:    {Name: "nvfp4", DType: "ggml_nvfp4", Bits: 4, BlockSize: 32, Quantized: true, Known: true},
}

func ggufTensorTypeDetails(tensorType uint32) ggufTensorTypeDetailsInfo {
	if tensorType < uint32(len(ggufTensorTypeDetailsTable)) {
		return ggufTensorTypeDetailsTable[tensorType]
	}
	return ggufTensorTypeDetailsInfo{}
}

// buildGGUFTensorInfos fills the derived dtype/quantisation fields of an
// already-parsed TensorInfo slice in place and returns it alongside any
// validation issues. parseGGUF hands over a slice whose base fields
// (Name/Type/Shape/Offset) are set; the TypeName/DType/Bits/BlockSize/
// Elements/Quantized fields arrive zero-valued and are completed here. The
// slice is mutated and returned directly — no second allocation, no copy
// loop.
func buildGGUFTensorInfos(tensors []TensorInfo) ([]TensorInfo, []ValidationIssue) {
	var issues []ValidationIssue
	for i := range tensors {
		tensor := &tensors[i]
		details := ggufTensorTypeDetails(tensor.Type)
		// Base fields (Name/Type/Shape/Offset) are already populated by
		// parseGGUF — Shape ownership was transferred there. Fill only the
		// derived fields in place.
		tensor.TypeName = details.Name
		tensor.DType = details.DType
		tensor.Bits = details.Bits
		tensor.BlockSize = details.BlockSize
		tensor.Elements = ggufTensorElements(tensor.Shape)
		tensor.Quantized = details.Quantized

		if !details.Known {
			issues = append(issues, ValidationIssue{
				Severity: GGUFValidationError,
				Code:     "unknown_tensor_type",
				Message:  "tensor has unknown GGML type id " + strconv.FormatUint(uint64(tensor.Type), 10),
				Tensor:   tensor.Name,
			})
		}
		if len(tensor.Shape) == 0 {
			issues = append(issues, ValidationIssue{
				Severity: GGUFValidationError,
				Code:     "invalid_tensor_shape",
				Message:  "tensor has no shape dimensions",
				Tensor:   tensor.Name,
			})
		}
		if slices.Contains(tensor.Shape, 0) {
			issues = append(issues, ValidationIssue{
				Severity: GGUFValidationError,
				Code:     "invalid_tensor_dimension",
				Message:  "tensor shape contains a zero dimension",
				Tensor:   tensor.Name,
			})
		}
		if details.Known && details.Quantized && details.BlockSize > 0 && len(tensor.Shape) > 0 && tensor.Shape[0] > 0 && tensor.Shape[0]%uint64(details.BlockSize) != 0 {
			issues = append(issues, ValidationIssue{
				Severity: GGUFValidationError,
				Code:     "tensor_shape_not_block_aligned",
				Message:  "tensor first dimension " + strconv.FormatUint(tensor.Shape[0], 10) + " is not divisible by GGML block size " + strconv.Itoa(details.BlockSize),
				Tensor:   tensor.Name,
			})
		}
	}
	return tensors, issues
}

func ggufTensorElements(shape []uint64) uint64 {
	if len(shape) == 0 {
		return 0
	}
	total := uint64(1)
	for _, dim := range shape {
		if dim == 0 {
			return 0
		}
		total *= dim
	}
	return total
}

func inferGGUFQuantization(metadata map[string]any, tensors []TensorInfo) QuantizationInfo {
	tensorTypes := summarizeGGUFTensorTypes(tensors)
	fileType, fileTypePresent := metadataIntIfPresent(metadata, "general.file_type")
	var fileTypeName string
	var fileTypeBits int
	if fileTypePresent {
		fileTypeName, fileTypeBits = ggufFileTypeQuantization(fileType)
	}
	explicitType := NormalizeQuantType(core.FirstNonBlank(
		metadataString(metadata["general.quantization_type"]),
		metadataString(metadata["quantization.type"]),
		metadataString(metadata["quantization.name"]),
		metadataString(metadata["general.quantization"]),
	))
	majorityType, majorityBits, majorityGroup := majorityGGUFQuantizedTensorType(tensorTypes)
	quantType := core.FirstNonBlank(explicitType, fileTypeName, majorityType)
	bits := core.FirstPositive(quantBitsFromTypeName(quantType), fileTypeBits, majorityBits)
	family := quantFamilyForType(quantType)
	if family == "" && majorityType != "" {
		family = quantFamilyForType(majorityType)
	}
	group := core.FirstPositive(metadataInt(metadata["quantization.group_size"]), metadataInt(metadata["general.quantization_group_size"]), majorityGroup)
	return QuantizationInfo{
		Type:         quantType,
		Family:       family,
		Bits:         bits,
		GroupSize:    group,
		FileType:     fileType,
		FileTypeName: fileTypeName,
		Version:      metadataInt(metadata["general.quantization_version"]),
		Mixed:        ggufQuantizationIsMixed(quantType, tensorTypes),
		TensorTypes:  tensorTypes,
	}
}

func metadataIntIfPresent(metadata map[string]any, key string) (int, bool) {
	value, ok := metadata[key]
	if !ok {
		return 0, false
	}
	return metadataInt(value), true
}

func summarizeGGUFTensorTypes(tensors []TensorInfo) []TensorTypeSummary {
	// Real GGUF files surface ~2-10 distinct tensor types (often just
	// f32 + one quant variant). A linear search over a small slice is
	// faster than a map allocation + hashing per-tensor here, and skips
	// the materialise-then-copy round-trip into the output slice.
	if len(tensors) == 0 {
		return nil
	}
	out := make([]TensorTypeSummary, 0, 8)
	for i := range tensors {
		t := &tensors[i]
		found := false
		for j := range out {
			if out[j].Type == t.Type && out[j].Name == t.TypeName {
				out[j].Count++
				found = true
				break
			}
		}
		if !found {
			out = append(out, TensorTypeSummary{
				Type:      t.Type,
				Name:      t.TypeName,
				DType:     t.DType,
				Bits:      t.Bits,
				BlockSize: t.BlockSize,
				Quantized: t.Quantized,
				Count:     1,
			})
		}
	}
	if len(out) > 1 {
		sort.Slice(out, func(i, j int) bool {
			if out[i].Count != out[j].Count {
				return out[i].Count > out[j].Count
			}
			return out[i].Name < out[j].Name
		})
	}
	return out
}

func majorityGGUFQuantizedTensorType(summaries []TensorTypeSummary) (string, int, int) {
	var best TensorTypeSummary
	for _, summary := range summaries {
		if !summary.Quantized {
			continue
		}
		if summary.Count > best.Count || (summary.Count == best.Count && summary.Bits > best.Bits) {
			best = summary
		}
	}
	return best.Name, best.Bits, best.BlockSize
}

func quantizationGroupFromTensorTypes(summaries []TensorTypeSummary) int {
	_, _, group := majorityGGUFQuantizedTensorType(summaries)
	return group
}

// ggufFileTypeQuantizationTable — direct lookup table by GGUF file_type.
// Replaces the case-by-case switch; lives in .rodata. Index 5, 6 unused
// in the spec — those slots hold zero values (matching the prior default
// arm "", 0).
type ggufFileTypeEntry struct {
	Name string
	Bits int
}

var ggufFileTypeQuantizationTable = [40]ggufFileTypeEntry{
	0:  {"f32", 32},
	1:  {"f16", 16},
	2:  {"q4_0", 4},
	3:  {"q4_1", 4},
	4:  {"q4_1_some_f16", 4},
	7:  {"q8_0", 8},
	8:  {"q5_0", 5},
	9:  {"q5_1", 5},
	10: {"q2_k", 2},
	11: {"q3_k_s", 3},
	12: {"q3_k_m", 3},
	13: {"q3_k_l", 3},
	14: {"q4_k_s", 4},
	15: {"q4_k_m", 4},
	16: {"q5_k_s", 5},
	17: {"q5_k_m", 5},
	18: {"q6_k", 6},
	19: {"iq2_xxs", 2},
	20: {"iq2_xs", 2},
	21: {"q2_k_s", 2},
	22: {"iq3_xs", 3},
	23: {"iq3_xxs", 3},
	24: {"iq1_s", 1},
	25: {"iq4_nl", 4},
	26: {"iq3_s", 3},
	27: {"iq3_m", 3},
	28: {"iq2_s", 2},
	29: {"iq2_m", 2},
	30: {"iq4_xs", 4},
	31: {"iq1_m", 1},
	32: {"bf16", 16},
	33: {"q4_0_4_4", 4},
	34: {"q4_0_4_8", 4},
	35: {"q4_0_8_8", 4},
	36: {"tq1_0", 1},
	37: {"tq2_0", 2},
	38: {"mxfp4", 4},
	39: {"nvfp4", 4},
}

func ggufFileTypeQuantization(fileType int) (string, int) {
	if fileType >= 0 && fileType < len(ggufFileTypeQuantizationTable) {
		e := ggufFileTypeQuantizationTable[fileType]
		return e.Name, e.Bits
	}
	return "", 0
}

// NormalizeQuantType lowercases a GGUF/GGML quantisation type name and
// folds '-' and ' ' separators to '_' (e.g. "Q4-K M" → "q4_k_m").
//
//	gguf.NormalizeQuantType("Q4_K_M") // "q4_k_m"
func NormalizeQuantType(value string) string {
	value = core.Lower(core.Trim(value))
	value = core.Replace(value, "-", "_")
	value = core.Replace(value, " ", "_")
	return value
}

func quantBitsFromTypeName(name string) int {
	name = NormalizeQuantType(name)
	switch {
	case name == "":
		return 0
	case core.Contains(name, "bf16") || core.Contains(name, "f16"):
		return 16
	case core.Contains(name, "f32"):
		return 32
	case core.Contains(name, "f64"):
		return 64
	case core.Contains(name, "nvfp4") || core.Contains(name, "mxfp4") || core.Contains(name, "iq4") || core.Contains(name, "q4"):
		return 4
	case core.Contains(name, "iq5") || core.Contains(name, "q5"):
		return 5
	case core.Contains(name, "iq8") || core.Contains(name, "q8"):
		return 8
	case core.Contains(name, "iq6") || core.Contains(name, "q6"):
		return 6
	case core.Contains(name, "iq3") || core.Contains(name, "q3"):
		return 3
	case core.Contains(name, "iq2") || core.Contains(name, "q2"):
		return 2
	case core.Contains(name, "iq1") || core.Contains(name, "tq1"):
		return 1
	default:
		return 0
	}
}

func quantFamilyForType(name string) string {
	name = NormalizeQuantType(name)
	switch {
	case name == "":
		return ""
	case core.HasPrefix(name, "iq"):
		return "iq"
	case core.HasPrefix(name, "mxfp"):
		return "mxfp"
	case core.HasPrefix(name, "nvfp"):
		return "nvfp"
	case core.Contains(name, "_k"):
		return "qk"
	case core.HasPrefix(name, "q8"):
		return "q8"
	case core.HasPrefix(name, "q5"):
		return "q5"
	case core.HasPrefix(name, "q4"):
		return "q4"
	case core.HasPrefix(name, "q3"):
		return "q3"
	case core.HasPrefix(name, "q2"):
		return "q2"
	case core.HasPrefix(name, "tq"):
		return "tq"
	case name == "f16" || name == "f32" || name == "bf16" || name == "f64":
		return "dense"
	default:
		return ""
	}
}

func ggufQuantizationIsMixed(quantType string, summaries []TensorTypeSummary) bool {
	quantType = NormalizeQuantType(quantType)
	if core.HasSuffix(quantType, "_m") || core.Contains(quantType, "some_f16") {
		return true
	}
	// summaries is the output of summarizeGGUFTensorTypes, which already
	// deduplicates by (Type, TypeName). Just count the quantised entries
	// directly — no need for a map.
	quantisedCount := 0
	for i := range summaries {
		if summaries[i].Quantized && summaries[i].Name != "" {
			quantisedCount++
			if quantisedCount > 1 {
				return true
			}
		}
	}
	return false
}

func indexString(s, substr string) int {
	if substr == "" {
		return 0
	}
	if len(substr) > len(s) {
		return -1
	}
	for i := range len(s) - len(substr) + 1 {
		if s[i:i+len(substr)] == substr {
			return i
		}
	}
	return -1
}
