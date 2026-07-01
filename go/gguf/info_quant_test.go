// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

func TestInfoQuant_NormalizeQuantType_Good(t *testing.T) {
	cases := []struct{ in, want string }{
		{"Q4_K_M", "q4_k_m"},
		{"Q4-K M", "q4_k_m"},
		{"  q8_0  ", "q8_0"},
		{"", ""},
	}
	for _, tc := range cases {
		if got := NormalizeQuantType(tc.in); got != tc.want {
			t.Errorf("NormalizeQuantType(%q) = %q, want %q", tc.in, got, tc.want)
		}
	}
}

func TestInfoQuant_ggufTensorBits_Good(t *testing.T) {
	if got := ggufTensorBits(TensorTypeQ4_0); got != 4 {
		t.Errorf("ggufTensorBits(Q4_0) = %d, want 4", got)
	}
	if got := ggufTensorBits(ggufTensorTypeF32); got != 0 {
		t.Errorf("ggufTensorBits(F32) = %d, want 0 (not quantised)", got)
	}
}

func TestInfoQuant_ggufTensorTypeDetails_Good(t *testing.T) {
	details := ggufTensorTypeDetails(ggufTensorTypeQ6K)
	if !details.Known || !details.Quantized || details.Bits != 6 || details.BlockSize != 256 {
		t.Errorf("ggufTensorTypeDetails(Q6_K) = %+v, want Known+Quantized bits=6 block=256", details)
	}
}

func TestInfoQuant_ggufTensorTypeDetails_Bad(t *testing.T) {
	details := ggufTensorTypeDetails(9999)
	if details.Known {
		t.Errorf("ggufTensorTypeDetails(out-of-range) = %+v, want Known=false", details)
	}
}

func TestInfoQuant_buildGGUFTensorInfos_Good(t *testing.T) {
	tensors := []TensorInfo{
		{Name: "t0", Type: TensorTypeQ4_0, Shape: []uint64{32, 4}},
	}
	built, issues := buildGGUFTensorInfos(tensors)
	if len(issues) != 0 {
		t.Fatalf("issues = %+v, want none", issues)
	}
	if built[0].DType != "ggml_q4_0" || built[0].Bits != 4 || built[0].BlockSize != 32 {
		t.Errorf("built[0] = %+v, want ggml_q4_0/4bit/block32", built[0])
	}
	if built[0].Elements != 128 {
		t.Errorf("Elements = %d, want 128", built[0].Elements)
	}
	if !built[0].Quantized {
		t.Errorf("Quantized = false, want true")
	}
}

func TestInfoQuant_buildGGUFTensorInfos_Bad(t *testing.T) {
	tensors := []TensorInfo{{Name: "unknown", Type: 9999, Shape: []uint64{4}}}
	_, issues := buildGGUFTensorInfos(tensors)
	if !ggufValidationHasCode(issues, "unknown_tensor_type") {
		t.Errorf("issues = %+v, want unknown_tensor_type", issues)
	}
}

func TestInfoQuant_buildGGUFTensorInfos_Ugly(t *testing.T) {
	// Q4_0 has BlockSize 32; a first dimension of 5 is not block-aligned.
	tensors := []TensorInfo{{Name: "misaligned", Type: TensorTypeQ4_0, Shape: []uint64{5}}}
	_, issues := buildGGUFTensorInfos(tensors)
	if !ggufValidationHasCode(issues, "tensor_shape_not_block_aligned") {
		t.Errorf("issues = %+v, want tensor_shape_not_block_aligned", issues)
	}

	empty := []TensorInfo{{Name: "no-shape", Type: ggufTensorTypeF32}}
	_, issues = buildGGUFTensorInfos(empty)
	if !ggufValidationHasCode(issues, "invalid_tensor_shape") {
		t.Errorf("issues = %+v, want invalid_tensor_shape", issues)
	}

	zeroDim := []TensorInfo{{Name: "zero-dim", Type: ggufTensorTypeF32, Shape: []uint64{4, 0}}}
	_, issues = buildGGUFTensorInfos(zeroDim)
	if !ggufValidationHasCode(issues, "invalid_tensor_dimension") {
		t.Errorf("issues = %+v, want invalid_tensor_dimension", issues)
	}
}

func TestInfoQuant_ggufTensorElements_Good(t *testing.T) {
	if got := ggufTensorElements([]uint64{2, 3, 4}); got != 24 {
		t.Errorf("ggufTensorElements = %d, want 24", got)
	}
	if got := ggufTensorElements(nil); got != 0 {
		t.Errorf("ggufTensorElements(nil) = %d, want 0", got)
	}
	if got := ggufTensorElements([]uint64{4, 0}); got != 0 {
		t.Errorf("ggufTensorElements(with zero dim) = %d, want 0", got)
	}
}

func TestInfoQuant_summarizeGGUFTensorTypes_Good(t *testing.T) {
	tensors, _ := buildGGUFTensorInfos([]TensorInfo{
		{Name: "a", Type: TensorTypeQ4_0, Shape: []uint64{32}},
		{Name: "b", Type: TensorTypeQ4_0, Shape: []uint64{32}},
		{Name: "c", Type: ggufTensorTypeF32, Shape: []uint64{4}},
	})
	summaries := summarizeGGUFTensorTypes(tensors)
	if len(summaries) != 2 {
		t.Fatalf("summaries = %+v, want 2 distinct types", summaries)
	}
	// Highest count sorts first.
	if summaries[0].Name != "q4_0" || summaries[0].Count != 2 {
		t.Errorf("summaries[0] = %+v, want q4_0 count=2 first", summaries[0])
	}
}

func TestInfoQuant_summarizeGGUFTensorTypes_Bad(t *testing.T) {
	if got := summarizeGGUFTensorTypes(nil); got != nil {
		t.Errorf("summarizeGGUFTensorTypes(nil) = %+v, want nil", got)
	}
}

func TestInfoQuant_majorityGGUFQuantizedTensorType_Good(t *testing.T) {
	summaries := []TensorTypeSummary{
		{Name: "f32", Bits: 32, Quantized: false, Count: 5},
		{Name: "q4_0", Bits: 4, BlockSize: 32, Quantized: true, Count: 3},
	}
	name, bits, group := majorityGGUFQuantizedTensorType(summaries)
	if name != "q4_0" || bits != 4 || group != 32 {
		t.Errorf("majorityGGUFQuantizedTensorType = (%q, %d, %d), want (q4_0, 4, 32)", name, bits, group)
	}
}

func TestInfoQuant_ggufFileTypeQuantization_Good(t *testing.T) {
	name, bits := ggufFileTypeQuantization(15)
	if name != "q4_k_m" || bits != 4 {
		t.Errorf("ggufFileTypeQuantization(15) = (%q, %d), want (q4_k_m, 4)", name, bits)
	}
}

func TestInfoQuant_ggufFileTypeQuantization_Bad(t *testing.T) {
	name, bits := ggufFileTypeQuantization(-1)
	if name != "" || bits != 0 {
		t.Errorf("ggufFileTypeQuantization(-1) = (%q, %d), want (\"\", 0)", name, bits)
	}
	name, bits = ggufFileTypeQuantization(9999)
	if name != "" || bits != 0 {
		t.Errorf("ggufFileTypeQuantization(9999) = (%q, %d), want (\"\", 0)", name, bits)
	}
}

func TestInfoQuant_quantBitsFromTypeName_Good(t *testing.T) {
	cases := []struct {
		name string
		want int
	}{
		{"q4_k_m", 4}, {"q8_0", 8}, {"q6_k", 6}, {"q3_k", 3}, {"q2_k", 2},
		{"iq1_s", 1}, {"f16", 16}, {"f32", 32}, {"f64", 64}, {"", 0}, {"nonsense", 0},
	}
	for _, tc := range cases {
		if got := quantBitsFromTypeName(tc.name); got != tc.want {
			t.Errorf("quantBitsFromTypeName(%q) = %d, want %d", tc.name, got, tc.want)
		}
	}
}

func TestInfoQuant_quantFamilyForType_Good(t *testing.T) {
	cases := []struct {
		name string
		want string
	}{
		{"iq4_xs", "iq"}, {"mxfp4", "mxfp"}, {"nvfp4", "nvfp"}, {"q4_k", "qk"},
		{"q8_0", "q8"}, {"q5_0", "q5"}, {"q4_0", "q4"}, {"q3_k", "qk"},
		{"q2_k", "qk"}, {"tq1_0", "tq"}, {"f32", "dense"}, {"", ""}, {"unknown", ""},
	}
	for _, tc := range cases {
		if got := quantFamilyForType(tc.name); got != tc.want {
			t.Errorf("quantFamilyForType(%q) = %q, want %q", tc.name, got, tc.want)
		}
	}
}

func TestInfoQuant_ggufQuantizationIsMixed_Good(t *testing.T) {
	if !ggufQuantizationIsMixed("q3_k_m", nil) {
		t.Errorf("q3_k_m should be reported mixed by suffix")
	}
	summaries := []TensorTypeSummary{
		{Name: "q4_0", Quantized: true},
		{Name: "q8_0", Quantized: true},
	}
	if !ggufQuantizationIsMixed("q4_0", summaries) {
		t.Errorf("two distinct quantised tensor types should be reported mixed")
	}
}

func TestInfoQuant_ggufQuantizationIsMixed_Bad(t *testing.T) {
	summaries := []TensorTypeSummary{{Name: "q4_0", Quantized: true}}
	if ggufQuantizationIsMixed("q4_0", summaries) {
		t.Errorf("a single quantised tensor type should not be reported mixed")
	}
}

func TestInfoQuant_inferGGUFQuantization_Good(t *testing.T) {
	tensors, _ := buildGGUFTensorInfos([]TensorInfo{
		{Name: "a", Type: TensorTypeQ4_0, Shape: []uint64{32}},
	})
	metadata := map[string]any{"general.file_type": uint32(2)}
	quant := inferGGUFQuantization(metadata, tensors)
	if quant.Type != "q4_0" || quant.Bits != 4 || quant.FileType != 2 {
		t.Errorf("inferGGUFQuantization = %+v, want type=q4_0 bits=4 file_type=2", quant)
	}
}

func TestInfoQuant_indexString_Good(t *testing.T) {
	if got := indexString("blk.5.attn_q.weight", "blk."); got != 0 {
		t.Errorf("indexString = %d, want 0", got)
	}
	if got := indexString("", ""); got != 0 {
		t.Errorf("indexString(empty, empty) = %d, want 0", got)
	}
}

func TestInfoQuant_indexString_Bad(t *testing.T) {
	if got := indexString("short", "much-longer-substring"); got != -1 {
		t.Errorf("indexString(short haystack) = %d, want -1", got)
	}
	if got := indexString("no marker here", "blk."); got != -1 {
		t.Errorf("indexString(no match) = %d, want -1", got)
	}
}
