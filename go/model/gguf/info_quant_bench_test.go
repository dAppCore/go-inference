// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// The info-quant benches baseline the quantisation-classification path (AX-11): the string
// helpers (NormalizeQuantType, quantBitsFromTypeName, quantFamilyForType) resolve a quant
// type from GGUF metadata, and the tensor-summary path (buildGGUFTensorInfos →
// summarizeGGUFTensorTypes → inferGGUFQuantization) derives a checkpoint's quantisation
// profile from its tensor directory — run once per GGUF load. buildGGUFTensorInfos fills
// derived fields in place (zero-alloc); summarize + infer build the small type-summary.
// Synthetic tensor directory — no file.

func benchTensorInfos(n int) []TensorInfo {
	out := make([]TensorInfo, n)
	for i := range out {
		var typ uint32 = ggufTensorTypeQ4K
		if i%4 == 0 {
			typ = ggufTensorTypeF32 // a few dense tensors alongside the quantised majority
		}
		out[i] = TensorInfo{Name: "blk.weight", Type: typ, Shape: []uint64{256, 2048}}
	}
	return out
}

// BenchmarkNormalizeQuantType — the type-name normalise (lower + '-'/' '→'_'): the per-key
// canonicalisation on the metadata-classification path.
func BenchmarkNormalizeQuantType(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = NormalizeQuantType("Q4-K M")
	}
}

// BenchmarkQuantBitsFromTypeName — the bit-width classify: a normalise + a substring switch.
func BenchmarkQuantBitsFromTypeName(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if quantBitsFromTypeName("q4_k_m") != 4 {
			b.Fatal("misclassified")
		}
	}
}

// BenchmarkQuantFamilyForType — the family classify (iq/qk/q8/…): a normalise + prefix switch.
func BenchmarkQuantFamilyForType(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = quantFamilyForType("q4_k_m")
	}
}

// BenchmarkBuildGGUFTensorInfos — the derived-field fill over a 300-tensor directory: in-place
// TypeName/DType/Bits/BlockSize/Elements/Quantized completion, expected zero allocation on the
// all-valid path.
func BenchmarkBuildGGUFTensorInfos(b *testing.B) {
	tensors := benchTensorInfos(300)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = buildGGUFTensorInfos(tensors)
	}
}

// BenchmarkInferGGUFQuantization — the whole quantisation-profile derivation: summarise the
// tensor types + resolve the majority quant against the metadata. The type-summary slice is
// the allocation.
func BenchmarkInferGGUFQuantization(b *testing.B) {
	tensors, _ := buildGGUFTensorInfos(benchTensorInfos(300))
	metadata := map[string]any{"general.file_type": 15, "general.quantization_version": 2}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = inferGGUFQuantization(metadata, tensors)
	}
}
