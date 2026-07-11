// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// The quantize-dispatch benches baseline the format-resolution surface (AX-11), all per
// quantise request, none per element: resolveGGUFQuantizeFormat normalises a requested format
// to the kernel it maps to; ggufQuantizeLayout returns a format's (tensor-type, block-size,
// bytes-per-block); ValidationSummary joins issue codes for a failure report. Pure lookups +
// small string joins — no file, no kernel.

// BenchmarkResolveGGUFQuantizeFormat — the requested→used format resolve: a NormalizeQuantType
// + a switch. The per-request dispatch a quantise call pays once.
func BenchmarkResolveGGUFQuantizeFormat(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := resolveGGUFQuantizeFormat(QuantizeQ4_K_M); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkGGUFQuantizeLayout — the format→layout lookup: a switch returning the block
// geometry, no allocation. Read once per tensor to validate block alignment.
func BenchmarkGGUFQuantizeLayout(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, _, _, err := ggufQuantizeLayout(QuantizeQ4_K); err != nil {
			b.Fatal(err)
		}
	}
}

// BenchmarkValidationSummary — joining a handful of validation issue codes into a report
// string: the parts slice + the join, the cost of surfacing a failed GGUF validation.
func BenchmarkValidationSummary(b *testing.B) {
	issues := []ValidationIssue{
		{Code: "unknown_tensor_type", Tensor: "blk.0.attn_q.weight"},
		{Code: "invalid_tensor_shape", Tensor: "blk.1.ffn_down.weight"},
		{Code: "tensor_shape_not_block_aligned", Tensor: "blk.2.attn_k.weight"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = ValidationSummary(issues)
	}
}
