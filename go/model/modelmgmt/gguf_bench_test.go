// SPDX-Licence-Identifier: EUPL-1.2

package modelmgmt

import "testing"

// Per-tensor key/dtype translation — hit once per tensor on every
// MLX→GGUF conversion. ParseLayerFromTensorName re-compiles a regex on
// every call (24 allocs/op baseline); locked in as shape, candidate for
// hoisting if convert path is profiled hot.

func BenchmarkMLXTensorToGGUF(b *testing.B) {
	// Success path: the regex requires the "model." prefix, so this is the
	// realistic per-tensor key seen during an MLX→GGUF conversion.
	key := "model.layers.0.self_attn.q_proj.lora_a"
	b.ReportAllocs()
	for b.Loop() {
		_ = MLXTensorToGGUF(key)
	}
}

func BenchmarkSafetensorsDtypeToGGML_F16(b *testing.B) {
	b.ReportAllocs()
	for b.Loop() {
		_ = SafetensorsDtypeToGGML("F16")
	}
}

func BenchmarkSafetensorsDtypeToGGML_Unknown(b *testing.B) {
	// Error path — extra Sprintf/E.
	b.ReportAllocs()
	for b.Loop() {
		_ = SafetensorsDtypeToGGML("Q4_0")
	}
}

func BenchmarkParseLayerFromTensorName(b *testing.B) {
	name := "blk.5.attn_q.weight"
	b.ReportAllocs()
	for b.Loop() {
		_ = ParseLayerFromTensorName(name)
	}
}

func BenchmarkParseLayerFromTensorName_NoMatch(b *testing.B) {
	// Error path — no layer number.
	name := "output_norm.weight"
	b.ReportAllocs()
	for b.Loop() {
		_ = ParseLayerFromTensorName(name)
	}
}
