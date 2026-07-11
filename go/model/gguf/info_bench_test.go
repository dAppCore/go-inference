// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "testing"

// The info benches baseline the metadata-coercion helpers (AX-11): metadataInt / metadataString
// coerce a GGUF metadata value (arriving as an any of varied concrete type) to a Go int/string,
// run per probed key on the info path; normalizeArchitectureName + hasASCIIInsensitiveSuffix
// are the string helpers the arch resolution + file-suffix checks lean on. All pure, mostly
// allocation-free. (Info.Valid / ReadInfo need a parsed header from a .gguf file — benched via
// the model-load path, not here.)

// BenchmarkMetadataInt — coercing a float64-typed metadata value to int: the common GGUF wire
// shape (JSON-style numbers arrive as float64), a type switch, no allocation.
func BenchmarkMetadataInt(b *testing.B) {
	var v any = float64(262144)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if metadataInt(v) != 262144 {
			b.Fatal("mis-coerced")
		}
	}
}

// BenchmarkMetadataString — coercing a string-typed metadata value: the identity type switch,
// no allocation.
func BenchmarkMetadataString(b *testing.B) {
	var v any = "gemma3"
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if metadataString(v) != "gemma3" {
			b.Fatal("mis-coerced")
		}
	}
}

// BenchmarkNormalizeArchitectureName — the architecture-name normalise: the canonicalisation
// the arch resolution runs on config/metadata values.
func BenchmarkNormalizeArchitectureName(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = normalizeArchitectureName("Gemma3ForCausalLM")
	}
}

// BenchmarkHasASCIIInsensitiveSuffix — the allocation-free case-insensitive suffix check the
// weight-file validation runs per path (avoids a core.Lower allocation).
func BenchmarkHasASCIIInsensitiveSuffix(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if !hasASCIIInsensitiveSuffix("model-00001-of-00004.SafeTensors", ".safetensors") {
			b.Fatal("suffix not matched")
		}
	}
}
