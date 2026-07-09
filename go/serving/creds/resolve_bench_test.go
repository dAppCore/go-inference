// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contract for credential resolution (AX-11). Resolve is the
// per-request credential pick (RFC §6.17): the BYOK, local-hit and stored-hit
// paths each return a Credential value with no heap work, so these benches pin
// them to zero.
//
// Run: go test -bench=. -benchmem -run='^$' ./creds/
package creds

import "testing"

var (
	benchCred Credential
	benchErr  error
)

func benchResolver() *Resolver {
	r := New()
	r.MarkLocal("local-metal")
	_ = r.Set(Credential{Provider: "openai", Secret: "sk-bench-secret"})
	return r
}

func BenchmarkResolver_Resolve_Stored(b *testing.B) {
	r := benchResolver()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCred, benchErr = r.Resolve("openai", nil)
	}
}

func BenchmarkResolver_Resolve_Local(b *testing.B) {
	r := benchResolver()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCred, benchErr = r.Resolve("local-metal", nil)
	}
}

func BenchmarkResolver_Resolve_BYOK(b *testing.B) {
	r := benchResolver()
	byok := &Credential{Provider: "openrouter", Secret: "sk-byok"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchCred, benchErr = r.Resolve("openrouter", byok)
	}
}
