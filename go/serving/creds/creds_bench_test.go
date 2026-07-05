// SPDX-Licence-Identifier: EUPL-1.2

// Allocation contracts for credential rendering and key-policy routing (AX-11).
// String returns a freshly built masked string — one inherent allocation (the
// returned string). Allows is a pure membership test over the allow-list and
// must not allocate.
//
// Run: go test -bench=. -benchmem -run='^$' ./creds/
package creds

import "testing"

var (
	benchString string
	benchBool   bool
)

func BenchmarkCredential_String(b *testing.B) {
	c := Credential{Provider: "openai", Secret: "sk-secret"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchString = c.String()
	}
}

func BenchmarkKeyPolicy_Allows(b *testing.B) {
	p := KeyPolicy{AllowedProviders: []string{"local-metal", "openai", "openrouter"}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBool = p.Allows("openrouter")
	}
}
