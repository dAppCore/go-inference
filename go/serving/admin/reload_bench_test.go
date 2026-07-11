// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the reload path-containment check. Per AX-11 — pathWithinDir
// is the security gate on every hot-swap /reload request: it decides whether a
// resolved model path stays inside the models dir (a traversal-escape refusal).
// It runs once per admin reload (low multiplier), but it is on the security
// path, so its allocation profile — a PathRel plus prefix checks — is worth
// pinning. The within, escape and equal cases take different branches.
//
// Run:    go test -bench=PathWithinDir -benchmem -run='^$' ./serving/admin/
package admin

import "testing"

var adminBenchSinkBool bool

func BenchmarkPathWithinDir_Within(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adminBenchSinkBool = pathWithinDir("/models", "/models/gemma-4-31b")
	}
}

// Escape: a sibling dir sharing the root's prefix must be refused — the branch
// a traversal attempt exercises.
func BenchmarkPathWithinDir_Escape(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adminBenchSinkBool = pathWithinDir("/models", "/models-evil/payload")
	}
}

// Equal: the exact-root fast path (resolved == rootResolved).
func BenchmarkPathWithinDir_Equal(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		adminBenchSinkBool = pathWithinDir("/models", "/models")
	}
}
