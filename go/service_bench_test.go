// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the inference service registration shape — NewService
// factory + RegisterCore imperative variant. Per AX-11 — these fire
// once per Core construction, but anything embedded into the boot path
// of an SDK consumer or test fixture pays this cost on every startup.
//
// Run:    go test -bench='BenchmarkService' -benchmem -run='^$' .

package inference

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE.
var (
	serviceBenchSinkCore    *core.Core
	serviceBenchSinkResult  core.Result
	serviceBenchSinkFactory func(*core.Core) core.Result
)

// --- NewService factory construction (pure builder) ---

func BenchmarkService_NewService_Factory(b *testing.B) {
	opts := Options{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		serviceBenchSinkFactory = NewService(opts)
	}
}

// --- Full wire-up via core.WithService — what consumers actually pay. ---

func BenchmarkService_NewService_WiredIntoCore(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		serviceBenchSinkCore = core.New(core.WithService(NewService(Options{})))
	}
}

// --- RegisterCore imperative variant — same end-state, different entry. ---

func BenchmarkService_RegisterCore(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		serviceBenchSinkCore = core.New(core.WithService(RegisterCore))
	}
}

// --- RegisterCore invoked against a pre-built Core (no WithService). ---

func BenchmarkService_RegisterCore_OnExistingCore(b *testing.B) {
	c := core.New()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		serviceBenchSinkResult = RegisterCore(c)
	}
}
