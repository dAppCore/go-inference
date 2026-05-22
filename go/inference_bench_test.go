// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the inference orchestration types — backend registry
// lookups + LoadModel routing + AttentionSnapshot.HasQueries helper.
// Per AX-11 — Register fires once per backend init, but Get / List / All /
// Default run on every model load and every consumer that wants to
// enumerate available backends; HasQueries fires per attention snapshot.
//
// Run:    go test -bench='BenchmarkInference' -benchmem -run='^$' .

package inference

import (
	"testing"

	core "dappco.re/go"
)

// Sinks defeat compiler DCE. Distinct names from the gguf bench file.
var (
	inferenceBenchSinkBool    bool
	inferenceBenchSinkBackend Backend
	inferenceBenchSinkBackOK  bool
	inferenceBenchSinkNames   []string
	inferenceBenchSinkResult  core.Result
	inferenceBenchSinkCount   int
	inferenceBenchSinkSampler SamplerConfig
	inferenceBenchSinkGen     GenerateConfig
)

// benchRegisterPreferred wipes the global registry and primes it with
// preferred backends (metal, rocm, llama_cpp) plus n custom backends.
// All preferred are available; custom availability is alternating.
func benchRegisterPreferred(b *testing.B, custom int) {
	b.Helper()
	backendsMu.Lock()
	backends = map[string]Backend{}
	backendsMu.Unlock()
	Register(&inferenceBenchBackend{name: "metal", available: true})
	Register(&inferenceBenchBackend{name: "rocm", available: true})
	Register(&inferenceBenchBackend{name: "llama_cpp", available: true})
	for i := 0; i < custom; i++ {
		Register(&inferenceBenchBackend{
			name:      core.Sprintf("custom_%d", i),
			available: i%2 == 0,
		})
	}
}

// inferenceBenchBackend is a no-op Backend so the registry-level benches
// don't drag a real loader into the hot path. Distinct name from the
// existing test stubBackend to avoid colliding when the bench files share
// the package. LoadModel is never invoked from these benches, so we keep
// it minimal — the registered backend's role is to populate the registry
// for Get / List / All / Default.
type inferenceBenchBackend struct {
	name      string
	available bool
}

func (b *inferenceBenchBackend) Name() string    { return b.name }
func (b *inferenceBenchBackend) Available() bool { return b.available }
func (b *inferenceBenchBackend) LoadModel(_ string, _ ...LoadOption) (TextModel, error) {
	return nil, nil
}

// --- AttentionSnapshot.HasQueries (per-snapshot helper, pure scan) ---

func BenchmarkInference_HasQueries_True(b *testing.B) {
	snap := &AttentionSnapshot{
		NumLayers: 28,
		Queries:   make([][][]float32, 28),
	}
	for i := range snap.Queries {
		snap.Queries[i] = make([][]float32, 8)
		for j := range snap.Queries[i] {
			snap.Queries[i][j] = make([]float32, 128)
		}
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inferenceBenchSinkBool = snap.HasQueries()
	}
}

func BenchmarkInference_HasQueries_NilQueries(b *testing.B) {
	snap := &AttentionSnapshot{
		NumLayers: 28,
		Queries:   nil,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inferenceBenchSinkBool = snap.HasQueries()
	}
}

func BenchmarkInference_HasQueries_NilSnapshot(b *testing.B) {
	var snap *AttentionSnapshot
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inferenceBenchSinkBool = snap.HasQueries()
	}
}

// --- Registry: Get (per-lookup hot path on every LoadModel) ---

func BenchmarkInference_Get_Hit(b *testing.B) {
	benchRegisterPreferred(b, 0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inferenceBenchSinkBackend, inferenceBenchSinkBackOK = Get("metal")
	}
}

func BenchmarkInference_Get_Miss(b *testing.B) {
	benchRegisterPreferred(b, 0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inferenceBenchSinkBackend, inferenceBenchSinkBackOK = Get("nonexistent")
	}
}

// --- Registry: List (full snapshot + sort) ---

func BenchmarkInference_List_Three(b *testing.B) {
	benchRegisterPreferred(b, 0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inferenceBenchSinkNames = List()
	}
}

func BenchmarkInference_List_TwentyBackends(b *testing.B) {
	benchRegisterPreferred(b, 17)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inferenceBenchSinkNames = List()
	}
}

// --- Registry: All (iter.Seq2 snapshot + ranged yield) ---

func BenchmarkInference_All_Three(b *testing.B) {
	benchRegisterPreferred(b, 0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range All() {
			count++
		}
		inferenceBenchSinkCount = count
	}
}

func BenchmarkInference_All_TwentyBackends(b *testing.B) {
	benchRegisterPreferred(b, 17)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range All() {
			count++
		}
		inferenceBenchSinkCount = count
	}
}

// --- Registry: Default (preference-order scan) ---

func BenchmarkInference_Default_AllPreferred(b *testing.B) {
	benchRegisterPreferred(b, 0)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inferenceBenchSinkResult = Default()
	}
}

// Worst-case: metal + rocm + llama_cpp unavailable, fall through to a
// custom backend — exercises the second loop body.
func BenchmarkInference_Default_FallbackToCustom(b *testing.B) {
	backendsMu.Lock()
	backends = map[string]Backend{}
	backendsMu.Unlock()
	Register(&inferenceBenchBackend{name: "metal", available: false})
	Register(&inferenceBenchBackend{name: "rocm", available: false})
	Register(&inferenceBenchBackend{name: "llama_cpp", available: false})
	Register(&inferenceBenchBackend{name: "custom_vulkan", available: true})
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inferenceBenchSinkResult = Default()
	}
}

// --- Identity-bridge converters (per Generate call boundary) ---

func BenchmarkInference_SamplerConfigFromGenerateConfig(b *testing.B) {
	cfg := GenerateConfig{
		MaxTokens:     256,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		StopTokens:    []int32{2, 1, 0, 42, 1024},
		ReturnLogits:  true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inferenceBenchSinkSampler = SamplerConfigFromGenerateConfig(cfg)
	}
}

func BenchmarkInference_GenerateConfigFromSamplerConfig(b *testing.B) {
	cfg := SamplerConfig{
		MaxTokens:     256,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		StopTokens:    []int32{2, 1, 0, 42, 1024},
		ReturnLogits:  true,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		inferenceBenchSinkGen = GenerateConfigFromSamplerConfig(cfg)
	}
}
