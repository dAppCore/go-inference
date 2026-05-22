// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the identity / state-bundle surface.
// Per AX-11 — SamplerConfigFromGenerateConfig fires per request when
// state primitives capture the active sampler, and the reverse
// conversion fires per session resume. ProjectSeed.WakeRequest fires
// per wake; CheckWakeCompatibility fires per wake to validate the
// bundle against the live runtime — its allocation profile matters
// because every wake pays it.
//
// Run:    go test -bench=BenchmarkIdentity -benchmem -run='^$' .

package inference

import (
	"testing"
)

// Sinks defeat compiler DCE.
var (
	identityBenchSinkSampler        SamplerConfig
	identityBenchSinkGenerateCfg    GenerateConfig
	identityBenchSinkSeed           ProjectSeed
	identityBenchSinkWakeRequest    AgentMemoryWakeRequest
	identityBenchSinkCompatibility  WakeCompatibilityReport
	identityBenchSinkBundle         StateBundle
	identityBenchSinkModelIdentity  ModelIdentity
	identityBenchSinkAdapterIdent   AdapterIdentity
	identityBenchSinkTokenizerIdent TokenizerIdentity
	identityBenchSinkRuntimeIdent   RuntimeIdentity
)

// benchGenerateConfigMinimal — the floor (just MaxTokens set).
func benchGenerateConfigMinimal() GenerateConfig {
	return GenerateConfig{
		MaxTokens: 128,
	}
}

// benchGenerateConfigTypical — knob-set seen in real chat requests.
func benchGenerateConfigTypical() GenerateConfig {
	return GenerateConfig{
		MaxTokens:     256,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		StopTokens:    []int32{2},
		RepeatPenalty: 1.1,
	}
}

// benchGenerateConfigHeavy — large stop-set, logits on (classification path).
func benchGenerateConfigHeavy() GenerateConfig {
	return GenerateConfig{
		MaxTokens:     2048,
		Temperature:   0.8,
		TopK:          50,
		TopP:          0.95,
		StopTokens:    []int32{0, 1, 2, 3, 4, 5, 6, 7},
		RepeatPenalty: 1.15,
		ReturnLogits:  true,
	}
}

// benchSamplerConfigTypical — sampler-side shape, sized like the
// generate-config above but in its serialisable form.
func benchSamplerConfigTypical() SamplerConfig {
	return SamplerConfig{
		MaxTokens:     256,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		RepeatPenalty: 1.1,
		StopTokens:    []int32{2},
	}
}

func benchSamplerConfigHeavy() SamplerConfig {
	return SamplerConfig{
		MaxTokens:     2048,
		Temperature:   0.8,
		TopK:          50,
		TopP:          0.95,
		RepeatPenalty: 1.15,
		StopTokens:    []int32{0, 1, 2, 3, 4, 5, 6, 7},
		StopSequences: []string{"</s>", "[END]"},
		ReturnLogits:  true,
	}
}

// benchStateBundleTypical — what a session checkpoint actually carries
// — model + tokenizer + adapter + sampler + a few KV refs.
func benchStateBundleTypical() StateBundle {
	return StateBundle{
		Version: "1",
		Model: ModelIdentity{
			Architecture:  "qwen3",
			Hash:          "sha256:model-a",
			QuantBits:     4,
			ContextLength: 32768,
			NumLayers:     28,
			HiddenSize:    2048,
			VocabSize:     151936,
		},
		Tokenizer: TokenizerIdentity{
			Kind:  "sentencepiece",
			Hash:  "sha256:tok-a",
			EOSID: 2,
			BOSID: 1,
		},
		Adapter: AdapterIdentity{
			Hash:       "sha256:adapter-a",
			Format:     "lora",
			Rank:       16,
			Alpha:      32,
			TargetKeys: []string{"q_proj", "v_proj"},
		},
		Sampler: benchSamplerConfigTypical(),
		Runtime: RuntimeIdentity{
			Backend:       "metal",
			Device:        "M3 Ultra",
			NativeRuntime: true,
		},
		PromptTokens:    256,
		GeneratedTokens: 128,
		KVRefs: []StateRef{
			{Kind: "kv", URI: "state://lthn/snap/0", SizeBytes: 1 << 24, Encoding: "paged-q8"},
			{Kind: "kv", URI: "state://lthn/snap/1", SizeBytes: 1 << 24, Encoding: "paged-q8"},
		},
	}
}

// --- SamplerConfigFromGenerateConfig (per-request capture) ---

func BenchmarkIdentity_SamplerConfigFromGenerateConfig_Minimal(b *testing.B) {
	cfg := benchGenerateConfigMinimal()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkSampler = SamplerConfigFromGenerateConfig(cfg)
	}
}

func BenchmarkIdentity_SamplerConfigFromGenerateConfig_Typical(b *testing.B) {
	cfg := benchGenerateConfigTypical()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkSampler = SamplerConfigFromGenerateConfig(cfg)
	}
}

func BenchmarkIdentity_SamplerConfigFromGenerateConfig_Heavy(b *testing.B) {
	cfg := benchGenerateConfigHeavy()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkSampler = SamplerConfigFromGenerateConfig(cfg)
	}
}

// Empty config → empty sampler — no slice clone cost.
func BenchmarkIdentity_SamplerConfigFromGenerateConfig_Empty(b *testing.B) {
	cfg := GenerateConfig{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkSampler = SamplerConfigFromGenerateConfig(cfg)
	}
}

// --- GenerateConfigFromSamplerConfig (per-session resume) ---

func BenchmarkIdentity_GenerateConfigFromSamplerConfig_Typical(b *testing.B) {
	sampler := benchSamplerConfigTypical()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkGenerateCfg = GenerateConfigFromSamplerConfig(sampler)
	}
}

func BenchmarkIdentity_GenerateConfigFromSamplerConfig_Heavy(b *testing.B) {
	sampler := benchSamplerConfigHeavy()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkGenerateCfg = GenerateConfigFromSamplerConfig(sampler)
	}
}

func BenchmarkIdentity_GenerateConfigFromSamplerConfig_Empty(b *testing.B) {
	sampler := SamplerConfig{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkGenerateCfg = GenerateConfigFromSamplerConfig(sampler)
	}
}

// --- Identity construction (per-LoadModel / per-checkpoint cost) ---

func BenchmarkIdentity_ModelIdentity_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkModelIdentity = ModelIdentity{
			Architecture:  "qwen3",
			Hash:          "sha256:model-a",
			QuantBits:     4,
			ContextLength: 32768,
			NumLayers:     28,
			HiddenSize:    2048,
			VocabSize:     151936,
		}
	}
}

func BenchmarkIdentity_TokenizerIdentity_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkTokenizerIdent = TokenizerIdentity{
			Kind:  "sentencepiece",
			Hash:  "sha256:tok-a",
			EOSID: 2,
			BOSID: 1,
		}
	}
}

func BenchmarkIdentity_AdapterIdentity_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkAdapterIdent = AdapterIdentity{
			Hash:       "sha256:adapter-a",
			Format:     "lora",
			Rank:       16,
			Alpha:      32,
			TargetKeys: []string{"q_proj", "v_proj"},
		}
	}
}

func BenchmarkIdentity_RuntimeIdentity_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkRuntimeIdent = RuntimeIdentity{
			Backend:       "metal",
			Device:        "M3 Ultra",
			NativeRuntime: true,
		}
	}
}

// --- StateBundle construction (per-checkpoint cost) ---

func BenchmarkIdentity_StateBundle_ConstructTypical(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkBundle = benchStateBundleTypical()
	}
}

// --- ProjectSeed (per session-bootstrap cost) ---

func BenchmarkIdentity_NewProjectSeed_Defaults(b *testing.B) {
	opts := ProjectSeedOptions{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkSeed = NewProjectSeed(opts)
	}
}

func BenchmarkIdentity_NewProjectSeed_BaseAndProject(b *testing.B) {
	opts := ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkSeed = NewProjectSeed(opts)
	}
}

func BenchmarkIdentity_NewProjectSeed_Full(b *testing.B) {
	opts := ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
		EntryURI:  "state://lthn/projects/core/go-mlx/seed",
		BundleURI: "state://lthn/projects/core/go-mlx/seed/bundle",
		IndexURI:  "state://lthn/projects/core/go-mlx/seed/index",
		Title:     "core/go-mlx project seed",
		Labels:    map[string]string{"project_id": "core/go-mlx", "env": "dev"},
		Metadata:  map[string]string{"created_by": "cladius"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkSeed = NewProjectSeed(opts)
	}
}

// --- ProjectSeed.WakeRequest (per wake) ---

func BenchmarkIdentity_ProjectSeed_WakeRequest_Minimal(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})
	opts := ProjectSeedWakeOptions{
		Model:     ModelIdentity{Hash: "sha256:model-a"},
		Tokenizer: TokenizerIdentity{Hash: "sha256:tok-a"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkWakeRequest = seed.WakeRequest(opts)
	}
}

func BenchmarkIdentity_ProjectSeed_WakeRequest_Typical(b *testing.B) {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
		Labels:    map[string]string{"env": "dev"},
	})
	opts := ProjectSeedWakeOptions{
		Model: ModelIdentity{
			Architecture: "qwen3",
			Hash:         "sha256:model-a",
			NumLayers:    28,
		},
		Tokenizer: TokenizerIdentity{
			Kind: "sentencepiece",
			Hash: "sha256:tok-a",
		},
		Adapter: AdapterIdentity{Hash: "sha256:adapter-a", Format: "lora"},
		Runtime: RuntimeIdentity{Backend: "metal"},
		Labels:  map[string]string{"session": "s-7"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkWakeRequest = seed.WakeRequest(opts)
	}
}

// --- CheckWakeCompatibility (per-wake validation) ---
// Iterates over model/tokenizer/adapter/runtime identity fields —
// pays the field-compare cost every wake.

func BenchmarkIdentity_CheckWakeCompatibility_Skip(b *testing.B) {
	bundle := benchStateBundleTypical()
	req := AgentMemoryWakeRequest{SkipCompatibilityCheck: true}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkCompatibility = CheckWakeCompatibility(bundle, req)
	}
}

func BenchmarkIdentity_CheckWakeCompatibility_Match(b *testing.B) {
	bundle := benchStateBundleTypical()
	req := AgentMemoryWakeRequest{
		Model:     bundle.Model,
		Tokenizer: bundle.Tokenizer,
		Adapter:   bundle.Adapter,
		Runtime:   bundle.Runtime,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkCompatibility = CheckWakeCompatibility(bundle, req)
	}
}

func BenchmarkIdentity_CheckWakeCompatibility_HashMismatch(b *testing.B) {
	bundle := benchStateBundleTypical()
	req := AgentMemoryWakeRequest{
		Model:     ModelIdentity{Hash: "sha256:other-model", Architecture: "gemma3", NumLayers: 12},
		Tokenizer: TokenizerIdentity{Hash: "sha256:other-tok"},
		Adapter:   AdapterIdentity{Hash: "sha256:other-adapter"},
		Runtime:   RuntimeIdentity{Backend: "rocm"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkCompatibility = CheckWakeCompatibility(bundle, req)
	}
}

func BenchmarkIdentity_CheckWakeCompatibility_Empty(b *testing.B) {
	bundle := StateBundle{}
	req := AgentMemoryWakeRequest{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identityBenchSinkCompatibility = CheckWakeCompatibility(bundle, req)
	}
}
