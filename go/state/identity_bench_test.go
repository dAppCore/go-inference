// SPDX-Licence-Identifier: EUPL-1.2

// Benchmarks for the backend-neutral identity primitives.
// Per AX-11 — ModelIdentity / TokenizerIdentity / AdapterIdentity /
// RuntimeIdentity travel inside every WakeRequest, SleepRequest, and
// Bundle. Bundle itself is the durable envelope written on every
// Sleep and re-read on every Wake. The struct fields are flat but
// the slices (KVRefs, ProbeRefs, StateRefs) carry the per-bundle
// allocation cost.
//
// Run:    go test -bench='Benchmark' -benchmem -run='^$' ./state

package state

import "testing"

// Sinks defeat compiler DCE. Distinct names per state-package bench file.
var (
	identitySinkModel     ModelIdentity
	identitySinkTokenizer TokenizerIdentity
	identitySinkAdapter   AdapterIdentity
	identitySinkRuntime   RuntimeIdentity
	identitySinkSampler   SamplerConfig
	identitySinkBundle    Bundle
	identitySinkStateRef  StateRef
)

// --- ModelIdentity (per-bundle, per-wake, per-sleep) ---

func BenchmarkIdentity_Model_Construct_Minimal(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkModel = ModelIdentity{
			ID:           "gemma4",
			Architecture: "gemma4_text",
			Hash:         "model-a",
			NumLayers:    28,
		}
	}
}

func BenchmarkIdentity_Model_Construct_Full(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkModel = ModelIdentity{
			ID:            "gemma4",
			Path:          "/Users/snider/Lethean/models/gemma4-27b",
			Architecture:  "gemma4_text",
			Revision:      "main",
			Hash:          "sha256:abcdefabcdef",
			QuantBits:     4,
			QuantGroup:    64,
			QuantType:     "jangtq",
			ContextLength: 262144,
			NumLayers:     28,
			HiddenSize:    4096,
			VocabSize:     262144,
		}
	}
}

func BenchmarkIdentity_Model_Construct_FullWithLabels(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkModel = ModelIdentity{
			ID:            "gemma4",
			Path:          "/Users/snider/Lethean/models/gemma4-27b",
			Architecture:  "gemma4_text",
			Hash:          "sha256:abcdefabcdef",
			QuantBits:     4,
			QuantGroup:    64,
			QuantType:     "jangtq",
			ContextLength: 262144,
			NumLayers:     28,
			HiddenSize:    4096,
			VocabSize:     262144,
			Labels: map[string]string{
				"vendor":   "google",
				"family":   "gemma",
				"size":     "27b",
				"variant":  "text",
				"licence":  "gemma-tos",
				"upstream": "huggingface",
			},
		}
	}
}

// --- TokenizerIdentity (per-bundle) ---

func BenchmarkIdentity_Tokenizer_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkTokenizer = TokenizerIdentity{
			Kind:         "sentencepiece",
			Path:         "/Users/snider/Lethean/models/gemma4-27b/tokenizer.model",
			Hash:         "sha256:tok-abc",
			ChatTemplate: "gemma-it",
			BOSID:        2,
			EOSID:        1,
			PADID:        0,
		}
	}
}

// --- AdapterIdentity (per-bundle, per-wake compatibility check) ---

func BenchmarkIdentity_Adapter_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkAdapter = AdapterIdentity{
			Path:          "/Users/snider/Lethean/adapters/cladius.lora",
			Hash:          "sha256:adapter-abc",
			Format:        "lora",
			Rank:          8,
			Alpha:         16,
			BaseModelHash: "sha256:abcdefabcdef",
		}
	}
}

func BenchmarkIdentity_Adapter_Construct_WithTargetKeys(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkAdapter = AdapterIdentity{
			Path:   "/Users/snider/Lethean/adapters/cladius.lora",
			Hash:   "sha256:adapter-abc",
			Format: "lora",
			Rank:   8,
			Alpha:  16,
			TargetKeys: []string{
				"q_proj", "k_proj", "v_proj", "o_proj",
				"gate_proj", "up_proj", "down_proj",
			},
			BaseModelHash: "sha256:abcdefabcdef",
		}
	}
}

// --- RuntimeIdentity (per-bundle) ---

func BenchmarkIdentity_Runtime_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkRuntime = RuntimeIdentity{
			Backend:       "metal",
			Device:        "Apple M3 Ultra",
			Version:       "26.0.0",
			CacheMode:     "paged-q8",
			NativeRuntime: true,
		}
	}
}

// --- SamplerConfig (per-generation step, per-bundle) ---

func BenchmarkIdentity_Sampler_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkSampler = SamplerConfig{
			MaxTokens:     4096,
			Temperature:   0.7,
			TopK:          40,
			TopP:          0.9,
			RepeatPenalty: 1.1,
			StopTokens:    []int32{1, 2, 0},
			StopSequences: []string{"</s>", "<|end|>"},
			ReturnLogits:  false,
		}
	}
}

// --- StateRef (per-block during bundle assembly) ---

func BenchmarkIdentity_StateRef_Construct(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkStateRef = StateRef{
			Kind:      "kv",
			URI:       "state://kv/blocks/0",
			Hash:      "sha256:block-abc",
			SizeBytes: 65536,
			Encoding:  "raw",
		}
	}
}

// --- Bundle (durable envelope — every Sleep writes one) ---

func BenchmarkIdentity_Bundle_Construct_Minimal(b *testing.B) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkBundle = Bundle{
			Version:       "v1",
			CreatedAtUnix: 1700000000,
			Model:         ModelIdentity{ID: "gemma4", Hash: "model-a"},
			PromptTokens:  2048,
		}
	}
}

func BenchmarkIdentity_Bundle_Construct_KVRefs_10(b *testing.B) {
	model := ModelIdentity{ID: "gemma4", Hash: "model-a", NumLayers: 28}
	tok := TokenizerIdentity{Hash: "tok-a"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kv := make([]StateRef, 0, 10)
		for j := 0; j < 10; j++ {
			kv = append(kv, StateRef{Kind: "kv", URI: "state://kv/blocks", SizeBytes: 65536})
		}
		identitySinkBundle = Bundle{
			Version:       "v1",
			CreatedAtUnix: 1700000000,
			Model:         model,
			Tokenizer:     tok,
			KVRefs:        kv,
			PromptTokens:  2048,
		}
	}
}

func BenchmarkIdentity_Bundle_Construct_KVRefs_100(b *testing.B) {
	model := ModelIdentity{ID: "gemma4", Hash: "model-a", NumLayers: 28}
	tok := TokenizerIdentity{Hash: "tok-a"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kv := make([]StateRef, 0, 100)
		for j := 0; j < 100; j++ {
			kv = append(kv, StateRef{Kind: "kv", URI: "state://kv/blocks", SizeBytes: 65536})
		}
		identitySinkBundle = Bundle{
			Version:       "v1",
			CreatedAtUnix: 1700000000,
			Model:         model,
			Tokenizer:     tok,
			KVRefs:        kv,
			PromptTokens:  2048,
		}
	}
}

func BenchmarkIdentity_Bundle_Construct_KVRefs_1000(b *testing.B) {
	model := ModelIdentity{ID: "gemma4", Hash: "model-a", NumLayers: 28}
	tok := TokenizerIdentity{Hash: "tok-a"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		kv := make([]StateRef, 0, 1000)
		for j := 0; j < 1000; j++ {
			kv = append(kv, StateRef{Kind: "kv", URI: "state://kv/blocks", SizeBytes: 65536})
		}
		identitySinkBundle = Bundle{
			Version:       "v1",
			CreatedAtUnix: 1700000000,
			Model:         model,
			Tokenizer:     tok,
			KVRefs:        kv,
			PromptTokens:  2048,
		}
	}
}

// --- Bundle copy (pure struct shape, no slice alloc) ---
// The Bundle struct copy fires on every WakeResult / SleepResult
// return; the slice headers are shared so this measures just the
// scalar+header cost.

func BenchmarkIdentity_Bundle_Copy(b *testing.B) {
	src := Bundle{
		Version:       "v1",
		CreatedAtUnix: 1700000000,
		Model:         ModelIdentity{ID: "gemma4", Hash: "model-a", NumLayers: 28},
		Tokenizer:     TokenizerIdentity{Hash: "tok-a"},
		Adapter:       AdapterIdentity{Hash: "adapter-a", Rank: 8},
		Runtime:       RuntimeIdentity{Backend: "metal"},
		PromptTokens:  2048,
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkBundle = src
	}
}

// StateBundle is the long-form type alias for Bundle — confirm zero overhead.

func BenchmarkIdentity_StateBundle_AliasCopy(b *testing.B) {
	src := StateBundle{
		Version: "v1",
		Model:   ModelIdentity{ID: "gemma4"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		identitySinkBundle = src
	}
}
