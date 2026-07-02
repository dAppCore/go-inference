// SPDX-Licence-Identifier: EUPL-1.2

package inference

import "testing"

func TestIdentity_SamplerConfigFromGenerateConfig_Good(t *testing.T) {
	cfg := GenerateConfig{
		MaxTokens:     64,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.9,
		MinP:          0.05,
		StopTokens:    []int32{1, 2},
		RepeatPenalty: 1.1,
		ReturnLogits:  true,
	}
	sampler := SamplerConfigFromGenerateConfig(cfg)
	cfg.StopTokens[0] = 99

	checkEqual(t, []int32{1, 2}, sampler.StopTokens)
	checkEqual(t, 64, sampler.MaxTokens)
	checkEqual(t, float32(0.7), sampler.Temperature)
	checkEqual(t, 40, sampler.TopK)
	checkEqual(t, float32(0.9), sampler.TopP)
	checkEqual(t, float32(0.05), sampler.MinP)
	checkEqual(t, float32(1.1), sampler.RepeatPenalty)
	checkTrue(t, sampler.ReturnLogits)
}

func TestIdentity_SamplerConfigFromGenerateConfig_Bad(t *testing.T) {
	sampler := SamplerConfigFromGenerateConfig(GenerateConfig{})

	checkEqual(t, 0, sampler.MaxTokens)
	checkEmpty(t, sampler.StopTokens)
	checkFalse(t, sampler.ReturnLogits)
}

func TestIdentity_SamplerConfigFromGenerateConfig_Ugly(t *testing.T) {
	cfg := GenerateConfig{StopTokens: []int32{}}

	sampler := SamplerConfigFromGenerateConfig(cfg)
	cfg.StopTokens = append(cfg.StopTokens, 7)

	checkEmpty(t, sampler.StopTokens)
	checkEqual(t, []int32{7}, cfg.StopTokens)
}

func TestIdentity_GenerateConfigFromSamplerConfig_Good(t *testing.T) {
	sampler := SamplerConfig{
		MaxTokens:     128,
		Temperature:   0.2,
		TopK:          8,
		TopP:          0.5,
		MinP:          0.03,
		StopTokens:    []int32{3, 4},
		RepeatPenalty: 1.2,
		ReturnLogits:  true,
	}
	cfg := GenerateConfigFromSamplerConfig(sampler)
	sampler.StopTokens[0] = 99

	checkEqual(t, []int32{3, 4}, cfg.StopTokens)
	checkEqual(t, 128, cfg.MaxTokens)
	checkEqual(t, float32(0.2), cfg.Temperature)
	checkEqual(t, 8, cfg.TopK)
	checkEqual(t, float32(0.5), cfg.TopP)
	checkEqual(t, float32(0.03), cfg.MinP)
	checkEqual(t, float32(1.2), cfg.RepeatPenalty)
	checkTrue(t, cfg.ReturnLogits)
}

func TestIdentity_GenerateConfigFromSamplerConfig_Bad(t *testing.T) {
	cfg := GenerateConfigFromSamplerConfig(SamplerConfig{})

	checkEqual(t, 0, cfg.MaxTokens)
	checkEmpty(t, cfg.StopTokens)
	checkFalse(t, cfg.ReturnLogits)
}

func TestIdentity_GenerateConfigFromSamplerConfig_Ugly(t *testing.T) {
	sampler := SamplerConfig{StopTokens: []int32{}}

	cfg := GenerateConfigFromSamplerConfig(sampler)
	sampler.StopTokens = append(sampler.StopTokens, 7)

	checkEmpty(t, cfg.StopTokens)
	checkEqual(t, []int32{7}, sampler.StopTokens)
}

func TestIdentity_StateBundle_Good(t *testing.T) {
	bundle := StateBundle{
		Version: "1",
		Model: ModelIdentity{
			Architecture:  "qwen3",
			QuantBits:     4,
			ContextLength: 32768,
		},
		Tokenizer: TokenizerIdentity{
			Kind:  "sentencepiece",
			EOSID: 2,
		},
		Adapter: AdapterIdentity{
			Rank:       16,
			Alpha:      32,
			TargetKeys: []string{"q_proj", "v_proj"},
		},
		Runtime: RuntimeIdentity{
			Backend:       "metal",
			NativeRuntime: true,
		},
		Sampler: SamplerConfig{
			MaxTokens: 256,
		},
		KVRefs: []StateRef{{
			Kind: "kv",
			URI:  "file:///tmp/state.kvbin",
		}},
	}

	checkEqual(t, "qwen3", bundle.Model.Architecture)
	checkEqual(t, int32(2), bundle.Tokenizer.EOSID)
	checkEqual(t, 16, bundle.Adapter.Rank)
	checkTrue(t, bundle.Runtime.NativeRuntime)
	checkLen(t, bundle.KVRefs, 1)
}

func TestIdentity_StateBundle_Bad_EmptyAllowed(t *testing.T) {
	bundle := StateBundle{}

	checkEqual(t, "", bundle.Model.Architecture)
	checkEqual(t, 0, bundle.Sampler.MaxTokens)
	checkEmpty(t, bundle.KVRefs)
}

func TestIdentity_ProjectSeedAliases_Good(t *testing.T) {
	seed := NewProjectSeed(ProjectSeedOptions{BaseURI: "state://lthn/projects", ProjectID: "core/go-mlx"})
	wake := seed.WakeRequest(ProjectSeedWakeOptions{
		Model:     ModelIdentity{Hash: "model-a"},
		Tokenizer: TokenizerIdentity{Hash: "tok-a"},
	})

	report := CheckWakeCompatibility(StateBundle{
		Model:        ModelIdentity{Hash: "model-a"},
		Tokenizer:    TokenizerIdentity{Hash: "tok-a"},
		PromptTokens: 16,
	}, wake)

	checkEqual(t, "state://lthn/projects/core/go-mlx/seed", wake.EntryURI)
	checkTrue(t, report.Compatible)
}

func TestIdentity_AdapterIdentity_Ugly_MetadataOnly(t *testing.T) {
	adapter := AdapterIdentity{
		Hash:          "sha256:abc",
		Format:        "lora",
		BaseModelHash: "sha256:base",
		Labels:        map[string]string{"source": "unit"},
	}

	checkEqual(t, "sha256:abc", adapter.Hash)
	checkEqual(t, "unit", adapter.Labels["source"])
	checkEmpty(t, adapter.TargetKeys)
}
