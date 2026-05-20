// SPDX-Licence-Identifier: EUPL-1.2

package inference

import core "dappco.re/go"

func ExampleStateBundle() {
	bundle := StateBundle{
		Model: ModelIdentity{
			Architecture: "gemma4",
			QuantBits:    4,
		},
		Runtime: RuntimeIdentity{
			Backend:       "metal",
			NativeRuntime: true,
		},
	}

	core.Println(bundle.Model.Architecture, bundle.Runtime.Backend)
	// Output: gemma4 metal
}

func ExampleSamplerConfigFromGenerateConfig() {
	sampler := SamplerConfigFromGenerateConfig(GenerateConfig{
		MaxTokens:  32,
		TopK:       8,
		StopTokens: []int32{2},
	})

	core.Println(sampler.MaxTokens, sampler.TopK, sampler.StopTokens)
	// Output: 32 8 [2]
}

func ExampleGenerateConfigFromSamplerConfig() {
	cfg := GenerateConfigFromSamplerConfig(SamplerConfig{
		MaxTokens:     64,
		Temperature:   0.2,
		RepeatPenalty: 1.1,
	})

	core.Println(cfg.MaxTokens, cfg.Temperature, cfg.RepeatPenalty)
	// Output: 64 0.2 1.1
}
