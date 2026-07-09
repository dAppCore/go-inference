// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import "dappco.re/go/inference/model"

// derive.go bridges the literal config copy (Gemma4TextConfig — the faithful parse with the
// wrapper-merge, validation, rope-merge and don't-guess discipline) to the neutral model.Arch the
// backends consume. The arch DERIVATION (rope folding, per-layer head-dim/KV geometry, DeriveLayers,
// MoE) already lives in Config.Arch() and is exercised by the existing tests, so Gemma4TextConfig
// reuses it via toConfig rather than re-deriving — the head_dim/layer_types guesses in Config.Arch()
// never fire here because the loader has resolved head_dim from the weight shapes (infer.go) and the
// copied parse REQUIRES layer_types. The metal-vs-neutral parity test proves the result matches metal.

// toConfig copies the resolved Gemma4TextConfig into the neutral Config (int32 → int, *int32 → int with
// nil = absent, RopeParams → RopeParam) — the arch-relevant subset Config.Arch() derives from.
func (c *Gemma4TextConfig) toConfig() Config {
	cfg := Config{
		HiddenSize:              int(c.HiddenSize),
		NumHiddenLayers:         int(c.NumHiddenLayers),
		IntermediateSize:        int(c.IntermediateSize),
		NumAttentionHeads:       int(c.NumAttentionHeads),
		NumKeyValueHeads:        int(c.NumKeyValueHeads),
		HeadDim:                 int(c.HeadDim),
		GlobalHeadDim:           int(c.GlobalHeadDim),
		VocabSize:               int(c.VocabSize),
		RMSNormEps:              c.RMSNormEps,
		FinalLogitSoftcapping:   c.FinalLogitSoftcapping,
		SlidingWindow:           int(c.SlidingWindow),
		NumKVSharedLayers:       int(c.NumKVSharedLayers),
		LayerTypes:              c.LayerTypes,
		AttentionKEqV:           c.AttentionKEqV,
		VocabSizePerLayerInput:  int(c.VocabSizePerLayerInput),
		HiddenSizePerLayerInput: int(c.HiddenSizePerLayerInput),
		EnableMoEBlock:          c.EnableMoEBlock,
		Quantization:            c.Quantization,
	}
	if c.NumGlobalKeyValueHeads != nil {
		cfg.NumGlobalKeyValueHeads = int(*c.NumGlobalKeyValueHeads)
	}
	if c.NumExperts != nil {
		cfg.NumExperts = int(*c.NumExperts)
	}
	if c.TopKExperts != nil {
		cfg.TopKExperts = int(*c.TopKExperts)
	}
	if c.MoEIntermediateSize != nil {
		cfg.MoEIntermediateSize = int(*c.MoEIntermediateSize)
	}
	if len(c.RopeParameters) > 0 {
		cfg.RopeParameters = make(map[string]RopeParam, len(c.RopeParameters))
		for k, rp := range c.RopeParameters {
			cfg.RopeParameters[k] = RopeParam{
				RopeTheta:           float32(rp.RopeTheta),
				PartialRotaryFactor: rp.PartialRotaryFactor,
				RopeType:            rp.RopeType,
				Factor:              rp.Factor,
			}
		}
	}
	return cfg
}

// Arch builds the backend-agnostic model.Arch from the faithfully-parsed, weight-resolved config.
func (c *Gemma4TextConfig) Arch() (model.Arch, error) { return c.toConfig().Arch() }

// ParseConfig is the exported entry to the literal-copied parser — for the metal-vs-neutral parity
// test that proves this copy (+ its cgo adaptation) stays identical to metal's parseGemma4Config until
// pkg/metal is deleted. Internally Load uses parseGemma4Config directly.
func ParseConfig(data []byte) (*Gemma4TextConfig, error) { return parseGemma4Config(data) }
