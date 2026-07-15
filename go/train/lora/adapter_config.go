// SPDX-Licence-Identifier: EUPL-1.2

// Adapter config: the adapter_config.json metadata surface (rank/alpha/scale/
// target keys) plus the alias normalisation PEFT, mlx-lm, and go-mlx saved
// adapters use, so any engine's loader reads one canonical shape regardless
// of which tool produced the adapter on disk.

package lora

import core "dappco.re/go"

// AdapterConfig is the shared adapter_config.json metadata surface read by
// adapter inspection (see Inspect) and by an engine's own adapter loader.
type AdapterConfig struct {
	Rank          int      `json:"rank"`
	R             int      `json:"r"`
	Alpha         float32  `json:"alpha"`
	LoRAAlpha     float32  `json:"lora_alpha"`
	Scale         float32  `json:"scale"`
	NumLayers     int      `json:"num_layers"`
	TargetKeys    []string `json:"target_keys"`
	TargetModules []string `json:"target_modules"`
	LoRALayers    []string `json:"lora_layers"`
}

// ParseAdapterConfig parses adapter_config.json bytes and applies lossless
// aliases (see NormalizeAdapterConfig). It does not fabricate required
// metadata such as rank; public inspection and fusion validation need to know
// when an adapter omitted those fields.
//
//	cfg, err := lora.ParseAdapterConfig(data)
func ParseAdapterConfig(data []byte) (AdapterConfig, error) {
	var cfg AdapterConfig
	if result := core.JSONUnmarshal(data, &cfg); !result.OK {
		return AdapterConfig{}, core.E("lora.ParseAdapterConfig", "parse adapter_config.json", nil)
	}
	return NormalizeAdapterConfig(cfg), nil
}

// NormalizeAdapterConfig applies the adapter metadata aliases used by PEFT,
// mlx-lm, and go-mlx saved adapters without inventing missing required
// metadata.
//
//	cfg := lora.NormalizeAdapterConfig(lora.AdapterConfig{R: 4, LoRAAlpha: 12})
func NormalizeAdapterConfig(cfg AdapterConfig) AdapterConfig {
	if cfg.Rank <= 0 && cfg.R > 0 {
		cfg.Rank = cfg.R
	}
	if cfg.Alpha == 0 {
		switch {
		case cfg.LoRAAlpha != 0:
			cfg.Alpha = cfg.LoRAAlpha
		case cfg.Scale != 0 && cfg.Rank > 0:
			cfg.Alpha = cfg.Scale * float32(cfg.Rank)
		}
	}
	if cfg.Scale == 0 && cfg.Rank > 0 && cfg.Alpha != 0 {
		cfg.Scale = cfg.Alpha / float32(cfg.Rank)
	}
	if len(cfg.TargetKeys) == 0 {
		switch {
		case len(cfg.TargetModules) > 0:
			cfg.TargetKeys = cfg.TargetModules
		case len(cfg.LoRALayers) > 0:
			cfg.TargetKeys = cfg.LoRALayers
		}
	}
	return cfg
}

// NormalizeAdapterConfigForLoad applies the default adapter values an
// engine-side loader can accept when adapter_config.json omits them: a
// missing rank defaults to 8, and alpha derives from scale (or 2x rank) when
// absent. Keep this separate from ParseAdapterConfig so public metadata
// validation can still reject incomplete adapter_config.json files.
//
//	cfg := lora.NormalizeAdapterConfigForLoad(lora.AdapterConfig{})
func NormalizeAdapterConfigForLoad(cfg AdapterConfig) AdapterConfig {
	cfg = NormalizeAdapterConfig(cfg)
	if cfg.Rank <= 0 {
		cfg.Rank = 8
	}
	if cfg.Alpha == 0 {
		switch {
		case cfg.Scale != 0:
			cfg.Alpha = cfg.Scale * float32(cfg.Rank)
		default:
			cfg.Alpha = float32(cfg.Rank) * 2
		}
	}
	if cfg.Scale == 0 && cfg.Rank > 0 && cfg.Alpha != 0 {
		cfg.Scale = cfg.Alpha / float32(cfg.Rank)
	}
	return cfg
}
