// SPDX-Licence-Identifier: EUPL-1.2

// Runnable examples for adapter_config.go — kept separate from
// adapter_config_test.go so the godoc-attached usage snippets stay readable.

package lora

import core "dappco.re/go"

// ExampleParseAdapterConfig shows a PEFT-style adapter_config.json (r /
// lora_alpha / target_modules) read into the canonical AdapterConfig shape:
// ParseAdapterConfig resolves the aliases and derives Scale from Rank+Alpha.
func ExampleParseAdapterConfig() {
	cfg, err := ParseAdapterConfig([]byte(`{"r":8,"lora_alpha":16,"target_modules":["q_proj","v_proj"]}`))
	if err != nil {
		core.Println("error:", err)
		return
	}
	core.Println(cfg.Rank, cfg.Alpha, cfg.Scale)
	core.Println(cfg.TargetKeys)
	// Output:
	// 8 16 2
	// [q_proj v_proj]
}

// ExampleNormalizeAdapterConfigForLoad shows the engine-load defaulting
// applied to an adapter_config.json that omits rank and alpha entirely: the
// loader gets a usable rank/alpha/scale rather than failing on missing
// metadata.
func ExampleNormalizeAdapterConfigForLoad() {
	cfg := NormalizeAdapterConfigForLoad(AdapterConfig{})
	core.Println(cfg.Rank, cfg.Alpha, cfg.Scale)
	// Output:
	// 8 16 2
}
