// SPDX-Licence-Identifier: EUPL-1.2

package dflash_test

import (
	core "dappco.re/go"
	dflash "dappco.re/go/inference/model/arch/z-lab/dflash"
)

// ExampleParseConfig recognises a z-lab-convention DFlash drafter from its
// config.json bytes and reads both halves of the contract: the drafter-facing
// block parameters and the drafter's own qwen3-style decoder geometry.
func ExampleParseConfig() {
	cfg, ok := dflash.ParseConfig([]byte(`{
		"architectures": ["DFlashDraftModel"],
		"block_size": 16,
		"dflash_config": {"mask_token_id": 151669, "target_layer_ids": [1, 9, 17, 25, 33]},
		"head_dim": 128,
		"hidden_size": 2560,
		"intermediate_size": 9728,
		"model_type": "qwen3",
		"num_attention_heads": 32,
		"num_hidden_layers": 5,
		"num_key_value_heads": 8
	}`))
	core.Println(ok)
	core.Println(cfg.Block.BlockSize, cfg.NumAux(), cfg.Hidden, cfg.NumLayers)
	// Output:
	// true
	// 16 5 2560 5
}
