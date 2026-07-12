// SPDX-Licence-Identifier: EUPL-1.2

package dflash_test

import (
	"fmt"

	"dappco.re/go/inference/decode/dflash"
	"dappco.re/go/inference/decode/specctl"
)

// ExampleParseConfig recognises a real DFlash speculator checkpoint by its
// config.json marker and reads the drafter contract (block size, the fused
// verifier layers, and the target model it verifies against).
func ExampleParseConfig() {
	data := []byte(`{"speculators_model_type":"dflash","block_size":8,` +
		`"aux_hidden_state_layer_ids":[3,13,23,32,42],` +
		`"speculators_config":{"verifier":{"name":"deepseek-ai/DeepSeek-V4-Flash"}}}`)
	cfg, ok := dflash.ParseConfig(data)
	fmt.Println(ok, cfg.BlockSize, len(cfg.AuxHiddenLayerIDs), cfg.Verifier)
	// Output:
	// true 8 5 deepseek-ai/DeepSeek-V4-Flash
}

// ExampleGenerate runs the speculative loop against a tiny deterministic target
// (the next token is the running length mod 4, so the continuation cycles
// 0,1,2,3,…). The model-free lookup drafter predicts the repeats, yet the output
// is byte-identical to plain greedy decode — speculation is lossless.
func ExampleGenerate() {
	next := func(prefix []int) int { return len(prefix) % 4 }
	p := dflash.NewLookupProposer(dflash.Config{BlockSize: 4})

	out, _ := dflash.Generate(nil, 8, p, next)
	base := dflash.Autoregress(nil, 8, next)
	fmt.Println("out: ", out)
	fmt.Println("base:", base)
	fmt.Println("lossless:", eqInts(out, base))
	// Output:
	// out:  [0 1 2 3 0 1 2 3]
	// base: [0 1 2 3 0 1 2 3]
	// lossless: true
}

// Example_adaptiveBlock composes a DFlash run's Stats with the specctl controller:
// the accept-rate the run reports sizes the next block — speculate hard when
// drafts land, pull back when they miss. (Window 1 tracks the last round exactly,
// for a clear illustration.)
func Example_adaptiveBlock() {
	c := specctl.New(specctl.Controller{Min: 1, Max: 8, Window: 1})

	landed := dflash.Stats{ProposedTokens: 8, AcceptedTokens: 8} // every draft token kept
	c.Record(landed.ProposedTokens, landed.AcceptedTokens)
	fmt.Println("landed:", c.NextLength())

	missed := dflash.Stats{ProposedTokens: 8, AcceptedTokens: 0} // all rejected
	c.Record(missed.ProposedTokens, missed.AcceptedTokens)
	fmt.Println("missed:", c.NextLength())
	// Output:
	// landed: 8
	// missed: 1
}
