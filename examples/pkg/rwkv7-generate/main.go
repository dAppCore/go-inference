// SPDX-Licence-Identifier: EUPL-1.2

// rwkv7-generate is the LIBRARY-LEVEL acceptance example for the RWKV-7 "Goose" host port (#36): it does
// NOT go through inference.LoadModel / examples/internal/engine (rwkv7 has no load-hook wired into the
// native engine yet — that lands separately, at merge) — it drives model/arch/rwkv7 directly: read the
// checkpoint's safetensors + config.json, LoadRWKV7Model, tokenise with the RWKV World tokenizer, and
// greedy-generate through RWKV7Session, exactly the path real_checkpoint_test.go proves against the numpy
// oracle. Point it at RWKV/RWKV7-Goose-World2.8-0.1B-HF (or a 1.5B/2.9B fla-hub sibling — same tokenizer,
// same tensor-name conventions).
//
//	go run ./pkg/rwkv7-generate -model ~/.cache/huggingface/hub/models--RWKV--RWKV7-Goose-World2.8-0.1B-HF/snapshots/<rev>
package main

import (
	"flag"
	"fmt"
	"os"

	"dappco.re/go/inference/model/arch/rwkv7"
	"dappco.re/go/inference/model/safetensors"
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "RWKV-7 checkpoint snapshot directory (config.json + model.safetensors)")
	vocab := flag.String("vocab", defaultVocabPath(), "RWKV World tokenizer vocab (the derived .hex fixture — see model/arch/rwkv7/tokenizer.go)")
	prompt := flag.String("prompt", "The capital of France is", "text to continue — no chat template applied")
	maxTokens := flag.Int("max-tokens", 40, "tokens to greedily generate")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to an RWKV-7 checkpoint snapshot directory")
		os.Exit(2)
	}

	tok, err := rwkv7.LoadWorldTokenizerHex(*vocab)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load tokenizer:", err)
		os.Exit(1)
	}

	dm, err := safetensors.LoadDirMmap(*model)
	if err != nil {
		fmt.Fprintln(os.Stderr, "load safetensors:", err)
		os.Exit(1)
	}
	defer func() { _ = dm.Close() }()
	cfgBytes, err := os.ReadFile(*model + "/config.json")
	if err != nil {
		fmt.Fprintln(os.Stderr, "read config.json:", err)
		os.Exit(1)
	}
	m, err := rwkv7.LoadRWKV7Model(dm.Tensors, cfgBytes)
	if err != nil {
		fmt.Fprintln(os.Stderr, "LoadRWKV7Model:", err)
		os.Exit(1)
	}
	fmt.Fprintf(os.Stderr, "loaded: %d layers, hidden=%d, vocab=%d\n", len(m.Layers), m.D, m.Vocab)

	promptIDs := tok.Encode(*prompt)
	gen, err := rwkv7.NewSession(m).Generate(promptIDs, *maxTokens, -1)
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate:", err)
		os.Exit(1)
	}
	fmt.Println(*prompt + tok.Decode(gen))
}

// defaultVocabPath points at the RWKV World tokenizer fixture this repo ships (every RWKV-7 checkpoint
// shares the same vocab — see model/arch/rwkv7/tokenizer.go's package doc). Resolved relative to this
// example's own source tree so `go run ./pkg/rwkv7-generate` works from examples/ with no extra setup.
func defaultVocabPath() string {
	return "../go/model/arch/rwkv7/testdata/rwkv_vocab_v20230424.hex"
}
