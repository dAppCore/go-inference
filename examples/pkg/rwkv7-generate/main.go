// SPDX-Licence-Identifier: EUPL-1.2

// rwkv7-generate is the LIBRARY-LEVEL acceptance example for the RWKV-7 "Goose" host port (#36): it
// drives model/arch/rwkv7 directly rather than through inference.LoadModel / examples/internal/engine —
// read the checkpoint's safetensors + config.json, LoadRWKV7Model, tokenise with the RWKV World
// tokenizer, and greedy-generate through RWKV7Session, exactly the path real_checkpoint_test.go proves
// against the numpy oracle. The native engine's load hook for model_type rwkv7 IS wired (engine/metal's
// LoadModel routes it through the generic model.SessionModel serve arm — see examples/pkg/generate for
// the equivalent inference.LoadModel + TextModel.Generate path, which now works unchanged for this
// checkpoint); this example stays as the low-level demonstration that model/arch/rwkv7 works standalone,
// no engine required. Point it at RWKV/RWKV7-Goose-World2.8-0.1B-HF (or a 1.5B/2.9B fla-hub sibling —
// same tokenizer, same tensor-name conventions).
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
	vocab := flag.String("vocab", "", "RWKV World tokenizer vocab override: an on-disk hex-per-line fixture (see model/arch/rwkv7/tokenizer.go); empty (the default) uses the tokenizer's own embedded canonical vocab, no file needed")
	prompt := flag.String("prompt", "The capital of France is", "text to continue — no chat template applied")
	maxTokens := flag.Int("max-tokens", 40, "tokens to greedily generate")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to an RWKV-7 checkpoint snapshot directory")
		os.Exit(2)
	}

	var tok *rwkv7.WorldTokenizer
	var err error
	if *vocab == "" {
		tok, err = rwkv7.NewWorldTokenizer()
	} else {
		tok, err = rwkv7.LoadWorldTokenizerHex(*vocab)
	}
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
