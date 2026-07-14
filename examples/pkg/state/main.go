// SPDX-Licence-Identifier: EUPL-1.2

// Durable conversation state — the no-prompt-replay loop: `lem generate -state
// <name>` wakes a named slot from disk (KV restored in ~1-3ms), appends ONLY
// the new turn, generates, then sleeps the session back for the next call.
// This example calls the same library seam cmd/lem's -state flag delegates to
// (dappco.re/go/inference/decode/generate) twice against one state name,
// proving the second turn answers from the first turn's content without
// resending it.
//
//	go run ./pkg/state -model ~/models/gemma-4-e2b-it-4bit
package main

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"dappco.re/go/inference/decode/generate"

	_ "dappco.re/go/inference/examples/internal/engine" // registers the platform engine
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (config.json + *.safetensors)")
	store := flag.String("store", filepath.Join(os.TempDir(), "go-inference-example-state.kv"), "state store file (an example scratch file — never ~/Lethean)")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a model snapshot directory")
		os.Exit(2)
	}

	const name = "chat-example"
	turn(*model, *store, name, "My name is Marker.")
	turn(*model, *store, name, "What is my name?")
}

// turn runs one -state turn through generate.RunGenerate: it wakes the named
// slot if the store already holds one (no re-prefill of earlier turns),
// appends prompt as the new turn, generates, and sleeps the session back to
// store — the same wake/append/generate/sleep loop `lem generate -state` runs
// per invocation.
func turn(model, store, name, prompt string) {
	var out bytes.Buffer
	err := generate.RunGenerate(context.Background(), generate.Config{
		ModelPath:  model,
		Prompt:     prompt,
		MaxTokens:  64,
		Temp:       0,
		StateName:  name,
		StateStore: store,
		Out:        &out,
		Log:        os.Stderr,
	})
	if err != nil {
		fmt.Fprintln(os.Stderr, "generate:", err)
		os.Exit(1)
	}
	fmt.Printf("--- %q ---\n%s", prompt, out.String())
}
