// SPDX-Licence-Identifier: EUPL-1.2

// Tokenise text without loading a model or touching the GPU: tokenizer.json
// is a standalone artifact every model snapshot carries, so token counting
// and the encode/decode plumbing work with no engine import at all — like
// pkg/discover, this example never blank-imports examples/internal/engine.
//
//	go run ./pkg/tokenizer -model ~/models/gemma-4-e2b-it-4bit
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"

	"dappco.re/go/inference/decode/tokenizer"
)

func main() {
	model := flag.String("model", os.Getenv("LEM_MODEL"), "model snapshot directory (only tokenizer.json is read)")
	flag.Parse()
	if *model == "" {
		fmt.Fprintln(os.Stderr, "set -model (or LEM_MODEL) to a model snapshot directory")
		os.Exit(2)
	}

	tok, err := tokenizer.LoadTokenizer(filepath.Join(*model, "tokenizer.json"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "load:", err)
		os.Exit(1)
	}

	// Token budgeting: knowing the cost of a prompt against a context window
	// before you ever load the model or spend a generation call.
	paragraph := "A lighthouse keeper's work is equal parts vigilance and " +
		"routine: the lamp must turn, the log must be kept, and the weather " +
		"must be read before it reads you."
	ids := tok.Encode(paragraph)
	fmt.Printf("%d tokens for %d bytes of text\n", len(ids), len(paragraph))

	// Round trip: ids back to text should reproduce the input (modulo the
	// BOS token and any whitespace normalisation the tokeniser applies).
	fmt.Println("round trip:", tok.Decode(ids))

	// DecodeToken renders a single id — the per-token path a streaming
	// generation loop calls once for every emitted token.
	// ids[0] is usually BOS (decodes empty), so demo a content token.
	mid := ids[len(ids)/2]
	fmt.Printf("token %d decodes to %q\n", mid, tok.DecodeToken(mid))
}
