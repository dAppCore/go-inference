// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/decode/tokenizer"
)

// ExampleReadSentencePieceModel reads a checkpoint's tokenizer.model and prints
// the count and a representative entry — the id-ordered pieces feed a GGUF
// "llama" tokenizer header's token/score/type arrays.
func ExampleReadSentencePieceModel() {
	pieces, err := tokenizer.ReadSentencePieceModel("/models/gemma3-1b/tokenizer.model")
	if err != nil {
		return // a missing/corrupt tokenizer.model is a loud error, not a panic
	}
	core.Println(len(pieces), pieces[2].Piece, pieces[2].Type == tokenizer.SPMTokenControl)
}
