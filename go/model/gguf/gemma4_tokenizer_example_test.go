// SPDX-Licence-Identifier: EUPL-1.2

package gguf

import "fmt"

// Example shows gemma4Tokenizer resolving the end-of-turn EOS id and classifying
// a byte-fallback token from a checkpoint's tokenizer.json.
func Example_gemma4Tokenizer() {
	entries, _ := gemma4Tokenizer([]byte(gemma4TestTokenizer))
	for _, e := range entries {
		switch e.Key {
		case "tokenizer.ggml.eos_token_id":
			fmt.Println("eos", e.Value)
		case "tokenizer.ggml.token_type":
			fmt.Println("byte-token type", e.Value.([]int32)[6])
		}
	}
	// Output:
	// byte-token type 6
	// eos 5
}
