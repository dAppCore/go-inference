// SPDX-Licence-Identifier: EUPL-1.2

package rwkv7

import (
	"os"
	"path/filepath"

	core "dappco.re/go"
)

// exampleVocabHexPath writes the same tiny fixture mkVocabHexFile(t) builds for the _test.go cases,
// without needing a *testing.T (Example functions take none).
func exampleVocabHexPath() string {
	dir, _ := os.MkdirTemp("", "rwkv7-vocab-example")
	path := filepath.Join(dir, "vocab.hex")
	// ids: 1="a" 2="b" 3="ab" 4=" " 5="ba" (hex of each token's UTF-8 bytes, one per line)
	_ = os.WriteFile(path, []byte("61\n62\n6162\n20\n6261\n"), 0o644)
	return path
}

func ExampleLoadWorldTokenizerHex() {
	tok, err := LoadWorldTokenizerHex(exampleVocabHexPath())
	core.Println(err == nil, len(tok.toBytes))
	// Output: true 5
}

func ExampleWorldTokenizer_Encode() {
	tok, _ := LoadWorldTokenizerHex(exampleVocabHexPath())
	core.Println(tok.Encode("ab"))
	// Output: [3]
}

func ExampleWorldTokenizer_Decode() {
	tok, _ := LoadWorldTokenizerHex(exampleVocabHexPath())
	core.Println(tok.Decode([]int32{1, 2}))
	// Output: ab
}
