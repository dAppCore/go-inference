// SPDX-Licence-Identifier: EUPL-1.2

package tokenizer

import core "dappco.re/go"

func ExampleLoadTokenizer() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok != nil, tok.BOSToken(), tok.EOSToken())
	// Output: true 100 101
}

func ExampleTokenizer_Encode() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.Encode("hello"))
	// Output: [100 4 5 6 3]
}

func ExampleTokenizer_Decode() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.Decode([]int32{100, 4, 5, 6, 3}))
	// Output: hello
}

func ExampleTokenizer_DecodeToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.DecodeToken(5), tok.DecodeToken(7))
	// Output: he  h
}

func ExampleTokenizer_BOSToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.BOSToken())
	// Output: 100
}

func ExampleTokenizer_EOSToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.EOSToken())
	// Output: 101
}

func ExampleTokenizer_HasBOSToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.HasBOSToken())
	// Output: true
}

func ExampleTokenizer_HasEOSToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.HasEOSToken())
	// Output: true
}

func ExampleTokenizer_BOS() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.BOS())
	// Output: 100
}

func ExampleTokenizer_EOS() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.EOS())
	// Output: 101
}

func ExampleTokenizer_TokenID() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	id, ok := tok.TokenID("he")
	core.Println(id, ok)
	// Output: 5 true
}

func ExampleTokenizer_IDToken() {
	tok, cleanup := mustExampleTokenizer()
	defer cleanup()

	core.Println(tok.IDToken(6))
	// Output: ll
}

func ExampleFormatGemmaPrompt() {
	core.Println(FormatGemmaPrompt("What is 2+2?"))
	// Output:
	// <bos><start_of_turn>user
	// What is 2+2?<end_of_turn>
	// <start_of_turn>model
}

func mustExampleTokenizer() (*Tokenizer, func()) {
	dirResult := core.MkdirTemp("", "go-mlx-metal-tokenizer-example-*")
	if !dirResult.OK {
		panic(dirResult.Value)
	}
	dir := dirResult.Value.(string)
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte(minimalTokenizerJSON), 0o644); !result.OK {
		core.RemoveAll(dir)
		panic(result.Value)
	}
	tok, err := LoadTokenizer(path)
	if err != nil {
		core.RemoveAll(dir)
		panic(err)
	}
	return tok, func() { core.RemoveAll(dir) }
}
