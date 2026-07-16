// SPDX-Licence-Identifier: EUPL-1.2

package bert_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/arch/bert"
)

// ExampleNewTokenizer builds a WordPiece tokeniser from a vocab blob and frames
// a sentence as [CLS] … [SEP] ids.
func ExampleNewTokenizer() {
	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\nfox\n"
	tk, err := bert.NewTokenizer([]byte(vocab), true)
	if err != nil {
		panic(err)
	}
	ids := tk.Encode("The fox")
	// ids == [2 5 6 3] -> [CLS] the fox [SEP]
	println(len(ids))
	// Output:
}

func ExampleTokenizer_PadID() {
	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\nfox\n"
	tk, err := bert.NewTokenizer([]byte(vocab), true)
	if err != nil {
		panic(err)
	}
	core.Println(tk.PadID())
	// Output: 0
}

func ExampleTokenizer_Encode() {
	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nthe\nfox\n"
	tk, err := bert.NewTokenizer([]byte(vocab), true)
	if err != nil {
		panic(err)
	}
	core.Println(tk.Encode("the fox"))
	// Output: [2 5 6 3]
}

func ExampleTokenizer_EncodePair() {
	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nquery\npassage\n"
	tk, err := bert.NewTokenizer([]byte(vocab), true)
	if err != nil {
		panic(err)
	}
	ids, tokenTypes := tk.EncodePair("query", "passage")
	core.Println(ids)
	core.Println(tokenTypes)
	// Output:
	// [2 5 3 6 3]
	// [0 0 0 1 1]
}
