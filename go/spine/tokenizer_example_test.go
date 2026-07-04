// SPDX-Licence-Identifier: EUPL-1.2

package spine_test

import (
	"fmt"

	"dappco.re/go/inference/spine"
)

// exampleTokenizer is a tiny in-memory TokenizerImpl for the runnable
// examples — it maps a fixed three-word vocabulary and carries a BOS.
type exampleTokenizer struct{}

func (exampleTokenizer) Encode(text string) []int32 {
	// Always emit a leading BOS (1) so the wrapper's implicit-BOS strip
	// is visible in the example output.
	switch text {
	case "hi":
		return []int32{1, 10}
	default:
		return []int32{1}
	}
}
func (exampleTokenizer) Decode(ids []int32) string {
	if len(ids) == 1 && ids[0] == 10 {
		return "hi"
	}
	return ""
}
func (exampleTokenizer) DecodeOne(id int32) string {
	if id == 10 {
		return "hi"
	}
	return ""
}
func (exampleTokenizer) TokenID(s string) (int32, bool) {
	if s == "hi" {
		return 10, true
	}
	return 0, false
}
func (exampleTokenizer) IDToken(id int32) string {
	switch id {
	case 1:
		return "<s>"
	case 10:
		return "hi"
	}
	return ""
}
func (exampleTokenizer) BOS() int32        { return 1 }
func (exampleTokenizer) EOS() int32        { return 2 }
func (exampleTokenizer) HasBOSToken() bool { return true }

func ExampleNewTokenizer() {
	tok := spine.NewTokenizer(exampleTokenizer{})
	fmt.Println(tok.Valid())
	// A nil implementation produces an invalid wrapper.
	fmt.Println(spine.NewTokenizer(nil).Valid())
	// Output:
	// true
	// false
}

func ExampleTokenizer_Encode() {
	tok := spine.NewTokenizer(exampleTokenizer{})
	// The model-internal implicit BOS (1) is stripped from the result.
	ids, err := tok.Encode("hi")
	fmt.Println(ids, err)
	// Output: [10] <nil>
}

func ExampleTokenizer_BOS() {
	tok := spine.NewTokenizer(exampleTokenizer{})
	fmt.Println(tok.BOS(), tok.EOS())
	// Output: 1 2
}
