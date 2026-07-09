// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"fmt"

	"dappco.re/go/inference/model/spine"
	"dappco.re/go/inference/model/state/session/internal/sessionfake"
)

// ExampleDefaultNewSessionText shows the engine default alias a framework
// reads when opening a session before any real user prompt has arrived.
func ExampleDefaultNewSessionText() {
	fmt.Println(DefaultNewSessionText == DefaultLemmaNewSessionText)
	// Output:
	// true
}

// ExampleDefaultLemmaNewSessionText prefills a fresh session with the raw
// Lemma-family seed text.
func ExampleDefaultLemmaNewSessionText() {
	sess := New(&sessionfake.Handle{}, spine.ModelInfo{Architecture: "gemma4_text"}, nil)

	err := sess.Prefill(DefaultLemmaNewSessionText)

	fmt.Println("prefill error:", err)
	// Output:
	// prefill error: <nil>
}
