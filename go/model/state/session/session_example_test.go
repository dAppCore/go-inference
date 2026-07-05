// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"fmt"

	"dappco.re/go/inference"
	"dappco.re/go/inference/model/spine"
	"dappco.re/go/inference/model/state/session/internal/sessionfake"
)

// ExampleNew wraps an already-created native session handle — the
// construction seam the root mlx package builds on. Tests and callers
// hand New a handle, the model info, and a tokenizer.
func ExampleNew() {
	handle := &sessionfake.Handle{
		Tokens: []inference.Token{{ID: 1, Text: "Hi"}, {ID: 2, Text: " there"}},
	}
	sess := New(handle, spine.ModelInfo{Architecture: "gemma4_text"}, nil)

	_ = sess.Prefill("stable context")
	reply, _ := sess.Generate(optMaxTokens(8))

	fmt.Println(reply)
	// Output: Hi there
}

// ExampleSession_Valid reports whether a session still holds a live native
// handle — the exported form of the internal nil/closed guard.
func ExampleSession_Valid() {
	sess := New(&sessionfake.Handle{}, spine.ModelInfo{}, nil)
	fmt.Println("after new:", sess.Valid())

	_ = sess.Close()
	fmt.Println("after close:", sess.Valid())

	var nilSession *Session
	fmt.Println("nil session:", nilSession.Valid())
	// Output:
	// after new: true
	// after close: false
	// nil session: false
}

// ExampleSession_Native returns the underlying native session handle — the
// accessor callers outside the package use instead of reaching the
// unexported field. A nil *Session yields a nil handle.
func ExampleSession_Native() {
	handle := &sessionfake.Handle{}
	sess := New(handle, spine.ModelInfo{}, nil)

	fmt.Println("same handle:", sess.Native() == handle)

	var nilSession *Session
	fmt.Println("nil handle:", nilSession.Native() == nil)
	// Output:
	// same handle: true
	// nil handle: true
}
