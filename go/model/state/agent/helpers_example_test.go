// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"fmt"

	"dappco.re/go/inference/model/bundle"
)

// Example_stateHash shows the canonical state-bundle hashing helper —
// a deterministic 64-char SHA-256 hex digest over its input. (Named as a
// package-level example because stateHash is unexported.)
func Example_stateHash() {
	fmt.Println(len(stateHash("hello")))
	fmt.Println(stateHash("hello") == stateHash("hello")) // same input twice — pins determinism
	fmt.Println(stateHash("hello") == stateHash("world")) // differing input — pins discrimination
	// Output:
	// 64
	// true
	// false
}

// Example_stateBundleTokenizer shows that a fully-populated tokenizer
// passes through normalisation with its hashes intact.
func Example_stateBundleTokenizer() {
	out := stateBundleTokenizer(bundle.Tokenizer{Hash: "tok-a", ChatTemplateHash: "chat-a"})
	fmt.Println(out.Hash)
	fmt.Println(out.ChatTemplateHash)
	// Output:
	// tok-a
	// chat-a
}

// Example_cloneStringMap shows that the clone is independent of its
// source — and that empty/nil inputs clone to nil rather than an empty
// map.
func Example_cloneStringMap() {
	src := map[string]string{"agent": "cladius"}
	clone := cloneStringMap(src)
	clone["agent"] = "mutated"
	fmt.Println(src["agent"])
	fmt.Println(clone["agent"])
	fmt.Println(cloneStringMap(nil) == nil)
	// Output:
	// cladius
	// mutated
	// true
}
