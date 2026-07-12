// SPDX-Licence-Identifier: EUPL-1.2

package prefixindex_test

import (
	"fmt"

	"dappco.re/go/inference/kv/prefixindex"
)

// ExampleIndex shows the cross-conversation share: A publishes its framed
// prompt, and B — opening with the same system prefix but a different first
// user turn — finds the shared run and A's backing bundle to wake instead of
// re-prefilling.
func ExampleIndex() {
	ix := prefixindex.New(prefixindex.Config{MaxEntries: 4096})

	system := []int32{1, 2, 3, 4, 5, 6, 7, 8} // a shared system prompt
	convA := append(append([]int32{}, system...), 100, 101)
	convB := append(append([]int32{}, system...), 200, 201, 202)

	ix.Publish(convA, prefixindex.Entry{BundleURI: "state://conv-a", BlockSize: 4, TokenCount: len(convA)})

	entry, shared, ok := ix.Match(convB)
	fmt.Println(ok, shared, entry.BundleURI)
	// Output: true 8 state://conv-a
}
