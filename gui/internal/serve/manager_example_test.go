// SPDX-Licence-Identifier: EUPL-1.2

package serve

import (
	core "dappco.re/go"
)

// ExampleManager shows a freshly built manager owns no process until Start is
// called. In use, Start can add chat, embeddings and scheduler flags, and Stop
// terminates the child with a graceful SIGTERM.
func ExampleManager() {
	m := NewManager("lem", ":36911")

	core.Println(m.Managed(), m.Addr())
	// Output:
	// false :36911
}
