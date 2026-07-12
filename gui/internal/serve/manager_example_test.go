// SPDX-Licence-Identifier: EUPL-1.2

package serve

import (
	core "dappco.re/go"
)

// ExampleManager shows a freshly built manager owns no process until Start is
// called. In use, Start spawns `lem serve --addr :36911 --model <path>` and Stop
// terminates it with a graceful SIGTERM.
func ExampleManager() {
	m := NewManager("lem", ":36911")

	core.Println(m.Managed(), m.Addr())
	// Output:
	// false :36911
}
