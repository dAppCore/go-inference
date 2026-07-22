// SPDX-License-Identifier: EUPL-1.2

package gitserver

import (
	core "dappco.re/go"
)

func ExampleDefaultOptions() {
	result := DefaultOptions("/tmp/lem/soft-serve")
	options := result.Value.(Options)
	core.Println(options.ListenAddress, options.PublicURL)
	// Output: 127.0.0.1:23231 ssh://127.0.0.1:23231
}
