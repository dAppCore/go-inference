// SPDX-License-Identifier: EUPL-1.2

package gitserver

import (
	core "dappco.re/go"
)

func ExampleNewSoftServe() {
	options := DefaultOptions("/tmp/lem/soft-serve").Value.(Options)
	options.ListenAddress = "127.0.0.1:0"
	options.PublicURL = "ssh://127.0.0.1:0"
	result := NewSoftServe(options)
	core.Println(result.OK)
	// Output: true
}
