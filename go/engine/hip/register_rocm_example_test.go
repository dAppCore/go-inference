//go:build linux && amd64

package hip

import core "dappco.re/go"

func ExampleROCmAvailable() {
	available := ROCmAvailable()
	core.Println(available || !available) /* Output: true */
}
