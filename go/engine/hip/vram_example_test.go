//go:build linux && amd64

package hip

import core "dappco.re/go"

func ExampleGetVRAMInfo() {
	info, err := GetVRAMInfo()
	core.Println(err == nil || info == (VRAMInfo{})) /* Output: true */
}
