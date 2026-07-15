//go:build !linux || !amd64

package hip

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleROCmAvailable() { core.Println(ROCmAvailable()) /* Output: false */ }
func ExampleGetVRAMInfo()   { _, err := GetVRAMInfo(); core.Println(err != nil) /* Output: true */ }
func ExampleROCmAvailable_backendRegistration() {
	backend, ok := inference.Get("rocm")
	core.Println(ok, backend.Available())
	// Output: true false
}
