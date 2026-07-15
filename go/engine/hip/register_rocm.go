//go:build linux && amd64

package hip

import "dappco.re/go/inference"

func init() {
	inference.Register(&rocmBackend{})
}

//	if ROCmAvailable() {
//	    fmt.Println("ROCm code path compiled in")
//	}
//
// ROCmAvailable reports whether ROCm GPU inference is available.
func ROCmAvailable() bool { return (&rocmBackend{}).Available() }
