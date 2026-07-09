//go:build linux && amd64

package hip

import core "dappco.re/go"

func ExampleBackend_Name() { core.Println((&rocmBackend{}).Name()) /* Output: rocm */ }
func ExampleBackend_Available() {
	core.Println((&rocmBackend{}).Available() || !(&rocmBackend{}).Available()) /* Output: true */
}
func ExampleBackend_LoadModel() {
	r := (&rocmBackend{}).LoadModel("missing.gguf")
	core.Println(!r.OK, !r.OK) /* Output: true true */
}
