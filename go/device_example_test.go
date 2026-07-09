// SPDX-Licence-Identifier: EUPL-1.2

package inference_test

import (
	"fmt"

	"dappco.re/go/inference"
)

// BackendDeviceInfo probes a registered backend for its accelerator without
// loading a model — false when the backend is absent or cannot say.
func ExampleBackendDeviceInfo() {
	if info, ok := inference.BackendDeviceInfo("metal"); ok {
		fmt.Printf("%s, %d GiB\n", info.Name, info.MemorySize>>30)
		return
	}
	fmt.Println("metal backend not registered")
	// Output: metal backend not registered
}
