// SPDX-Licence-Identifier: EUPL-1.2

package serve

import (
	core "dappco.re/go"
)

// ExampleDefaultModelsDir shows the tray's default model-discovery root, the
// same store the daemon downloads into.
func ExampleDefaultModelsDir() {
	core.Println(core.HasSuffix(DefaultModelsDir(), "Lethean/lem/models"))
	// Output:
	// true
}
