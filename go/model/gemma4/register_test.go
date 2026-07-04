// SPDX-Licence-Identifier: EUPL-1.2

package gemma4

import (
	"testing"

	"dappco.re/go/inference/model"
)

// TestRegistersArch pins that gemma4's init() registered an ArchSpec for every model_type id the family
// uses, so the engine's reactive loader (model.Load → model.LookupArch) dispatches to gemma4 with no
// central switch.
func TestRegistersArch(t *testing.T) {
	for _, mt := range []string{"gemma4", "gemma4_text", "gemma4_unified", "diffusion_gemma"} {
		if _, ok := model.LookupArch(mt); !ok {
			t.Fatalf("gemma4 init() should register an ArchSpec for model_type %q", mt)
		}
	}
}
