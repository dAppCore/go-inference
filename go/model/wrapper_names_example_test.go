// SPDX-Licence-Identifier: EUPL-1.2

package model

import (
	core "dappco.re/go"

	"dappco.re/go/inference/model/safetensors"
)

// ExampleNormalizeWrapperNames shows the multimodal-wrapper alias: a "language_model."
// nested tensor becomes ALSO addressable by its stripped "model.…" name, so an assembler's
// bare lookups work whether or not the checkpoint nests the text model under the wrapper.
func ExampleNormalizeWrapperNames() {
	tensors := map[string]safetensors.Tensor{
		"language_model.model.norm.weight": {Shape: []int{4}},
	}
	out := NormalizeWrapperNames(tensors)
	_, wrapped := out["language_model.model.norm.weight"]
	_, bare := out["model.norm.weight"]
	core.Println(wrapped) // the original nested name is kept
	core.Println(bare)    // the stripped alias is added
	// Output:
	// true
	// true
}
