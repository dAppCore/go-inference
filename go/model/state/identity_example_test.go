// SPDX-Licence-Identifier: EUPL-1.2

package state

import "fmt"

// ExampleModelIdentity builds the backend-neutral model metadata carried
// on a Bundle or WakeRequest.
func ExampleModelIdentity() {
	model := ModelIdentity{ID: "gemma4", Architecture: "gemma4_text", NumLayers: 28, ContextLength: 8192}
	fmt.Println(model.ID, model.NumLayers)
	// Output:
	// gemma4 28
}

// ExampleBundle builds a portable state envelope combining model,
// tokenizer, and runtime identity with prompt/generation counters.
func ExampleBundle() {
	bundle := Bundle{
		Version:      "1",
		Model:        ModelIdentity{ID: "gemma4", Hash: "model-a"},
		PromptTokens: 2048,
	}
	fmt.Println(bundle.Model.ID, bundle.PromptTokens)
	// Output:
	// gemma4 2048
}
