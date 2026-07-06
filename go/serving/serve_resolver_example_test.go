// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"fmt"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// ExampleModelLoader shows a custom ModelLoader implementation — the seam
// RunServe injects to reach an engine other than the registered "metal"
// default (a test double here; a non-Apple engine in practice).
func ExampleModelLoader() {
	var loader ModelLoader = func(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
		return &mockTextModel{modelType: path}, nil
	}
	m, err := loader("gemma3-1b")
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(m.ModelType())
	// Output:
	// gemma3-1b
}

// ExampleSpeculativeLoader shows a custom SpeculativeLoader implementation —
// the seam armed when reactive drafter detection finds a target+draft pair
// and the registered engine exposes a speculative path.
func ExampleSpeculativeLoader() {
	var loader SpeculativeLoader = func(targetPath, draftPath string, draftBlock int, _ ...inference.LoadOption) (inference.TextModel, error) {
		return &mockTextModel{modelType: core.Sprintf("%s+%s(block=%d)", targetPath, draftPath, draftBlock)}, nil
	}
	m, err := loader("target", "assistant", 5)
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(m.ModelType())
	// Output:
	// target+assistant(block=5)
}
