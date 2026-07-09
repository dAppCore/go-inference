// SPDX-Licence-Identifier: EUPL-1.2

package serving

import "fmt"

// ExampleDraftDetection_Active shows the Active gate: a resolved source with
// a non-empty path is engaged, the zero value never is.
func ExampleDraftDetection_Active() {
	forced := DraftDetection{Source: DraftSourceFlag, DraftPath: "/models/drafter"}
	var none DraftDetection
	fmt.Println(forced.Active(), none.Active())
	// Output:
	// true false
}

// ExampleDetectGemma4DraftPath shows an explicit --draft path winning the
// ladder outright, regardless of the target model.
func ExampleDetectGemma4DraftPath() {
	det := DetectGemma4DraftPath("/models/target", "/models/drafter", DraftDetectOptions{})
	fmt.Println(det.Source, det.DraftPath)
	// Output:
	// flag /models/drafter
}

// ExampleResolveServeDraft shows the "auto" flag routing into the reactive
// ladder; a non-gemma4 target (no config.json here) gets no auto drafter.
func ExampleResolveServeDraft() {
	det := ResolveServeDraft("/models/plain-llama", "auto", true)
	fmt.Println(det.Active())
	// Output:
	// false
}
