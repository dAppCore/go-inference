// SPDX-Licence-Identifier: EUPL-1.2

package fusion

import (
	"context"

	core "dappco.re/go"
)

// ExampleRun demonstrates a fusion deliberation (RFC.md §6.9): a panel of
// analysis models runs in parallel, the judge synthesises their responses,
// and Run returns the final answer plus the structured Analysis.
func ExampleRun() {
	panel := []Model{
		&fakeModel{id: "gemma-31b", reply: "consensus: photons scatter"},
		&fakeModel{id: "gemma-26b", reply: "unique: Rayleigh scattering"},
	}
	judge := &fakeModel{id: "judge", reply: "the sky is blue because of Rayleigh scattering"}

	cfg := Config{AnalysisModels: panel, Judge: judge, Enabled: true}

	res, err := Run(context.Background(), "why is the sky blue?", cfg)
	if err != nil {
		core.Println(err)
		return
	}
	core.Println(res.Answer)
	core.Println(len(res.Analysis.Panel))
	// Output:
	// the sky is blue because of Rayleigh scattering
	// 2
}
