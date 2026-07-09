// SPDX-Licence-Identifier: EUPL-1.2

package state

import "fmt"

// ExampleNewProjectSeed derives the durable-state URI family for a
// project from just a base URI and project ID.
func ExampleNewProjectSeed() {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})
	fmt.Println(seed.EntryURI)
	// Output:
	// state://lthn/projects/core/go-mlx/seed
}

// ExampleProjectSeed_WakeRequest builds a WakeRequest addressed at the
// seed's own entry/index URIs, tagging it with the project's label.
func ExampleProjectSeed_WakeRequest() {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})

	// setProjectLabel only ever writes into an already-non-nil Labels map
	// (see TestProjectSeed_WakeRequest_Bad), so a caller who wants
	// project_id back on the request needs to seed at least one label —
	// here, its own scope tag.
	wake := seed.WakeRequest(ProjectSeedWakeOptions{
		Model:  ModelIdentity{ID: "gemma4"},
		Labels: map[string]string{"scope": "repo"},
	})
	fmt.Println(wake.EntryURI, wake.Labels["project_id"])
	// Output:
	// state://lthn/projects/core/go-mlx/seed core/go-mlx
}

// ExampleProjectSeed_PlanContinuation plans the next durable-state action
// once a task using the seed has finished.
func ExampleProjectSeed_PlanContinuation() {
	seed := NewProjectSeed(ProjectSeedOptions{
		BaseURI:   "state://lthn/projects",
		ProjectID: "core/go-mlx",
	})

	plan := seed.PlanContinuation(ProjectSeedContinuationOptions{Mode: ProjectSeedStateCheckpoint})
	fmt.Println(plan.PersistState, plan.Sleep.EntryURI)
	// Output:
	// true state://lthn/projects/core/go-mlx/checkpoints/latest
}

// ExampleCheckWakeCompatibility compares a durable bundle's identity
// against a wake request, reporting whether the runtime can safely resume
// from it.
func ExampleCheckWakeCompatibility() {
	bundle := Bundle{Model: ModelIdentity{Hash: "model-a", Architecture: "gemma4_text"}}
	req := WakeRequest{Model: ModelIdentity{Hash: "model-a", Architecture: "gemma4_text"}}

	report := CheckWakeCompatibility(bundle, req)
	fmt.Println(report.Compatible, report.SummaryRequired)
	// Output:
	// true false
}
