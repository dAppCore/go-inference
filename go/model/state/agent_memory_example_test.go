// SPDX-Licence-Identifier: EUPL-1.2

package state

import (
	"context"
	"fmt"
)

// ExampleRef builds the durable-state envelope a runtime hands back after
// waking or sleeping a session.
func ExampleRef() {
	ref := Ref{
		URI:        "state://agents/cladius/seed",
		Kind:       "agent_memory",
		TokenStart: 0,
		TokenCount: 4096,
	}
	fmt.Println(ref.Kind, ref.TokenCount)
	// Output:
	// agent_memory 4096
}

// ExampleWakeRequest selects a durable state prefix to restore into a
// live session.
func ExampleWakeRequest() {
	req := WakeRequest{
		IndexURI: "state://lthn/projects/core/go-mlx/seed/index",
		EntryURI: "state://lthn/projects/core/go-mlx/seed",
		Model:    ModelIdentity{ID: "gemma4"},
	}
	fmt.Println(req.EntryURI, req.Model.ID)
	// Output:
	// state://lthn/projects/core/go-mlx/seed gemma4
}

// ExampleSleepRequest asks a live session to persist its current state.
func ExampleSleepRequest() {
	req := SleepRequest{
		EntryURI:          "state://lthn/projects/core/go-mlx/checkpoints/latest",
		ReuseParentPrefix: true,
		BlockSize:         512,
	}
	fmt.Println(req.EntryURI, req.BlockSize)
	// Output:
	// state://lthn/projects/core/go-mlx/checkpoints/latest 512
}

// ExampleForker demonstrates the ForkState contract a runtime implements
// to create an independent live session from durable state.
func ExampleForker() {
	var forker Forker = fakeForker{}

	session, wake, err := forker.ForkState(context.Background(), WakeRequest{IndexURI: "state://index"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(wake.Entry.URI, wake.PrefixTokens)

	sleep, err := session.SleepState(context.Background(), SleepRequest{EntryURI: "state://entry"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}
	fmt.Println(sleep.TokenCount)
	// Output:
	// state://index/entry 12
	// 12
}
