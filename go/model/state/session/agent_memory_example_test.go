// SPDX-Licence-Identifier: EUPL-1.2

package session

import (
	"context"
	"fmt"

	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/model/spine"
	memvid "dappco.re/go/inference/model/state"
	"dappco.re/go/inference/model/state/agent"
	"dappco.re/go/inference/model/state/session/internal/sessionfake"
)

// ExampleSession_Sleep streams the session's retained KV state to a state
// store, then a fresh session Wakes from the durable index — the
// agent-memory lifecycle round-trip.
func ExampleSession_Sleep() {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}

	source := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info, nil)
	sleep, err := source.Sleep(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/example",
		Title:        "example state",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		fmt.Println("sleep error:", err)
		return
	}

	awake := New(&sessionfake.Handle{}, info, nil)
	wake, err := awake.Wake(ctx, store, agent.WakeOptions{
		IndexURI: sleep.IndexURI,
		EntryURI: sleep.EntryURI,
	})
	if err != nil {
		fmt.Println("wake error:", err)
		return
	}

	fmt.Println("slept tokens:", sleep.TokenCount)
	fmt.Println("woke tokens:", wake.PrefixTokens)
	fmt.Println("strategy:", wake.RestoreStrategy)
	// Output:
	// slept tokens: 2
	// woke tokens: 2
	// strategy: kv-blocks
}

// ExampleSession_AppendAndSleep records a new observation onto the retained
// state and persists it without generating a reply — the agent-memory
// "remember this, no answer needed" flow. The appended prompt reaches the
// native session before the sleep streams the state to the store.
func ExampleSession_AppendAndSleep() {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}

	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	sess := New(native, info, nil)

	report, err := sess.AppendAndSleep(ctx, "repo observation: tests green", store, agent.SleepOptions{
		EntryURI:     "mlx://agent/observation",
		Title:        "observation",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		fmt.Println("append-and-sleep error:", err)
		return
	}

	fmt.Println("appended:", native.AppendPromptSeen)
	fmt.Println("generate calls:", native.GenerateCalls)
	fmt.Println("entry:", report.EntryURI)
	fmt.Println("tokens:", report.TokenCount)
	// Output:
	// appended: repo observation: tests green
	// generate calls: 0
	// entry: mlx://agent/observation
	// tokens: 2
}

// ExampleSession_GenerateAndSleep produces a reply from the retained state
// and persists the post-answer state in one call — the "answer the user,
// then remember the turn" flow. The seeded tokens become the returned reply
// and the sleep writes the durable index.
func ExampleSession_GenerateAndSleep() {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}

	native := &sessionfake.Handle{
		KV:     sessionfake.TestKVSnapshot(),
		Tokens: []inference.Token{{ID: 1, Text: "All "}, {ID: 2, Text: "green."}},
	}
	sess := New(native, info, nil)

	reply, report, err := sess.GenerateAndSleep(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/turn",
		Title:        "turn",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		fmt.Println("generate-and-sleep error:", err)
		return
	}

	fmt.Println("reply:", reply)
	fmt.Println("entry:", report.EntryURI)
	fmt.Println("tokens:", report.TokenCount)
	// Output:
	// reply: All green.
	// entry: mlx://agent/turn
	// tokens: 2
}

// ExampleSession_Wake_foldedPrefill sleeps with the folded-state label, then
// wakes — the wake folds the durable prefix back in via a token prefill
// rather than a raw KV-block restore. The "folded-state" label written at
// sleep steers shouldPrefillFoldedAgentMemory at wake.
func ExampleSession_Wake_foldedPrefill() {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}

	source := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info, nil)
	sleep, err := source.Sleep(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/folded",
		Title:        "folded state",
		Labels:       []string{"folded-state"},
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		fmt.Println("sleep error:", err)
		return
	}

	awakeNative := &sessionfake.Handle{}
	awake := New(awakeNative, info, nil)
	wake, err := awake.Wake(ctx, store, agent.WakeOptions{
		IndexURI: sleep.IndexURI,
		EntryURI: sleep.EntryURI,
	})
	if err != nil {
		fmt.Println("wake error:", err)
		return
	}

	fmt.Println("strategy:", wake.RestoreStrategy)
	fmt.Println("prefilled tokens:", len(awakeNative.PrefillTokensSeen))
	// Output:
	// strategy: folded-prefill
	// prefilled tokens: 2
}

// ExampleSession_Fork forks a slept session — the fork starts from the same
// retained state and carries the parent's agent-memory linkage, so its next
// sleep records the parent as its lineage.
func ExampleSession_Fork() {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1, QuantBits: 4, ContextLength: 8}

	parentNative := &sessionfake.Handle{
		KV:     sessionfake.TestKVSnapshot(),
		Forked: &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()},
	}
	parent := New(parentNative, info, nil)
	parentSleep, err := parent.Sleep(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/parent",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		fmt.Println("parent sleep error:", err)
		return
	}

	forked, err := parent.Fork()
	if err != nil {
		fmt.Println("fork error:", err)
		return
	}
	childSleep, err := forked.Sleep(ctx, store, agent.SleepOptions{
		EntryURI:     "mlx://agent/child",
		BlockOptions: kv.StateBlockOptions{BlockSize: 1},
	})
	if err != nil {
		fmt.Println("child sleep error:", err)
		return
	}

	fmt.Println("child entry:", childSleep.EntryURI)
	fmt.Println("inherited parent:", childSleep.ParentEntryURI == parentSleep.EntryURI)
	// Output:
	// child entry: mlx://agent/child
	// inherited parent: true
}
