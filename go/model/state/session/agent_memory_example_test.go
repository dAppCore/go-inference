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

// ExampleSession_WakeAgentMemory restores a session from a durable indexed
// KV prefix written by an earlier SleepAgentMemory.
func ExampleSession_WakeAgentMemory() {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}

	asleep := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info, nil)
	sleep, err := asleep.SleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: "mlx://agent/wake-example"})
	if err != nil {
		fmt.Println("sleep error:", err)
		return
	}

	awake := New(&sessionfake.Handle{}, info, nil)
	report, err := awake.WakeAgentMemory(ctx, store, agent.WakeOptions{IndexURI: sleep.IndexURI, EntryURI: sleep.EntryURI})
	if err != nil {
		fmt.Println("wake error:", err)
		return
	}

	fmt.Println("strategy:", report.RestoreStrategy)
	// Output:
	// strategy: kv-blocks
}

// ExampleSession_WakeState implements the backend-neutral go-inference
// agent-memory contract over an ordinary state.Store.
func ExampleSession_WakeState() {
	ctx := context.Background()
	store := memvid.NewInMemoryStore(nil)
	info := spine.ModelInfo{Architecture: "gemma4_text", NumLayers: 1}

	asleep := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, info, nil)
	sleep, err := asleep.SleepAgentMemory(ctx, store, agent.SleepOptions{EntryURI: "mlx://agent/wakestate-example"})
	if err != nil {
		fmt.Println("sleep error:", err)
		return
	}

	awake := New(&sessionfake.Handle{}, info, nil)
	result, err := awake.WakeState(ctx, inference.AgentMemoryWakeRequest{Store: store, IndexURI: sleep.IndexURI, EntryURI: sleep.EntryURI})
	if err != nil {
		fmt.Println("wake error:", err)
		return
	}

	fmt.Println("entry:", result.Entry.URI)
	// Output:
	// entry: mlx://agent/wakestate-example
}

// ExampleSession_SleepAgentMemory streams the session's retained KV state
// to durable storage and writes a bundle manifest plus wake index.
func ExampleSession_SleepAgentMemory() {
	sess := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)
	store := memvid.NewInMemoryStore(nil)

	report, err := sess.SleepAgentMemory(context.Background(), store, agent.SleepOptions{EntryURI: "mlx://agent/sleep-example"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("entry:", report.EntryURI)
	fmt.Println("tokens:", report.TokenCount)
	// Output:
	// entry: mlx://agent/sleep-example
	// tokens: 2
}

// ExampleSession_SleepState implements the backend-neutral go-inference
// agent-memory contract over an ordinary state.Writer.
func ExampleSession_SleepState() {
	sess := New(&sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}, spine.ModelInfo{}, nil)
	store := memvid.NewInMemoryStore(nil)

	result, err := sess.SleepState(context.Background(), inference.AgentMemorySleepRequest{Store: store, EntryURI: "mlx://agent/sleepstate-example"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("entry:", result.Entry.URI)
	// Output:
	// entry: mlx://agent/sleepstate-example
}

// ExampleSession_AppendAndSleepAgentMemory appends new prompt material and
// then streams the resulting state to durable storage without generating a
// reply.
func ExampleSession_AppendAndSleepAgentMemory() {
	native := &sessionfake.Handle{KV: sessionfake.TestKVSnapshot()}
	sess := New(native, spine.ModelInfo{}, nil)
	store := memvid.NewInMemoryStore(nil)

	report, err := sess.AppendAndSleepAgentMemory(context.Background(), "observation", store, agent.SleepOptions{EntryURI: "mlx://agent/append-example"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("appended:", native.AppendPromptSeen)
	fmt.Println("entry:", report.EntryURI)
	// Output:
	// appended: observation
	// entry: mlx://agent/append-example
}

// ExampleSession_GenerateAndSleepAgentMemory generates a reply from the
// retained state and streams the post-answer KV state to durable storage in
// one call.
func ExampleSession_GenerateAndSleepAgentMemory() {
	sess := New(&sessionfake.Handle{
		KV:     sessionfake.TestKVSnapshot(),
		Tokens: []inference.Token{{ID: 1, Text: "hello"}},
	}, spine.ModelInfo{}, nil)
	store := memvid.NewInMemoryStore(nil)

	text, report, err := sess.GenerateAndSleepAgentMemory(context.Background(), store, agent.SleepOptions{EntryURI: "mlx://agent/gen-example"})
	if err != nil {
		fmt.Println("error:", err)
		return
	}

	fmt.Println("reply:", text)
	fmt.Println("entry:", report.EntryURI)
	// Output:
	// reply: hello
	// entry: mlx://agent/gen-example
}

// ExampleWakeOptionsFromInference maps a go-inference wake request onto
// agent wake options — the shared shape Model.ForkState builds on.
func ExampleWakeOptionsFromInference() {
	req := inference.AgentMemoryWakeRequest{
		IndexURI: "mlx://index",
		EntryURI: "mlx://entry",
	}

	opts := WakeOptionsFromInference(req)

	fmt.Println("index:", opts.IndexURI)
	fmt.Println("entry:", opts.EntryURI)
	// Output:
	// index: mlx://index
	// entry: mlx://entry
}

// ExampleToInferenceWakeResult maps a wake report onto the go-inference
// result shape — the shared shape Model.ForkState builds on.
func ExampleToInferenceWakeResult() {
	report := &agent.WakeReport{
		EntryURI:     "mlx://entry",
		PrefixTokens: 4,
	}

	result := ToInferenceWakeResult(report)

	fmt.Println("entry:", result.Entry.URI)
	fmt.Println("prefix tokens:", result.PrefixTokens)
	// Output:
	// entry: mlx://entry
	// prefix tokens: 4
}
