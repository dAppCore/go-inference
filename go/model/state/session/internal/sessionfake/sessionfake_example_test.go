// SPDX-Licence-Identifier: EUPL-1.2

package sessionfake

import (
	"context"
	"fmt"

	"dappco.re/go/inference"
	"dappco.re/go/inference/kv"
)

// ExampleHandle_Prefill records the prompt the caller loaded into the fake
// session.
func ExampleHandle_Prefill() {
	h := &Handle{}

	err := h.Prefill(context.Background(), "stable context")

	fmt.Println("error:", err)
	fmt.Println("recorded:", h.PrefillPrompt)
	// Output:
	// error: <nil>
	// recorded: stable context
}

// ExampleHandle_PrefillChunks collects a bounded prompt-chunk sequence in
// order.
func ExampleHandle_PrefillChunks() {
	h := &Handle{}

	err := h.PrefillChunks(context.Background(), seqOf("stable ", "context"))

	fmt.Println("error:", err)
	fmt.Println("recorded:", h.PrefillChunksSeen)
	// Output:
	// error: <nil>
	// recorded: [stable  context]
}

// ExampleHandle_PrefillTokens copies the model-native token IDs the caller
// loaded.
func ExampleHandle_PrefillTokens() {
	h := &Handle{}

	err := h.PrefillTokens(context.Background(), []int32{11, 12})

	fmt.Println("error:", err)
	fmt.Println("recorded:", h.PrefillTokensSeen)
	// Output:
	// error: <nil>
	// recorded: [11 12]
}

// ExampleHandle_AppendPrompt records the appended prompt text.
func ExampleHandle_AppendPrompt() {
	h := &Handle{}

	err := h.AppendPrompt(context.Background(), "Question: who?")

	fmt.Println("error:", err)
	fmt.Println("recorded:", h.AppendPromptSeen)
	// Output:
	// error: <nil>
	// recorded: Question: who?
}

// ExampleHandle_AppendPromptChunks collects the appended chunk sequence.
func ExampleHandle_AppendPromptChunks() {
	h := &Handle{}

	err := h.AppendPromptChunks(context.Background(), seqOf("Question: ", "who?"))

	fmt.Println("error:", err)
	fmt.Println("recorded:", h.AppendChunksSeen)
	// Output:
	// error: <nil>
	// recorded: [Question:  who?]
}

// ExampleHandle_AppendTokens copies the appended token IDs.
func ExampleHandle_AppendTokens() {
	h := &Handle{}

	err := h.AppendTokens(context.Background(), []int32{21, 22})

	fmt.Println("error:", err)
	fmt.Println("recorded:", h.AppendTokensSeen)
	// Output:
	// error: <nil>
	// recorded: [21 22]
}

// ExampleHandle_Generate drains the seeded tokens, recording the generate
// config as a side effect.
func ExampleHandle_Generate() {
	h := &Handle{Tokens: []inference.Token{{ID: 1, Text: "Hi"}, {ID: 2, Text: " there"}}}

	var got string
	for tok := range h.Generate(context.Background(), inference.GenerateConfig{MaxTokens: 8}) {
		got += tok.Text
	}

	fmt.Println(got)
	fmt.Println("calls:", h.GenerateCalls)
	// Output:
	// Hi there
	// calls: 1
}

// ExampleHandle_CaptureKV returns the seeded snapshot.
func ExampleHandle_CaptureKV() {
	h := &Handle{KV: TestKVSnapshot()}

	snap, err := h.CaptureKV(context.Background())

	fmt.Println("error:", err)
	fmt.Println("architecture:", snap.Architecture)
	// Output:
	// error: <nil>
	// architecture: gemma4_text
}

// ExampleHandle_RangeKVBlocks iterates the seeded KV blocks, or synthesises
// the whole KV as one block when none are seeded.
func ExampleHandle_RangeKVBlocks() {
	h := &Handle{KV: TestKVSnapshot()}

	var seen int
	err := h.RangeKVBlocks(context.Background(), 0, kv.CaptureOptions{}, func(kv.Block) (bool, error) {
		seen++
		return true, nil
	})

	fmt.Println("error:", err)
	fmt.Println("blocks:", seen)
	// Output:
	// error: <nil>
	// blocks: 1
}

// ExampleHandle_RestoreKV records the restored snapshot.
func ExampleHandle_RestoreKV() {
	h := &Handle{}

	err := h.RestoreKV(context.Background(), TestKVSnapshot())

	fmt.Println("error:", err)
	fmt.Println("restored:", h.RestoredKV != nil)
	// Output:
	// error: <nil>
	// restored: true
}

// ExampleHandle_RestoreKVBlocks loads blocks from a source up to the prefix
// boundary.
func ExampleHandle_RestoreKVBlocks() {
	snap := TestKVSnapshot()
	h := &Handle{}
	src := kv.BlockSource{
		BlockCount:   1,
		PrefixTokens: 2,
		Load: func(_ context.Context, i int) (kv.Block, error) {
			return kv.Block{Index: i, TokenStart: 0, TokenCount: 2, Snapshot: snap}, nil
		},
	}

	err := h.RestoreKVBlocks(context.Background(), src)

	fmt.Println("error:", err)
	fmt.Println("restored:", h.RestoredKV != nil)
	// Output:
	// error: <nil>
	// restored: true
}

// ExampleHandle_Fork returns the seeded fork handle.
func ExampleHandle_Fork() {
	child := &Handle{}
	h := &Handle{Forked: child}

	got, err := h.Fork(context.Background())

	fmt.Println("error:", err)
	fmt.Println("same handle:", got == inference.SessionHandle(child))
	// Output:
	// error: <nil>
	// same handle: true
}

// ExampleHandle_Reset counts the call.
func ExampleHandle_Reset() {
	h := &Handle{}

	h.Reset()

	fmt.Println("reset calls:", h.ResetCalls)
	// Output:
	// reset calls: 1
}

// ExampleHandle_Close counts the call and returns the seeded error.
func ExampleHandle_Close() {
	h := &Handle{}

	err := h.Close()

	fmt.Println("error:", err)
	fmt.Println("close calls:", h.CloseCalls)
	// Output:
	// error: <nil>
	// close calls: 1
}

// ExampleHandle_Err returns the seeded error value.
func ExampleHandle_Err() {
	h := &Handle{}

	fmt.Println("error:", h.Err())
	// Output:
	// error: <nil>
}

// ExampleTestKVSnapshot builds the canonical two-token gemma4 KV snapshot
// the session and agent-memory tests sleep/wake against.
func ExampleTestKVSnapshot() {
	snap := TestKVSnapshot()

	fmt.Println("architecture:", snap.Architecture)
	fmt.Println("tokens:", len(snap.Tokens))
	// Output:
	// architecture: gemma4_text
	// tokens: 2
}
