// SPDX-Licence-Identifier: EUPL-1.2

package dataset_test

import (
	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
)

// ExamplePairContent shows the wire shape a KindPair Item's Content
// carries — the same shape the pairs-jsonl export writer produces.
func ExamplePairContent() {
	encoded := core.JSONMarshal(dataset.PairContent{
		Prompt:   "explain your reasoning",
		Response: "the sky looks blue because of Rayleigh scattering",
	})
	core.Println(core.AsString(encoded.Value.([]byte)))
	// Output:
	// {"prompt":"explain your reasoning","response":"the sky looks blue because of Rayleigh scattering"}
}

// ExampleMessagesContent shows the wire shape a KindMessages Item's
// Content carries — identical to the `lem train --sft` sft-jsonl row.
func ExampleMessagesContent() {
	encoded := core.JSONMarshal(dataset.MessagesContent{
		Messages: []dataset.MessageTurn{
			{Role: "user", Content: "hello"},
			{Role: "assistant", Content: "hi there"},
		},
	})
	core.Println(core.AsString(encoded.Value.([]byte)))
	// Output:
	// {"messages":[{"role":"user","content":"hello"},{"role":"assistant","content":"hi there"}]}
}

// ExampleMessagesContent_LastExchange reduces a multi-turn conversation
// to the final prompt/response pair a heuristic scorer or a pairs-jsonl
// export can consume.
func ExampleMessagesContent_LastExchange() {
	mc := dataset.MessagesContent{Messages: []dataset.MessageTurn{
		{Role: "user", Content: "hello"},
		{Role: "assistant", Content: "hi"},
		{Role: "user", Content: "how are you"},
		{Role: "assistant", Content: "well, thanks"},
	}}
	pc, ok := mc.LastExchange()
	core.Println(ok, pc.Response)
	// Output:
	// true well, thanks
}

// ExampleJudgeScoreKind builds a namespaced ScoreKind for a named judge
// template — the Score.Kind a judge-tier row carries.
func ExampleJudgeScoreKind() {
	core.Println(dataset.JudgeScoreKind("helpfulness"))
	// Output:
	// judge:helpfulness
}

// ExampleScoreKind_IsJudge distinguishes the fixed heuristic kinds from
// dynamic judge-tier kinds.
func ExampleScoreKind_IsJudge() {
	core.Println(dataset.ScoreKindLEK.IsJudge())
	core.Println(dataset.JudgeScoreKind("helpfulness").IsJudge())
	// Output:
	// false
	// true
}

// ExampleScoreKind_JudgeName recovers the template name from a
// judge-tier ScoreKind.
func ExampleScoreKind_JudgeName() {
	core.Println(dataset.JudgeScoreKind("helpfulness").JudgeName())
	// Output:
	// helpfulness
}

// ExampleValidateItemContent validates a KindPair Item's Content before
// it is hashed and stored.
func ExampleValidateItemContent() {
	content := core.JSONMarshal(dataset.PairContent{Prompt: "hi", Response: "hello"}).Value.([]byte)
	r := dataset.ValidateItemContent(dataset.KindPair, content)
	core.Println(r.OK)
	// Output:
	// true
}

// ExampleNewID demonstrates the identifier shape every Dataset/Item/
// Score/Export row uses. The value itself is random — this only checks
// the deterministic properties (length, uniqueness).
func ExampleNewID() {
	a := dataset.NewID()
	b := dataset.NewID()
	core.Println(len(a) == 36, a != b)
	// Output:
	// true true
}
