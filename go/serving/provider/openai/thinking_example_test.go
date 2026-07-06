// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func ExampleNewThinkingExtractor() {
	extractor := NewThinkingExtractor()

	content, thought := extractor.Process(inference.Token{Text: "hello"})

	core.Println(content)
	core.Println(thought)
	// Output:
	// hello
	//
}

func ExampleThinkingExtractor_Process() {
	extractor := NewThinkingExtractor()

	content, thought := extractor.Process(inference.Token{Text: "<think>plan</think>answer"})

	core.Println(content)
	core.Println(thought)
	// Output:
	// answer
	// plan
}

func ExampleThinkingExtractor_Flush() {
	extractor := NewThinkingExtractor()
	extractor.Process(inference.Token{Text: "hello <thi"})

	content, thought := extractor.Flush()

	core.Println(content)
	core.Println(thought)
	// Output:
	// <thi
	//
}

func ExampleThinkingExtractor_Content() {
	extractor := NewThinkingExtractor()
	extractor.Process(inference.Token{Text: "before <think>hidden</think> after"})

	core.Println(extractor.Content())
	// Output:
	// before  after
}

func ExampleThinkingExtractor_Thinking() {
	extractor := NewThinkingExtractor()
	extractor.Process(inference.Token{Text: "<think>plan</think>answer"})

	core.Println(extractor.Thinking())
	// Output:
	// plan
}
