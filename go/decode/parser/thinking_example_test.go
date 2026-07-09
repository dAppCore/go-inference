// SPDX-Licence-Identifier: EUPL-1.2

package parser

import core "dappco.re/go"

func ExampleFilter() {
	result := Filter("<think>plan</think>answer", Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	core.Println(result.Text)
	core.Println(result.Reasoning)
	// Output:
	// answer
	// plan
}

func ExampleNewProcessor() {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	core.Println(p.Process("<think>plan</think>answer"))
	// Output: answer
}

func ExampleNormaliseMode() {
	core.Println(NormaliseMode(""))
	core.Println(NormaliseMode(Capture))
	// Output:
	// show
	// capture
}

func ExampleProcessor_Process() {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	core.Println(p.Process("visible <think>hidden</think>tail"))
	// Output: visible tail
}

func ExampleProcessor_Flush() {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	p.Process("<thi")
	core.Println(p.Flush())
	// Output: <thi
}

func ExampleProcessor_Reasoning() {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	p.Process("<think>plan</think>answer")
	core.Println(p.Reasoning())
	// Output: plan
}

func ExampleProcessor_Chunks() {
	p := NewProcessor(Config{Mode: Hide}, Hint{Architecture: "qwen3"})
	p.Process("<think>plan</think>answer")
	chunks := p.Chunks()
	core.Println(chunks[0].Text, chunks[0].Channel)
	// Output: plan thinking
}
