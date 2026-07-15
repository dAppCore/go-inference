// SPDX-Licence-Identifier: EUPL-1.2

package openai

import core "dappco.re/go"

func ExampleChatCompletionRequest_UnmarshalJSON() {
	var req ChatCompletionRequest
	in := []byte(`{"model":"qwen","messages":[{"role":"user","content":"hi"}]}`)

	if err := req.UnmarshalJSON(in); err != nil {
		core.Println(err)
		return
	}

	core.Println(req.Model)
	core.Println(len(req.Messages))
	// Output:
	// qwen
	// 1
}

func ExampleResponseRequest_UnmarshalJSON() {
	var req ResponseRequest
	in := []byte(`{"model":"qwen","input":[{"role":"user","content":"hi"}]}`)

	if err := req.UnmarshalJSON(in); err != nil {
		core.Println(err)
		return
	}

	core.Println(req.Model)
	core.Println(len(req.Input))
	// Output:
	// qwen
	// 1
}
