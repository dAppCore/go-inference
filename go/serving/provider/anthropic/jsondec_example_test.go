// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import core "dappco.re/go"

func ExampleMessageRequest_UnmarshalJSON() {
	var req MessageRequest
	data := []byte(`{"model":"gemma-4","messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}],"max_tokens":128}`)
	if err := req.UnmarshalJSON(data); err != nil {
		core.Println(err)
		return
	}
	core.Println(req.Model, req.MaxTokens, req.Messages[0].Content[0].Text)
	// Output:
	// gemma-4 128 hi
}

func ExampleMessageResponse_UnmarshalJSON() {
	var resp MessageResponse
	data := []byte(`{"id":"msg_1","type":"message","role":"assistant","model":"gemma-4","content":[{"type":"text","text":"hi"}],"usage":{"input_tokens":3,"output_tokens":1}}`)
	if err := resp.UnmarshalJSON(data); err != nil {
		core.Println(err)
		return
	}
	core.Println(resp.ID, resp.Content[0].Text, resp.Usage.OutputTokens)
	// Output:
	// msg_1 hi 1
}
