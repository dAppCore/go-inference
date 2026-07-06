// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import core "dappco.re/go"

func ExampleAppendMessageStartEvent() {
	msg := MessageResponse{ID: "msg_1", Type: "message", Role: "assistant", Model: "gemma-4", Usage: Usage{InputTokens: 5}}
	core.Println(string(AppendMessageStartEvent(nil, msg)))
	// Output:
	// {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"gemma-4","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}
}

func ExampleAppendContentBlockStartEvent() {
	core.Println(string(AppendContentBlockStartEvent(nil, 0)))
	// Output:
	// {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}
}

func ExampleAppendContentBlockDeltaEvent() {
	core.Println(string(AppendContentBlockDeltaEvent(nil, 0, "hi")))
	// Output:
	// {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}
}

func ExampleAppendContentBlockStopEvent() {
	core.Println(string(AppendContentBlockStopEvent(nil, 0)))
	// Output:
	// {"type":"content_block_stop","index":0}
}

func ExampleAppendContentBlockStartToolUseEvent() {
	core.Println(string(AppendContentBlockStartToolUseEvent(nil, 0, "toolu_1", "get_weather")))
	// Output:
	// {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}
}

func ExampleAppendInputJSONDeltaEvent() {
	core.Println(string(AppendInputJSONDeltaEvent(nil, 0, `{"city":"Paris"}`)))
	// Output:
	// {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}
}

func ExampleAppendMessageDeltaEvent() {
	core.Println(string(AppendMessageDeltaEvent(nil, "end_turn", "", 12)))
	// Output:
	// {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}}
}
