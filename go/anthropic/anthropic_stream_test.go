// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import (
	"testing"

	core "dappco.re/go"
)

func TestAppendMessageStartEvent_Good(t *testing.T) {
	msg := MessageResponse{ID: "msg_1", Type: "message", Role: "assistant", Model: "lemer", Usage: Usage{InputTokens: 5}}
	core.AssertEqual(t,
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"lemer","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}`,
		string(AppendMessageStartEvent(nil, msg)))
}

func TestAppendContentBlockStartEvent_Good(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		string(AppendContentBlockStartEvent(nil, 0)))
}

func TestAppendContentBlockDeltaEvent_Good(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		string(AppendContentBlockDeltaEvent(nil, 0, "hello")))
}

func TestAppendContentBlockDeltaEvent_Ugly_Escapes(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"a\"b\nc"}}`,
		string(AppendContentBlockDeltaEvent(nil, 2, "a\"b\nc")))
}

func TestAppendContentBlockStopEvent_Good(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_stop","index":0}`,
		string(AppendContentBlockStopEvent(nil, 0)))
}

func TestAppendMessageDeltaEvent_Good_EndTurn(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}}`,
		string(AppendMessageDeltaEvent(nil, "end_turn", "", 12)))
}

func TestAppendMessageDeltaEvent_Bad_StopSequence(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"message_delta","delta":{"stop_reason":"stop_sequence","stop_sequence":"</s>"},"usage":{"output_tokens":3}}`,
		string(AppendMessageDeltaEvent(nil, "stop_sequence", "</s>", 3)))
}

func TestStaticStreamPayloads_Good(t *testing.T) {
	core.AssertEqual(t, `{"type":"message_stop"}`, MessageStopPayload)
	core.AssertEqual(t, `{"type":"ping"}`, PingPayload)
}

// AppendToExisting pins that the builders append to a non-empty buffer rather
// than assuming buf starts empty — the streaming handler reuses one buffer.
func TestStreamEvents_Ugly_AppendToExisting(t *testing.T) {
	buf := []byte("PRE")
	buf = AppendContentBlockStopEvent(buf, 1)
	core.AssertEqual(t, `PRE{"type":"content_block_stop","index":1}`, string(buf))
}
