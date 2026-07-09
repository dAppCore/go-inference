// SPDX-Licence-Identifier: EUPL-1.2

package anthropic

import (
	"encoding/json"
	"testing"

	core "dappco.re/go"
)

// contentBlockStartPayload decodes a content_block_start event body —
// shared by the text-block and tool_use variants of AppendContentBlockStartEvent
// / AppendContentBlockStartToolUseEvent so escape-heavy Ugly cases can assert
// against a real decode rather than a hand-escaped literal.
type contentBlockStartPayload struct {
	Type         string `json:"type"`
	Index        int    `json:"index"`
	ContentBlock struct {
		Type  string         `json:"type"`
		Text  string         `json:"text"`
		ID    string         `json:"id"`
		Name  string         `json:"name"`
		Input map[string]any `json:"input"`
	} `json:"content_block"`
}

// contentBlockDeltaPayload decodes a content_block_delta event body — shared
// by the text_delta and input_json_delta variants.
type contentBlockDeltaPayload struct {
	Type  string `json:"type"`
	Index int    `json:"index"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
	} `json:"delta"`
}

// messageDeltaPayload decodes a message_delta event body.
type messageDeltaPayload struct {
	Type  string `json:"type"`
	Delta struct {
		StopReason   string  `json:"stop_reason"`
		StopSequence *string `json:"stop_sequence"`
	} `json:"delta"`
	Usage struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

func TestAnthropicStream_AppendMessageStartEvent_Good(t *testing.T) {
	msg := MessageResponse{ID: "msg_1", Type: "message", Role: "assistant", Model: "lemer", Usage: Usage{InputTokens: 5}}
	core.AssertEqual(t,
		`{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"lemer","content":[],"usage":{"input_tokens":5,"output_tokens":0}}}`,
		string(AppendMessageStartEvent(nil, msg)))
}

// TestAnthropicStream_AppendMessageStartEvent_Bad pins the zero-value edge —
// every field empty still produces well-formed JSON.
func TestAnthropicStream_AppendMessageStartEvent_Bad(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"message_start","message":{"id":"","type":"","role":"","model":"","content":[],"usage":{"input_tokens":0,"output_tokens":0}}}`,
		string(AppendMessageStartEvent(nil, MessageResponse{})))
}

// TestAnthropicStream_AppendMessageStartEvent_Ugly combines two edges:
// appending onto a non-empty caller buffer, and an escape-heavy ID.
// Verified via decode (through MessageResponse's own UnmarshalJSON) rather
// than a hand-escaped literal.
func TestAnthropicStream_AppendMessageStartEvent_Ugly(t *testing.T) {
	buf := []byte("PRE")
	msg := MessageResponse{ID: `msg "1"`, Type: "message", Role: "assistant", Model: "gemma-4"}
	buf = AppendMessageStartEvent(buf, msg)
	if string(buf[:3]) != "PRE" {
		t.Fatalf("prefix not preserved: %s", buf)
	}
	var env struct {
		Type    string          `json:"type"`
		Message MessageResponse `json:"message"`
	}
	if err := json.Unmarshal(buf[3:], &env); err != nil {
		t.Fatalf("json.Unmarshal: %v\nbody: %s", err, buf)
	}
	if env.Type != "message_start" || env.Message.ID != msg.ID {
		t.Fatalf("round-trip mismatch: %+v", env)
	}
}

func TestAnthropicStream_AppendContentBlockStartEvent_Good(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
		string(AppendContentBlockStartEvent(nil, 0)))
}

// TestAnthropicStream_AppendContentBlockStartEvent_Bad pins a negative
// index — an edge value the caller should never send, but the encoder must
// not panic or mangle it.
func TestAnthropicStream_AppendContentBlockStartEvent_Bad(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_start","index":-1,"content_block":{"type":"text","text":""}}`,
		string(AppendContentBlockStartEvent(nil, -1)))
}

// TestAnthropicStream_AppendContentBlockStartEvent_Ugly combines appending
// onto a non-empty buffer with a large index value.
func TestAnthropicStream_AppendContentBlockStartEvent_Ugly(t *testing.T) {
	buf := []byte("PRE")
	buf = AppendContentBlockStartEvent(buf, 999999)
	core.AssertEqual(t,
		`PRE{"type":"content_block_start","index":999999,"content_block":{"type":"text","text":""}}`,
		string(buf))
}

func TestAnthropicStream_AppendContentBlockDeltaEvent_Good(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}`,
		string(AppendContentBlockDeltaEvent(nil, 0, "hello")))
}

// TestAnthropicStream_AppendContentBlockDeltaEvent_Bad pins the edge case of
// an empty text delta — still a well-formed text_delta event.
func TestAnthropicStream_AppendContentBlockDeltaEvent_Bad(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}`,
		string(AppendContentBlockDeltaEvent(nil, 0, "")))
}

func TestAnthropicStream_AppendContentBlockDeltaEvent_Ugly(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"a\"b\nc"}}`,
		string(AppendContentBlockDeltaEvent(nil, 2, "a\"b\nc")))
}

func TestAnthropicStream_AppendContentBlockStopEvent_Good(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_stop","index":0}`,
		string(AppendContentBlockStopEvent(nil, 0)))
}

// TestAnthropicStream_AppendContentBlockStopEvent_Bad pins a negative index.
func TestAnthropicStream_AppendContentBlockStopEvent_Bad(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_stop","index":-1}`,
		string(AppendContentBlockStopEvent(nil, -1)))
}

// TestAnthropicStream_AppendContentBlockStopEvent_Ugly pins that the
// builders append to a non-empty buffer rather than assuming buf starts
// empty — the streaming handler reuses one buffer.
func TestAnthropicStream_AppendContentBlockStopEvent_Ugly(t *testing.T) {
	buf := []byte("PRE")
	buf = AppendContentBlockStopEvent(buf, 1)
	core.AssertEqual(t, `PRE{"type":"content_block_stop","index":1}`, string(buf))
}

// TestAnthropicStream_AppendContentBlockStartToolUseEvent_Good pins the
// tool_use content_block_start payload.
func TestAnthropicStream_AppendContentBlockStartToolUseEvent_Good(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"get_weather","input":{}}}`,
		string(AppendContentBlockStartToolUseEvent(nil, 0, "toolu_1", "get_weather")))
}

// TestAnthropicStream_AppendContentBlockStartToolUseEvent_Bad pins the edge
// case of empty id/name — still a well-formed tool_use block open.
func TestAnthropicStream_AppendContentBlockStartToolUseEvent_Bad(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"","name":"","input":{}}}`,
		string(AppendContentBlockStartToolUseEvent(nil, 0, "", "")))
}

// TestAnthropicStream_AppendContentBlockStartToolUseEvent_Ugly combines
// appending onto a non-empty buffer with an escape-heavy tool name,
// verified via decode rather than a hand-escaped literal.
func TestAnthropicStream_AppendContentBlockStartToolUseEvent_Ugly(t *testing.T) {
	buf := []byte("PRE")
	buf = AppendContentBlockStartToolUseEvent(buf, 2, "toolu_2", `get "the" weather`)
	if string(buf[:3]) != "PRE" {
		t.Fatalf("prefix not preserved: %s", buf)
	}
	var payload contentBlockStartPayload
	if err := json.Unmarshal(buf[3:], &payload); err != nil {
		t.Fatalf("json.Unmarshal: %v\nbody: %s", err, buf)
	}
	if payload.Index != 2 || payload.ContentBlock.Name != `get "the" weather` {
		t.Fatalf("payload = %+v", payload)
	}
}

// TestAnthropicStream_AppendInputJSONDeltaEvent_Good pins one tool_use
// arguments delta.
func TestAnthropicStream_AppendInputJSONDeltaEvent_Good(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}}`,
		string(AppendInputJSONDeltaEvent(nil, 0, `{"city":"Paris"}`)))
}

// TestAnthropicStream_AppendInputJSONDeltaEvent_Bad pins the edge case of an
// empty partial-JSON fragment.
func TestAnthropicStream_AppendInputJSONDeltaEvent_Bad(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}`,
		string(AppendInputJSONDeltaEvent(nil, 0, "")))
}

// TestAnthropicStream_AppendInputJSONDeltaEvent_Ugly pins a fragment
// carrying backslashes and newlines — verified via decode so the escaping
// doesn't need to be hand-computed.
func TestAnthropicStream_AppendInputJSONDeltaEvent_Ugly(t *testing.T) {
	frag := `{"path":"C:\\tmp","note":"line1\nline2"}`
	buf := AppendInputJSONDeltaEvent(nil, 3, frag)
	var payload contentBlockDeltaPayload
	if err := json.Unmarshal(buf, &payload); err != nil {
		t.Fatalf("json.Unmarshal: %v\nbody: %s", err, buf)
	}
	if payload.Index != 3 || payload.Delta.PartialJSON != frag {
		t.Fatalf("payload = %+v, want partial_json round-tripping to %q", payload, frag)
	}
}

func TestAnthropicStream_AppendMessageDeltaEvent_Good(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":12}}`,
		string(AppendMessageDeltaEvent(nil, "end_turn", "", 12)))
}

func TestAnthropicStream_AppendMessageDeltaEvent_Bad(t *testing.T) {
	core.AssertEqual(t,
		`{"type":"message_delta","delta":{"stop_reason":"stop_sequence","stop_sequence":"</s>"},"usage":{"output_tokens":3}}`,
		string(AppendMessageDeltaEvent(nil, "stop_sequence", "</s>", 3)))
}

// TestAnthropicStream_AppendMessageDeltaEvent_Ugly pins an escape-heavy
// stop_reason plus a negative output-token count, verified via decode.
func TestAnthropicStream_AppendMessageDeltaEvent_Ugly(t *testing.T) {
	reason := `weird "reason"`
	buf := AppendMessageDeltaEvent(nil, reason, "", -1)
	var payload messageDeltaPayload
	if err := json.Unmarshal(buf, &payload); err != nil {
		t.Fatalf("json.Unmarshal: %v\nbody: %s", err, buf)
	}
	if payload.Delta.StopReason != reason || payload.Delta.StopSequence != nil {
		t.Fatalf("payload.Delta = %+v", payload.Delta)
	}
	if payload.Usage.OutputTokens != -1 {
		t.Fatalf("payload.Usage.OutputTokens = %d, want -1", payload.Usage.OutputTokens)
	}
}

// TestStaticStreamPayloads_Good pins the two fixed event payloads both as
// literal strings and as valid, decodable JSON carrying the expected type.
func TestStaticStreamPayloads_Good(t *testing.T) {
	core.AssertEqual(t, `{"type":"message_stop"}`, MessageStopPayload)
	core.AssertEqual(t, `{"type":"ping"}`, PingPayload)

	var stop struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal([]byte(MessageStopPayload), &stop); err != nil || stop.Type != "message_stop" {
		t.Fatalf("MessageStopPayload decode = %+v, err %v", stop, err)
	}
	var ping struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal([]byte(PingPayload), &ping); err != nil || ping.Type != "ping" {
		t.Fatalf("PingPayload decode = %+v, err %v", ping, err)
	}
}
