// SPDX-License-Identifier: EUPL-1.2

package provider

import (
	"testing"

	core "dappco.re/go"
)

func TestParser_ParseFinalStatus_Good(t *testing.T) {
	tests := []struct {
		name   string
		input  string
		status FinalStatus
	}{
		{
			name:   "completed",
			input:  `finished\n<<<LEM_STATUS>>>{"status":"completed","summary":"tests pass"}<<<END_LEM_STATUS>>>`,
			status: FinalStatus{Status: "completed", Summary: "tests pass"},
		},
		{
			name:   "waiting",
			input:  `<<<LEM_STATUS>>>{"status":"waiting","question":"Which API should be canonical?"}<<<END_LEM_STATUS>>>`,
			status: FinalStatus{Status: "waiting", Question: "Which API should be canonical?"},
		},
		{
			name:   "failed",
			input:  `<<<LEM_STATUS>>>{"status":"failed","reason":"validation failed"}<<<END_LEM_STATUS>>>`,
			status: FinalStatus{Status: "failed", Reason: "validation failed"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			result := ParseFinalStatus(test.input)
			core.AssertTrue(t, result.OK, result.Error())
			core.AssertEqual(t, test.status, result.Value.(FinalStatus))
		})
	}
}

func TestParser_ParseFinalStatus_Bad(t *testing.T) {
	tests := []string{
		`<<<LEM_STATUS>>>{"status":"waiting"}<<<END_LEM_STATUS>>>`,
		`<<<LEM_STATUS>>>{"status":"failed"}<<<END_LEM_STATUS>>>`,
		`<<<LEM_STATUS>>>{"status":"unknown","question":"do not wait"}<<<END_LEM_STATUS>>>`,
		`<<<LEM_STATUS>>>{not-json}<<<END_LEM_STATUS>>>`,
		`<<<LEM_STATUS>>>{"status":"waiting","question":" "}<<<END_LEM_STATUS>>>`,
	}
	for _, input := range tests {
		result := ParseFinalStatus(input)
		core.AssertFalse(t, result.OK)
	}
}

func TestParser_ParseFinalStatus_Ugly(t *testing.T) {
	core.AssertFalse(t, ParseFinalStatus("").OK)
	core.AssertFalse(t, ParseFinalStatus("ordinary output").OK)
	core.AssertFalse(t, ParseFinalStatus(`<<<END_LEM_STATUS>>><<<LEM_STATUS>>>{"status":"waiting","question":"no"}`).OK)
	core.AssertFalse(t, ParseFinalStatus(`<<<LEM_STATUS>>>{"status":"completed"}`).OK)
	core.AssertFalse(t, ParseFinalStatus(`<<<LEM_STATUS>>>{"status":"waiting","question":"first"}<<<END_LEM_STATUS>>><<<LEM_STATUS>>>{"status":"waiting","question":"second"}<<<END_LEM_STATUS>>>`).OK)
	core.AssertFalse(t, ParseFinalStatus(`<<<LEM_STATUS>>>{"status":"completed"}<<<END_LEM_STATUS>>> trailing`).OK)
}

func TestParserProviderFixtures(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	tests := []struct {
		name     string
		provider string
		stream   string
		line     string
		kind     string
		text     string
	}{
		{name: "codex text", provider: "codex", stream: "stdout", line: `{"type":"item.completed","item":{"type":"agent_message","text":"Codex done"}}`, kind: "text", text: "Codex done"},
		{name: "codex progress", provider: "codex", stream: "stdout", line: `{"type":"thread.started","thread_id":"thread-1"}`, kind: "progress", text: "thread.started"},
		{name: "codex usage", provider: "codex", stream: "stdout", line: `{"type":"turn.completed","usage":{"input_tokens":10,"output_tokens":4}}`, kind: "usage"},
		{name: "claude text", provider: "claude", stream: "stdout", line: `{"type":"assistant","message":{"content":[{"type":"text","text":"Claude done"}]}}`, kind: "text", text: "Claude done"},
		{name: "claude stream event", provider: "claude", stream: "stdout", line: `{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":"Streaming"}}}`, kind: "text", text: "Streaming"},
		{name: "claude result", provider: "claude", stream: "stdout", line: `{"type":"result","result":"Final words","usage":{"input_tokens":8}}`, kind: "text", text: "Final words"},
		{name: "opencode part", provider: "opencode", stream: "stdout", line: `{"type":"text","part":{"text":"OpenCode done"}}`, kind: "text", text: "OpenCode done"},
		{name: "plain stdout", provider: "opencode", stream: "stdout", line: "plain output", kind: "text", text: "plain output"},
		{name: "plain stderr", provider: "claude", stream: "stderr", line: "warning output", kind: "stderr", text: "warning output"},
		{name: "malformed JSON", provider: "codex", stream: "stdout", line: `{"type":`, kind: "raw", text: `{"type":`},
		{name: "unknown JSON", provider: "opencode", stream: "stdout", line: `{"mystery":true}`, kind: "raw", text: `{"mystery":true}`},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			adapter := registry.Adapter(test.provider).Value.(Adapter)
			outputs := adapter.ParseLine(test.stream, test.line)
			core.AssertTrue(t, len(outputs) >= 1)
			core.AssertEqual(t, test.kind, outputs[0].Kind)
			if test.text != "" {
				core.AssertEqual(t, test.text, outputs[0].Text)
			}
			if test.kind == "usage" {
				core.AssertContains(t, outputs[0].UsageJSON, "input_tokens")
			}
			if test.line[0] == '{' {
				core.AssertEqual(t, test.line, outputs[0].DetailJSON)
			}
		})
	}
}

func TestParserRateLimitFixtures(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("claude").Value.(Adapter)

	outputs := adapter.ParseLine("stderr", "Rate limit exceeded; try again in 30m.")
	core.AssertEqual(t, 1, len(outputs))
	core.AssertEqual(t, "rate_limit", outputs[0].Kind)
	core.AssertEqual(t, "30m", outputs[0].RetryAfter)

	outputs = adapter.ParseLine("stderr", "429 too many requests; retry later")
	core.AssertEqual(t, "rate_limit", outputs[0].Kind)
	core.AssertEqual(t, "", outputs[0].RetryAfter)

	outputs = adapter.ParseLine("stderr", "rate limit; try again in nonsense")
	core.AssertEqual(t, "rate_limit", outputs[0].Kind)
	core.AssertEqual(t, "", outputs[0].RetryAfter)

	outputs = adapter.ParseLine("stdout", `{"type":"error","error":{"message":"rate limit","retry_after":"45s"}}`)
	core.AssertEqual(t, "rate_limit", outputs[0].Kind)
	core.AssertEqual(t, "45s", outputs[0].RetryAfter)
	core.AssertTrue(t, outputs[0].DetailJSON != "")

	outputs = adapter.ParseLine("stdout", `{"type":"error","error":{"message":"rate limit","retryAfter":"2m"}}`)
	core.AssertEqual(t, "2m", outputs[0].RetryAfter)

	outputs = adapter.ParseLine("stderr", "rate limit; retry after")
	core.AssertEqual(t, "", outputs[0].RetryAfter)

	outputs = adapter.ParseLine("stderr", "rate limit; retry in -1m")
	core.AssertEqual(t, "", outputs[0].RetryAfter)

	outputs = adapter.ParseLine("stderr", "rate limit; retry in 0s")
	core.AssertEqual(t, "", outputs[0].RetryAfter)
}

func TestParserNestedEdges(t *testing.T) {
	registry := providerTestRegistry(t, nil)
	adapter := registry.Adapter("opencode").Value.(Adapter)

	outputs := adapter.ParseLine("stdout", `{"type":"text","part":{"content":[{"text":"one"},17,{"text":"two"}]}}`)
	core.AssertEqual(t, "one\ntwo", outputs[0].Text)

	outputs = adapter.ParseLine("stdout", `[{"usage":{"tokens":1}}]`)
	core.AssertEqual(t, "raw", outputs[0].Kind)

	core.AssertEqual(t, "", retryDuration(`{"message":"rate limit","retry_after":17}`))
}
