// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"encoding/json"
	"testing"
)

// TestChatMessageDelta_MarshalJSON_RoundTrip locks the hand-rolled
// encoder shape against encoding/json's deserialiser. The encoder
// is on the streaming hot path — every SSE delta + priming + close
// chunk routes through it, so its output must round-trip cleanly
// back into ChatMessageDelta with no field drift.
//
// Cases cover every branch the encoder walks:
//   - empty struct  -> "{}"
//   - role-only     -> emits both role and content:"" (priming chunk)
//   - content-only  -> emits content only
//   - both set      -> both fields
//   - escape body   -> control/quote/backslash characters in content
func TestChatMessageDelta_MarshalJSON_RoundTrip(t *testing.T) {
	cases := []struct {
		name string
		in   ChatMessageDelta
		want string
	}{
		{"empty", ChatMessageDelta{}, `{}`},
		{"role-only", ChatMessageDelta{Role: "assistant"}, `{"role":"assistant","content":""}`},
		{"content-only", ChatMessageDelta{Content: "hello"}, `{"content":"hello"}`},
		{"both", ChatMessageDelta{Role: "assistant", Content: "world"}, `{"role":"assistant","content":"world"}`},
		{"escapes", ChatMessageDelta{Content: "quote \" backslash \\ tab\tnewline\n"},
			`{"content":"quote \" backslash \\ tab\tnewline\n"}`},
		{"control", ChatMessageDelta{Content: "\x01\x02"}, `{"content":"\u0001\u0002"}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoded, err := tc.in.MarshalJSON()
			if err != nil {
				t.Fatalf("MarshalJSON() error = %v", err)
			}
			if string(encoded) != tc.want {
				t.Fatalf("MarshalJSON() = %s, want %s", encoded, tc.want)
			}
			// Round-trip via encoding/json — the streaming chunk
			// types wrap ChatMessageDelta and the proxy clients
			// consuming the stream feed it back into the same Go
			// types.
			var back ChatMessageDelta
			if err := json.Unmarshal(encoded, &back); err != nil {
				t.Fatalf("json.Unmarshal(%s) error = %v", encoded, err)
			}
			if back.Role != tc.in.Role || back.Content != tc.in.Content {
				t.Fatalf("round-trip: got %+v, want %+v", back, tc.in)
			}
		})
	}
}
