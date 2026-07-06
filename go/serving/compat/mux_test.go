// SPDX-Licence-Identifier: EUPL-1.2

package compat

import (
	"testing"

	core "dappco.re/go"
)

// TestJSONHTMLSafe pins the pass-through predicate encoding/json's HTML-safe
// default keys on: printable ASCII passes, but the control range, the JSON
// structural bytes (" \), and the HTML-meta bytes (< > &) must be escaped.
func TestJSONHTMLSafe(t *testing.T) {
	for _, b := range []byte{'a', ' ', '~', '{', '}', '@'} {
		if !jsonHTMLSafe(b) {
			t.Fatalf("jsonHTMLSafe(%q) = false, want true (ordinary byte)", b)
		}
	}
	for _, b := range []byte{0x00, 0x1f, '"', '\\', '<', '>', '&'} {
		if jsonHTMLSafe(b) {
			t.Fatalf("jsonHTMLSafe(%q) = true, want false (must be escaped)", b)
		}
	}
}

// TestHexNibble pins the lowercase-hex digit mapping the \u00XX / \u202X escape
// branches emit, including that only the low nibble is read.
func TestHexNibble(t *testing.T) {
	cases := map[byte]byte{0x0: '0', 0x9: '9', 0xa: 'a', 0xf: 'f', 0x1f: 'f'}
	for in, want := range cases {
		if got := hexNibble(in); got != want {
			t.Fatalf("hexNibble(%#x) = %q, want %q", in, got, want)
		}
	}
}

// TestAppendJSONStringHTML pins the hand-rolled encoder against the contract it
// claims: byte-identity with encoding/json's HTML-safe Marshal (core
// .JSONMarshalString of a string). The HTML-meta escaping is the load-bearing
// case — a streamed code/markup delta routinely carries < > &, and a naive
// encoder that skips them corrupts the wire.
func TestAppendJSONStringHTML(t *testing.T) {
	cases := []string{
		"",
		"hello world",
		`a"b\c`,
		"tab\tnewline\ncarriage\r",
		"<div>&nbsp;</div>",
		"null\x00byte\x01",
		"café — naïve",     // multibyte pass-through
		" line par",        // separators encoding/json escapes
		"\xff\xfe invalid", // invalid UTF-8 -> �
	}
	for _, s := range cases {
		got := string(appendJSONStringHTML(nil, s))
		want := core.JSONMarshalString(s)
		if got != want {
			t.Fatalf("appendJSONStringHTML(%q) = %q, want %q (encoding/json parity)", s, got, want)
		}
	}
	// Explicitly document the HTML-meta contract: < > & become the escaped
	// < > & sequences, never raw.
	if got, want := string(appendJSONStringHTML(nil, "<>&")), "\"\\u003c\\u003e\\u0026\""; got != want {
		t.Fatalf("HTML-meta escaping = %q, want %q", got, want)
	}
}

// FuzzAppendJSONStringHTML is the equivalence lock the appendJSONStringHTML doc
// references: for ANY input the hand-rolled fast-path encoder must produce the
// exact bytes encoding/json would (core.JSONMarshalString), so the streaming
// wire encoders can stay off the reflect path without drifting from it.
func FuzzAppendJSONStringHTML(f *testing.F) {
	for _, s := range []string{"", "plain", `q"o\o`, "<>&", "\x00\x1f", " ", "\xff", "café"} {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		got := string(appendJSONStringHTML(nil, s))
		if want := core.JSONMarshalString(s); got != want {
			t.Fatalf("appendJSONStringHTML(%q) = %q, want %q", s, got, want)
		}
	})
}

// TestWriteSSEData pins the OpenAI streaming frame shape: "data: <payload>\n\n".
func TestWriteSSEData(t *testing.T) {
	buf := core.NewBuffer()
	writeSSEData(buf, `{"x":1}`)
	if got, want := buf.String(), "data: {\"x\":1}\n\n"; got != want {
		t.Fatalf("writeSSEData = %q, want %q", got, want)
	}
}

// TestWriteSSEEvent pins the Anthropic streaming frame shape:
// "event: <name>\ndata: <payload>\n\n".
func TestWriteSSEEvent(t *testing.T) {
	buf := core.NewBuffer()
	writeSSEEvent(buf, "message_start", `{"y":2}`)
	if got, want := buf.String(), "event: message_start\ndata: {\"y\":2}\n\n"; got != want {
		t.Fatalf("writeSSEEvent = %q, want %q", got, want)
	}
}

// TestWriteNDJSONLine pins the Ollama streaming frame shape: "<payload>\n".
func TestWriteNDJSONLine(t *testing.T) {
	buf := core.NewBuffer()
	writeNDJSONLine(buf, `{"z":3}`)
	if got, want := buf.String(), "{\"z\":3}\n"; got != want {
		t.Fatalf("writeNDJSONLine = %q, want %q", got, want)
	}
}

// TestWriteResponseDeltaFrame pins the /v1/responses per-token delta frame and
// proves the delta is JSON-escaped (HTML-safe) rather than spliced raw.
func TestWriteResponseDeltaFrame(t *testing.T) {
	buf := core.NewBuffer()
	writeResponseDeltaFrame(buf, nil, "a<b")
	want := `data: {"type":"response.output_text.delta","delta":` + core.JSONMarshalString("a<b") + "}\n\n"
	if got := buf.String(); got != want {
		t.Fatalf("writeResponseDeltaFrame = %q, want %q", got, want)
	}
}

// TestWriteOllamaChatFrame pins the /api/chat per-token frame, model and content
// both JSON-escaped into the fixed punctuation.
func TestWriteOllamaChatFrame(t *testing.T) {
	buf := core.NewBuffer()
	writeOllamaChatFrame(buf, nil, "gemma", "hi&bye")
	want := `{"model":` + core.JSONMarshalString("gemma") +
		`,"message":{"role":"assistant","content":` + core.JSONMarshalString("hi&bye") +
		"},\"done\":false}\n"
	if got := buf.String(); got != want {
		t.Fatalf("writeOllamaChatFrame = %q, want %q", got, want)
	}
}

// TestWriteOllamaGenerateFrame pins the /api/generate per-token frame shape.
func TestWriteOllamaGenerateFrame(t *testing.T) {
	buf := core.NewBuffer()
	writeOllamaGenerateFrame(buf, nil, "gemma", "42>0")
	want := `{"model":` + core.JSONMarshalString("gemma") +
		`,"response":` + core.JSONMarshalString("42>0") +
		",\"done\":false}\n"
	if got := buf.String(); got != want {
		t.Fatalf("writeOllamaGenerateFrame = %q, want %q", got, want)
	}
}
