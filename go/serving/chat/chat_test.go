// SPDX-Licence-Identifier: EUPL-1.2

package chat

import core "dappco.re/go"

// --- Role: the canonical message author enum (§6.1) ---

func TestChat_Role_Good(t *core.T) {
	// Each canonical wire role parses and round-trips through String.
	for _, want := range []Role{System, Developer, User, Assistant, Tool} {
		got, err := ParseRole(want.String())
		core.AssertNoError(t, err)
		core.AssertEqual(t, want, got)
		core.AssertTrue(t, got.Valid(), "canonical role is valid")
	}
	// Parsing is whitespace- and case-tolerant (raw request values).
	r, err := ParseRole("  ASSISTANT ")
	core.AssertNoError(t, err)
	core.AssertEqual(t, Assistant, r)
}

func TestChat_Role_Bad(t *core.T) {
	// An unknown role is rejected, not silently coerced.
	_, err := ParseRole("robot")
	core.AssertError(t, err, "unknown role")
	core.AssertFalse(t, Role("robot").Valid(), "unknown role invalid")
}

func TestChat_Role_Ugly(t *core.T) {
	// The zero value is not a valid role, and empty input is rejected.
	core.AssertFalse(t, Role("").Valid(), "empty role invalid")
	_, err := ParseRole("   ")
	core.AssertError(t, err, "unknown role")
}

// --- Content: build content blocks and concatenate text (§6.1, §6.12) ---

func TestChat_Content_Good(t *core.T) {
	// Constructors set the right kind + payload.
	txt := Text("hello")
	core.AssertEqual(t, KindText, txt.Kind)
	core.AssertEqual(t, "hello", txt.Text)

	img := Image([]byte{0x89, 0x50}, "image/png")
	core.AssertEqual(t, KindImage, img.Kind)
	core.AssertEqual(t, "image/png", img.MIME)
	core.AssertEqual(t, 2, len(img.Data))

	iurl := ImageURL("https://cdn.example/x.png", "image/png")
	core.AssertEqual(t, KindImage, iurl.Kind)
	core.AssertEqual(t, "https://cdn.example/x.png", iurl.URL)

	aud := Audio([]byte{0x01, 0x02, 0x03}, "audio/wav")
	core.AssertEqual(t, KindAudio, aud.Kind)
	core.AssertEqual(t, "audio/wav", aud.MIME)

	f := File([]byte{0x25, 0x50, 0x44, 0x46}, "report.pdf", "application/pdf")
	core.AssertEqual(t, KindFile, f.Kind)
	core.AssertEqual(t, "report.pdf", f.FileName)
	core.AssertEqual(t, "application/pdf", f.MIME)
	core.AssertEqual(t, 4, len(f.Data))

	// CacheControl is opt-in and off by default.
	core.AssertFalse(t, txt.CacheControl, "cache-control off by default")
	cached := Text("system preamble").Cached()
	core.AssertTrue(t, cached.CacheControl, "Cached() sets cache-control")

	// Message.Text concatenates only the text blocks, in order.
	m := Message{
		Role: User,
		Content: []ContentBlock{
			Text("see "), img, Text("and "), aud, Text("done"),
		},
	}
	core.AssertEqual(t, "see and done", m.Text(), "text blocks concatenated, media skipped")

	// UserText is the common single-text-message constructor.
	u := UserText("what is 2+2?")
	core.AssertEqual(t, User, u.Role)
	core.AssertEqual(t, 1, len(u.Content))
	core.AssertEqual(t, "what is 2+2?", u.Text())
}

func TestChat_Content_Bad(t *core.T) {
	// A block with no payload is empty; a populated one is not.
	core.AssertTrue(t, ContentBlock{Kind: KindImage}.IsEmpty(), "no payload is empty")
	core.AssertFalse(t, Text("x").IsEmpty())
	core.AssertFalse(t, ImageURL("https://h/x.png", "image/png").IsEmpty(), "URL counts as payload")
	core.AssertFalse(t, File([]byte{1}, "a.bin", "application/octet-stream").IsEmpty())

	// A media-only message flattens to empty text, not a panic.
	m := Message{Role: Assistant, Content: []ContentBlock{
		Image([]byte{1}, "image/png"),
		Audio([]byte{2}, "audio/wav"),
	}}
	core.AssertEqual(t, "", m.Text(), "no text blocks -> empty body")
}

func TestChat_Content_Ugly(t *core.T) {
	// A message with no content flattens to empty text.
	core.AssertEqual(t, "", Message{Role: User}.Text())
	core.AssertEqual(t, "", Message{Role: User, Content: nil}.Text())

	// Empty text blocks contribute nothing but do not break concatenation.
	m := Message{Role: User, Content: []ContentBlock{Text(""), Text("body"), Text("")}}
	core.AssertEqual(t, "body", m.Text())

	// Kind reports its own wire string and validity.
	core.AssertEqual(t, "image", KindImage.String())
	core.AssertTrue(t, KindVideo.Valid(), "video is a known kind")
	core.AssertFalse(t, Kind("").Valid(), "empty kind invalid")
	core.AssertFalse(t, Kind("hologram").Valid())
}

// --- Validate: the request guard (§6.1) ---

func validReq() Request {
	return Request{
		Model:    "gemma-4-e4b",
		Messages: []Message{UserText("hi")},
	}
}

func TestChat_Validate_Good(t *core.T) {
	core.AssertNoError(t, validReq().Validate())

	// models-only (fallback chain, no primary) is valid.
	r := Request{
		Models:   []string{"local-metal/gemma-4-31b", "nim/qwen"},
		Messages: []Message{UserText("hi")},
	}
	core.AssertNoError(t, r.Validate())

	// A full multi-role transcript with a tool result validates.
	r = Request{
		Model: "gemma-4-e4b",
		Messages: []Message{
			{Role: System, Content: []ContentBlock{Text("be helpful")}},
			{Role: Developer, Content: []ContentBlock{Text("use UK English")}},
			UserText("weather?"),
			{Role: Assistant, Content: []ContentBlock{Text("checking")}},
			{Role: Tool, ToolCallID: "call_1", Content: []ContentBlock{Text("sunny")}},
		},
	}
	core.AssertNoError(t, r.Validate())
}

func TestChat_Validate_Bad(t *core.T) {
	// Neither model nor models -> error.
	r := Request{Messages: []Message{UserText("hi")}}
	core.AssertError(t, r.Validate(), "model")

	// No messages -> error.
	r = Request{Model: "m"}
	core.AssertError(t, r.Validate(), "at least one message")

	// A message carrying an unknown role -> error.
	r = validReq()
	r.Messages = append(r.Messages, Message{Role: Role("robot"), Content: []ContentBlock{Text("x")}})
	core.AssertError(t, r.Validate(), "role")
}

func TestChat_Validate_Ugly(t *core.T) {
	// A tool message without a ToolCallID is invalid (can't bind the result).
	r := validReq()
	r.Messages = append(r.Messages, Message{Role: Tool, Content: []ContentBlock{Text("result")}})
	core.AssertError(t, r.Validate(), "tool_call_id")

	// A non-tool message that sets ToolCallID is invalid (only tool replies bind).
	r = Request{Model: "m", Messages: []Message{
		{Role: User, ToolCallID: "call_x", Content: []ContentBlock{Text("hi")}},
	}}
	core.AssertError(t, r.Validate(), "tool_call_id")

	// A message with an empty role is invalid.
	r = Request{Model: "m", Messages: []Message{{Content: []ContentBlock{Text("hi")}}}}
	core.AssertError(t, r.Validate(), "role")

	// A blank model string with no models list is rejected (whitespace only).
	r = Request{Model: "   ", Messages: []Message{UserText("hi")}}
	core.AssertError(t, r.Validate(), "model")

	// An assistant message with empty content but a tool-less body is allowed
	// (assistants may emit tool calls carried opaquely), so it must validate.
	r = Request{Model: "m", Messages: []Message{
		UserText("go"),
		{Role: Assistant},
	}}
	core.AssertNoError(t, r.Validate())
}

// --- FallbackChain / PrimaryModel: routing helpers (§6.1, §6.2) ---

func TestChat_FallbackChain_Good(t *core.T) {
	// Primary model leads, models list appended in order, deduped.
	r := Request{
		Model:  "a",
		Models: []string{"b", "c"},
	}
	core.AssertEqual(t, "a", r.PrimaryModel())
	chain := r.FallbackChain()
	core.AssertEqual(t, 3, len(chain))
	core.AssertEqual(t, "a", chain[0])
	core.AssertEqual(t, "b", chain[1])
	core.AssertEqual(t, "c", chain[2])

	// models-only: first entry is the primary.
	r = Request{Models: []string{"x", "y"}}
	core.AssertEqual(t, "x", r.PrimaryModel())
	chain = r.FallbackChain()
	core.AssertEqual(t, 2, len(chain))
	core.AssertEqual(t, "x", chain[0])
	core.AssertEqual(t, "y", chain[1])
}

func TestChat_FallbackChain_Bad(t *core.T) {
	// Empty request: no primary, empty chain (not nil-deref).
	r := Request{}
	core.AssertEqual(t, "", r.PrimaryModel())
	core.AssertEqual(t, 0, len(r.FallbackChain()))
}

func TestChat_FallbackChain_Ugly(t *core.T) {
	// Duplicates across model + models collapse, first-seen order kept.
	r := Request{Model: "a", Models: []string{"a", "b", "b", "a"}}
	chain := r.FallbackChain()
	core.AssertEqual(t, 2, len(chain), "duplicates removed")
	core.AssertEqual(t, "a", chain[0])
	core.AssertEqual(t, "b", chain[1])

	// Whitespace-only / empty entries are skipped, real ones trimmed.
	r = Request{Model: "  ", Models: []string{"", "  spaced  ", "b"}}
	core.AssertEqual(t, "spaced", r.PrimaryModel(), "first real entry trimmed")
	chain = r.FallbackChain()
	core.AssertEqual(t, 2, len(chain))
	core.AssertEqual(t, "spaced", chain[0])
	core.AssertEqual(t, "b", chain[1])
}
