// SPDX-Licence-Identifier: EUPL-1.2

package chat

import core "dappco.re/go"

// --- ParseRole: wire-string reader, tolerant of case + whitespace (§6.1) ---

func TestChat_ParseRole_Good(t *core.T) {
	// Each canonical wire role parses and round-trips through String.
	for _, want := range []Role{System, Developer, User, Assistant, Tool} {
		got, err := ParseRole(want.String())
		core.AssertNoError(t, err)
		core.AssertEqual(t, want, got)
	}
	// Parsing is whitespace- and case-tolerant (raw request values).
	r, err := ParseRole("  ASSISTANT ")
	core.AssertNoError(t, err)
	core.AssertEqual(t, Assistant, r)
}

func TestChat_ParseRole_Bad(t *core.T) {
	// An unknown role is rejected, not silently coerced, and the error
	// names the offending value for a useful diagnostic.
	_, err := ParseRole("robot")
	core.AssertError(t, err, "unknown role")
	core.AssertError(t, err, "robot")
}

func TestChat_ParseRole_Ugly(t *core.T) {
	// Whitespace-only input has nothing to trim to and is rejected, same
	// as truly empty input.
	_, err := ParseRole("   ")
	core.AssertError(t, err, "unknown role")
	r, err2 := ParseRole("")
	core.AssertError(t, err2, "unknown role")
	core.AssertEqual(t, Role(""), r)
}

// --- Role.String: canonical lower-case wire form (§6.1) ---

func TestChat_Role_String_Good(t *core.T) {
	core.AssertEqual(t, "assistant", Assistant.String())
	core.AssertEqual(t, "tool", Tool.String())
	core.AssertEqual(t, "system", System.String())
}

func TestChat_Role_String_Bad(t *core.T) {
	// String never validates -- an unknown role still stringifies verbatim,
	// for any garbage input, not just one example.
	core.AssertEqual(t, "robot", Role("robot").String())
	core.AssertEqual(t, "ROBOT", Role("ROBOT").String(), "no case-folding")
	core.AssertEqual(t, "123", Role("123").String(), "no character-class check")
}

func TestChat_Role_String_Ugly(t *core.T) {
	// The zero value stringifies to "", not a panic; String applies no
	// trimming or normalisation at all -- that is ParseRole's job.
	core.AssertEqual(t, "", Role("").String())
	core.AssertEqual(t, "  ", Role("  ").String(), "no trimming in String")
	core.AssertEqual(t, "a\nb", Role("a\nb").String(), "no character filtering")
}

// --- Role.Valid: canonical-role guard (§6.1) ---

func TestChat_Role_Valid_Good(t *core.T) {
	for _, r := range []Role{System, Developer, User, Assistant, Tool} {
		core.AssertTrue(t, r.Valid(), "canonical role is valid")
	}
}

func TestChat_Role_Valid_Bad(t *core.T) {
	core.AssertFalse(t, Role("robot").Valid(), "unknown role invalid")
	core.AssertFalse(t, Role("System").Valid(), "case-sensitive: capitalised is not canonical")
	core.AssertFalse(t, Role("assistant ").Valid(), "trailing space breaks the match")
}

func TestChat_Role_Valid_Ugly(t *core.T) {
	core.AssertFalse(t, Role("").Valid(), "empty role invalid")
	core.AssertFalse(t, Role(" ").Valid(), "whitespace-only is not canonical")
	core.AssertFalse(t, Role("\t").Valid(), "control whitespace is not canonical")
}

// --- Kind.String: canonical lower-case wire form (§6.1, §6.12) ---

func TestChat_Kind_String_Good(t *core.T) {
	core.AssertEqual(t, "image", KindImage.String())
	core.AssertEqual(t, "file", KindFile.String())
	core.AssertEqual(t, "text", KindText.String())
}

func TestChat_Kind_String_Bad(t *core.T) {
	// String never validates -- an unknown kind still stringifies verbatim.
	core.AssertEqual(t, "hologram", Kind("hologram").String())
	core.AssertEqual(t, "HOLOGRAM", Kind("HOLOGRAM").String(), "no case-folding")
	core.AssertEqual(t, "123", Kind("123").String(), "no character-class check")
}

func TestChat_Kind_String_Ugly(t *core.T) {
	core.AssertEqual(t, "", Kind("").String())
	core.AssertEqual(t, "  ", Kind("  ").String(), "no trimming")
	core.AssertEqual(t, "a\tb", Kind("a\tb").String(), "no character filtering")
}

// --- Kind.Valid: known-content-kind guard (§6.1, §6.12) ---

func TestChat_Kind_Valid_Good(t *core.T) {
	for _, k := range []Kind{KindText, KindImage, KindAudio, KindVideo, KindFile} {
		core.AssertTrue(t, k.Valid(), "known kind is valid")
	}
}

func TestChat_Kind_Valid_Bad(t *core.T) {
	core.AssertFalse(t, Kind("hologram").Valid())
	core.AssertFalse(t, Kind("Image").Valid(), "case-sensitive: capitalised is not canonical")
	core.AssertFalse(t, Kind("txt").Valid(), "close-but-not-canonical is rejected")
}

func TestChat_Kind_Valid_Ugly(t *core.T) {
	core.AssertFalse(t, Kind("").Valid(), "empty kind invalid")
	core.AssertFalse(t, Kind(" ").Valid(), "whitespace is not a canonical kind")
	core.AssertFalse(t, Kind("text ").Valid(), "trailing space breaks the match")
}

// --- Text: text content-block constructor (§6.1, §6.12) ---

func TestChat_Text_Good(t *core.T) {
	b := Text("hello")
	core.AssertEqual(t, KindText, b.Kind)
	core.AssertEqual(t, "hello", b.Text)
	core.AssertFalse(t, b.IsEmpty())
}

func TestChat_Text_Bad(t *core.T) {
	// No guard against a blank payload -- caller checks IsEmpty.
	b := Text("")
	core.AssertEqual(t, KindText, b.Kind)
	core.AssertTrue(t, b.IsEmpty(), "blank text has no payload")
}

func TestChat_Text_Ugly(t *core.T) {
	// Unicode and control bytes pass through untouched.
	b := Text("emoji \U0001F44D\x00tab\t")
	core.AssertEqual(t, "emoji \U0001F44D\x00tab\t", b.Text)
	core.AssertEqual(t, KindText, b.Kind, "kind unaffected by payload content")
}

// --- Image: inline-bytes image block constructor (§6.1, §6.12) ---

func TestChat_Image_Good(t *core.T) {
	b := Image([]byte{0x89, 0x50}, "image/png")
	core.AssertEqual(t, KindImage, b.Kind)
	core.AssertEqual(t, "image/png", b.MIME)
	core.AssertEqual(t, 2, len(b.Data))
	core.AssertFalse(t, b.IsEmpty())
}

func TestChat_Image_Bad(t *core.T) {
	// No inline bytes and no URL -- constructs but carries no payload.
	b := Image(nil, "image/png")
	core.AssertEqual(t, "image/png", b.MIME, "MIME is kept even without data")
	core.AssertTrue(t, b.IsEmpty(), "no data and no URL is empty")
}

func TestChat_Image_Ugly(t *core.T) {
	// No MIME validation -- an empty MIME still constructs.
	b := Image([]byte{1}, "")
	core.AssertEqual(t, "", b.MIME)
	core.AssertFalse(t, b.IsEmpty(), "data alone is a payload")
}

// --- ImageURL: URL-referenced image block constructor (§6.1, §6.12) ---

func TestChat_ImageURL_Good(t *core.T) {
	b := ImageURL("https://cdn.example/x.png", "image/png")
	core.AssertEqual(t, KindImage, b.Kind)
	core.AssertEqual(t, "https://cdn.example/x.png", b.URL)
	core.AssertFalse(t, b.IsEmpty(), "URL counts as payload")
}

func TestChat_ImageURL_Bad(t *core.T) {
	// A blank URL and no inline data means there is no payload at all.
	b := ImageURL("", "image/png")
	core.AssertEqual(t, "image/png", b.MIME, "MIME kept even with a blank URL")
	core.AssertTrue(t, b.IsEmpty(), "blank URL has no payload")
}

func TestChat_ImageURL_Ugly(t *core.T) {
	// No trimming -- a whitespace-only URL is stored verbatim and still
	// counts as non-empty (unlike ParseRole, which does trim).
	b := ImageURL("   ", "image/png")
	core.AssertEqual(t, "   ", b.URL)
	core.AssertFalse(t, b.IsEmpty())
}

// --- Audio: inline-bytes audio block constructor (§6.1, §6.12) ---

func TestChat_Audio_Good(t *core.T) {
	b := Audio([]byte{0x01, 0x02, 0x03}, "audio/wav")
	core.AssertEqual(t, KindAudio, b.Kind)
	core.AssertEqual(t, "audio/wav", b.MIME)
	core.AssertFalse(t, b.IsEmpty())
}

func TestChat_Audio_Bad(t *core.T) {
	// No inline bytes -- constructs but carries no payload.
	b := Audio(nil, "audio/wav")
	core.AssertEqual(t, "audio/wav", b.MIME, "MIME kept even without data")
	core.AssertTrue(t, b.IsEmpty(), "no data is empty")
}

func TestChat_Audio_Ugly(t *core.T) {
	// No MIME validation -- an empty MIME still constructs.
	b := Audio([]byte{1}, "")
	core.AssertEqual(t, "", b.MIME)
	core.AssertFalse(t, b.IsEmpty())
}

// --- File: file-attachment block constructor (§6.1, §6.12) ---

func TestChat_File_Good(t *core.T) {
	b := File([]byte{0x25, 0x50, 0x44, 0x46}, "report.pdf", "application/pdf")
	core.AssertEqual(t, KindFile, b.Kind)
	core.AssertEqual(t, "report.pdf", b.FileName)
	core.AssertEqual(t, "application/pdf", b.MIME)
	core.AssertEqual(t, 4, len(b.Data))
	core.AssertFalse(t, b.IsEmpty())
}

func TestChat_File_Bad(t *core.T) {
	// No inline bytes -- constructs but carries no payload, even with a name.
	b := File(nil, "a.bin", "application/octet-stream")
	core.AssertEqual(t, "a.bin", b.FileName)
	core.AssertTrue(t, b.IsEmpty(), "no data is empty even with a name")
}

func TestChat_File_Ugly(t *core.T) {
	// IsEmpty only looks at Text/Data/URL -- a FileName alone never counts
	// as a payload, so a name-only block is (perhaps surprisingly) empty.
	b := File(nil, "orphan.bin", "")
	core.AssertEqual(t, "orphan.bin", b.FileName)
	core.AssertTrue(t, b.IsEmpty())
}

// --- ContentBlock.Cached: prefix-cache boundary marker (§6.11) ---

func TestChat_ContentBlock_Cached_Good(t *core.T) {
	b := Text("system preamble")
	core.AssertFalse(t, b.CacheControl, "cache-control off by default")
	cached := b.Cached()
	core.AssertTrue(t, cached.CacheControl, "Cached() sets cache-control")
	core.AssertEqual(t, b.Text, cached.Text, "payload unchanged")
}

func TestChat_ContentBlock_Cached_Bad(t *core.T) {
	// Value receiver -- calling Cached() does not mutate the original.
	b := Text("x")
	_ = b.Cached()
	core.AssertFalse(t, b.CacheControl, "original left untouched")
}

func TestChat_ContentBlock_Cached_Ugly(t *core.T) {
	// Idempotent: caching an already-cached block stays cached.
	b := Text("x").Cached().Cached()
	core.AssertTrue(t, b.CacheControl)
	core.AssertEqual(t, "x", b.Text, "payload untouched by repeated caching")
}

// --- ContentBlock.IsEmpty: no-payload guard (§6.1, §6.12) ---

func TestChat_ContentBlock_IsEmpty_Good(t *core.T) {
	core.AssertFalse(t, Text("x").IsEmpty())
	core.AssertFalse(t, Image([]byte{1}, "image/png").IsEmpty())
	core.AssertFalse(t, Audio([]byte{1}, "audio/wav").IsEmpty())
}

func TestChat_ContentBlock_IsEmpty_Bad(t *core.T) {
	core.AssertTrue(t, ContentBlock{Kind: KindImage}.IsEmpty(), "no payload is empty")
	core.AssertTrue(t, ContentBlock{Kind: KindText}.IsEmpty(), "same for a bare text block")
	core.AssertTrue(t, ContentBlock{}.IsEmpty(), "zero value is empty")
}

func TestChat_ContentBlock_IsEmpty_Ugly(t *core.T) {
	// URL-only counts as a payload even with no inline bytes or text.
	core.AssertFalse(t, ImageURL("https://h/x.png", "image/png").IsEmpty(), "URL counts as payload")
	// A FileName alone does not: IsEmpty only inspects Text/Data/URL.
	core.AssertTrue(t, File(nil, "name.bin", "").IsEmpty(), "FileName is not a payload field")
	core.AssertFalse(t, ContentBlock{Text: " "}.IsEmpty(), "even whitespace-only text counts")
}

// --- Message.Text: ordered text-block concatenation (§6.1) ---

func TestChat_Message_Text_Good(t *core.T) {
	m := Message{
		Role: User,
		Content: []ContentBlock{
			Text("see "), Image([]byte{1}, "image/png"), Text("and "), Audio([]byte{2}, "audio/wav"), Text("done"),
		},
	}
	core.AssertEqual(t, "see and done", m.Text(), "text blocks concatenated, media skipped")
}

func TestChat_Message_Text_Bad(t *core.T) {
	// A media-only message flattens to empty text, not a panic.
	m := Message{Role: Assistant, Content: []ContentBlock{
		Image([]byte{1}, "image/png"),
		Audio([]byte{2}, "audio/wav"),
	}}
	core.AssertEqual(t, "", m.Text(), "no text blocks -> empty body")
}

func TestChat_Message_Text_Ugly(t *core.T) {
	// A message with no content flattens to empty text.
	core.AssertEqual(t, "", Message{Role: User}.Text())
	core.AssertEqual(t, "", Message{Role: User, Content: nil}.Text())

	// Empty text blocks contribute nothing but do not break concatenation.
	m := Message{Role: User, Content: []ContentBlock{Text(""), Text("body"), Text("")}}
	core.AssertEqual(t, "body", m.Text())
}

// --- UserText: single-text-message constructor (§6.1) ---

func TestChat_UserText_Good(t *core.T) {
	u := UserText("what is 2+2?")
	core.AssertEqual(t, User, u.Role)
	core.AssertEqual(t, 1, len(u.Content))
	core.AssertEqual(t, "what is 2+2?", u.Text())
}

func TestChat_UserText_Bad(t *core.T) {
	// No validation -- an empty string still builds a (empty) text block.
	u := UserText("")
	core.AssertEqual(t, User, u.Role)
	core.AssertTrue(t, u.Content[0].IsEmpty())
}

func TestChat_UserText_Ugly(t *core.T) {
	// Unicode passes through untouched.
	u := UserText("café \U0001F44D")
	core.AssertEqual(t, "café \U0001F44D", u.Text())
	core.AssertEqual(t, User, u.Role, "role unaffected by payload content")
}

// --- Request.PrimaryModel: first-model-tried resolver (§6.1, §6.2) ---

func TestChat_Request_PrimaryModel_Good(t *core.T) {
	core.AssertEqual(t, "a", Request{Model: "a"}.PrimaryModel())
	core.AssertEqual(t, "x", Request{Models: []string{"x", "y"}}.PrimaryModel())
	core.AssertEqual(t, "a", Request{Model: "a", Models: []string{"z"}}.PrimaryModel(), "Model wins over Models when both set")
}

func TestChat_Request_PrimaryModel_Bad(t *core.T) {
	core.AssertEqual(t, "", Request{}.PrimaryModel(), "neither field set")
	core.AssertEqual(t, "", Request{Models: nil}.PrimaryModel(), "nil Models is the same as unset")
	core.AssertEqual(t, "", Request{Model: "", Models: []string{}}.PrimaryModel(), "empty slice is the same as unset")
}

func TestChat_Request_PrimaryModel_Ugly(t *core.T) {
	// Blank Model falls through to the first real, trimmed Models entry.
	r := Request{Model: "  ", Models: []string{"", "  spaced  ", "b"}}
	core.AssertEqual(t, "spaced", r.PrimaryModel())
	core.AssertEqual(t, "b", Request{Models: []string{"  ", "b"}}.PrimaryModel(), "skips whitespace-only entries")
}

// --- Request.FallbackChain: ordered, de-duplicated model list (§6.1, §6.2) ---

func TestChat_Request_FallbackChain_Good(t *core.T) {
	r := Request{Model: "a", Models: []string{"b", "c"}}
	chain := r.FallbackChain()
	core.AssertEqual(t, 3, len(chain))
	core.AssertEqual(t, "a", chain[0])
	core.AssertEqual(t, "b", chain[1])
	core.AssertEqual(t, "c", chain[2])
}

func TestChat_Request_FallbackChain_Bad(t *core.T) {
	// Empty request: no primary, empty chain -- not a nil-deref.
	core.AssertEqual(t, 0, len(Request{}.FallbackChain()))
	core.AssertEqual(t, 0, len(Request{Models: []string{}}.FallbackChain()))
	core.AssertEqual(t, 0, len(Request{Model: "  "}.FallbackChain()), "whitespace-only model yields nothing")
}

func TestChat_Request_FallbackChain_Ugly(t *core.T) {
	// Duplicates across Model + Models collapse, first-seen order kept.
	r := Request{Model: "a", Models: []string{"a", "b", "b", "a"}}
	chain := r.FallbackChain()
	core.AssertEqual(t, 2, len(chain), "duplicates removed")
	core.AssertEqual(t, "a", chain[0])
	core.AssertEqual(t, "b", chain[1])

	// Whitespace-only / empty entries are skipped, real ones trimmed.
	r = Request{Model: "  ", Models: []string{"", "  spaced  ", "b"}}
	chain = r.FallbackChain()
	core.AssertEqual(t, 2, len(chain))
	core.AssertEqual(t, "spaced", chain[0])
	core.AssertEqual(t, "b", chain[1])
}

// --- Request.Validate: the request well-formedness guard (§6.1) ---

func validReq() Request {
	return Request{
		Model:    "gemma-4-e4b",
		Messages: []Message{UserText("hi")},
	}
}

func TestChat_Request_Validate_Good(t *core.T) {
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

func TestChat_Request_Validate_Bad(t *core.T) {
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

func TestChat_Request_Validate_Ugly(t *core.T) {
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
