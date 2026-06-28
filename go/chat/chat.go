// SPDX-Licence-Identifier: EUPL-1.2

// Package chat is the canonical chat request/response shape for the serving
// surface (RFC.md §6.1, with multimodal content per §6.12). It is the shared
// vocabulary the model-adjacent concern packages — provider routing, batching,
// streaming, usage, structured output, provider translation — are meant to
// converge on, so one request shape crosses the whole pipeline rather than each
// package inventing its own.
//
// It is a pure-Go type package: no I/O, no dependency beyond the core framework,
// so it stays trivially unit-testable and importable by any concern package
// without pulling in a backend. Heavy logic (routing decisions, wire
// translation, token counting) lives in the concern packages and the layers
// below (§6 "Layer ownership"); this package only carries the shapes.
//
//	req := chat.Request{
//		Model:    "gemma-4-e4b",
//		Messages: []chat.Message{chat.UserText("what is 2+2?")},
//	}
//	if err := req.Validate(); err != nil { return err }
//	for _, model := range req.FallbackChain() { /* route */ }
//
// Convergence note: pkg/modality, pkg/session, and the provider/router packages
// each hold near-identical Role / content-block / message shapes today. They are
// intended to adopt these canonical types over time; this package introduces the
// shared definitions without refactoring the existing ones.
package chat

import core "dappco.re/go"

// Role is the author of a message (§6.1). The wire form is the lower-case
// string; use ParseRole to read caller input and String to emit it.
type Role string

const (
	// System is the system prompt — top-level instructions for the model.
	System Role = "system"
	// Developer is a developer-authored instruction, ranked between system and
	// user on backends that distinguish it (OpenAI's developer role).
	Developer Role = "developer"
	// User is an end-user turn.
	User Role = "user"
	// Assistant is a model-authored turn (may carry tool calls, opaquely).
	Assistant Role = "assistant"
	// Tool is a tool/function result, bound to a prior call via ToolCallID.
	Tool Role = "tool"
)

// String returns the canonical lower-case wire form.
//
//	chat.Assistant.String() == "assistant"
func (r Role) String() string { return string(r) }

// Valid reports whether r is one of the canonical roles. The zero value is not
// valid.
//
//	chat.User.Valid() == true
//	chat.Role("robot").Valid() == false
func (r Role) Valid() bool {
	switch r {
	case System, Developer, User, Assistant, Tool:
		return true
	default:
		return false
	}
}

// ParseRole reads a wire string into a Role, tolerant of surrounding whitespace
// and case (callers pass raw request values). Unknown values error.
//
//	r, err := chat.ParseRole(" ASSISTANT ") // -> Assistant, nil
//	_, err := chat.ParseRole("robot")       // -> error
func ParseRole(s string) (Role, error) {
	r := Role(core.Lower(core.Trim(s)))
	if !r.Valid() {
		return "", core.E("chat", "unknown role: "+s, nil)
	}
	return r, nil
}

// Kind is the content kind of one message block (§6.1, §6.12). Text blocks
// carry a string; media blocks carry inline Data + MIME or a URL + MIME; file
// blocks add a FileName.
type Kind string

const (
	// KindText is a text block.
	KindText Kind = "text"
	// KindImage is an image block (inline Data or URL).
	KindImage Kind = "image"
	// KindAudio is an audio block (inline Data or URL).
	KindAudio Kind = "audio"
	// KindVideo is a video block (inline Data or URL).
	KindVideo Kind = "video"
	// KindFile is a file attachment (inline Data or URL) with a FileName.
	KindFile Kind = "file"
)

// String returns the canonical lower-case wire form.
//
//	chat.KindImage.String() == "image"
func (k Kind) String() string { return string(k) }

// Valid reports whether k is one of the known content kinds. The zero value is
// not valid.
//
//	chat.KindVideo.Valid() == true
//	chat.Kind("hologram").Valid() == false
func (k Kind) Valid() bool {
	switch k {
	case KindText, KindImage, KindAudio, KindVideo, KindFile:
		return true
	default:
		return false
	}
}

// ContentBlock is one part of a message's multimodal content (§6.1, §6.12). A
// text block carries Text; a media block (image / audio / video) carries inline
// Data + MIME or a URL + MIME; a file block adds FileName. CacheControl marks a
// block as a cacheable prefix boundary (§6.11) — e.g. a long system preamble
// prefilled once.
//
//	chat.Text("hello")
//	chat.Image(pngBytes, "image/png")
//	chat.ImageURL("https://cdn/x.png", "image/png")
//	chat.Audio(wavBytes, "audio/wav")
//	chat.File(pdfBytes, "report.pdf", "application/pdf")
type ContentBlock struct {
	Kind         Kind   `json:"kind"`
	Text         string `json:"text,omitempty"`
	Data         []byte `json:"data,omitempty"`
	URL          string `json:"url,omitempty"`
	MIME         string `json:"mime,omitempty"`
	FileName     string `json:"file_name,omitempty"`
	CacheControl bool   `json:"cache_control,omitempty"`
}

// Text builds a text content block.
//
//	b := chat.Text("the answer is 42")
func Text(text string) ContentBlock { return ContentBlock{Kind: KindText, Text: text} }

// Image builds an image block from inline bytes + its MIME type.
//
//	b := chat.Image(pngBytes, "image/png")
func Image(data []byte, mime string) ContentBlock {
	return ContentBlock{Kind: KindImage, Data: data, MIME: mime}
}

// ImageURL builds an image block that references a URL rather than carrying
// inline bytes (some callers and backends pass a link).
//
//	b := chat.ImageURL("https://cdn/x.png", "image/png")
func ImageURL(url, mime string) ContentBlock {
	return ContentBlock{Kind: KindImage, URL: url, MIME: mime}
}

// Audio builds an audio block from inline bytes + its MIME type.
//
//	b := chat.Audio(wavBytes, "audio/wav")
func Audio(data []byte, mime string) ContentBlock {
	return ContentBlock{Kind: KindAudio, Data: data, MIME: mime}
}

// File builds a file-attachment block from inline bytes, a display name, and its
// MIME type.
//
//	b := chat.File(pdfBytes, "report.pdf", "application/pdf")
func File(data []byte, name, mime string) ContentBlock {
	return ContentBlock{Kind: KindFile, Data: data, FileName: name, MIME: mime}
}

// Cached returns a copy of the block with CacheControl set, marking it a
// cacheable prefix boundary (§6.11).
//
//	preamble := chat.Text(longSystemPrompt).Cached()
func (b ContentBlock) Cached() ContentBlock {
	b.CacheControl = true
	return b
}

// IsEmpty reports whether the block carries no payload at all — no text, no
// inline data, and no URL. Used to tell a meaningful block from a placeholder.
//
//	chat.ContentBlock{Kind: chat.KindImage}.IsEmpty() == true
func (b ContentBlock) IsEmpty() bool {
	return b.Text == "" && len(b.Data) == 0 && b.URL == ""
}

// Message is one chat turn: a Role, an ordered list of content blocks, and —
// for a Tool reply — the ToolCallID it answers (§6.1, §6.4).
//
//	chat.Message{Role: chat.System, Content: []chat.ContentBlock{chat.Text("be helpful")}}
//	chat.Message{Role: chat.Tool, ToolCallID: "call_1", Content: []chat.ContentBlock{chat.Text("sunny")}}
type Message struct {
	Role       Role           `json:"role"`
	Content    []ContentBlock `json:"content,omitempty"`
	ToolCallID string         `json:"tool_call_id,omitempty"`
}

// Text returns the concatenated text of the message's text blocks, in order,
// skipping media blocks. A message with no text blocks yields "".
//
//	m := chat.Message{Content: []chat.ContentBlock{chat.Text("a"), img, chat.Text("b")}}
//	m.Text() == "ab"
func (m Message) Text() string {
	// Single pass to size the output: sum the text-block lengths and
	// remember the first one. The common no-text and single-text cases
	// then return without allocating, and the many-block case writes
	// straight into one pre-sized Builder — the earlier []string + Join
	// also allocated an intermediate slice for the parts.
	var (
		n     int
		count int
		first string
	)
	for _, b := range m.Content {
		if b.Kind == KindText {
			if count == 0 {
				first = b.Text
			}
			n += len(b.Text)
			count++
		}
	}
	switch count {
	case 0:
		return ""
	case 1:
		return first
	}
	var sb core.Builder
	sb.Grow(n)
	for _, b := range m.Content {
		if b.Kind == KindText {
			sb.WriteString(b.Text)
		}
	}
	return sb.String()
}

// UserText is the common single-text-message constructor.
//
//	m := chat.UserText("what is 2+2?")
func UserText(text string) Message {
	return Message{Role: User, Content: []ContentBlock{Text(text)}}
}

// Request is the canonical chat request (§6.1): the OpenAI fields plus the
// OpenRouter routing extensions the inference stack serves. The Tools / ToolChoice fields are
// opaque (any) so this package never imports pkg/tools — a router resolves them.
//
//	req := chat.Request{Model: "gemma-4-e4b", Messages: []chat.Message{chat.UserText("hi")}}
//	err := req.Validate()
type Request struct {
	// Model is the primary model; Models is an ordered fallback list tried in
	// turn (§6.2). At least one of the two must be set.
	Model  string   `json:"model,omitempty"`
	Models []string `json:"models,omitempty"`

	Messages []Message `json:"messages"`

	// Sampling (§6.1). TopK / MinP are local-model extensions.
	Temperature      float64  `json:"temperature,omitempty"`
	TopP             float64  `json:"top_p,omitempty"`
	TopK             float64  `json:"top_k,omitempty"`
	MinP             float64  `json:"min_p,omitempty"`
	MaxTokens        int      `json:"max_tokens,omitempty"`
	Stop             []string `json:"stop,omitempty"`
	Seed             int      `json:"seed,omitempty"`
	FrequencyPenalty float64  `json:"frequency_penalty,omitempty"`
	PresencePenalty  float64  `json:"presence_penalty,omitempty"`

	// Tools / ToolChoice are opaque to keep this package import-light (§6.4);
	// the router/translation layer types them.
	Tools      any `json:"tools,omitempty"`
	ToolChoice any `json:"tool_choice,omitempty"`

	// ResponseFormat selects structured output — "", "text", "json_object",
	// "json_schema", "grammar", or "python" (§6.3, §6.15).
	ResponseFormat string `json:"response_format,omitempty"`
	// Reasoning is the reasoning effort for reasoning models — e.g. "low",
	// "medium", "high" (§6.1).
	Reasoning string `json:"reasoning,omitempty"`

	Stream    bool              `json:"stream,omitempty"`
	SessionID string            `json:"session_id,omitempty"`
	Metadata  map[string]string `json:"metadata,omitempty"`
	User      string            `json:"user,omitempty"`
}

// PrimaryModel is the first model the router tries: the Model field when set,
// else the first usable entry of Models. Whitespace-only entries are skipped and
// the result is trimmed. Returns "" when neither is set.
//
//	chat.Request{Model: "a"}.PrimaryModel()              // "a"
//	chat.Request{Models: []string{"x", "y"}}.PrimaryModel() // "x"
func (r Request) PrimaryModel() string {
	// The primary is the first non-blank, trimmed entry scanning Model
	// then Models in order — the head of FallbackChain. Compute it
	// directly so routing's per-request lookup builds no chain slice (and
	// core.Trim returns a sub-string, so this allocates nothing).
	if m := core.Trim(r.Model); m != "" {
		return m
	}
	for _, raw := range r.Models {
		if m := core.Trim(raw); m != "" {
			return m
		}
	}
	return ""
}

// FallbackChain is the ordered, de-duplicated list of models the router tries:
// the primary Model first (when set), then Models in order. Whitespace-only
// entries are dropped and each entry is trimmed, so a malformed request yields a
// clean chain rather than blank candidates (§6.2).
//
//	chat.Request{Model: "a", Models: []string{"a", "b"}}.FallbackChain() // ["a", "b"]
func (r Request) FallbackChain() []string {
	out := make([]string, 0, 1+len(r.Models))
	seen := make(map[string]bool, 1+len(r.Models))
	add := func(raw string) {
		m := core.Trim(raw)
		if m == "" || seen[m] {
			return
		}
		seen[m] = true
		out = append(out, m)
	}
	add(r.Model)
	for _, m := range r.Models {
		add(m)
	}
	return out
}

// Validate checks the request is well-formed before routing (§6.1):
//   - a model or a models list must be present (a usable, non-blank one),
//   - there must be at least one message,
//   - every message role must be a canonical Role,
//   - a Tool message must carry a ToolCallID, and only a Tool message may.
//
// Returns a core.E("chat", …) on the first violation.
//
//	if err := req.Validate(); err != nil { return err }
func (r Request) Validate() error {
	if r.PrimaryModel() == "" {
		return core.E("chat", "request needs a model or models list", nil)
	}
	if len(r.Messages) == 0 {
		return core.E("chat", "request needs at least one message", nil)
	}
	for i, m := range r.Messages {
		if !m.Role.Valid() {
			return core.E("chat", "message "+core.Itoa(i)+" has invalid role: "+m.Role.String(), nil)
		}
		if m.Role == Tool && m.ToolCallID == "" {
			return core.E("chat", "tool message "+core.Itoa(i)+" needs a tool_call_id", nil)
		}
		if m.Role != Tool && m.ToolCallID != "" {
			return core.E("chat", "non-tool message "+core.Itoa(i)+" must not set tool_call_id", nil)
		}
	}
	return nil
}

// Response is the canonical chat response (§6.1): the assistant message(s), a
// flattened text body, the finish reason, and opaque usage (typed by the usage
// package, §6.6) to keep this package import-light.
//
//	resp := chat.Response{
//		Messages:     []chat.Message{{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text("4")}}},
//		Text:         "4",
//		FinishReason: "stop",
//	}
type Response struct {
	Messages     []Message `json:"messages,omitempty"`
	Text         string    `json:"text,omitempty"`
	FinishReason string    `json:"finish_reason,omitempty"`
	Usage        any       `json:"usage,omitempty"`
}
