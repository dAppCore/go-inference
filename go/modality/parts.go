// SPDX-Licence-Identifier: EUPL-1.2

package modality

import core "dappco.re/go"

// Kind is the content kind of one assistant output part — the part-level
// counterpart of a Modality. Image and audio parts carry a payload + MIME type;
// text parts carry a string.
type Kind string

const (
	KindText  Kind = "text"
	KindImage Kind = "image"
	KindAudio Kind = "audio"
)

// Role is the author of a message. Assembled output is always the assistant.
type Role string

const (
	RoleAssistant Role = "assistant"
)

// ContentPart is one part of an assistant's multimodal output (§6.1). A text
// part carries Text; an image part carries inline Data + MIME or a URL + MIME;
// an audio part carries Data + MIME. Tokens / Units are the backend-reported
// per-part accounting that feeds usage (§6.6) — left zero when the backend does
// not report them.
//
//	TextPart("hello")
//	ImagePart(pngBytes, "image/png")
//	AudioPart(wavBytes, "audio/wav")
type ContentPart struct {
	Kind   Kind   `json:"kind"`
	Text   string `json:"text,omitempty"`
	Data   []byte `json:"data,omitempty"`
	URL    string `json:"url,omitempty"`
	MIME   string `json:"mime,omitempty"`
	Tokens int    `json:"tokens,omitempty"` // backend-reported token count for this part
	Units  int    `json:"units,omitempty"`  // backend-reported unit count (e.g. images, seconds)
}

// TextPart builds a text output part.
//
//	p := modality.TextPart("the answer is 42")
func TextPart(text string) ContentPart {
	return ContentPart{Kind: KindText, Text: text}
}

// ImagePart builds an image output part from inline bytes + its MIME type.
//
//	p := modality.ImagePart(pngBytes, "image/png")
func ImagePart(data []byte, mime string) ContentPart {
	return ContentPart{Kind: KindImage, Data: data, MIME: mime}
}

// ImageURLPart builds an image output part that references a URL rather than
// carrying inline bytes (some backends return a link).
//
//	p := modality.ImageURLPart("https://cdn/x.png", "image/png")
func ImageURLPart(url, mime string) ContentPart {
	return ContentPart{Kind: KindImage, URL: url, MIME: mime}
}

// AudioPart builds an audio output part from inline bytes + its MIME type.
//
//	p := modality.AudioPart(wavBytes, "audio/wav")
func AudioPart(data []byte, mime string) ContentPart {
	return ContentPart{Kind: KindAudio, Data: data, MIME: mime}
}

// IsEmpty reports whether the part carries no payload at all — no text, no
// inline data, and no URL. Used to tell a meaningful part from a placeholder.
//
//	modality.ContentPart{Kind: modality.KindImage}.IsEmpty() == true
func (p ContentPart) IsEmpty() bool {
	return p.Text == "" && len(p.Data) == 0 && p.URL == ""
}

// Message is an assembled assistant output: the concatenated text body plus the
// ordered list of every part (text and media). Parts is the source of truth;
// Text is the convenience flattening of the text parts.
type Message struct {
	Role  Role          `json:"role"`
	Text  string        `json:"text,omitempty"`
	Parts []ContentPart `json:"parts,omitempty"`
}

// Assemble collects backend output parts into one assistant Message: text parts
// are concatenated in order into the Text body, and every part — text and media
// alike — is retained in Parts in its original order, so callers that care about
// interleaving keep it while callers that only want the text read Message.Text.
//
//	msg := modality.Assemble([]modality.ContentPart{
//		modality.TextPart("see "), img, modality.TextPart("above"),
//	})
//	msg.Text  // "see above"
//	msg.Parts // [text, image, text]
func Assemble(parts []ContentPart) Message {
	msg := Message{Role: RoleAssistant}
	if len(parts) == 0 {
		return msg
	}
	text := make([]string, 0, len(parts))
	msg.Parts = make([]ContentPart, 0, len(parts))
	for _, p := range parts {
		if p.Kind == KindText {
			text = append(text, p.Text)
		}
		msg.Parts = append(msg.Parts, p)
	}
	msg.Text = core.Join("", text...)
	return msg
}

// OutputCounts is the per-kind tally of an assistant output, for usage reporting
// (§6.6): how many parts of each kind, and the summed backend-reported token /
// unit counts across those parts.
type OutputCounts struct {
	TextParts  int `json:"text_parts"`
	ImageParts int `json:"image_parts"`
	AudioParts int `json:"audio_parts"`

	TextTokens  int `json:"text_tokens"`
	ImageTokens int `json:"image_tokens"`
	AudioTokens int `json:"audio_tokens"`

	ImageUnits int `json:"image_units"` // e.g. number of generated images
	AudioUnits int `json:"audio_units"` // e.g. seconds of audio
}

// Counts tallies output parts by kind for usage accounting (§6.6): it counts the
// parts of each kind and sums any backend-reported per-part Tokens / Units into
// the matching totals. A nil or text-only input yields a zeroed media tally.
//
//	c := modality.Counts(msg.Parts)
//	c.ImageParts  // how many image parts
//	c.AudioTokens // summed audio tokens for the usage record
func Counts(parts []ContentPart) OutputCounts {
	var c OutputCounts
	for _, p := range parts {
		switch p.Kind {
		case KindText:
			c.TextParts++
			c.TextTokens += p.Tokens
		case KindImage:
			c.ImageParts++
			c.ImageTokens += p.Tokens
			c.ImageUnits += p.Units
		case KindAudio:
			c.AudioParts++
			c.AudioTokens += p.Tokens
			c.AudioUnits += p.Units
		}
	}
	return c
}
