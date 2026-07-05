// SPDX-Licence-Identifier: EUPL-1.2

// Multimodal content parsing for the chat completions route: OpenAI allows
// message content as a plain string OR an array of typed parts. Text parts
// concatenate into Content; image_url parts must be base64 data: URLs and
// decode into Images — this is a LOCAL engine, so remote image URLs are
// refused rather than fetched (no SSRF surface, no silent network I/O on
// behalf of a prompt).

package openai

import (
	core "dappco.re/go"
)

// maxDecodedImageBytes caps one decoded image. The vision front-end resizes
// onto a fixed patch budget anyway, so anything past this is either a mistake
// or an attack on the decoder.
const maxDecodedImageBytes = 32 << 20

// maxImagesPerRequest bounds the per-request vision work.
const maxImagesPerRequest = 16

type chatContentPart struct {
	Type     string               `json:"type"`
	Text     string               `json:"text"`
	ImageURL *chatContentImageURL `json:"image_url"`
}

type chatContentImageURL struct {
	URL string `json:"url"`
}

// rawJSON captures a field's raw bytes during unmarshal without importing
// encoding/json for RawMessage.
type rawJSON []byte

func (r *rawJSON) UnmarshalJSON(data []byte) error {
	*r = append((*r)[:0], data...)
	return nil
}

// UnmarshalJSON accepts both content shapes:
//
//	{"role":"user","content":"plain text"}
//	{"role":"user","content":[
//	    {"type":"text","text":"What is in this image?"},
//	    {"type":"image_url","image_url":{"url":"data:image/png;base64,…"}}]}
func (m *ChatMessage) UnmarshalJSON(data []byte) error {
	var wire struct {
		Role    string  `json:"role"`
		Content rawJSON `json:"content"`
	}
	if result := core.JSONUnmarshal(data, &wire); !result.OK {
		return result.Err()
	}
	m.Role = wire.Role
	m.Content = ""
	m.Images = nil

	content := trimJSONSpace(wire.Content)
	if len(content) == 0 || string(content) == "null" {
		return nil
	}
	switch content[0] {
	case '"':
		var text string
		if result := core.JSONUnmarshal(content, &text); !result.OK {
			return result.Err()
		}
		m.Content = text
		return nil
	case '[':
		var parts []chatContentPart
		if result := core.JSONUnmarshal(content, &parts); !result.OK {
			return result.Err()
		}
		return m.applyContentParts(parts)
	default:
		return core.E("openai.ChatMessage", "content must be a string or a content-part array", nil)
	}
}

func (m *ChatMessage) applyContentParts(parts []chatContentPart) error {
	var text core.Builder
	for index, part := range parts {
		switch part.Type {
		case "text":
			if text.Len() > 0 {
				text.WriteString("\n")
			}
			text.WriteString(part.Text)
		case "image_url":
			if part.ImageURL == nil || part.ImageURL.URL == "" {
				return core.E("openai.ChatMessage", core.Sprintf("content[%d].image_url.url is required", index), nil)
			}
			if len(m.Images) >= maxImagesPerRequest {
				return core.E("openai.ChatMessage", core.Sprintf("too many images — at most %d per request", maxImagesPerRequest), nil)
			}
			decoded, err := decodeImageDataURL(part.ImageURL.URL)
			if err != nil {
				return err
			}
			m.Images = append(m.Images, decoded)
		default:
			return core.E("openai.ChatMessage", core.Sprintf("content[%d].type %q is not supported (text, image_url)", index, part.Type), nil)
		}
	}
	m.Content = text.String()
	return nil
}

// decodeImageDataURL decodes "data:image/png;base64,…" into raw image bytes.
// Only data: URLs are accepted — a local engine never fetches a remote URL
// embedded in a prompt.
func decodeImageDataURL(url string) ([]byte, error) {
	if !core.HasPrefix(url, "data:") {
		return nil, core.E("openai.ChatMessage", "image_url must be a base64 data: URL — this engine does not fetch remote images", nil)
	}
	comma := core.Index(url, ",")
	if comma < 0 {
		return nil, core.E("openai.ChatMessage", "malformed data: URL — missing payload separator", nil)
	}
	if !core.HasSuffix(url[:comma], ";base64") {
		return nil, core.E("openai.ChatMessage", "data: URL must be base64-encoded", nil)
	}
	payload := url[comma+1:]
	// Base64 expands 3 bytes to 4 chars; bound the ENCODED length before
	// decoding so an oversized payload never allocates its decoded form.
	if len(payload) > (maxDecodedImageBytes/3+1)*4 {
		return nil, core.E("openai.ChatMessage", core.Sprintf("image exceeds the %d MiB cap", maxDecodedImageBytes>>20), nil)
	}
	decoded := core.Base64Decode(payload)
	if !decoded.OK {
		return nil, core.E("openai.ChatMessage", "image base64 payload is invalid", decoded.Err())
	}
	bytes, ok := decoded.Value.([]byte)
	if !ok {
		text, textOK := decoded.Value.(string)
		if !textOK {
			return nil, core.E("openai.ChatMessage", "image base64 decode returned an unexpected type", nil)
		}
		bytes = []byte(text)
	}
	if len(bytes) == 0 {
		return nil, core.E("openai.ChatMessage", "image payload is empty", nil)
	}
	return bytes, nil
}

func trimJSONSpace(data []byte) []byte {
	start := 0
	for start < len(data) {
		switch data[start] {
		case ' ', '\t', '\n', '\r':
			start++
		default:
			return data[start:]
		}
	}
	return nil
}
