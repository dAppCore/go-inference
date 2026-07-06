// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"iter"
	"net/http/httptest"
	"strings"
	"testing"

	"dappco.re/go/inference"
)

// visionStubModel is a stubModel that accepts images.
type visionStubModel struct {
	stubModel
	gotMessages []inference.Message
}

func (m *visionStubModel) AcceptsImages() bool { return true }

func (m *visionStubModel) Chat(_ context.Context, messages []inference.Message, _ ...inference.GenerateOption) iter.Seq[inference.Token] {
	m.gotMessages = messages
	return m.seq()
}

func imageDataURL(payload string) string {
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString([]byte(payload))
}

// Plain-string content keeps its wire shape — the union decode must be
// invisible to every existing client.
func TestChatMessage_DecodeRequest_StringContent_Good(t *testing.T) {
	req, err := DecodeRequest(strings.NewReader(`{"model":"m","messages":[{"role":"user","content":"hello"}]}`))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if req.Messages[0].Content != "hello" || len(req.Messages[0].Images) != 0 {
		t.Fatalf("message = %+v", req.Messages[0])
	}
}

// The multimodal array shape: text parts concatenate, image_url data: URLs
// decode to raw bytes in part order.
func TestChatMessage_DecodeRequest_ContentParts_Good(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":[` +
		`{"type":"text","text":"What is in"},` +
		`{"type":"image_url","image_url":{"url":"` + imageDataURL("PNG-ONE") + `"}},` +
		`{"type":"text","text":"this image?"},` +
		`{"type":"image_url","image_url":{"url":"` + imageDataURL("PNG-TWO") + `"}}]}]}`
	req, err := DecodeRequest(strings.NewReader(body))
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	msg := req.Messages[0]
	if msg.Content != "What is in\nthis image?" {
		t.Fatalf("content = %q", msg.Content)
	}
	if len(msg.Images) != 2 || string(msg.Images[0]) != "PNG-ONE" || string(msg.Images[1]) != "PNG-TWO" {
		t.Fatalf("images = %d decoded", len(msg.Images))
	}
}

// A local engine never fetches remote URLs out of a prompt, and malformed
// payloads fail loudly at the door.
func TestChatMessage_DecodeRequest_ContentParts_Bad(t *testing.T) {
	cases := map[string]string{
		"remote url":   `[{"type":"image_url","image_url":{"url":"https://example.com/cat.png"}}]`,
		"no separator": `[{"type":"image_url","image_url":{"url":"data:image/png;base64"}}]`,
		"not base64":   `[{"type":"image_url","image_url":{"url":"data:image/png,plain"}}]`,
		"bad payload":  `[{"type":"image_url","image_url":{"url":"data:image/png;base64,!!!"}}]`,
		"missing url":  `[{"type":"image_url"}]`,
		"odd type":     `[{"type":"input_video","text":"x"}]`,
		"object":       `{"oops":true}`,
	}
	for name, content := range cases {
		body := `{"model":"m","messages":[{"role":"user","content":` + content + `}]}`
		if _, err := DecodeRequest(strings.NewReader(body)); err == nil {
			t.Fatalf("%s: decode accepted bad content", name)
		}
	}
}

// TestChatMessage_DecodeRequest_ContentParts_Bad_TooManyImages covers
// the maxImagesPerRequest cap — one part over the limit must be
// rejected before decoding it.
func TestChatMessage_DecodeRequest_ContentParts_Bad_TooManyImages(t *testing.T) {
	url := imageDataURL("x")
	var parts strings.Builder
	for i := 0; i < maxImagesPerRequest+1; i++ {
		if i > 0 {
			parts.WriteByte(',')
		}
		parts.WriteString(`{"type":"image_url","image_url":{"url":"` + url + `"}}`)
	}
	body := `{"model":"m","messages":[{"role":"user","content":[` + parts.String() + `]}]}`

	if _, err := DecodeRequest(strings.NewReader(body)); err == nil {
		t.Fatal("DecodeRequest() accepted a content array over the per-request image cap")
	}
}

// TestContent_ChatMessage_UnmarshalJSON_Good exercises ChatMessage's
// own UnmarshalJSON directly via encoding/json.Unmarshal (rather than
// through DecodeRequest/parseChatMessage — the hand-rolled top-level
// walker in unmarshal.go never calls this method; it is a separate
// json.Unmarshaler entry point for anything that decodes a ChatMessage
// standalone, e.g. a container type outside this package's own
// request shape).
func TestContent_ChatMessage_UnmarshalJSON_Good(t *testing.T) {
	var empty ChatMessage
	if err := json.Unmarshal([]byte(`{"role":"user"}`), &empty); err != nil {
		t.Fatalf("Unmarshal(no content field) error = %v", err)
	}
	if empty.Content != "" || empty.Images != nil {
		t.Fatalf("Unmarshal(no content field) = %+v, want zero content/images", empty)
	}

	var nulled ChatMessage
	if err := json.Unmarshal([]byte(`{"role":"user","content":null}`), &nulled); err != nil {
		t.Fatalf("Unmarshal(content:null) error = %v", err)
	}
	if nulled.Content != "" {
		t.Fatalf("Unmarshal(content:null) = %+v, want empty content", nulled)
	}

	var stringContent ChatMessage
	if err := json.Unmarshal([]byte(`{"role":"user","content":"hello"}`), &stringContent); err != nil {
		t.Fatalf("Unmarshal(string content) error = %v", err)
	}
	if stringContent.Content != "hello" {
		t.Fatalf("Unmarshal(string content) = %+v, want content=hello", stringContent)
	}
}

// TestContent_ChatMessage_UnmarshalJSON_Bad drives the shape-rejection
// branches — a non-object outer document, and a content field of the
// wrong JSON type at both the top level and inside a content-part array.
func TestContent_ChatMessage_UnmarshalJSON_Bad(t *testing.T) {
	cases := map[string]string{
		// "42" is syntactically valid top-level JSON (so encoding/json's
		// whole-document checkValid pass lets it through to our custom
		// UnmarshalJSON), but is the wrong shape for the {role,content}
		// wire struct this method decodes into.
		"outer-wrong-shape":  `42`,
		"content-wrong-type": `{"role":"user","content":42}`,
		// Syntactically valid (42 is a legal JSON number), but
		// chatContentPart.Type is a string field — the content-part
		// array's own core.JSONUnmarshal call fails, distinct from the
		// parseChatMessage hand-rolled walker's equivalent branch
		// (unmarshal.go), which this method never calls.
		"content-array-element-wrong-type": `{"role":"user","content":[{"type":42}]}`,
	}
	for name, in := range cases {
		t.Run(name, func(t *testing.T) {
			var msg ChatMessage
			if err := json.Unmarshal([]byte(in), &msg); err == nil {
				t.Fatalf("Unmarshal(%q) returned nil error", in)
			}
		})
	}
}

// TestContent_ChatMessage_UnmarshalJSON_Ugly covers the multimodal
// content-part array shape — multiple text parts join with a newline
// rather than concatenating flush.
func TestContent_ChatMessage_UnmarshalJSON_Ugly(t *testing.T) {
	var arrayContent ChatMessage
	in := `{"role":"user","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}`
	if err := json.Unmarshal([]byte(in), &arrayContent); err != nil {
		t.Fatalf("Unmarshal(array content) error = %v", err)
	}
	if arrayContent.Content != "a\nb" {
		t.Fatalf("Unmarshal(array content) = %+v, want joined text", arrayContent)
	}
}

// TestTrimJSONSpace_Good_Bad_Ugly pins the whitespace-skip contract
// used ahead of ChatMessage's content-shape sniff.
func TestTrimJSONSpace_Good_Bad_Ugly(t *testing.T) {
	if got := trimJSONSpace([]byte(`  "x"`)); string(got) != `"x"` {
		t.Fatalf("trimJSONSpace(leading spaces) = %q", got)
	}
	if got := trimJSONSpace([]byte("\t\n\r[1]")); string(got) != "[1]" {
		t.Fatalf("trimJSONSpace(tab/newline/CR) = %q", got)
	}
	if got := trimJSONSpace(nil); got != nil {
		t.Fatalf("trimJSONSpace(nil) = %v, want nil", got)
	}
	if got := trimJSONSpace([]byte("   ")); got != nil {
		t.Fatalf("trimJSONSpace(all whitespace) = %v, want nil", got)
	}
}

// TestDecodeImageDataURL_Bad_EmptyPayload covers the
// decodes-to-zero-bytes rejection — a syntactically valid empty
// base64 payload after the comma.
func TestDecodeImageDataURL_Bad_EmptyPayload(t *testing.T) {
	if _, err := decodeImageDataURL("data:image/png;base64,"); err == nil {
		t.Fatal("decodeImageDataURL(empty payload) error = nil, want empty-payload rejection")
	}
}

// TestDecodeImageDataURL_Ugly_OversizedPayloadRejectedBeforeDecoding
// covers the pre-decode length guard — an ENCODED payload longer than
// the cap must be rejected without ever calling into the base64
// decoder (guards against a decode-bomb allocating the full decoded
// size for a payload that was going to be rejected anyway).
func TestDecodeImageDataURL_Ugly_OversizedPayloadRejectedBeforeDecoding(t *testing.T) {
	oversized := "data:image/png;base64," + strings.Repeat("A", (maxDecodedImageBytes/3+1)*4+1)
	if _, err := decodeImageDataURL(oversized); err == nil {
		t.Fatal("decodeImageDataURL(oversized) error = nil, want cap rejection")
	}
}

// The capability gate: image requests against a text-only model answer 400
// before any generation work; a vision model receives the decoded bytes.
func TestHandler_ImageCapabilityGate_Good(t *testing.T) {
	body := `{"model":"m","messages":[{"role":"user","content":[` +
		`{"type":"text","text":"describe"},` +
		`{"type":"image_url","image_url":{"url":"` + imageDataURL("PNG") + `"}}]}]}`

	textOnly := NewHandler(NewStaticResolver(map[string]inference.TextModel{"m": &stubModel{}}))
	rec := httptest.NewRecorder()
	textOnly.ServeHTTP(rec, httptest.NewRequest("POST", DefaultChatCompletionsPath, strings.NewReader(body)))
	if rec.Code != 400 || !strings.Contains(rec.Body.String(), "does not accept image input") {
		t.Fatalf("text-only model: status %d body %s", rec.Code, rec.Body.String())
	}

	vision := &visionStubModel{stubModel: stubModel{tokens: []inference.Token{{ID: 1, Text: "a cat"}}}}
	handler := NewHandler(NewStaticResolver(map[string]inference.TextModel{"m": vision}))
	rec = httptest.NewRecorder()
	handler.ServeHTTP(rec, httptest.NewRequest("POST", DefaultChatCompletionsPath, strings.NewReader(body)))
	if rec.Code != 200 {
		t.Fatalf("vision model: status %d body %s", rec.Code, rec.Body.String())
	}
	if len(vision.gotMessages) != 1 || len(vision.gotMessages[0].Images) != 1 || string(vision.gotMessages[0].Images[0]) != "PNG" {
		t.Fatalf("vision model messages = %+v", vision.gotMessages)
	}
	if !strings.Contains(rec.Body.String(), "a cat") {
		t.Fatalf("response body = %s", rec.Body.String())
	}
}
