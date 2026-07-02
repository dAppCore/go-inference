// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"context"
	"encoding/base64"
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
