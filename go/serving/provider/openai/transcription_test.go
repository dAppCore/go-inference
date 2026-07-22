// SPDX-Licence-Identifier: EUPL-1.2

package openai

import (
	"bytes"
	"encoding/base64"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
)

// stubTranscriber is a minimal inference.Transcriber: returns fixed text/language, or err when set —
// mirrors serviceModel's err-switch shape (services_test.go) for the equivalent capability-error branch.
type stubTranscriber struct {
	text, language string
	gotWav         []byte
	gotLanguage    string
	err            error
}

func (s *stubTranscriber) TranscribeAudio(wavBytes []byte, language string) (string, string, error) {
	s.gotWav, s.gotLanguage = wavBytes, language
	if s.err != nil {
		return "", "", s.err
	}
	return s.text, s.language, nil
}

func audioDataURL(payload string) string {
	return "data:audio/wav;base64," + base64.StdEncoding.EncodeToString([]byte(payload))
}

// multipartAudioRequest builds a POST DefaultTranscriptionsPath request with a "file" field (audio bytes)
// plus any extra form fields (language, response_format) — the real OpenAI wire shape.
func multipartAudioRequest(t *testing.T, audio []byte, fields map[string]string) *http.Request {
	t.Helper()
	var body bytes.Buffer
	w := multipart.NewWriter(&body)
	if audio != nil {
		part, err := w.CreateFormFile("file", "clip.wav")
		if err != nil {
			t.Fatalf("CreateFormFile: %v", err)
		}
		if _, err := part.Write(audio); err != nil {
			t.Fatalf("write audio part: %v", err)
		}
	}
	for k, v := range fields {
		if err := w.WriteField(k, v); err != nil {
			t.Fatalf("WriteField(%s): %v", k, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}
	req := httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, &body)
	req.Header.Set("Content-Type", w.FormDataContentType())
	return req
}

// TestOpenAI_TranscriptionHandler_Good_JSONDataURL is the compat-shape gate for this engine's JSON
// extension: request field "audio" (a data: URL, the image_url precedent) decodes, language forwards to
// TranscribeAudio, and the response is the OpenAI-shaped {"text":...} plus this engine's "language"
// superset field (TranscriptionResponse's doc comment).
func TestOpenAI_TranscriptionHandler_Good_JSONDataURL(t *testing.T) {
	transcriber := &stubTranscriber{text: "the quick brown fox", language: "en"}
	handler := NewTranscriptionHandler(transcriber)
	body := `{"model":"whisper-tiny","audio":"` + audioDataURL("RIFF...") + `","language":"en"}`
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"text":"the quick brown fox"`) || !strings.Contains(rec.Body.String(), `"language":"en"`) {
		t.Fatalf("response = %s, want the OpenAI {text} shape plus language", rec.Body.String())
	}
	if string(transcriber.gotWav) != "RIFF..." || transcriber.gotLanguage != "en" {
		t.Fatalf("TranscribeAudio got wav=%q language=%q, want the decoded payload and forwarded language", transcriber.gotWav, transcriber.gotLanguage)
	}
}

// TestOpenAI_TranscriptionHandler_Good_Multipart proves the real OpenAI multipart shape (field "file")
// decodes and routes through exactly like the JSON variant — the other half of the compat-shape gate.
func TestOpenAI_TranscriptionHandler_Good_Multipart(t *testing.T) {
	transcriber := &stubTranscriber{text: "hello world", language: "en"}
	handler := NewTranscriptionHandler(transcriber)
	req := multipartAudioRequest(t, []byte("RIFF-fake-wav-bytes"), map[string]string{"model": "whisper-tiny", "language": "en"})
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d body=%s, want 200", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"text":"hello world"`) {
		t.Fatalf("response = %s, want text=%q", rec.Body.String(), "hello world")
	}
	if string(transcriber.gotWav) != "RIFF-fake-wav-bytes" {
		t.Fatalf("TranscribeAudio got wav=%q, want the multipart file field's bytes", transcriber.gotWav)
	}
}

// TestOpenAI_TranscriptionHandler_Good_ResponseFormatText proves response_format=text returns the bare
// transcript as a text/plain body instead of the JSON envelope.
func TestOpenAI_TranscriptionHandler_Good_ResponseFormatText(t *testing.T) {
	transcriber := &stubTranscriber{text: "plain text reply", language: "en"}
	handler := NewTranscriptionHandler(transcriber)
	body := `{"audio":"` + audioDataURL("RIFF...") + `","response_format":"text"}`
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(body)))

	if rec.Code != http.StatusOK || rec.Body.String() != "plain text reply" {
		t.Fatalf("status=%d body=%q, want 200 \"plain text reply\"", rec.Code, rec.Body.String())
	}
	if ct := rec.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/plain") {
		t.Fatalf("Content-Type = %q, want text/plain", ct)
	}
}

// TestOpenAI_TranscriptionHandler_Bad_CapabilityRefusal is THE capability-refusal gate: a nil
// transcriber (no Whisper checkpoint loaded — serve started with a non-whisper --model or none at all)
// answers with a clean 400, mirroring the vision/audio "model does not accept image/audio input" 400
// pattern (handler.go) rather than a 404 or a panic.
func TestOpenAI_TranscriptionHandler_Bad_CapabilityRefusal(t *testing.T) {
	handler := NewTranscriptionHandler(nil)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(`{"audio":"`+audioDataURL("x")+`"}`)))

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "does not support audio transcription") {
		t.Fatalf("status=%d body=%s, want 400 \"does not support audio transcription\"", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_TranscriptionHandler_Bad_WrongMethod proves GET is refused with 405, matching every other
// handler's requireServiceMethod gate.
func TestOpenAI_TranscriptionHandler_Bad_WrongMethod(t *testing.T) {
	handler := NewTranscriptionHandler(&stubTranscriber{})
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, DefaultTranscriptionsPath, nil))

	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

// TestOpenAI_TranscriptionHandler_Bad_Validation covers the request-shape rejections common to both
// wire forms: no audio/file at all, a remote (non data:) URL (the no-remote-fetch rule — the design's
// explicit "data-URL only, no remote fetch"), and an unsupported response_format.
func TestOpenAI_TranscriptionHandler_Bad_Validation(t *testing.T) {
	cases := map[string]*http.Request{
		"json-audio-empty":       httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(`{"audio":""}`)),
		"json-audio-missing":     httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(`{"model":"whisper-tiny"}`)),
		"json-audio-remote-url":  httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(`{"audio":"https://example.com/clip.wav"}`)),
		"json-audio-not-base64":  httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(`{"audio":"data:audio/wav,plain"}`)),
		"json-response-format":   httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(`{"audio":"`+audioDataURL("x")+`","response_format":"srt"}`)),
		"json-malformed":         httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(`{`)),
		"multipart-file-missing": multipartAudioRequest(t, nil, map[string]string{"model": "whisper-tiny"}),
	}
	for name, req := range cases {
		t.Run(name, func(t *testing.T) {
			handler := NewTranscriptionHandler(&stubTranscriber{text: "unused"})
			rec := httptest.NewRecorder()
			handler.ServeHTTP(rec, req)
			if rec.Code != http.StatusBadRequest {
				t.Fatalf("status = %d body=%s, want 400", rec.Code, rec.Body.String())
			}
		})
	}
}

// TestOpenAI_TranscriptionHandler_Bad_TranscribeError covers TranscribeAudio's error-propagation branch
// (e.g. audio past the checkpoint's fixed window) — the error text reaches the caller, no partial/zero
// success body.
func TestOpenAI_TranscriptionHandler_Bad_TranscribeError(t *testing.T) {
	transcriber := &stubTranscriber{err: core.E("test", "audio exceeds the 30s window", nil)}
	handler := NewTranscriptionHandler(transcriber)
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(`{"audio":"`+audioDataURL("x")+`"}`)))

	if rec.Code != http.StatusBadRequest || !strings.Contains(rec.Body.String(), "audio exceeds the 30s window") {
		t.Fatalf("status=%d body=%s, want 400 naming the transcribe error", rec.Code, rec.Body.String())
	}
}

// TestOpenAI_TranscriptionHandler_Ugly_NilHandler proves a nil *TranscriptionHandler answers the same
// clean capability refusal rather than panicking on the nil receiver — distinct from _Bad's populated-
// but-empty transcriber case (mirrors handler.go's own nil-receiver ServeHTTP guard).
func TestOpenAI_TranscriptionHandler_Ugly_NilHandler(t *testing.T) {
	var handler *TranscriptionHandler
	rec := httptest.NewRecorder()

	handler.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, DefaultTranscriptionsPath, strings.NewReader(`{}`)))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400 (nil handler must not panic)", rec.Code)
	}
}
