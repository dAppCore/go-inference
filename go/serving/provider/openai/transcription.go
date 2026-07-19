// SPDX-Licence-Identifier: EUPL-1.2

// transcription.go serves POST /v1/audio/transcriptions: OpenAI-compatible multipart upload (the real
// API's own shape) OR this engine's JSON+base64-data-URL extension (mirroring image_url's data: URL
// convention — content.go — per the design's explicit ask: "data-URL only, no remote fetch"). Unlike
// every other handler in this package, this one does NOT resolve a model by name through a Resolver: a
// Whisper checkpoint never enters the TextModel factory (model/arch/openai/whisper.Model's own doc
// comment), so there is nothing for h.resolver.ResolveModel to find. TranscriptionHandler instead holds
// an inference.Transcriber directly — nil when the serving process was not started with a Whisper
// --model (the v1 "one process, one purpose" shape; see dappco.re/go/inference/serving's
// detectAndLoadWhisper) — and every request against a nil transcriber gets the clean capability refusal,
// mirroring the vision/audio 400 pattern (handler.go's messagesCarryImages/messagesCarryAudios gates):
// the route is always mounted, never a 404.
package openai

import (
	"io"
	"mime"
	"net/http"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// DefaultTranscriptionsPath is the OpenAI-compatible ASR route.
const DefaultTranscriptionsPath = "/v1/audio/transcriptions"

// maxTranscriptionRequestBytes bounds one transcription request's total body — the same 32 MiB-class cap
// content.go's image/audio data: URL decoders use (the design's open question #3: moot while the v1
// window is 30s of 16 kHz WAV, ~1 MiB, but the bound outlives v1), plus a little headroom for multipart
// framing overhead (boundary markers, the other form fields).
const maxTranscriptionRequestBytes = maxDecodedImageBytes + 64<<10

// TranscriptionResponse is the OpenAI-compatible JSON response body for the default response_format
// ("json"): OpenAI's own shape is {"text": "..."} only; Language is this engine's addition — the same
// superset `lem transcribe --json` already returns (cli/transcribe.go) — harmless to a strict OpenAI
// client, which ignores unknown fields.
type TranscriptionResponse struct {
	Text     string `json:"text"`
	Language string `json:"language,omitempty"`
}

// transcriptionJSONRequest is this engine's JSON extension to the transcriptions route: the real OpenAI
// API is multipart-only (decodeTranscriptionMultipart matches it exactly), but every other route in this
// package also accepts inline base64 media in a JSON body (chat completions' image_url/input_audio
// content parts). Audio mirrors image_url's data: URL convention specifically, not input_audio's bare-
// base64 shape — the design's explicit ask.
type transcriptionJSONRequest struct {
	Model          string `json:"model,omitempty"`
	Audio          string `json:"audio"`
	Language       string `json:"language,omitempty"`
	ResponseFormat string `json:"response_format,omitempty"`
}

// TranscriptionHandler serves DefaultTranscriptionsPath. See the file doc comment for why it holds a
// transcriber directly instead of resolving one through a Resolver.
type TranscriptionHandler struct {
	transcriber inference.Transcriber
}

// NewTranscriptionHandler wraps transcriber (nil when no Whisper checkpoint is loaded — every request
// then gets the capability refusal, never a panic or a 404).
func NewTranscriptionHandler(transcriber inference.Transcriber) *TranscriptionHandler {
	return &TranscriptionHandler{transcriber: transcriber}
}

func (h *TranscriptionHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if !requireServiceMethod(w, r, http.MethodPost) {
		return
	}
	if h == nil || h.transcriber == nil {
		writeError(w, http.StatusBadRequest, "model does not support audio transcription", "model")
		return
	}
	wavBytes, language, responseFormat, ok := decodeTranscriptionRequest(w, r)
	if !ok {
		return
	}
	format := core.Lower(responseFormat)
	if format != "" && format != "json" && format != "text" {
		writeError(w, http.StatusBadRequest,
			"response_format "+core.Sprintf("%q", responseFormat)+" is not supported (json, text — this v1 lane emits no timestamps/segments)",
			"response_format")
		return
	}
	text, detectedLanguage, err := h.transcriber.TranscribeAudio(wavBytes, language)
	if err != nil {
		// Whisper's own errors are request-shape problems (audio past the fixed window, a malformed WAV
		// header) far more often than a backend crash — a clean 400 naming what's wrong is more honest
		// here than the 500 the embeddings/rerank forward-call failures use for their very different
		// failure shape.
		writeError(w, http.StatusBadRequest, err.Error(), "audio")
		return
	}
	if format == "text" {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		w.WriteHeader(http.StatusOK)
		core.WriteString(w, text)
		return
	}
	writeJSON(w, http.StatusOK, TranscriptionResponse{Text: text, Language: detectedLanguage})
}

// decodeTranscriptionRequest dispatches on Content-Type: a multipart body (the real OpenAI shape) or
// anything else (this engine's JSON+data-URL extension, the default when Content-Type is absent/unknown
// — matching how every other route here expects JSON by default).
func decodeTranscriptionRequest(w http.ResponseWriter, r *http.Request) (wavBytes []byte, language, responseFormat string, ok bool) {
	mediaType, _, _ := mime.ParseMediaType(r.Header.Get("Content-Type"))
	if mediaType == "multipart/form-data" {
		return decodeTranscriptionMultipart(w, r)
	}
	return decodeTranscriptionJSON(w, r)
}

// decodeTranscriptionMultipart reads the standard OpenAI form fields: file (required, the audio binary),
// language and response_format (both optional). model is accepted but not consulted — this v1 lane
// serves exactly one Whisper checkpoint per process (see the file doc comment), so there is nothing to
// route by name yet.
func decodeTranscriptionMultipart(w http.ResponseWriter, r *http.Request) (wavBytes []byte, language, responseFormat string, ok bool) {
	r.Body = http.MaxBytesReader(w, r.Body, maxTranscriptionRequestBytes)
	if err := r.ParseMultipartForm(maxDecodedImageBytes); err != nil {
		writeError(w, http.StatusBadRequest, "invalid multipart request: "+err.Error(), "file")
		return nil, "", "", false
	}
	file, _, err := r.FormFile("file")
	if err != nil {
		writeError(w, http.StatusBadRequest, "file is required (multipart audio upload)", "file")
		return nil, "", "", false
	}
	defer func() { _ = file.Close() }()
	data, err := io.ReadAll(file)
	if err != nil {
		writeError(w, http.StatusBadRequest, "read audio file failed", "file")
		return nil, "", "", false
	}
	if len(data) == 0 {
		writeError(w, http.StatusBadRequest, "audio file is empty", "file")
		return nil, "", "", false
	}
	return data, core.Trim(r.FormValue("language")), core.Trim(r.FormValue("response_format")), true
}

// decodeTranscriptionJSON reads this engine's JSON extension body.
func decodeTranscriptionJSON(w http.ResponseWriter, r *http.Request) (wavBytes []byte, language, responseFormat string, ok bool) {
	var req transcriptionJSONRequest
	if !decodeServiceRequest(w, r, &req, "openai.TranscriptionHandler") {
		return nil, "", "", false
	}
	if core.Trim(req.Audio) == "" {
		writeError(w, http.StatusBadRequest, "audio is required (a base64 data: URL — this engine does not fetch remote audio)", "audio")
		return nil, "", "", false
	}
	decoded, err := decodeAudioDataURL(req.Audio)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error(), "audio")
		return nil, "", "", false
	}
	return decoded, core.Trim(req.Language), core.Trim(req.ResponseFormat), true
}

// decodeAudioDataURL decodes "data:audio/wav;base64,…" into raw WAV bytes — audio's twin of content.go's
// decodeImageDataURL (data: URL, not the bare-base64 input_audio.data shape): only data: URLs are
// accepted, a local engine never fetches a remote URL embedded in a request.
func decodeAudioDataURL(url string) ([]byte, error) {
	if !core.HasPrefix(url, "data:") {
		return nil, core.E("openai.TranscriptionHandler", "audio must be a base64 data: URL — this engine does not fetch remote audio", nil)
	}
	comma := core.Index(url, ",")
	if comma < 0 {
		return nil, core.E("openai.TranscriptionHandler", "malformed data: URL — missing payload separator", nil)
	}
	if !core.HasSuffix(url[:comma], ";base64") {
		return nil, core.E("openai.TranscriptionHandler", "data: URL must be base64-encoded", nil)
	}
	payload := url[comma+1:]
	// Base64 expands 3 bytes to 4 chars; bound the ENCODED length before decoding so an oversized
	// payload never allocates its decoded form (mirrors decodeImageDataURL's own bound exactly).
	if len(payload) > (maxDecodedImageBytes/3+1)*4 {
		return nil, core.E("openai.TranscriptionHandler", core.Sprintf("audio exceeds the %d MiB cap", maxDecodedImageBytes>>20), nil)
	}
	decoded := core.Base64Decode(payload)
	if !decoded.OK {
		return nil, core.E("openai.TranscriptionHandler", "audio base64 payload is invalid", decoded.Err())
	}
	bytes := decoded.Bytes()
	if len(bytes) == 0 {
		return nil, core.E("openai.TranscriptionHandler", "audio payload is empty", nil)
	}
	return bytes, nil
}
