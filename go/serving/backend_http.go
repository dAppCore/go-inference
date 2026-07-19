package serving

import (
	"context"
	"net/http"
	"time"

	"dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving/provider/openai"
	coreio "dappco.re/go/io"
)

// HTTPBackend talks to an OpenAI-compatible chat completions API.
type HTTPBackend struct {
	baseURL    string
	chatURL    string // precomputed baseURL + completions path (immutable)
	model      string
	maxTokens  int
	httpClient *http.Client
	medium     coreio.Medium
}

// HTTPOption configures an HTTPBackend at construction time.
//
//	b := serving.NewHTTPBackend("http://localhost:11434", "llama3",
//	    serving.WithHTTPClient(myClient),
//	    serving.WithMedium(io.S3("models.lthn.io")),
//	)
type HTTPOption func(*HTTPBackend)

// WithHTTPClient overrides the default net/http.Client used for requests.
func WithHTTPClient(client *http.Client) HTTPOption {
	return func(b *HTTPBackend) {
		if client != nil {
			b.httpClient = client
		}
	}
}

// WithMedium attaches an io.Medium so model artefacts (LoRA adapters,
// GGUF blobs, streamed responses) can be loaded or staged from any
// supported backend (local disk, S3, in-memory, etc.).
//
//	b := serving.NewHTTPBackend(url, model, serving.WithMedium(io.S3("models.lthn.io")))
func WithMedium(medium coreio.Medium) HTTPOption {
	return func(b *HTTPBackend) {
		b.medium = medium
	}
}

// WithHTTPMaxTokens sets the default maximum token count for requests.
func WithHTTPMaxTokens(n int) HTTPOption {
	return func(b *HTTPBackend) {
		b.maxTokens = n
	}
}

// openaiMessages converts serving messages to the shared OpenAI wire type so
// HTTPBackend and provider/openai encode requests through one codec.
func openaiMessages(messages []Message) []openai.ChatMessage {
	out := make([]openai.ChatMessage, len(messages))
	for i, m := range messages {
		out[i] = openai.ChatMessage{Role: m.Role, Content: m.Content, Images: m.Images}
	}
	return out
}

// retryableError marks errors that should be retried.
type retryableError struct {
	err error
}

func (e *retryableError) Error() string { return e.err.Error() }
func (e *retryableError) Unwrap() error { return e.err }

// NewHTTPBackend creates an HTTPBackend for the given base URL and model.
// Additional options configure the HTTP client, default max tokens, or an
// io.Medium used for staging model artefacts.
//
//	b := serving.NewHTTPBackend("http://localhost:11434", "llama3")
//	b := serving.NewHTTPBackend(url, model, serving.WithMedium(io.S3("models.lthn.io")))
func NewHTTPBackend(baseURL, model string, opts ...HTTPOption) *HTTPBackend {
	b := &HTTPBackend{
		baseURL: baseURL,
		chatURL: baseURL + openai.DefaultChatCompletionsPath,
		model:   model,
		httpClient: &http.Client{
			Timeout: 300 * time.Second,
		},
	}
	for _, opt := range opts {
		opt(b)
	}
	return b
}

// Medium returns the io.Medium configured via WithMedium, or nil if none
// was supplied.
func (b *HTTPBackend) Medium() coreio.Medium { return b.medium }

// Name returns "http".
func (b *HTTPBackend) Name() string { return "http" }

// Available always returns true for HTTP backends.
func (b *HTTPBackend) Available() bool { return b.baseURL != "" }

// Model returns the configured model name.
func (b *HTTPBackend) Model() string { return b.model }

// BaseURL returns the configured base URL.
func (b *HTTPBackend) BaseURL() string { return b.baseURL }

// SetMaxTokens sets the maximum token count for requests.
func (b *HTTPBackend) SetMaxTokens(n int) { b.maxTokens = n }

// LoadModel satisfies inference.Backend by wrapping the HTTPBackend as an
// inference.TextModel. The path argument is ignored — HTTP backends talk to
// a remote server which already has the model loaded. Spec §2.3.
//
//	backend := serving.NewHTTPBackend("http://localhost:11434", "llama2")
//	result := backend.LoadModel("dummy")
//	for tok := range model.Generate(ctx, "hello") {
//	    fmt.Print(tok.Text)
//	}
func (b *HTTPBackend) LoadModel(_ string, _ ...inference.LoadOption) core.Result {
	return core.Ok(NewHTTPTextModel(b))
}

// Generate sends a single prompt and returns the response.
//
//	r := b.Generate(ctx, "hello", serving.DefaultGenOpts())
//	if !r.OK { return r }
//	resp := r.Value.(serving.Result)
func (b *HTTPBackend) Generate(ctx context.Context, prompt string, opts GenOpts) core.Result {
	return b.Chat(ctx, []Message{{Role: "user", Content: prompt}}, opts)
}

// Chat sends a multi-turn conversation and returns the response.
// Retries up to 3 times with exponential backoff on transient failures.
//
//	r := b.Chat(ctx, messages, serving.DefaultGenOpts())
//	if !r.OK { return r }
//	resp := r.Value.(serving.Result)
func (b *HTTPBackend) Chat(ctx context.Context, messages []Message, opts GenOpts) core.Result {
	model := b.model
	if opts.Model != "" {
		model = opts.Model
	}
	maxTokens := b.maxTokens
	if opts.MaxTokens > 0 {
		maxTokens = opts.MaxTokens
	}
	temp := float32(opts.Temperature)

	req := openai.ChatCompletionRequest{
		Model:       model,
		Messages:    openaiMessages(messages),
		Temperature: &temp,
	}
	if maxTokens > 0 {
		req.MaxTokens = &maxTokens
	}

	// JSONMarshalString hands back a string view of a freshly-marshalled,
	// single-owner buffer; AsBytes views it as []byte without re-copying the
	// whole request body. The body is only ever read (by the HTTP transport),
	// never mutated, so the zero-copy view is safe.
	body := core.AsBytes(core.JSONMarshalString(req))

	const maxAttempts = 3
	var lastErr error

	for attempt := range maxAttempts {
		if attempt > 0 {
			backoff := time.Duration(100<<uint(attempt-1)) * time.Millisecond
			time.Sleep(backoff)
		}

		r := b.doRequest(ctx, body)
		if r.OK {
			text := r.String()
			return core.Ok(newResult(applyStopSequences(text, opts.StopSequences), nil))
		}
		err := r.Value.(error)
		lastErr = err

		var re *retryableError
		if !core.As(err, &re) {
			return core.Fail(err)
		}
	}

	return core.Fail(core.E("serving.HTTPBackend.Chat", core.Sprintf("exhausted %d retries", maxAttempts), lastErr))
}

// doRequest sends a single HTTP request and parses the response.
//
//	r := b.doRequest(ctx, body)
//	if !r.OK { return r }
//	text := r.String()
func (b *HTTPBackend) doRequest(ctx context.Context, body []byte) core.Result {
	// chatURL is precomputed at construction; baseURL never changes, so this
	// avoids rebuilding the same string on every request.
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, b.chatURL, core.NewBuffer(body))
	if err != nil {
		return core.Fail(core.E("serving.HTTPBackend.doRequest", "create request", err))
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := b.httpClient.Do(httpReq)
	if err != nil {
		return core.Fail(&retryableError{core.E("serving.HTTPBackend.doRequest", "http request", err)})
	}
	defer resp.Body.Close()

	rBody := readAll(resp.Body)
	if !rBody.OK {
		return core.Fail(&retryableError{core.E("serving.HTTPBackend.doRequest", "read response", rBody.Value.(error))})
	}
	respBody := rBody.Bytes()

	if resp.StatusCode >= 500 {
		return core.Fail(&retryableError{core.E("serving.HTTPBackend.doRequest", core.Sprintf("server error %d: %s", resp.StatusCode, string(respBody)), nil)})
	}
	if resp.StatusCode != http.StatusOK {
		return core.Fail(core.E("serving.HTTPBackend.doRequest", core.Sprintf("unexpected status %d: %s", resp.StatusCode, string(respBody)), nil))
	}

	var chatResp openai.ChatCompletionResponse
	if r := core.JSONUnmarshal(respBody, &chatResp); !r.OK {
		return core.Fail(core.E("serving.HTTPBackend.doRequest", "unmarshal response", r.Value.(error)))
	}

	if len(chatResp.Choices) == 0 {
		return core.Fail(core.E("serving.HTTPBackend.doRequest", "no choices in response", nil))
	}

	return core.Ok(chatResp.Choices[0].Message.Content)
}
