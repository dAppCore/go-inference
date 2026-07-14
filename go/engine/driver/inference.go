// SPDX-License-Identifier: EUPL-1.2

package driver

import (
	// AX-6: bytes.Reader is the structural request-body source for the upstream forward.
	"bytes"
	"context"
	// AX-6: io is the structural stream boundary for response passthrough.
	"io"
	// AX-6: net/http is the structural client/transport boundary for the proxy.
	"net/http"
	// AX-6: sync.Pool reuses the per-request streaming-copy buffer in forward().
	"sync"

	core "dappco.re/go"
	coreapi "dappco.re/go/api"
	coreprovider "dappco.re/go/api/pkg/provider"
	"github.com/gin-gonic/gin"
)

// inferenceClient forwards chat to the driver. No client timeout — a streaming
// completion can run for minutes; the caller's request context bounds it.
var inferenceClient = &http.Client{}

// forwardBufPool supplies the 16KB streaming-copy buffer forward() borrows per
// request, so the proxy doesn't book a fresh 16KB heap allocation on every chat
// request. AX-11: BenchmarkForwardCopy_{Make,Pooled} — 16KB/2 allocs → 8B/1.
var forwardBufPool = sync.Pool{New: func() any { b := make([]byte, 16*1024); return &b }}

// charsPerToken is the crude bytes→tokens divisor for the capacity estimate.
// Authoritative counts come back in the response usage; this only sizes the
// pre-flight WaitForCapacity check and the rough usage record.
const charsPerToken = 4

// maxChatRequestBytes caps the buffered request body so a client can't force the
// host to allocate unbounded memory before the capacity gate runs. Generous for
// chat (a 128k-token context is well under this); streaming output is unbounded
// and bypasses this — only the request is buffered.
const maxChatRequestBytes = 8 << 20 // 8 MiB

// InferenceProvider proxies OpenAI chat completions through lthn-ai to the
// active driver: it gates on go-ratelimit capacity (the host owns capacity),
// forwards to the driver, streams the response back, then records usage.
// Mounted at /v1 so clients hit the standard /v1/chat/completions; the driver
// stays an implementation detail behind the host.
//
// Usage example:
//
//	engine.Register(driver.NewInferenceProvider(driverSvc))
type InferenceProvider struct {
	svc *Service
}

var (
	_ coreapi.RouteGroup       = (*InferenceProvider)(nil)
	_ coreprovider.Describable = (*InferenceProvider)(nil)
)

// NewInferenceProvider wraps a driver Service as the inference RouteGroup.
func NewInferenceProvider(svc *Service) *InferenceProvider { return &InferenceProvider{svc: svc} }

// Name implements api.RouteGroup.
func (p *InferenceProvider) Name() string { return "inference" }

// BasePath implements api.RouteGroup.
func (p *InferenceProvider) BasePath() string { return "/v1" }

// RegisterRoutes implements api.RouteGroup.
func (p *InferenceProvider) RegisterRoutes(rg *gin.RouterGroup) {
	if p == nil || rg == nil {
		return
	}
	// Gated inference — capacity-checked, body forwarded to the active driver.
	rg.POST("/chat/completions", p.chat)
	rg.POST("/completions", p.chat)
	rg.POST("/messages", p.chat)
	// Ungated read passthrough — the driver's loaded-model list (the desktop
	// polls this for its model picker + header).
	rg.GET("/models", p.models)
}

// Describe implements coreprovider.Describable so the gated inference routes
// appear in the OpenAPI document when core/api mounts the provider. These
// routes speak the OpenAI/Anthropic wire formats verbatim — request bodies are
// forwarded to the active driver and its response streams back untouched — so
// every description declares ResponseRaw with the real compat shapes (a typed
// SDK client deserialises exactly what arrives; the house envelope never
// appears on this surface).
func (p *InferenceProvider) Describe() []coreapi.RouteDescription {
	chatBody := map[string]any{
		"type":     "object",
		"required": []string{"model", "messages"},
		"properties": map[string]any{
			"model": map[string]any{"type": "string", "description": "cosmetic on a single-model serve — the loaded model answers every name"},
			"messages": map[string]any{"type": "array", "items": map[string]any{
				"type":     "object",
				"required": []string{"role"},
				"properties": map[string]any{
					"role":    map[string]any{"type": "string"},
					"content": map[string]any{"description": "string, or an array of typed content parts (text / image_url data: URLs)"},
				},
			}},
			"max_tokens":           map[string]any{"type": "integer", "description": "per-reply token budget — a thinking model spends from it"},
			"temperature":          map[string]any{"type": "number"},
			"top_p":                map[string]any{"type": "number"},
			"stream":               map[string]any{"type": "boolean"},
			"chat_template_kwargs": map[string]any{"type": "object", "description": "vendor template kwargs, e.g. {\"enable_thinking\": false}"},
		},
	}
	usageSchema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"prompt_tokens":     map[string]any{"type": "integer"},
			"completion_tokens": map[string]any{"type": "integer"},
			"total_tokens":      map[string]any{"type": "integer"},
		},
	}
	chatResponse := map[string]any{
		"type":     "object",
		"required": []string{"id", "object", "choices"},
		"properties": map[string]any{
			"id":      map[string]any{"type": "string"},
			"object":  map[string]any{"type": "string"},
			"created": map[string]any{"type": "integer"},
			"model":   map[string]any{"type": "string"},
			"choices": map[string]any{"type": "array", "items": map[string]any{
				"type":     "object",
				"required": []string{"index", "message"},
				"properties": map[string]any{
					"index": map[string]any{"type": "integer"},
					"message": map[string]any{
						"type":     "object",
						"required": []string{"role"},
						"properties": map[string]any{
							"role":    map[string]any{"type": "string"},
							"content": map[string]any{"type": "string"},
						},
					},
					"finish_reason": map[string]any{"type": "string"},
				},
			}},
			"usage": usageSchema,
			// The reasoning channel rides the RESPONSE root, split from
			// content — the SDK fleet found the earlier message-level
			// placement wrong against the wire.
			"thought": map[string]any{"type": "string", "description": "the reasoning channel, split from content (thinking models)"},
		},
	}
	completionResponse := map[string]any{
		"type":     "object",
		"required": []string{"id", "object", "choices"},
		"properties": map[string]any{
			"id":      map[string]any{"type": "string"},
			"object":  map[string]any{"type": "string"},
			"created": map[string]any{"type": "integer"},
			"model":   map[string]any{"type": "string"},
			"choices": map[string]any{"type": "array", "items": map[string]any{
				"type":     "object",
				"required": []string{"index"},
				"properties": map[string]any{
					"index":         map[string]any{"type": "integer"},
					"text":          map[string]any{"type": "string"},
					"finish_reason": map[string]any{"type": "string"},
				},
			}},
			"usage": usageSchema,
		},
	}
	messagesResponse := map[string]any{
		"type":     "object",
		"required": []string{"id", "type", "role", "content"},
		"properties": map[string]any{
			"id":   map[string]any{"type": "string"},
			"type": map[string]any{"type": "string"},
			"role": map[string]any{"type": "string"},
			"content": map[string]any{"type": "array", "items": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"type": map[string]any{"type": "string"},
					"text": map[string]any{"type": "string"},
				},
			}},
			"model":       map[string]any{"type": "string"},
			"stop_reason": map[string]any{"type": "string"},
			"usage": map[string]any{
				"type": "object",
				"properties": map[string]any{
					"input_tokens":  map[string]any{"type": "integer"},
					"output_tokens": map[string]any{"type": "integer"},
				},
			},
		},
	}
	modelsResponse := map[string]any{
		"type":     "object",
		"required": []string{"object", "data"},
		"properties": map[string]any{
			"object": map[string]any{"type": "string"},
			"data": map[string]any{"type": "array", "items": map[string]any{
				"type":     "object",
				"required": []string{"id"},
				"properties": map[string]any{
					"id":       map[string]any{"type": "string"},
					"object":   map[string]any{"type": "string"},
					"created":  map[string]any{"type": "integer"},
					"owned_by": map[string]any{"type": "string"},
				},
			}},
		},
	}
	return []coreapi.RouteDescription{
		{
			Method:      http.MethodPost,
			Path:        "/chat/completions",
			Summary:     "Create a chat completion",
			Description: "Capacity-gated OpenAI-compatible chat completion, proxied to the active driver. Streams when stream is true (SSE chunks; the documented schema is the non-streaming body).",
			Tags:        []string{"inference"},
			RequestBody: chatBody,
			Response:    chatResponse,
			ResponseRaw: true,
		},
		{
			Method:      http.MethodPost,
			Path:        "/completions",
			Summary:     "Create a text completion",
			Description: "Capacity-gated completion, proxied to the active driver.",
			Tags:        []string{"inference"},
			RequestBody: chatBody,
			Response:    completionResponse,
			ResponseRaw: true,
		},
		{
			Method:      http.MethodPost,
			Path:        "/messages",
			Summary:     "Create a messages completion",
			Description: "Capacity-gated Anthropic-style messages completion, proxied to the active driver.",
			Tags:        []string{"inference"},
			RequestBody: chatBody,
			Response:    messagesResponse,
			ResponseRaw: true,
		},
		{
			Method:      http.MethodGet,
			Path:        "/models",
			Summary:     "List the active driver's loaded models",
			Description: "Ungated passthrough of the active driver's loaded-model list (what the desktop polls for its model picker).",
			Tags:        []string{"inference"},
			Response:    modelsResponse,
			ResponseRaw: true,
		},
	}
}

// chat — POST /v1/chat/completions. Cap + read the body, resolve the active
// driver, gate on capacity keyed by the SERVED model (never the client-supplied
// one), forward, stream the response back, record usage. The body is forwarded
// to the driver verbatim — the driver owns request validation + the model.
func (p *InferenceProvider) chat(c *gin.Context) {
	c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxChatRequestBytes)
	body, err := io.ReadAll(c.Request.Body)
	if err != nil {
		var maxErr *http.MaxBytesError
		if core.As(err, &maxErr) {
			c.JSON(http.StatusRequestEntityTooLarge, fail("request body exceeds limit"))
			return
		}
		c.JSON(http.StatusBadRequest, fail("read body: "+err.Error()))
		return
	}

	target, model, ok := p.svc.Target()
	if !ok {
		c.JSON(http.StatusServiceUnavailable, fail("no driver ready — serve a model first"))
		return
	}

	// Size the gate against the whole payload, not a parsed subset, so content
	// hidden in fields the host doesn't model can't slip past the limiter.
	est := len(body) / charsPerToken
	if err := p.svc.WaitCapacity(c.Request.Context(), model, est); err != nil {
		c.JSON(http.StatusServiceUnavailable, fail("capacity wait: "+err.Error()))
		return
	}

	outBytes := p.forward(c, target, body)
	p.svc.Record(model, est, outBytes/charsPerToken)
}

// models — GET /v1/models. Ungated passthrough of the driver's loaded-model
// list (what the desktop polls); no body, no capacity gate.
func (p *InferenceProvider) models(c *gin.Context) {
	target, _, ok := p.svc.Target()
	if !ok {
		c.JSON(http.StatusServiceUnavailable, fail("no driver ready — serve a model first"))
		return
	}
	p.forward(c, target, nil)
}

// forward proxies the incoming request (method + path + optional body) to the
// active driver and streams the response back, flushing per chunk so SSE
// streaming works. Returns the number of response bytes copied (for the usage
// record on gated calls). A nil body means a bodyless request (e.g. GET /models).
func (p *InferenceProvider) forward(c *gin.Context, target string, body []byte) int {
	url := "http://" + target + c.Request.URL.Path
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	upReq, err := http.NewRequestWithContext(c.Request.Context(), c.Request.Method, url, reader)
	if err != nil {
		c.JSON(http.StatusInternalServerError, fail("build upstream request: "+err.Error()))
		return 0
	}
	if body != nil {
		upReq.Header.Set("Content-Type", "application/json")
	}

	resp, err := inferenceClient.Do(upReq)
	if err != nil {
		c.JSON(http.StatusBadGateway, fail("driver unreachable: "+err.Error()))
		return 0
	}
	defer func() { _ = resp.Body.Close() }()

	if ct := resp.Header.Get("Content-Type"); ct != "" {
		c.Header("Content-Type", ct)
	}
	c.Status(resp.StatusCode)

	flusher, _ := c.Writer.(http.Flusher)
	bufp := forwardBufPool.Get().(*[]byte)
	defer forwardBufPool.Put(bufp)
	buf := *bufp
	total := 0
	for {
		n, rerr := resp.Body.Read(buf)
		if n > 0 {
			if _, werr := c.Writer.Write(buf[:n]); werr != nil {
				break
			}
			total += n
			if flusher != nil {
				flusher.Flush()
			}
		}
		if rerr != nil {
			break
		}
	}
	return total
}

// Target returns the loopback address and served-model key of a ready driver,
// or ok=false if none is up. The model key is the driver's actual served model
// (the resource the limiter must account for) — never a client-supplied string,
// so usage can't be spread across buckets by varying the request's model field.
// Prefers mlx, then cuda, then amd; model-based routing across multiple live
// drivers lands with hot-swap.
func (s *Service) Target() (addr string, model string, ok bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, rt := range []string{RuntimeMLX, RuntimeCUDA, RuntimeAMD} {
		if sv := s.served[rt]; sv != nil && sv.Ready && s.running(sv.ProcessID) {
			key := sv.Model
			if key == "" {
				key = sv.Runtime
			}
			return sv.Addr, key, true
		}
	}
	return "", "", false
}

// WaitCapacity blocks until the limiter grants capacity for model — a no-op when
// no limiter is configured.
func (s *Service) WaitCapacity(ctx context.Context, model string, estTokens int) error {
	if s.limiter == nil {
		return nil
	}
	return s.limiter.WaitForCapacity(ctx, model, estTokens)
}

// Record books usage against the limiter — a no-op when no limiter is configured.
func (s *Service) Record(model string, promptTokens, outputTokens int) {
	if s.limiter == nil {
		return
	}
	s.limiter.RecordUsage(model, promptTokens, outputTokens)
}
