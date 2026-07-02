// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	core "dappco.re/go"
	ratelimit "dappco.re/go/ratelimit"
	"github.com/gin-gonic/gin"
)

// --- Name / BasePath ---

func TestInference_Name_Good(t *testing.T) {
	p := NewInferenceProvider(nil)
	if got := p.Name(); got != "inference" {
		t.Fatalf("Name() = %q, want %q", got, "inference")
	}
}

func TestInference_BasePath_Good(t *testing.T) {
	p := NewInferenceProvider(nil)
	if got := p.BasePath(); got != "/v1" {
		t.Fatalf("BasePath() = %q, want %q", got, "/v1")
	}
}

// --- RegisterRoutes ---

func TestInference_RegisterRoutes_Good(t *testing.T) {
	p := NewInferenceProvider(nil)
	engine := gin.New()
	grp := engine.Group(p.BasePath())
	p.RegisterRoutes(grp)

	want := map[string]bool{
		http.MethodPost + " /v1/chat/completions": false,
		http.MethodPost + " /v1/completions":      false,
		http.MethodPost + " /v1/messages":         false,
		http.MethodGet + " /v1/models":            false,
	}
	for _, route := range engine.Routes() {
		key := route.Method + " " + route.Path
		if _, ok := want[key]; ok {
			want[key] = true
		}
	}
	for key, seen := range want {
		if !seen {
			t.Fatalf("RegisterRoutes did not register %s", key)
		}
	}
}

// TestInference_RegisterRoutes_Bad covers the nil-receiver guard — a nil
// InferenceProvider must be a safe no-op, not a panic.
func TestInference_RegisterRoutes_Bad(t *testing.T) {
	engine := gin.New()
	grp := engine.Group("/v1")
	var p *InferenceProvider
	p.RegisterRoutes(grp)
	if len(engine.Routes()) != 0 {
		t.Fatalf("RegisterRoutes on a nil provider registered %d routes, want 0", len(engine.Routes()))
	}
}

// --- chat ---

func TestInference_Chat_Good(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"chatcmpl-1","choices":[]}`))
	}))
	t.Cleanup(backend.Close)
	target := core.TrimPrefix(backend.URL, "http://")

	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	svc := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, Model: "lthn/LEM-Gemma3-1B", Addr: target, ProcessID: pid, Running: true, Ready: true},
	}}
	p := NewInferenceProvider(svc)

	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	body := []byte(`{"model":"lthn/LEM-Gemma3-1B","messages":[{"role":"user","content":"hi"}]}`)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(body))

	p.chat(c)

	if w.Code != http.StatusOK {
		t.Fatalf("chat status = %d, want 200; body=%s", w.Code, w.Body.String())
	}
	if !core.Contains(w.Body.String(), "chatcmpl-1") {
		t.Fatalf("chat body = %s, want the backend's response forwarded", w.Body.String())
	}
}

func TestInference_Chat_Bad(t *testing.T) {
	p := NewInferenceProvider(&Service{served: map[string]*Served{}})
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader([]byte(`{}`)))

	p.chat(c)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("chat status = %d, want 503 with no driver ready", w.Code)
	}
}

// TestInference_Chat_Ugly covers the request-body size cap — a body over
// maxChatRequestBytes must be rejected before any driver lookup happens.
func TestInference_Chat_Ugly(t *testing.T) {
	p := NewInferenceProvider(&Service{served: map[string]*Served{}})
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	oversized := bytes.Repeat([]byte("a"), maxChatRequestBytes+1)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(oversized))

	p.chat(c)

	if w.Code != http.StatusRequestEntityTooLarge {
		t.Fatalf("chat status = %d, want 413 over the request body cap", w.Code)
	}
}

// --- models (handler) ---

func TestInference_Models_Good(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"models":["m1"]}`))
	}))
	t.Cleanup(backend.Close)
	target := core.TrimPrefix(backend.URL, "http://")

	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	svc := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, Addr: target, ProcessID: pid, Running: true, Ready: true},
	}}
	p := NewInferenceProvider(svc)

	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodGet, "/v1/models", nil)

	p.models(c)

	if w.Code != http.StatusOK || !core.Contains(w.Body.String(), "m1") {
		t.Fatalf("models status=%d body=%s, want the backend's model list forwarded", w.Code, w.Body.String())
	}
}

func TestInference_Models_Bad(t *testing.T) {
	p := NewInferenceProvider(&Service{served: map[string]*Served{}})
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodGet, "/v1/models", nil)

	p.models(c)

	if w.Code != http.StatusServiceUnavailable {
		t.Fatalf("models status = %d, want 503 with no driver ready", w.Code)
	}
}

// --- forward ---

func TestInference_Forward_Good(t *testing.T) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain")
		w.WriteHeader(http.StatusCreated)
		_, _ = w.Write([]byte("hello-world"))
	}))
	t.Cleanup(backend.Close)
	target := core.TrimPrefix(backend.URL, "http://")

	p := NewInferenceProvider(&Service{served: map[string]*Served{}})
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	n := p.forward(c, target, []byte(`{"a":1}`))

	if n != len("hello-world") {
		t.Fatalf("forward returned %d bytes copied, want %d", n, len("hello-world"))
	}
	if w.Code != http.StatusCreated {
		t.Fatalf("forward status = %d, want the backend's status propagated", w.Code)
	}
	if w.Header().Get("Content-Type") != "text/plain" {
		t.Fatalf("forward Content-Type = %q, want it propagated", w.Header().Get("Content-Type"))
	}
	if w.Body.String() != "hello-world" {
		t.Fatalf("forward body = %q, want the backend's body streamed through", w.Body.String())
	}
}

func TestInference_Forward_Bad(t *testing.T) {
	deadAddr := freeDeadAddr(t)
	p := NewInferenceProvider(&Service{served: map[string]*Served{}})
	engine := gin.New()
	w := httptest.NewRecorder()
	c := gin.CreateTestContextOnly(w, engine)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)

	n := p.forward(c, deadAddr, []byte(`{}`))

	if n != 0 {
		t.Fatalf("forward against an unreachable target copied %d bytes, want 0", n)
	}
	if w.Code != http.StatusBadGateway {
		t.Fatalf("forward status = %d, want 502 for an unreachable driver", w.Code)
	}
}

// --- Target ---

func TestInference_Target_Good(t *testing.T) {
	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, Model: "org/model", Addr: "127.0.0.1:1", ProcessID: pid, Running: true, Ready: true},
	}}
	addr, model, ok := s.Target()
	if !ok || addr != "127.0.0.1:1" || model != "org/model" {
		t.Fatalf("Target() = (%q, %q, %t), want the ready mlx entry", addr, model, ok)
	}
}

func TestInference_Target_Bad(t *testing.T) {
	s := &Service{served: map[string]*Served{}}
	if _, _, ok := s.Target(); ok {
		t.Fatal("Target() ok=true with nothing served")
	}
}

// TestInference_Target_Ugly proves the runtime priority order (mlx before
// cuda) and the model-less fallback key (the runtime name substitutes for an
// empty Model) together: mlx is tracked but not ready, cuda is ready and
// model-less.
func TestInference_Target_Ugly(t *testing.T) {
	proc := benchProcSvc(t)
	mlxPID := benchSleepProc(t, proc)
	cudaPID := benchSleepProc(t, proc)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX:  {Runtime: RuntimeMLX, Addr: "127.0.0.1:1", ProcessID: mlxPID, Running: true, Ready: false},
		RuntimeCUDA: {Runtime: RuntimeCUDA, Addr: "127.0.0.1:2", ProcessID: cudaPID, Running: true, Ready: true},
	}}
	addr, model, ok := s.Target()
	if !ok || addr != "127.0.0.1:2" || model != RuntimeCUDA {
		t.Fatalf("Target() = (%q, %q, %t), want the ready cuda entry with the runtime-name fallback key", addr, model, ok)
	}
}

// --- WaitCapacity ---

func TestInference_WaitCapacity_Good(t *testing.T) {
	s := &Service{}
	if err := s.WaitCapacity(context.Background(), "any-model", 100); err != nil {
		t.Fatalf("WaitCapacity with no limiter = %v, want nil (no-op)", err)
	}
}

func TestInference_WaitCapacity_Bad(t *testing.T) {
	rl, err := ratelimit.New()
	if err != nil {
		t.Fatalf("ratelimit.New: %v", err)
	}
	s := &Service{limiter: rl}
	if err := s.WaitCapacity(context.Background(), "org/model", -1); err == nil {
		t.Fatal("WaitCapacity with negative tokens succeeded, want a rejection")
	}
}

// TestInference_WaitCapacity_Ugly exercises the real limiter's allow path for
// a model with no configured quota (unlimited by policy) — it must return
// immediately rather than blocking on the limiter's retry timer.
func TestInference_WaitCapacity_Ugly(t *testing.T) {
	rl, err := ratelimit.New()
	if err != nil {
		t.Fatalf("ratelimit.New: %v", err)
	}
	s := &Service{limiter: rl}

	start := time.Now()
	if err := s.WaitCapacity(context.Background(), "lthn/LEM-Gemma3-1B", 100); err != nil {
		t.Fatalf("WaitCapacity for an unconfigured model failed: %v", err)
	}
	if elapsed := time.Since(start); elapsed > time.Second {
		t.Fatalf("WaitCapacity took %s for an unconfigured (unlimited) model, want near-instant", elapsed)
	}
}

// --- Record ---

func TestInference_Record_Good(t *testing.T) {
	rl, err := ratelimit.New()
	if err != nil {
		t.Fatalf("ratelimit.New: %v", err)
	}
	s := &Service{limiter: rl}
	s.Record("org/model", 10, 20)

	stats := rl.State["org/model"]
	if stats == nil || len(stats.Requests) != 1 || len(stats.Tokens) != 1 || stats.Tokens[0].Count != 30 {
		t.Fatalf("RateLimiter.State[org/model] = %+v, want one recorded usage entry summing to 30", stats)
	}
}

func TestInference_Record_Bad(t *testing.T) {
	s := &Service{} // no limiter configured
	s.Record("org/model", 10, 20)
	if s.limiter != nil {
		t.Fatal("Record with no configured limiter somehow acquired one")
	}
}

// TestInference_Record_Ugly proves repeated calls accumulate rather than
// overwrite the model's usage history.
func TestInference_Record_Ugly(t *testing.T) {
	rl, err := ratelimit.New()
	if err != nil {
		t.Fatalf("ratelimit.New: %v", err)
	}
	s := &Service{limiter: rl}
	s.Record("org/model", 1, 1)
	s.Record("org/model", 2, 2)

	stats := rl.State["org/model"]
	if stats == nil || len(stats.Requests) != 2 || len(stats.Tokens) != 2 {
		t.Fatalf("RateLimiter.State[org/model] = %+v, want two accumulated usage entries", stats)
	}
}
