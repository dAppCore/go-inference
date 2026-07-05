// SPDX-License-Identifier: EUPL-1.2

package driver

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	coreprocess "dappco.re/go/process"
	"github.com/gin-gonic/gin"
)

// These benchmarks exercise the per-request inference-dispatch path the proxy
// pays on every chat request: Service.Target (the ready-driver lookup),
// InferenceProvider.forward (build the upstream request + stream the reply),
// and InferenceProvider.chat end-to-end against a mock engine backend. They
// rank allocs/op + B/op so the genuinely-avoidable per-request allocations are
// separated from the inherent stdlib HTTP cost.

// Package sinks keep the benchmarked work from being optimised away.
var (
	benchTargetAddr  string
	benchTargetModel string
	benchTargetOK    bool
	benchForwardN    int
)

// benchProcSvc spins up a real go-process Service so running() observes a live
// process — the realistic state Target/chat see when a driver is up.
func benchProcSvc(tb testing.TB) *coreprocess.Service {
	tb.Helper()
	app := core.New(core.WithName("process", coreprocess.NewService(coreprocess.Options{})))
	svc, ok := core.ServiceFor[*coreprocess.Service](app, "process")
	if !ok {
		tb.Fatal("process supervisor not registered")
	}
	return svc
}

// benchSleepProc starts a long-lived child so running(proc.ID) stays true for
// the whole benchmark, and registers its kill as cleanup.
func benchSleepProc(tb testing.TB, proc *coreprocess.Service) string {
	tb.Helper()
	r := proc.StartWithOptions(core.Background(), coreprocess.RunOptions{
		Command:   "/bin/sleep",
		Args:      []string{"3600"},
		Detach:    true,
		KillGroup: true,
	})
	if !r.OK {
		tb.Skipf("cannot spawn helper process: %v", r.Value)
	}
	p := r.Value.(*coreprocess.Process)
	tb.Cleanup(func() { _ = proc.Kill(p.ID) })
	return p.ID
}

// benchChatBody is a realistic OpenAI chat-completion request payload.
var benchChatBody = []byte(`{"model":"lthn/LEM-Gemma3-1B","stream":true,"messages":[` +
	`{"role":"system","content":"You are a concise assistant."},` +
	`{"role":"user","content":"Summarise the Lethean LEM Runtime split in two sentences."}]}`)

// benchSSEResponse is a ~64KB SSE-shaped reply, the modelled completion stream
// the proxy copies back (matches inference_copybuf_bench_test's response size).
var benchSSEResponse = func() []byte {
	const chunk = "data: {\"choices\":[{\"delta\":{\"content\":\"token \"}}]}\n\n"
	out := make([]byte, 0, 64*1024)
	for len(out) < 64*1024 {
		out = append(out, chunk...)
	}
	return out
}()

// BenchmarkServiceTarget_NoneReady measures Target when no driver is up — the
// real pre-serve 503 path. It iterates the full runtime list, isolating any
// per-call allocation in the lookup itself (the runtime slice it ranges).
func BenchmarkServiceTarget_NoneReady(b *testing.B) {
	s := &Service{served: map[string]*Served{}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchTargetAddr, benchTargetModel, benchTargetOK = s.Target()
	}
}

// BenchmarkServiceTarget_Ready measures the per-request hit path: an mlx driver
// ready + running, Target returns its addr + model key. This is what chat pays
// on every request once a model is served.
func BenchmarkServiceTarget_Ready(b *testing.B) {
	proc := benchProcSvc(b)
	pid := benchSleepProc(b, proc)
	s := &Service{
		proc: proc,
		served: map[string]*Served{
			RuntimeMLX: {Runtime: RuntimeMLX, Model: "lthn/LEM-Gemma3-1B", Addr: "127.0.0.1:36911", ProcessID: pid, Running: true, Ready: true},
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchTargetAddr, benchTargetModel, benchTargetOK = s.Target()
	}
}

// BenchmarkInferenceForward measures the proxy core: build the upstream request
// and stream the mock engine's reply back through the pooled copy buffer.
func BenchmarkInferenceForward(b *testing.B) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(benchSSEResponse)
	}))
	b.Cleanup(backend.Close)
	target := core.TrimPrefix(backend.URL, "http://")

	p := NewInferenceProvider(&Service{served: map[string]*Served{}})
	engine := gin.New()
	gin.SetMode(gin.ReleaseMode)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		c := gin.CreateTestContextOnly(w, engine)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
		benchForwardN = p.forward(c, target, benchChatBody)
	}
}

// BenchmarkInferenceChat measures the full per-request handler: cap+read body,
// resolve the ready driver, gate (nil limiter = no-op), forward, record usage —
// against a live process for running() and a mock engine for the upstream.
func BenchmarkInferenceChat(b *testing.B) {
	backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write(benchSSEResponse)
	}))
	b.Cleanup(backend.Close)
	target := core.TrimPrefix(backend.URL, "http://")

	proc := benchProcSvc(b)
	pid := benchSleepProc(b, proc)
	p := NewInferenceProvider(&Service{
		proc: proc,
		served: map[string]*Served{
			RuntimeMLX: {Runtime: RuntimeMLX, Model: "lthn/LEM-Gemma3-1B", Addr: target, ProcessID: pid, Running: true, Ready: true},
		},
	})
	engine := gin.New()
	gin.SetMode(gin.ReleaseMode)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		c := gin.CreateTestContextOnly(w, engine)
		c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", bytes.NewReader(benchChatBody))
		p.chat(c)
	}
}
