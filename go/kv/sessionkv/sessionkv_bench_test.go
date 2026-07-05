// SPDX-License-Identifier: EUPL-1.2

package sessionkv

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	coreapi "dappco.re/go/api"
	"dappco.re/go/inference/state"
	"github.com/gin-gonic/gin"
)

// Package-level sinks so the compiler can't elide the benchmarked work.
var (
	benchCodeSink int
	benchDescSink []coreapi.RouteDescription
)

// benchHost opens a session.kv store with a single stored chunk so the
// chunkRef hit path resolves, and returns a router with the routes mounted.
func benchHost(b *testing.B) (*Host, *gin.Engine) {
	b.Helper()
	gin.SetMode(gin.TestMode)
	path := core.PathJoin(b.TempDir(), "session.kv")
	host, err := Open(context.Background(), path)
	if err != nil {
		b.Fatalf("Open: %v", err)
	}
	b.Cleanup(func() { host.Close() })
	if _, err := host.store.Put(context.Background(), "remembered", state.PutOptions{Kind: "note"}); err != nil {
		b.Fatalf("Put: %v", err)
	}
	r := gin.New()
	host.RegisterRoutes(r.Group(host.BasePath()))
	return host, r
}

// BenchmarkStatus drives GET /v1/state/status — the inspection status handler
// (gin.H map build + JSON render) per request.
func BenchmarkStatus(b *testing.B) {
	_, r := benchHost(b)
	req := httptest.NewRequest(http.MethodGet, "/v1/state/status", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		benchCodeSink = w.Code
	}
}

// BenchmarkChunkRefHit drives GET /v1/state/chunks/1 — id parse, store Resolve,
// gin.H ref render for a chunk that exists.
func BenchmarkChunkRefHit(b *testing.B) {
	_, r := benchHost(b)
	req := httptest.NewRequest(http.MethodGet, "/v1/state/chunks/1", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		benchCodeSink = w.Code
	}
}

// BenchmarkChunkRefMiss drives GET /v1/state/chunks/999 — id parse, Resolve
// miss, 404 error render.
func BenchmarkChunkRefMiss(b *testing.B) {
	_, r := benchHost(b)
	req := httptest.NewRequest(http.MethodGet, "/v1/state/chunks/999", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		benchCodeSink = w.Code
	}
}

// BenchmarkChunkRefBadID drives GET /v1/state/chunks/abc — non-integer id, 400
// error render (no store touch).
func BenchmarkChunkRefBadID(b *testing.B) {
	_, r := benchHost(b)
	req := httptest.NewRequest(http.MethodGet, "/v1/state/chunks/abc", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		r.ServeHTTP(w, req)
		benchCodeSink = w.Code
	}
}

// BenchmarkDescribe measures the OpenAPI route-description build.
func BenchmarkDescribe(b *testing.B) {
	host, _ := benchHost(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchDescSink = host.Describe()
	}
}
