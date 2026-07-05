// SPDX-License-Identifier: EUPL-1.2

package api_test

import (
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	mlapi "dappco.re/go/inference/serving/api"
	"github.com/gin-gonic/gin"
)

var (
	sinkRoutes  *mlapi.Routes
	sinkStrings []string
	sinkRoutesS string
)

func BenchmarkNewRoutes(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkRoutes = mlapi.NewRoutes(nil)
	}
}

func BenchmarkRoutesName(b *testing.B) {
	r := mlapi.NewRoutes(nil)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkRoutesS = r.Name()
	}
}

func BenchmarkRoutesBasePath(b *testing.B) {
	r := mlapi.NewRoutes(nil)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkRoutesS = r.BasePath()
	}
}

func BenchmarkRoutesChannels(b *testing.B) {
	r := mlapi.NewRoutes(nil)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkStrings = r.Channels()
	}
}

func BenchmarkRoutesListBackends(b *testing.B) {
	svc := newTestService()
	r := mlapi.NewRoutes(svc)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(w)
		ctx.Request, _ = http.NewRequest(http.MethodGet, "/v1/ml/backends", nil)
		r.ListBackends(ctx)
	}
}

func BenchmarkRoutesStatus(b *testing.B) {
	svc := newTestService()
	r := mlapi.NewRoutes(svc)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(w)
		ctx.Request, _ = http.NewRequest(http.MethodGet, "/v1/ml/status", nil)
		r.Status(ctx)
	}
}

func BenchmarkRoutesGenerateBadRequest(b *testing.B) {
	svc := newTestService()
	r := mlapi.NewRoutes(svc)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		w := httptest.NewRecorder()
		ctx, _ := gin.CreateTestContext(w)
		ctx.Request, _ = http.NewRequest(http.MethodPost, "/v1/ml/generate", core.NewReader(`not json`))
		ctx.Request.Header.Set("Content-Type", "application/json")
		r.Generate(ctx)
	}
}
