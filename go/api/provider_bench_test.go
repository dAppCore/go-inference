// SPDX-License-Identifier: EUPL-1.2

package api_test

import (
	"net/http"
	"net/http/httptest"
	"testing"

	coreapi "dappco.re/go/api"
	mlapi "dappco.re/go/inference/api"
	"github.com/gin-gonic/gin"
)

var (
	sinkProvider *mlapi.AIProvider
	sinkString   string
	sinkDescribe []coreapi.RouteDescription
	sinkInt      int
)

func BenchmarkNewProvider(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkProvider = mlapi.NewProvider()
	}
}

func BenchmarkNew(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkProvider = mlapi.New()
	}
}

func BenchmarkAIProviderName(b *testing.B) {
	p := mlapi.NewProvider()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkString = p.Name()
	}
}

func BenchmarkAIProviderBasePath(b *testing.B) {
	p := mlapi.NewProvider()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		sinkString = p.BasePath()
	}
}

func BenchmarkAIProviderDescribe(b *testing.B) {
	p := mlapi.NewProvider()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkDescribe = p.Describe()
	}
}

func BenchmarkAIProviderRegisterRoutes(b *testing.B) {
	p := mlapi.NewProvider()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		router := gin.New()
		p.RegisterRoutes(router.Group("/v1"))
		sinkInt = len(router.Routes())
	}
}

func BenchmarkAIProviderHealth(b *testing.B) {
	router := gin.New()
	mlapi.NewProvider().RegisterRoutes(router.Group("/v1"))
	req := httptest.NewRequest(http.MethodGet, "/v1/health", nil)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rec := httptest.NewRecorder()
		router.ServeHTTP(rec, req)
		sinkInt = rec.Code
	}
}
