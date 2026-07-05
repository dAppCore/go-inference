package mlservice

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/eval/score"
	"dappco.re/go/inference/serving"
)

var (
	benchStrings []string
	benchBackend serving.Backend
	benchResult  core.Result
	benchReport  inference.CapabilityReport
	benchBool    bool
)

// benchBackendNames is a realistic spread of registered inference backends.
var benchBackendNames = []string{"ollama", "llamacpp", "mlx", "openai", "anthropic", "local"}

func benchService(b *testing.B) *Service {
	b.Helper()
	r := NewService(Options{})(core.New())
	if !r.OK {
		b.Fatalf("new service: %s", r.Error())
	}
	svc := r.Value.(*Service)
	for _, name := range benchBackendNames {
		svc.RegisterBackend(name, &testBackend{name: name, available: true})
	}
	return svc
}

func BenchmarkService_Backends(b *testing.B) {
	svc := benchService(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStrings = svc.Backends()
	}
}

func BenchmarkService_BackendsIter(b *testing.B) {
	svc := benchService(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		count := 0
		for range svc.BackendsIter() {
			count++
		}
		benchBool = count > 0
	}
}

func BenchmarkService_Backend(b *testing.B) {
	svc := benchService(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBackend = svc.Backend("mlx")
	}
}

func BenchmarkService_DefaultBackend(b *testing.B) {
	svc := benchService(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchBackend = svc.DefaultBackend()
	}
}

func BenchmarkService_BackendCapabilities(b *testing.B) {
	svc := benchService(b)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchReport, benchBool = svc.BackendCapabilities("mlx")
	}
}

func BenchmarkService_Generate(b *testing.B) {
	svc := benchService(b)
	svc.RegisterBackend("mlx", &testBackend{name: "mlx", available: true, result: serving.Result{Text: "generated output"}})
	ctx := context.Background()
	opts := serving.GenOpts{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = svc.Generate(ctx, "mlx", "hello world", opts)
	}
}

func BenchmarkService_ScoreResponses(b *testing.B) {
	svc := benchService(b)
	svc.engine = score.NewEngine(score.NewJudge(&testBackend{result: serving.Result{Text: `{"sovereignty":1,"ethical_depth":1,"creative_expression":1,"self_concept":1}`}}), 1, "heuristic")
	ctx := context.Background()
	responses := []score.Response{
		{ID: "r1", Model: "m", Response: "a substantial response about sovereignty and ethical depth"},
		{ID: "r2", Model: "m", Response: "another response exploring creative expression and self concept"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = svc.ScoreResponses(ctx, responses)
	}
}
