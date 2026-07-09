package mcp

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Benchmarks isolate this package's own per-call allocations. The handlers'
// logging branch (gated by m.logger != nil) is left disabled here: it routes
// into dappco.re/go's core.Log.Info + core.Username (an OS user lookup) which
// allocate outside this package and are inherent to structured logging, so
// including them would measure dappco.re/go, not mlservice/mcp.

var (
	benchResult  core.Result
	benchStrings []string
)

func BenchmarkMLSubsystem_mlGenerate(b *core.B) {
	svc := newMCPTestService()
	sub := NewMLSubsystem(svc)
	sub.logger = nil
	ctx := context.Background()
	input := MLGenerateInput{Prompt: "hello world", Backend: "test", Temperature: 0.7, MaxTokens: 128}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = sub.mlGenerate(ctx, nil, input)
	}
}

func BenchmarkMLSubsystem_mlScore(b *core.B) {
	svc := newMCPTestService()
	sub := NewMLSubsystem(svc)
	sub.logger = nil
	ctx := context.Background()
	input := MLScoreInput{Prompt: "what is 2+2", Response: "The answer is 4."}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = sub.mlScore(ctx, nil, input)
	}
}

func BenchmarkMLSubsystem_mlProbe(b *core.B) {
	svc := newMCPTestService()
	sub := NewMLSubsystem(svc)
	sub.logger = nil
	ctx := context.Background()
	input := MLProbeInput{Backend: "test"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = sub.mlProbe(ctx, nil, input)
	}
}

func BenchmarkMLSubsystem_mlProbe_Filtered(b *core.B) {
	svc := newMCPTestService()
	sub := NewMLSubsystem(svc)
	sub.logger = nil
	ctx := context.Background()
	input := MLProbeInput{Backend: "test", Categories: "arithmetic"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = sub.mlProbe(ctx, nil, input)
	}
}

func BenchmarkMLSubsystem_mlBackends(b *core.B) {
	svc := newMCPTestService()
	sub := NewMLSubsystem(svc)
	sub.logger = nil
	ctx := context.Background()
	input := MLBackendsInput{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchResult = sub.mlBackends(ctx, nil, input)
	}
}

func BenchmarkMLSubsystem_capabilityIDStrings(b *core.B) {
	ids := []inference.CapabilityID{"chat", "generate", "embed", "vision", "tools"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchStrings = capabilityIDStrings(ids)
	}
}
