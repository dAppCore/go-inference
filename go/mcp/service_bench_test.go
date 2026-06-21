// SPDX-Licence-Identifier: EUPL-1.2

package mcp

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// AX-11 baseline benchmarks for the mcp/service surface.
//
// Tools / ToolNames fire on every tools/list MCP frame — every agent
// discovery pays this. RegisterTool is per-startup-per-tool but its
// alloc shape governs the floor of the whole tool catalogue.
//
// Run:
//   go test -bench=. -benchmem -benchtime=300ms ./mcp/...

// Sinks.
var (
	serviceBenchSinkRecords []ToolRecord
	serviceBenchSinkNames   []string
	serviceBenchSinkResult  core.Result
)

// --- Tools — per-tools/list-frame discovery ---

func BenchmarkService_Tools_BuiltInInventory(b *testing.B) {
	svc := benchService()
	if svc == nil {
		b.Skip("New() failed")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		serviceBenchSinkRecords = svc.Tools()
	}
}

func BenchmarkService_ToolNames_BuiltInInventory(b *testing.B) {
	svc := benchService()
	if svc == nil {
		b.Skip("New() failed")
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		serviceBenchSinkNames = svc.ToolNames()
	}
}

// --- RegisterTool — per-startup-per-tool ---

func BenchmarkService_RegisterTool_NewTool(b *testing.B) {
	tool := Tool{
		Name:        "bench.tool",
		Description: "bench fixture",
		Handler: func(ctx context.Context, _ RawMessage) core.Result {
			return core.Ok(nil)
		},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		svc := benchService()
		if svc == nil {
			b.Skip("New() failed")
		}
		serviceBenchSinkResult = svc.RegisterTool(tool)
	}
}

// --- AX-11 alloc-budget gates ---

// TestAllocBudget_Service_Tools locks the per-discovery clone cost.
// Each tool record gets a cloned InputSchema map; the cost scales
// with the registered tool count. The default catalogue is the floor
// every agent client pays on the first tools/list frame.
func TestAllocBudget_Service_Tools(t *testing.T) {
	svc := benchService()
	if svc == nil {
		t.Fatalf("New() failed")
	}

	// Behavioural lock — at least the built-in tools land in the catalogue.
	records := svc.Tools()
	if len(records) == 0 {
		t.Fatalf("Tools() returned empty catalogue — expected built-in inventory")
	}
	for i, rec := range records {
		if rec.Name == "" {
			t.Fatalf("Tools()[%d] has empty name", i)
		}
	}

	avg := testing.AllocsPerRun(5, func() {
		serviceBenchSinkRecords = svc.Tools()
	})
	// Ceiling: 115 — current measured 99 (Apple M3 Ultra), ~16%
	// headroom. Per-record: 1 alloc for the ToolRecord copy + N for
	// the cloned InputSchema map. A regression that adds an alloc
	// per tool fails this gate at the next discovery — keeps the
	// tools/list floor bounded.
	const budget = 115.0
	if avg > budget {
		t.Fatalf("Tools() alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Fires per tools/list MCP frame — per-discovery floor.\n"+
			"Profile: go test -bench=BenchmarkService_Tools -benchmem -memprofile=/tmp/t.mem",
			avg, budget)
	}
}

// TestAllocBudget_Service_ToolNames locks the per-call name list.
// slices.Clone of the tool order — 1 alloc for the backing array.
func TestAllocBudget_Service_ToolNames(t *testing.T) {
	svc := benchService()
	if svc == nil {
		t.Fatalf("New() failed")
	}

	// Behavioural lock — names come back in registration order.
	names := svc.ToolNames()
	if len(names) == 0 {
		t.Fatalf("ToolNames() returned empty list")
	}

	avg := testing.AllocsPerRun(5, func() {
		serviceBenchSinkNames = svc.ToolNames()
	})
	// Ceiling: 2 — current measured 1 (Apple M3 Ultra: backing array).
	// slices.Clone is the floor — anything more is regression.
	const budget = 2.0
	if avg > budget {
		t.Fatalf("ToolNames() alloc budget exceeded: %.1f allocs/call (budget=%.0f)",
			avg, budget)
	}
}
