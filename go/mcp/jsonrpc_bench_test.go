// SPDX-Licence-Identifier: EUPL-1.2

package mcp

import (
	"context"
	"testing"

	core "dappco.re/go"
)

// AX-11 baseline benchmarks for the mcp/jsonrpc hot path.
//
// HandleFrame is the per-frame entry — every inbound MCP message
// (tools/list, tools/call, initialize, ping) shells through it.
// decodeRPCRequest and marshalRPCResponse fire on every frame in
// both directions. handleMethod's switch is the dispatch core.
//
// Run:
//   go test -bench=. -benchmem -benchtime=300ms ./mcp/...

// Sinks.
var (
	jsonrpcBenchSinkResult core.Result
	jsonrpcBenchSinkBytes  []byte
)

// --- fixtures ---

func benchInitialiseFrame() []byte {
	return []byte(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{}}}`)
}

func benchPingFrame() []byte {
	return []byte(`{"jsonrpc":"2.0","id":2,"method":"ping"}`)
}

func benchToolsListFrame() []byte {
	return []byte(`{"jsonrpc":"2.0","id":3,"method":"tools/list"}`)
}

func benchToolsCallFrame() []byte {
	// lang_detect is a built-in tool with a real typed-input handler;
	// its path exercises typedHandler[I] (the generic wrapper that
	// every typed tool shares).
	return []byte(`{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"lang_detect","arguments":{"path":"main.go"}}}`)
}

func benchService() *Service {
	result := New()
	if !result.OK {
		return nil
	}
	return result.Value.(*Service)
}

// --- HandleFrame — per-frame entry ---

func BenchmarkJSONRPC_HandleFrame_Initialise(b *testing.B) {
	svc := benchService()
	if svc == nil {
		b.Skip("New() failed")
	}
	frame := benchInitialiseFrame()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonrpcBenchSinkResult = svc.HandleFrame(ctx, frame)
	}
}

func BenchmarkJSONRPC_HandleFrame_Ping(b *testing.B) {
	svc := benchService()
	if svc == nil {
		b.Skip("New() failed")
	}
	frame := benchPingFrame()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonrpcBenchSinkResult = svc.HandleFrame(ctx, frame)
	}
}

func BenchmarkJSONRPC_HandleFrame_ToolsList(b *testing.B) {
	svc := benchService()
	if svc == nil {
		b.Skip("New() failed")
	}
	frame := benchToolsListFrame()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonrpcBenchSinkResult = svc.HandleFrame(ctx, frame)
	}
}

// BenchmarkJSONRPC_HandleFrame_ToolsCall exercises the tools/call
// path including the typedHandler[I] wrapper that every typed tool
// shares. The lang_detect tool is built-in and accepts a single-field
// path argument — minimal payload that still walks the full
// decodeRPCRequest → handleToolCall → typedHandler → JSONMarshal
// response pipeline.
func BenchmarkJSONRPC_HandleFrame_ToolsCall(b *testing.B) {
	svc := benchService()
	if svc == nil {
		b.Skip("New() failed")
	}
	frame := benchToolsCallFrame()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonrpcBenchSinkResult = svc.HandleFrame(ctx, frame)
	}
}

// --- decodeRPCRequest — per-frame parse ---

func BenchmarkJSONRPC_decodeRPCRequest_Initialise(b *testing.B) {
	frame := benchInitialiseFrame()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonrpcBenchSinkResult = decodeRPCRequest(frame)
	}
}

func BenchmarkJSONRPC_decodeRPCRequest_Ping(b *testing.B) {
	frame := benchPingFrame()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonrpcBenchSinkResult = decodeRPCRequest(frame)
	}
}

func BenchmarkJSONRPC_decodeCallToolParams_Typical(b *testing.B) {
	raw := RawMessage(`{"name":"echo","arguments":{"message":"hi"}}`)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonrpcBenchSinkResult = decodeCallToolParams(raw)
	}
}

// --- marshalRPCResponse — per-response build ---

func BenchmarkJSONRPC_marshalRPCResponse_Success(b *testing.B) {
	resp := rpcResponse{
		JSONRPC: "2.0",
		ID:      float64(1),
		Result:  map[string]any{"ok": true},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonrpcBenchSinkBytes = marshalRPCResponse(resp)
	}
}

func BenchmarkJSONRPC_marshalRPCResponse_Error(b *testing.B) {
	resp := rpcResponse{
		JSONRPC: "2.0",
		ID:      float64(1),
		Error:   &rpcError{Code: -32601, Message: "method not found"},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		jsonrpcBenchSinkBytes = marshalRPCResponse(resp)
	}
}

// --- rpcCodeForError — error code dispatch (zero alloc target) ---

func BenchmarkJSONRPC_rpcCodeForError_InvalidRequest(b *testing.B) {
	err := errInvalidRequest
	var sink int
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sink = rpcCodeForError(err)
	}
	_ = sink
}

// --- AX-11 alloc-budget gates ---

// TestAllocBudget_JSONRPC_HandleFrame_Ping locks the cheapest method
// dispatch — ping has no params, no logic, just returns empty map.
// Should be the alloc floor for the whole HandleFrame surface.
func TestAllocBudget_JSONRPC_HandleFrame_Ping(t *testing.T) {
	svc := benchService()
	if svc == nil {
		t.Fatalf("New() failed")
	}
	frame := benchPingFrame()
	ctx := context.Background()

	// Behavioural lock — ping returns a valid JSON-RPC response.
	r := svc.HandleFrame(ctx, frame)
	if !r.OK {
		t.Fatalf("HandleFrame(ping) failed: %v", r.Value)
	}
	if len(r.Value.([]byte)) == 0 {
		t.Fatalf("HandleFrame(ping) returned empty response")
	}

	avg := testing.AllocsPerRun(5, func() {
		jsonrpcBenchSinkResult = svc.HandleFrame(ctx, frame)
	})
	// Ceiling: 36 — current measured 31 (Apple M3 Ultra), ~16%
	// headroom. The shape: decodeRPCRequest (JSON unmarshal to
	// map[string]any + per-field marshal-back into RawMessage),
	// handleMethod dispatch + constructed result map, marshalRPCResponse
	// JSON marshal back. Ping is the floor — bigger methods
	// (tools/list, tools/call) add proportionally more.
	const budget = 36.0
	if avg > budget {
		t.Fatalf("HandleFrame(ping) alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Fires per inbound MCP frame — per-request floor.\n"+
			"Profile: go test -bench=BenchmarkJSONRPC_HandleFrame_Ping -benchmem -memprofile=/tmp/h.mem",
			avg, budget)
	}
}

// TestAllocBudget_JSONRPC_decodeRPCRequest_Ping locks the per-frame
// parse cost for the smallest valid request shape.
func TestAllocBudget_JSONRPC_decodeRPCRequest_Ping(t *testing.T) {
	frame := benchPingFrame()

	// Behavioural lock — extracts jsonrpc + id + method.
	r := decodeRPCRequest(frame)
	if !r.OK {
		t.Fatalf("decodeRPCRequest(ping) failed: %v", r.Value)
	}
	req := r.Value.(rpcRequest)
	if req.JSONRPC != "2.0" || req.Method != "ping" {
		t.Fatalf("decodeRPCRequest(ping) wrong fields: %+v", req)
	}

	avg := testing.AllocsPerRun(5, func() {
		jsonrpcBenchSinkResult = decodeRPCRequest(frame)
	})
	// Ceiling: 23 — current measured 20 (Apple M3 Ultra), ~15%
	// headroom. The shape: json.Unmarshal into map[string]any
	// allocates per key + per value (jsonrpc string, id float64
	// boxed, method string). Then ID is re-marshalled to RawMessage.
	const budget = 23.0
	if avg > budget {
		t.Fatalf("decodeRPCRequest(ping) alloc budget exceeded: %.1f allocs/call (budget=%.0f)",
			avg, budget)
	}
}

// TestAllocBudget_JSONRPC_HandleFrame_ToolsCall locks the tools/call
// path through the typedHandler wrapper. Every typed-tool MCP call
// pays this floor: decodeRPCRequest → handleToolCall → typedHandler →
// JSON re-marshal of the tool's structured result.
func TestAllocBudget_JSONRPC_HandleFrame_ToolsCall(t *testing.T) {
	svc := benchService()
	if svc == nil {
		t.Fatalf("New() failed")
	}
	frame := benchToolsCallFrame()
	ctx := context.Background()

	// Behavioural lock — tools/call returns a valid JSON-RPC response
	// with a structured-content payload.
	r := svc.HandleFrame(ctx, frame)
	if !r.OK {
		t.Fatalf("HandleFrame(tools/call) failed: %v", r.Value)
	}
	if len(r.Value.([]byte)) == 0 {
		t.Fatalf("HandleFrame(tools/call) returned empty response")
	}

	avg := testing.AllocsPerRun(5, func() {
		jsonrpcBenchSinkResult = svc.HandleFrame(ctx, frame)
	})
	// Ceiling: 120 — current measured 106 (Apple M3 Ultra), ~13%
	// headroom. The shape: decodeRPCRequest (≈20), handleToolCall
	// inner decodeCallToolParams (≈24), typedHandler wrap (string
	// trim + JSON unmarshal into typed input ≈10), the tool body
	// (≈25), structured-content map + JSON re-marshal of the result
	// (≈25), final marshalRPCResponse (≈6).
	const budget = 120.0
	if avg > budget {
		t.Fatalf("HandleFrame(tools/call) alloc budget exceeded: %.1f allocs/call (budget=%.0f)\n"+
			"Fires per typed-tool tools/call MCP frame — per-call floor.\n"+
			"Profile: go test -bench=BenchmarkJSONRPC_HandleFrame_ToolsCall -benchmem -memprofile=/tmp/tc.mem",
			avg, budget)
	}
}

// TestAllocBudget_JSONRPC_rpcCodeForError locks the error→code dispatch.
// Pure switch on errors.Is — should be zero allocs.
func TestAllocBudget_JSONRPC_rpcCodeForError(t *testing.T) {
	err := errInvalidRequest

	// Behavioural lock — invalid-request error maps to -32600.
	if code := rpcCodeForError(err); code != -32600 {
		t.Fatalf("rpcCodeForError(errInvalidRequest) = %d, want -32600", code)
	}

	avg := testing.AllocsPerRun(5, func() {
		_ = rpcCodeForError(err)
	})
	// Ceiling: 0 — errors.Is is alloc-free on sentinel comparison.
	const budget = 0.0
	if avg > budget {
		t.Fatalf("rpcCodeForError alloc budget exceeded: %.1f allocs/call (budget=%.0f)",
			avg, budget)
	}
}
