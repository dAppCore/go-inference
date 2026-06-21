// SPDX-Licence-Identifier: EUPL-1.2

package tools

import (
	"context"
	"sync"

	core "dappco.re/go"
)

// ToolResult is the outcome of running one ToolCall. ID correlates it back to
// the call (and so to the model's tool-call message); Content is the executor's
// reply to feed back to the model; Err, when non-nil, marks this call as failed
// — an unknown tool or an executor error — without aborting the rest of the
// batch.
type ToolResult struct {
	ID      string
	Content string
	Err     error
}

// Executor runs one tool call and returns its result. the own MCP tool
// server (§4.6) is just one Executor registered under its tool names; a server
// tool (web_search, code_interpreter, …) is another; a caller-supplied function
// tool is a third. The orchestration layer doesn't care which — it dispatches
// every call the same way.
//
//	type weatherExec struct{}
//	func (weatherExec) Execute(ctx context.Context, c tools.ToolCall) (tools.ToolResult, error) {
//	    return tools.ToolResult{ID: c.ID, Content: lookup(c.Arguments)}, nil
//	}
type Executor interface {
	Execute(ctx context.Context, call ToolCall) (ToolResult, error)
}

// Registry maps a tool name to the Executor that runs it. Safe to share across
// goroutines: Register takes a write lock, lookups a read lock, so Dispatch can
// fan out concurrently over a registry other goroutines may still be filling.
//
//	reg := tools.NewRegistry()
//	reg.Register("web_search", mcpServer)
//	reg.Register("get_weather", weatherExec{})
type Registry struct {
	mu   sync.RWMutex
	exec map[string]Executor
}

// NewRegistry returns an empty Registry ready for Register.
func NewRegistry() *Registry {
	return &Registry{exec: make(map[string]Executor)}
}

// Register binds an Executor to a tool name, replacing any prior binding for
// that name (last registration wins — a host tool can override a default).
func (r *Registry) Register(name string, exec Executor) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.exec[name] = exec
}

// Lookup returns the Executor for a tool name and whether one is registered.
//
//	if exec, ok := reg.Lookup(call.Name); ok { exec.Execute(ctx, call) }
func (r *Registry) Lookup(name string) (Executor, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	exec, ok := r.exec[name]
	return exec, ok
}

// Dispatch runs every call through its registered Executor and collects the
// results in input order. When parallel is true the calls run concurrently (one
// goroutine each, results written to their own slot so no lock is needed); when
// false they run in sequence.
//
// A batch never aborts: an unknown tool, or an executor that errors or panics,
// becomes a ToolResult with Err set in that call's slot — the other calls still
// run and return their results. This is what lets parallel_tool_calls (§6.4)
// degrade gracefully when one of several calls fails.
//
//	results := tools.Dispatch(ctx, calls, registry, true)
//	for _, res := range results {
//	    if res.Err != nil { /* surface the failure for this call */ }
//	}
func Dispatch(ctx context.Context, calls []ToolCall, registry *Registry, parallel bool) []ToolResult {
	results := make([]ToolResult, len(calls))

	if !parallel {
		for i, call := range calls {
			results[i] = runOne(ctx, call, registry)
		}
		return results
	}

	var wg sync.WaitGroup
	wg.Add(len(calls))
	for i := range calls {
		go func(i int) {
			defer wg.Done()
			results[i] = runOne(ctx, calls[i], registry)
		}(i)
	}
	wg.Wait()
	return results
}

// runOne resolves one call's executor and runs it, turning every failure mode —
// unknown tool, executor error, executor panic — into a ToolResult carrying the
// call's ID and the error, so the batch never collapses on a single bad call.
func runOne(ctx context.Context, call ToolCall, registry *Registry) (res ToolResult) {
	exec, ok := registry.Lookup(call.Name)
	if !ok {
		return ToolResult{ID: call.ID, Err: core.E("tools", "no executor registered for tool: "+call.Name, nil)}
	}

	// A misbehaving executor must not take down the whole dispatch — recover its
	// panic into the result slot like any other failure.
	defer func() {
		if p := recover(); p != nil {
			res = ToolResult{ID: call.ID, Err: core.E("tools", "executor panicked", nil)}
		}
	}()

	out, err := exec.Execute(ctx, call)
	if err != nil {
		return ToolResult{ID: call.ID, Err: core.E("tools", "execute tool: "+call.Name, err)}
	}
	// Trust the executor's ID if it set one, but default to the call's ID so a
	// terse executor still produces a correlatable result.
	if out.ID == "" {
		out.ID = call.ID
	}
	return out
}
