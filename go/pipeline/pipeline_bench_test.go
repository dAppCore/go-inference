// SPDX-Licence-Identifier: EUPL-1.2

// Allocation benchmarks for the request-serving pipeline (RFC §6). The pipeline
// composes per request, so its steady-state allocation profile is squarely on
// the hot path: every request threads the stage chain (Complete), and the wired
// adapters project the canonical request/response onto each collaborator's view
// (toCacheRequest, routerAdapter.Select, userTurns, sessionAdapter.Load,
// fusionPrompt) on the way through. One benchmark per public function plus the
// per-request transform helpers. Fixtures are built once, outside the measured
// loop, and every collaborator used in the composition benches is
// allocation-free, so the numbers reflect the pipeline's own allocations rather
// than the fakes'.
//
// White-box (package pipeline) because the per-request transforms worth
// profiling — withCorrective, userTurns, toCacheRequest, the adapters — are
// unexported, and the existing test fakes / lenCounter / fixedCounter are reused.
//
// Run: go test -bench=. -benchmem -run='^$' ./pipeline/
package pipeline

import (
	"context"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/ai"
	"dappco.re/go/inference/kv/budget"
	chat "dappco.re/go/inference/chat"
	"dappco.re/go/inference/kv/respcache"
	"dappco.re/go/inference/safety"
	"dappco.re/go/inference/session"
	"dappco.re/go/inference/usage"
	"dappco.re/go/inference/welfare"
)

// Sinks defeat dead-code elimination — each benchmarked call writes its result
// to a package-level sink of the matching type.
var (
	benchSinkResp     chat.Response
	benchSinkErr      error
	benchSinkReq      chat.Request
	benchSinkPipe     *Pipeline
	benchSinkEndpts   []Endpoint
	benchSinkStr      string
	benchSinkStrs     []string
	benchSinkDecision Decision
	benchSinkBool     bool
	benchSinkCacheReq respcache.Request
)

// --- bench fakes: allocation-free so a composition bench measures only the
// pipeline's own allocations -------------------------------------------------

// benchBackend serves one canned response with no per-call recording (the
// existing fakeBackend appends to seenIDs/seenReqs, which would add the fake's
// allocations to a Complete measurement).
type benchBackend struct{ resp chat.Response }

func (b benchBackend) Complete(_ context.Context, _ Endpoint, _ chat.Request) (chat.Response, error) {
	return b.resp, nil
}

// benchMediateGuard passes input and mediates the first output of every pair,
// passing the second — so each Complete mediates exactly once (the regenerate
// path, exercising withCorrective) for the whole measured loop, where the
// scripted fakeGuard's "last value sticks" would stop mediating after iter 1.
type benchMediateGuard struct{ n int }

func (g *benchMediateGuard) CheckInput(_ chat.Request) Decision { return DecisionPass }

func (g *benchMediateGuard) CheckOutput(_ chat.Request, _ chat.Response) Decision {
	g.n++
	if g.n%2 == 1 {
		return DecisionMediate
	}
	return DecisionPass
}

// --- fixtures ---------------------------------------------------------------

// benchUserReq — the overwhelmingly common single-user-turn request.
func benchUserReq() chat.Request {
	return chat.Request{Model: "gemma-4-31b", Messages: []chat.Message{chat.UserText("what is the capital of France?")}}
}

// benchMultiTurnReq — a realistic multi-turn transcript (3 user turns
// interleaved with assistant replies + a system preamble), the shape the guard
// and session stages walk.
func benchMultiTurnReq() chat.Request {
	return chat.Request{
		Model: "gemma-4-31b",
		Messages: []chat.Message{
			{Role: chat.System, Content: []chat.ContentBlock{chat.Text("You are a helpful assistant.")}},
			chat.UserText("first question about the weather"),
			{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text("a considered first reply")}},
			chat.UserText("a follow-up about tomorrow"),
			{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text("a considered second reply")}},
			chat.UserText("and the weekend?"),
		},
	}
}

// benchPool — a single-provider routing pool the ai selector picks from.
func benchPool() []ai.Endpoint {
	return []ai.Endpoint{{Provider: "local-metal", Model: "gemma-4-31b", Local: true, Free: true}}
}

// --- New / NewWired (pipeline build) ----------------------------------------

func BenchmarkPipeline_New(b *core.B) {
	cache := &fakeCache{}
	router := &fakeRouter{endpoints: []Endpoint{{ID: "local-metal"}}}
	guard := &fakeGuard{}
	sink := &fakeSink{}
	backend := benchBackend{}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPipe = New(cache, router, guard, sink, backend)
	}
}

func BenchmarkPipeline_NewWired(b *core.B) {
	w := Wiring{
		Backend:     benchBackend{},
		Pool:        benchPool(),
		Cache:       respcache.New(nil),
		Welfare:     welfare.New(welfare.Config{}),
		Sessions:    session.NewManager(session.NewMemoryStore()),
		Counter:     lenCounter{},
		Window:      8192,
		Pricing:     usage.Pricing{PromptPer1K: 1, CompletionPer1K: 2},
		RecordUsage: func(chat.Request, chat.Response, float64) {},
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkPipe = NewWired(w)
	}
}

// --- Complete: the per-request stage chain ----------------------------------

// BenchmarkPipeline_Complete_Core measures the pure composition cost over the
// core five seams, every one allocation-free, so the number is the pipeline's
// own per-request overhead on the cache-miss happy path.
func BenchmarkPipeline_Complete_Core(b *core.B) {
	cache := &fakeCache{}
	router := &fakeRouter{endpoints: []Endpoint{{ID: "local-metal"}}}
	guard := &fakeGuard{}
	sink := &fakeSink{}
	backend := benchBackend{resp: chat.Response{Text: "hello", FinishReason: "stop"}}
	p := New(cache, router, guard, sink, backend)
	req := benchUserReq()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkResp, benchSinkErr = p.Complete(ctx, req)
	}
}

// BenchmarkPipeline_Complete_CacheHit measures the exact-hit short-circuit
// (cache → append → return, no inference).
func BenchmarkPipeline_Complete_CacheHit(b *core.B) {
	cache := &fakeCache{present: true, hit: chat.Response{Text: "cached"}}
	router := &fakeRouter{endpoints: []Endpoint{{ID: "local-metal"}}}
	guard := &fakeGuard{}
	sink := &fakeSink{}
	backend := benchBackend{}
	p := New(cache, router, guard, sink, backend)
	req := benchUserReq()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkResp, benchSinkErr = p.Complete(ctx, req)
	}
}

// BenchmarkPipeline_Complete_Mediate measures the output-mediate regenerate
// path (two inferences + withCorrective) once per iteration.
func BenchmarkPipeline_Complete_Mediate(b *core.B) {
	cache := &fakeCache{}
	router := &fakeRouter{endpoints: []Endpoint{{ID: "local-metal"}}}
	guard := &benchMediateGuard{}
	sink := &fakeSink{}
	backend := benchBackend{resp: chat.Response{Text: "regenerated", FinishReason: "stop"}}
	p := New(cache, router, guard, sink, backend)
	req := benchMultiTurnReq()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkResp, benchSinkErr = p.Complete(ctx, req)
	}
}

// BenchmarkPipeline_Complete_Wired measures a realistic multi-stage wired path
// on the cache-miss line: pool routing + welfare/safety guard (input + output)
// + context fit + usage accounting. No cache (so every iteration runs the full
// path) and no session (so nothing grows across the loop).
func BenchmarkPipeline_Complete_Wired(b *core.B) {
	p := NewWired(Wiring{
		Backend:     benchBackend{resp: chat.Response{Text: "a polite considered reply", FinishReason: "stop"}},
		Pool:        benchPool(),
		Welfare:     welfare.New(welfare.Config{}),
		Counter:     lenCounter{},
		Window:      100000,
		Pricing:     usage.Pricing{PromptPer1K: 1, CompletionPer1K: 2},
		RecordUsage: func(chat.Request, chat.Response, float64) {},
	})
	req := benchUserReq()
	ctx := context.Background()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkResp, benchSinkErr = p.Complete(ctx, req)
	}
}

// --- BudgetFits (placement predicate) ---------------------------------------

func BenchmarkPipeline_BudgetFits(b *core.B) {
	bg := budget.New(fixedCounter{n: 1000})
	ep := budget.Endpoint{ContextLen: 8192, MemoryBudget: 16 << 30, BytesPerToken: 2}
	msgs := []chat.Message{chat.UserText("anything")}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkBool = BudgetFits(bg, msgs, "gemma-4-31b", 512, ep)
	}
}

// --- per-stage transforms (unexported helpers) ------------------------------

func BenchmarkPipeline_withCorrective(b *core.B) {
	req := benchMultiTurnReq()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkReq = withCorrective(req)
	}
}

func BenchmarkPipeline_userTurns_Single(b *core.B) {
	req := benchUserReq()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkStr, benchSinkStrs = userTurns(req)
	}
}

func BenchmarkPipeline_userTurns_Multi(b *core.B) {
	req := benchMultiTurnReq()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkStr, benchSinkStrs = userTurns(req)
	}
}

func BenchmarkPipeline_toCacheRequest(b *core.B) {
	req := benchMultiTurnReq()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkCacheReq = toCacheRequest(req)
	}
}

func BenchmarkPipeline_fromCompletion(b *core.B) {
	c := respcache.Completion{Text: "a stored completion", Model: "gemma-4-31b", FinishReason: "stop"}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkResp = fromCompletion(c)
	}
}

func BenchmarkPipeline_routerAdapter_BarePool(b *core.B) {
	a := &routerAdapter{}
	req := benchUserReq()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkEndpts, benchSinkErr = a.Select(req)
	}
}

func BenchmarkPipeline_routerAdapter_Pool(b *core.B) {
	a := &routerAdapter{pool: benchPool()}
	req := benchUserReq()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkEndpts, benchSinkErr = a.Select(req)
	}
}

func BenchmarkPipeline_fusionPrompt(b *core.B) {
	req := benchMultiTurnReq()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkStr = fusionPrompt(req)
	}
}

func BenchmarkPipeline_sessionAdapter_Load(b *core.B) {
	mgr := session.NewManager(session.NewMemoryStore())
	sess := mgr.Open("gemma-4-31b")
	// Seed three prior exchanges the caller will not resend (0% replay, §6.10).
	_, _ = mgr.Append(sess.ID, chat.UserText("first earlier question"))
	_, _ = mgr.Append(sess.ID, chat.Message{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text("first earlier answer")}})
	_, _ = mgr.Append(sess.ID, chat.UserText("second earlier question"))
	_, _ = mgr.Append(sess.ID, chat.Message{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text("second earlier answer")}})
	a := &sessionAdapter{manager: mgr}
	req := chat.Request{Model: "gemma-4-31b", SessionID: sess.ID, Messages: []chat.Message{chat.UserText("the new turn")}}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkReq, benchSinkErr = a.Load(req)
	}
}

func BenchmarkPipeline_guardAdapter_CheckInput(b *core.B) {
	a := &guardAdapter{welfare: welfare.New(welfare.Config{}), policy: safety.DefaultPolicy()}
	req := benchMultiTurnReq()
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		benchSinkDecision = a.CheckInput(req)
	}
}
