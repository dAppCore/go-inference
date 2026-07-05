// SPDX-Licence-Identifier: EUPL-1.2

package pipeline

import (
	"context"

	core "dappco.re/go"
	chat "dappco.re/go/inference/serving/chat"
)

// --- Fakes -----------------------------------------------------------------
//
// Every collaborator is faked here so the pipeline is exercised in pure Go
// with no sibling-package imports. Each fake records its calls so a test can
// assert the orchestration order and short-circuits.

// userReq is the common single-user-turn request used across the tests.
//
//	req := userReq("gemma", "hi")
func userReq(model, text string) chat.Request {
	return chat.Request{Model: model, Messages: []chat.Message{chat.UserText(text)}}
}

// hasCorrective reports whether the request carries the prepended corrective
// system steer (RFC §6.18) — the marker that distinguishes a regeneration from
// the original attempt.
func hasCorrective(req chat.Request) bool {
	for _, m := range req.Messages {
		if m.Role == chat.System && m.Text() == correctiveInstruction {
			return true
		}
	}
	return false
}

// fakeCache records Get/Set and serves a canned hit when primed.
type fakeCache struct {
	hit      chat.Response // returned when present is true
	present  bool          // true → Get reports a hit
	getCalls int
	setCalls int
	setLast  chat.Response
}

func (c *fakeCache) Get(_ chat.Request) (chat.Response, bool) {
	c.getCalls++
	if c.present {
		return c.hit, true
	}
	return chat.Response{}, false
}

func (c *fakeCache) Set(_ chat.Request, resp chat.Response) {
	c.setCalls++
	c.setLast = resp
}

// fakeRouter returns a fixed endpoint list (or an error).
type fakeRouter struct {
	endpoints   []Endpoint
	err         error
	selectCalls int
}

func (r *fakeRouter) Select(_ chat.Request) ([]Endpoint, error) {
	r.selectCalls++
	return r.endpoints, r.err
}

// fakeGuard returns scripted input/output decisions.
type fakeGuard struct {
	in       Decision
	out      []Decision // consumed per CheckOutput call (last value sticks)
	inCalls  int
	outCalls int
}

func (g *fakeGuard) CheckInput(_ chat.Request) Decision {
	g.inCalls++
	if g.in == "" {
		return DecisionPass
	}
	return g.in
}

func (g *fakeGuard) CheckOutput(_ chat.Request, _ chat.Response) Decision {
	d := DecisionPass
	if len(g.out) > 0 {
		idx := g.outCalls
		if idx >= len(g.out) {
			idx = len(g.out) - 1
		}
		d = g.out[idx]
	}
	g.outCalls++
	return d
}

// fakeSink records usage writes.
type fakeSink struct {
	calls int
	last  chat.Response
}

func (s *fakeSink) Record(_ chat.Request, resp chat.Response) {
	s.calls++
	s.last = resp
}

// fakeBackend serves scripted per-endpoint responses/errors and remembers
// which endpoints and requests it saw (so fallback + regeneration are
// observable).
type fakeBackend struct {
	// byEndpoint maps an Endpoint.ID to its scripted outcome.
	byEndpoint map[string]backendStep
	calls      int
	seenIDs    []string
	seenReqs   []chat.Request
}

type backendStep struct {
	resp chat.Response
	err  error
}

func (b *fakeBackend) Complete(_ context.Context, ep Endpoint, req chat.Request) (chat.Response, error) {
	b.calls++
	b.seenIDs = append(b.seenIDs, ep.ID)
	b.seenReqs = append(b.seenReqs, req)
	step := b.byEndpoint[ep.ID]
	return step.resp, step.err
}

// --- Optional-seam fakes ---------------------------------------------------

// fakeTracer records the run lifecycle: a Start hands out a sequential handle,
// and Finish / Fail record which handle closed how.
type fakeTracer struct {
	starts    int
	finishes  int
	fails     int
	lastErr   error
	lastResp  chat.Response
	startReqs []chat.Request
}

type traceHandle struct{ n int }

func (tr *fakeTracer) Start(_ context.Context, req chat.Request) any {
	tr.starts++
	tr.startReqs = append(tr.startReqs, req)
	return &traceHandle{n: tr.starts}
}

func (tr *fakeTracer) Finish(_ any, resp chat.Response) {
	tr.finishes++
	tr.lastResp = resp
}

func (tr *fakeTracer) Fail(_ any, err error) {
	tr.fails++
	tr.lastErr = err
}

// fakeSessions records Load/Append. loadAppend, when set, is appended to the
// request's messages on Load (so a test can prove the prior transcript was
// recovered). loadErr / appendErr force the error paths.
type fakeSessions struct {
	loadCalls   int
	appendCalls int
	loadAppend  []chat.Message
	loadErr     error
	appendErr   error
	appendReq   chat.Request
	appendResp  chat.Response
}

func (s *fakeSessions) Load(req chat.Request) (chat.Request, error) {
	s.loadCalls++
	if s.loadErr != nil {
		return chat.Request{}, s.loadErr
	}
	if len(s.loadAppend) > 0 {
		req.Messages = append(append([]chat.Message{}, s.loadAppend...), req.Messages...)
	}
	return req, nil
}

func (s *fakeSessions) Append(req chat.Request, resp chat.Response) error {
	s.appendCalls++
	s.appendReq = req
	s.appendResp = resp
	return s.appendErr
}

// fakeFitter records Fit. shrink, when true, replaces the messages with a single
// turn (so a test can prove the transform ran). fitErr forces the error path.
type fakeFitter struct {
	calls  int
	shrink bool
	fitErr error
}

func (f *fakeFitter) Fit(req chat.Request) (chat.Request, error) {
	f.calls++
	if f.fitErr != nil {
		return chat.Request{}, f.fitErr
	}
	if f.shrink {
		req.Messages = []chat.Message{chat.UserText("compressed")}
	}
	return req, nil
}

// fakeFuser records Wants/Run. wants gates the fusion path; resp/err script the
// outcome.
type fakeFuser struct {
	wants     bool
	wantCalls int
	runCalls  int
	resp      chat.Response
	err       error
}

func (f *fakeFuser) Wants(_ chat.Request) bool {
	f.wantCalls++
	return f.wants
}

func (f *fakeFuser) Run(_ context.Context, _ chat.Request) (chat.Response, error) {
	f.runCalls++
	return f.resp, f.err
}

// fakePolicy records Do and simply runs fn once (the no-backoff identity), so a
// test can prove every backend call was routed through the retry wrapper.
type fakePolicy struct {
	calls int
}

func (p *fakePolicy) Do(_ context.Context, fn func() error) error {
	p.calls++
	return fn()
}

// retryingPolicy runs fn up to attempts times, stopping on the first nil — the
// retry-on-transient-error behaviour, without real sleeps.
type retryingPolicy struct {
	attempts int
	calls    int
}

func (p *retryingPolicy) Do(_ context.Context, fn func() error) error {
	var err error
	for i := 0; i < p.attempts; i++ {
		p.calls++
		err = fn()
		if err == nil {
			return nil
		}
	}
	return err
}

// fixture builds a pipeline over fresh fakes wired to sensible defaults (the
// core five seams only; optional seams are left nil and skipped).
func fixture() (*Pipeline, *fakeCache, *fakeRouter, *fakeGuard, *fakeSink, *fakeBackend) {
	cache := &fakeCache{}
	router := &fakeRouter{endpoints: []Endpoint{{ID: "local-metal"}}}
	guard := &fakeGuard{}
	sink := &fakeSink{}
	backend := &fakeBackend{byEndpoint: map[string]backendStep{}}
	p := New(cache, router, guard, sink, backend)
	return p, cache, router, guard, sink, backend
}

// --- Complete: cache → route → backend → usage + cache-set → return --------

func TestPipeline_Complete_Good(t *core.T) {
	p, cache, router, guard, sink, backend := fixture()
	backend.byEndpoint["local-metal"] = backendStep{
		resp: chat.Response{Text: "hello", FinishReason: "stop"},
	}

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "hello", resp.Text)
	core.AssertEqual(t, "stop", resp.FinishReason)

	// Cache miss → it consulted the router, scored the input, ran the backend
	// once, recorded usage, then populated the cache.
	core.AssertEqual(t, 1, cache.getCalls)
	core.AssertEqual(t, 1, router.selectCalls)
	core.AssertEqual(t, 1, guard.inCalls)
	core.AssertEqual(t, 1, backend.calls)
	core.AssertEqual(t, 1, sink.calls)
	core.AssertEqual(t, 1, cache.setCalls)
	core.AssertEqual(t, "hello", cache.setLast.Text)
}

// --- Complete: cache hit short-circuits everything -------------------------

func TestPipeline_Complete_Bad(t *core.T) {
	p, cache, router, guard, sink, backend := fixture()
	cache.present = true
	cache.hit = chat.Response{Text: "cached"}

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "cached", resp.Text)

	// A hit means no inference at all (RFC §6.11): router, guard, backend, sink,
	// and Set are never touched.
	core.AssertEqual(t, 1, cache.getCalls)
	core.AssertEqual(t, 0, router.selectCalls)
	core.AssertEqual(t, 0, guard.inCalls)
	core.AssertEqual(t, 0, backend.calls)
	core.AssertEqual(t, 0, sink.calls)
	core.AssertEqual(t, 0, cache.setCalls)
}

// --- Complete: every endpoint fails → typed error --------------------------

func TestPipeline_Complete_Ugly(t *core.T) {
	p, _, router, _, sink, backend := fixture()
	router.endpoints = []Endpoint{{ID: "a"}, {ID: "b"}}
	backend.byEndpoint["a"] = backendStep{err: core.E("backend", "a down", nil)}
	backend.byEndpoint["b"] = backendStep{err: core.E("backend", "b down", nil)}

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrAllEndpointsFailed)
	core.AssertEqual(t, "", resp.Text)

	// It tried every endpoint in order before giving up, and recorded nothing.
	core.AssertEqual(t, 2, backend.calls)
	core.AssertEqual(t, "a", backend.seenIDs[0])
	core.AssertEqual(t, "b", backend.seenIDs[1])
	core.AssertEqual(t, 0, sink.calls)
}

// --- Routing: empty endpoint set is an error -------------------------------

func TestPipeline_Router_Bad(t *core.T) {
	p, _, router, _, _, backend := fixture()
	router.endpoints = nil // router selected nothing

	_, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrNoEndpoints)
	core.AssertEqual(t, 0, backend.calls)
}

// --- Fallback: first endpoint errors, second succeeds ----------------------

func TestPipeline_Backend_Good(t *core.T) {
	p, cache, router, _, sink, backend := fixture()
	router.endpoints = []Endpoint{{ID: "first"}, {ID: "second"}}
	backend.byEndpoint["first"] = backendStep{err: core.E("backend", "first overloaded", nil)}
	backend.byEndpoint["second"] = backendStep{resp: chat.Response{Text: "served by second"}}

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "served by second", resp.Text)

	// First was tried and failed; the path fell through to the second.
	core.AssertEqual(t, 2, backend.calls)
	core.AssertEqual(t, "first", backend.seenIDs[0])
	core.AssertEqual(t, "second", backend.seenIDs[1])
	core.AssertEqual(t, 1, sink.calls)
	core.AssertEqual(t, 1, cache.setCalls)
}

// --- Guard: clean input + clean output passes through ----------------------

func TestPipeline_Guard_Good(t *core.T) {
	p, _, _, guard, sink, backend := fixture()
	guard.in = DecisionPass
	guard.out = []Decision{DecisionPass}
	backend.byEndpoint["local-metal"] = backendStep{resp: chat.Response{Text: "clean answer"}}

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "clean answer", resp.Text)
	core.AssertEqual(t, 1, guard.inCalls)
	core.AssertEqual(t, 1, guard.outCalls)
	core.AssertEqual(t, 1, backend.calls) // no regeneration
	core.AssertEqual(t, 1, sink.calls)
}

// --- Guard: input guard refuses before any inference -----------------------

func TestPipeline_Guard_Bad(t *core.T) {
	p, _, router, guard, sink, backend := fixture()
	guard.in = DecisionGuard

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrInputGuarded)
	core.AssertEqual(t, "", resp.Text)

	// Input was routed (endpoints chosen) but the guard refused before any
	// backend call, usage record, or cache write.
	core.AssertEqual(t, 1, router.selectCalls)
	core.AssertEqual(t, 1, guard.inCalls)
	core.AssertEqual(t, 0, backend.calls)
	core.AssertEqual(t, 0, sink.calls)
}

// --- Guard: output mediate triggers exactly one regeneration ---------------

func TestPipeline_Guard_Ugly(t *core.T) {
	p, _, _, guard, sink, backend := fixture()
	// First output mediates (steer + regenerate), second output passes.
	guard.out = []Decision{DecisionMediate, DecisionPass}
	backend.byEndpoint["local-metal"] = backendStep{resp: chat.Response{Text: "regenerated answer"}}

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "regenerated answer", resp.Text)

	// Exactly two backend calls: original + one corrective regeneration
	// (RFC §6.18 "regenerate, don't just block" — bounded to one).
	core.AssertEqual(t, 2, backend.calls)
	core.AssertEqual(t, 2, guard.outCalls)
	// The regeneration carried a corrective system steer the first call did not.
	core.AssertFalse(t, hasCorrective(backend.seenReqs[0]), "original carries no corrective steer")
	core.AssertTrue(t, hasCorrective(backend.seenReqs[1]), "regeneration carries the corrective steer")
	core.AssertEqual(t, 1, sink.calls)
}

// --- Guard: output guard refuses (even after a regeneration) ---------------

func TestPipeline_ErrOutputGuarded_Bad(t *core.T) {
	p, _, _, guard, sink, backend := fixture()
	// Output stays over-policy: mediate once, then a hard guard on the redo.
	guard.out = []Decision{DecisionMediate, DecisionGuard}
	backend.byEndpoint["local-metal"] = backendStep{resp: chat.Response{Text: "still bad"}}

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrOutputGuarded)
	core.AssertEqual(t, "", resp.Text)

	// One regeneration was attempted, then the output guard refused: nothing
	// recorded, nothing cached.
	core.AssertEqual(t, 2, backend.calls)
	core.AssertEqual(t, 0, sink.calls)
}

// --- Guard: output guard refuses on the FIRST check (no mediation) ---------

func TestPipeline_OutputGuard_Immediate(t *core.T) {
	// The very first output check returns guard — no mediation, no
	// regeneration. The pipeline refuses straight away (generate's DecisionGuard
	// arm) and records / caches nothing.
	p, _, _, guard, sink, backend := fixture()
	guard.out = []Decision{DecisionGuard}
	backend.byEndpoint["local-metal"] = backendStep{resp: chat.Response{Text: "disallowed"}}

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrOutputGuarded)
	core.AssertEqual(t, "", resp.Text)
	// Exactly one backend call (the original) and exactly one output check — no
	// regeneration was attempted.
	core.AssertEqual(t, 1, backend.calls, "an immediate output guard does not regenerate")
	core.AssertEqual(t, 1, guard.outCalls)
	core.AssertEqual(t, 0, sink.calls)
}

// --- Context cancellation surfaces -----------------------------------------

func TestPipeline_Context_Ugly(t *core.T) {
	p, _, _, _, _, backend := fixture()
	backend.byEndpoint["local-metal"] = backendStep{resp: chat.Response{Text: "won't get here"}}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled before the call

	_, err := p.Complete(ctx, userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertErrorIs(t, err, context.Canceled)
	core.AssertEqual(t, 0, backend.calls)
}

// --- Routing: the router itself errors (not merely empty) ------------------

func TestPipeline_Route_Ugly(t *core.T) {
	// A router that returns a non-nil error is distinct from one that returns an
	// empty set: the failure is wrapped with the router's cause, not the
	// ErrNoEndpoints sentinel, and nothing downstream runs.
	p, _, router, _, sink, backend := fixture()
	routeErr := core.E("router", "policy denied all providers", nil)
	router.err = routeErr
	router.endpoints = nil

	_, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "route request")
	core.AssertContains(t, err.Error(), "policy denied all providers")
	// It must NOT be reported as the empty-endpoints sentinel — the router failed.
	core.AssertFalse(t, core.Is(err, ErrNoEndpoints), "a router error is not ErrNoEndpoints")
	core.AssertEqual(t, 0, backend.calls)
	core.AssertEqual(t, 0, sink.calls)
}

// --- Regeneration backend failure surfaces ---------------------------------

func TestPipeline_Regenerate_Bad(t *core.T) {
	// Output mediates, so the pipeline regenerates once — but the regeneration's
	// backend call fails on every endpoint. That error surfaces from generate's
	// redo path (ErrAllEndpointsFailed), and nothing is recorded or cached.
	cache := &fakeCache{}
	router := &fakeRouter{endpoints: []Endpoint{{ID: "only"}}}
	guard := &fakeGuard{out: []Decision{DecisionMediate}} // first (and only) output mediates
	sink := &fakeSink{}

	// A backend that succeeds on the first attempt (no steer) but fails the redo
	// (corrective steer present) — so the regeneration's chain errors.
	backend := &flakyRedoBackend{}
	p := New(cache, router, guard, sink, backend)

	_, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrAllEndpointsFailed)
	core.AssertEqual(t, 2, backend.calls, "original succeeded, regeneration was attempted")
	core.AssertEqual(t, 0, sink.calls, "a failed regeneration records nothing")
	core.AssertEqual(t, 0, cache.setCalls, "a failed regeneration caches nothing")
}

// flakyRedoBackend succeeds on the first attempt (no corrective steer) and fails
// on the regeneration (corrective steer present), so the redo path's error
// branch is exercised.
type flakyRedoBackend struct {
	calls int
}

func (b *flakyRedoBackend) Complete(_ context.Context, _ Endpoint, req chat.Request) (chat.Response, error) {
	b.calls++
	if hasCorrective(req) {
		return chat.Response{}, core.E("backend", "regeneration failed", nil)
	}
	return chat.Response{Text: "first answer, will be mediated"}, nil
}

// --- Context cancelled mid-fallback-chain ----------------------------------

func TestPipeline_Context_Midchain(t *core.T) {
	// The context is cancelled after the first endpoint is tried but before the
	// second: backendChain's in-loop ctx.Err() guard fires, returning the
	// context error rather than falling through to the next endpoint.
	cache := &fakeCache{}
	router := &fakeRouter{endpoints: []Endpoint{{ID: "first"}, {ID: "second"}}}
	guard := &fakeGuard{}
	sink := &fakeSink{}

	ctx, cancel := context.WithCancel(context.Background())
	backend := &cancelMidchainBackend{cancel: cancel}
	p := New(cache, router, guard, sink, backend)

	_, err := p.Complete(ctx, userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertErrorIs(t, err, context.Canceled)
	// Only the first endpoint was attempted; the in-loop guard stopped the rest.
	core.AssertEqual(t, 1, backend.calls, "cancellation stops the fallback chain")
	core.AssertEqual(t, "first", backend.seen[0])
	core.AssertEqual(t, 0, sink.calls)
}

// cancelMidchainBackend fails the first endpoint and cancels the context as it
// does so, so the next loop iteration's ctx.Err() guard trips before the second
// endpoint is tried.
type cancelMidchainBackend struct {
	cancel context.CancelFunc
	calls  int
	seen   []string
}

func (b *cancelMidchainBackend) Complete(_ context.Context, ep Endpoint, _ chat.Request) (chat.Response, error) {
	b.calls++
	b.seen = append(b.seen, ep.ID)
	b.cancel() // cancel during the first attempt
	return chat.Response{}, core.E("backend", "first endpoint down", nil)
}

// --- Optional seams: full happy path with EVERY stage set ------------------

func TestPipeline_Tracer_Good(t *core.T) {
	// All optional seams set; the request flows through every one and the
	// invocation counts prove the order and that each ran exactly once.
	p, _, _, _, sink, backend := fixture()
	backend.byEndpoint["local-metal"] = backendStep{resp: chat.Response{Text: "final"}}

	tracer := &fakeTracer{}
	sessions := &fakeSessions{loadAppend: []chat.Message{{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text("prior")}}}}
	fitter := &fakeFitter{}
	fuser := &fakeFuser{wants: false} // present but this request does not want fusion
	policy := &fakePolicy{}
	p.Tracer = tracer
	p.Sessions = sessions
	p.Fitter = fitter
	p.Fuser = fuser
	p.Policy = policy

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "final", resp.Text)

	// The run was opened and finished (never failed).
	core.AssertEqual(t, 1, tracer.starts)
	core.AssertEqual(t, 1, tracer.finishes)
	core.AssertEqual(t, 0, tracer.fails)
	core.AssertEqual(t, "final", tracer.lastResp.Text)

	// Session loaded then appended; the prior transcript reached the backend.
	core.AssertEqual(t, 1, sessions.loadCalls)
	core.AssertEqual(t, 1, sessions.appendCalls)
	core.AssertEqual(t, "final", sessions.appendResp.Text)
	core.AssertEqual(t, 2, len(backend.seenReqs[0].Messages), "prior turn was prepended before placement")

	// Fitter ran; fusion was consulted (Wants) but not run; retry wrapped the call.
	core.AssertEqual(t, 1, fitter.calls)
	core.AssertEqual(t, 1, fuser.wantCalls)
	core.AssertEqual(t, 0, fuser.runCalls)
	core.AssertEqual(t, 1, policy.calls)
	core.AssertEqual(t, 1, backend.calls)
	core.AssertEqual(t, 1, sink.calls)
}

// --- Optional seams: all nil → original behaviour preserved ----------------

func TestPipeline_AllStages_Nil(t *core.T) {
	// The fixture leaves every optional seam nil; the path still completes,
	// proving each stage is genuinely skipped (not required).
	p, _, _, _, sink, backend := fixture()
	backend.byEndpoint["local-metal"] = backendStep{resp: chat.Response{Text: "plain"}}

	core.AssertTrue(t, p.Tracer == nil, "tracer unset")
	core.AssertTrue(t, p.Sessions == nil, "sessions unset")
	core.AssertTrue(t, p.Fitter == nil, "fitter unset")
	core.AssertTrue(t, p.Fuser == nil, "fuser unset")
	core.AssertTrue(t, p.Policy == nil, "policy unset")

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "plain", resp.Text)
	core.AssertEqual(t, 1, backend.calls)
	core.AssertEqual(t, 1, sink.calls)
}

// --- Tracer: a failure path closes the run as failed -----------------------

func TestPipeline_Tracer_Fail(t *core.T) {
	// The input guard refuses, so the request errors after the run opened. The
	// tracer must record a Fail (not a Finish) carrying that error.
	p, _, _, guard, _, _ := fixture()
	guard.in = DecisionGuard
	tracer := &fakeTracer{}
	p.Tracer = tracer

	_, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrInputGuarded)
	core.AssertEqual(t, 1, tracer.starts)
	core.AssertEqual(t, 0, tracer.finishes)
	core.AssertEqual(t, 1, tracer.fails)
	core.AssertErrorIs(t, tracer.lastErr, ErrInputGuarded)
}

// --- Tracer: a context cancel before any stage still opens then fails -------

func TestPipeline_Tracer_CancelledNoStart(t *core.T) {
	// The context is already cancelled, so Complete returns before opening the
	// run (the up-front ctx guard). No Start, no Fail — the tracer is untouched.
	p, _, _, _, _, _ := fixture()
	tracer := &fakeTracer{}
	p.Tracer = tracer
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := p.Complete(ctx, userReq("gemma", "hi"))

	core.AssertErrorIs(t, err, context.Canceled)
	core.AssertEqual(t, 0, tracer.starts)
	core.AssertEqual(t, 0, tracer.fails)
}

// --- Sessions: a cache hit is still appended to the session ----------------

func TestPipeline_Sessions_CacheHitAppends(t *core.T) {
	// A response-cache hit short-circuits inference but must still advance the
	// conversation — the cached turn is appended to the session.
	p, cache, _, _, _, _ := fixture()
	cache.present = true
	cache.hit = chat.Response{Text: "cached reply"}
	sessions := &fakeSessions{}
	p.Sessions = sessions

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "cached reply", resp.Text)
	core.AssertEqual(t, 1, sessions.loadCalls)
	core.AssertEqual(t, 1, sessions.appendCalls, "a cached turn is still appended")
	core.AssertEqual(t, "cached reply", sessions.appendResp.Text)
}

// --- Sessions: a load failure surfaces and fails the run -------------------

func TestPipeline_Sessions_LoadBad(t *core.T) {
	p, _, _, _, _, backend := fixture()
	tracer := &fakeTracer{}
	sessions := &fakeSessions{loadErr: core.E("session", "unknown session", nil)}
	p.Tracer = tracer
	p.Sessions = sessions

	_, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "session load")
	core.AssertEqual(t, 0, backend.calls, "a failed session load never reaches inference")
	core.AssertEqual(t, 1, tracer.fails)
}

// --- Sessions: an append failure surfaces ----------------------------------

func TestPipeline_Sessions_AppendBad(t *core.T) {
	// The completion succeeds but persisting the turn fails — the error
	// surfaces (the conversation would otherwise silently drift).
	p, _, _, _, sink, backend := fixture()
	backend.byEndpoint["local-metal"] = backendStep{resp: chat.Response{Text: "answer"}}
	sessions := &fakeSessions{appendErr: core.E("session", "store down", nil)}
	p.Sessions = sessions

	_, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "session append")
	core.AssertEqual(t, 1, backend.calls)
	core.AssertEqual(t, 1, sink.calls, "usage was already recorded before the append")
}

// --- Fitter: a transform compresses the request before placement -----------

func TestPipeline_Fitter_Compresses(t *core.T) {
	// The fitter shrinks the conversation; the backend must see the compressed
	// request, proving the transform ran before routing/placement.
	p, _, _, _, _, backend := fixture()
	backend.byEndpoint["local-metal"] = backendStep{resp: chat.Response{Text: "ok"}}
	fitter := &fakeFitter{shrink: true}
	p.Fitter = fitter

	resp, err := p.Complete(context.Background(), userReq("gemma", "a very long conversation"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "ok", resp.Text)
	core.AssertEqual(t, 1, fitter.calls)
	core.AssertEqual(t, "compressed", backend.seenReqs[0].Messages[0].Text(), "the compressed request was placed")
}

// --- Fitter: a fit failure surfaces ----------------------------------------

func TestPipeline_Fitter_Bad(t *core.T) {
	p, _, router, _, _, backend := fixture()
	fitter := &fakeFitter{fitErr: core.E("transform", "cannot fit window", nil)}
	p.Fitter = fitter

	_, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "context fit")
	core.AssertEqual(t, 0, router.selectCalls, "a fit failure never reaches routing")
	core.AssertEqual(t, 0, backend.calls)
}

// --- Fuser: a request that wants fusion takes the panel path ---------------

func TestPipeline_Fuser_Good(t *core.T) {
	// The request wants fusion, so the panel runs instead of the backend.
	p, _, _, guard, sink, backend := fixture()
	guard.out = []Decision{DecisionPass}
	fuser := &fakeFuser{wants: true, resp: chat.Response{Text: "fused answer"}}
	p.Fuser = fuser

	resp, err := p.Complete(context.Background(), userReq("gemma", "deliberate this"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "fused answer", resp.Text)
	core.AssertEqual(t, 1, fuser.wantCalls)
	core.AssertEqual(t, 1, fuser.runCalls)
	core.AssertEqual(t, 0, backend.calls, "fusion replaces the single-backend call")
	core.AssertEqual(t, 1, guard.outCalls, "the fused answer still passes output safety")
	core.AssertEqual(t, 1, sink.calls)
}

// --- Fuser: a fusion failure surfaces (no single-backend fallback) ---------

func TestPipeline_Fuser_Bad(t *core.T) {
	p, _, _, _, sink, backend := fixture()
	fuser := &fakeFuser{wants: true, err: core.E("fusion", "every analysis model failed", nil)}
	p.Fuser = fuser

	_, err := p.Complete(context.Background(), userReq("gemma", "deliberate this"))

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "fusion")
	core.AssertEqual(t, 1, fuser.runCalls)
	core.AssertEqual(t, 0, backend.calls, "a failed fusion does not fall back to the backend")
	core.AssertEqual(t, 0, sink.calls)
}

// --- Fuser: a mediated fusion answer regenerates through the panel ----------

func TestPipeline_Fuser_Mediate(t *core.T) {
	// The fused answer mediates, so the pipeline regenerates — and the
	// regeneration runs through the fusion path again (two Run calls).
	p, _, _, guard, sink, _ := fixture()
	guard.out = []Decision{DecisionMediate, DecisionPass}
	fuser := &fakeFuser{wants: true, resp: chat.Response{Text: "fused, then refined"}}
	p.Fuser = fuser

	resp, err := p.Complete(context.Background(), userReq("gemma", "deliberate this"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "fused, then refined", resp.Text)
	core.AssertEqual(t, 2, fuser.runCalls, "the regeneration re-ran the panel")
	core.AssertEqual(t, 2, guard.outCalls)
	core.AssertEqual(t, 1, sink.calls)
}

// --- Policy: a transient backend error is retried before falling through ---

func TestPipeline_Policy_RetriesTransient(t *core.T) {
	// The first endpoint fails once then succeeds; the retry policy retries it
	// in place, so the chain never advances to a second endpoint.
	cache := &fakeCache{}
	router := &fakeRouter{endpoints: []Endpoint{{ID: "flaky"}, {ID: "spare"}}}
	guard := &fakeGuard{}
	sink := &fakeSink{}
	backend := &transientBackend{failFirst: 1, ok: chat.Response{Text: "recovered"}}
	p := New(cache, router, guard, sink, backend)
	policy := &retryingPolicy{attempts: 3}
	p.Policy = policy

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "recovered", resp.Text)
	// Two backend calls total — both on the FIRST endpoint (retry, not fallback).
	core.AssertEqual(t, 2, backend.calls)
	core.AssertEqual(t, "flaky", backend.seen[0])
	core.AssertEqual(t, "flaky", backend.seen[1])
	core.AssertEqual(t, 2, policy.calls, "the policy ran the call twice")
	core.AssertEqual(t, 1, sink.calls)
}

// --- Policy: an exhausted retry advances the fallback chain -----------------

func TestPipeline_Policy_ExhaustedFallsThrough(t *core.T) {
	// The first endpoint fails on every retry; the policy gives up and the chain
	// advances to the second endpoint, which succeeds.
	cache := &fakeCache{}
	router := &fakeRouter{endpoints: []Endpoint{{ID: "dead"}, {ID: "live"}}}
	guard := &fakeGuard{}
	sink := &fakeSink{}
	backend := &fakeBackend{byEndpoint: map[string]backendStep{
		"dead": {err: core.E("backend", "always down", nil)},
		"live": {resp: chat.Response{Text: "from live"}},
	}}
	p := New(cache, router, guard, sink, backend)
	policy := &retryingPolicy{attempts: 2}
	p.Policy = policy

	resp, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertNoError(t, err)
	core.AssertEqual(t, "from live", resp.Text)
	// dead retried twice, then live once = three backend calls.
	core.AssertEqual(t, 3, backend.calls)
	core.AssertEqual(t, "dead", backend.seenIDs[0])
	core.AssertEqual(t, "dead", backend.seenIDs[1])
	core.AssertEqual(t, "live", backend.seenIDs[2])
	core.AssertEqual(t, 1, sink.calls)
}

// transientBackend fails its first failFirst calls, then serves ok. It records
// the endpoint of every call so a test can tell retry (same id) from fallback
// (different id).
type transientBackend struct {
	failFirst int
	ok        chat.Response
	calls     int
	seen      []string
}

func (b *transientBackend) Complete(_ context.Context, ep Endpoint, _ chat.Request) (chat.Response, error) {
	b.calls++
	b.seen = append(b.seen, ep.ID)
	if b.calls <= b.failFirst {
		return chat.Response{}, core.E("backend", "transient overload", nil)
	}
	return b.ok, nil
}
