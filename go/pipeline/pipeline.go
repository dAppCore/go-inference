// SPDX-Licence-Identifier: EUPL-1.2

package pipeline

import (
	"context"

	core "dappco.re/go"
	chat "dappco.re/go/inference/chat"
)

// correctiveInstruction is the steer prepended (as a system turn) to a
// regenerated request when the output guard mediates (RFC §6.18 "regenerate,
// don't just block"). The redo runs the same endpoint chain with this leading
// the conversation.
const correctiveInstruction = "Revise the previous answer to stay within policy: " +
	"remove hostile, sycophantic, or unsafe content while keeping it helpful."

// Pipeline composes the serving-path collaborators. Construct the core five
// seams with New (the optional stage seams are nil and skipped), or build a
// fully-wired pipeline from the real packages with NewWired. It holds no
// per-request state, so a single Pipeline is safe to share across goroutines
// provided its collaborators are.
//
// The first block is the required core seams; the second is the optional stage
// seams (nil ⇒ skipped). See types.go for each interface.
type Pipeline struct {
	// Core seams — always consulted.
	Cache     Cache
	Router    Router
	Guard     Guard
	UsageSink UsageSink
	Backend   Backend

	// Optional stage seams — nil ⇒ that stage is skipped.
	Tracer   Tracer   // observability run-tree (RFC.inference-stack §3.7)
	Sessions Sessions // stateful sessions (§6.10)
	Fitter   Fitter   // context fit / middle-out transform (§6.13, §6.11)
	Fuser    Fuser    // multi-model deliberation (§6.9)
	Policy   Policy   // per-call retry backoff (§6.7)
}

// New wires the core serving path over its five required collaborators
// (RFC §6). The optional stage seams (Tracer, Sessions, Fitter, Fuser, Policy)
// are left nil and therefore skipped — set them on the returned Pipeline to
// compose more of the path, or use NewWired to build them all from the real
// packages.
//
//	p := pipeline.New(cache, router, guard, sink, backend)
//	p.Tracer = myTracer // opt into the observability run
func New(cache Cache, router Router, guard Guard, usage UsageSink, backend Backend) *Pipeline {
	return &Pipeline{Cache: cache, Router: router, Guard: guard, UsageSink: usage, Backend: backend}
}

// Complete serves one chat request along the full path (RFC §6). Stages with a
// nil seam are skipped, so the order below degrades cleanly to the core five:
//
//  1. Tracer.Start (§3.7) — open the run; every error path below calls Fail,
//     the success path calls Finish.
//
//  2. Sessions.Load (§6.10) — prepend the prior transcript for SessionID.
//
//  3. Cache.Get (§6.11) — an exact hit returns with no inference and no further
//     steps (it is still appended to the session and traced as finished).
//
//  4. Fitter.Fit (§6.13) — count tokens; middle-out compress if over window.
//
//  5. Router.Select (§6.2) — an ordered endpoint list; empty is ErrNoEndpoints.
//
//  6. Guard.CheckInput (§6.18) — a guarded input is refused with ErrInputGuarded.
//
//  7. Inference — Fuser.Run when the request wants fusion (§6.9), else the
//     backend across the endpoint fallback chain (§6.7), each call wrapped in
//     Policy retry backoff (§6.7).
//
//  8. Guard.CheckOutput (§6.18) — a mediated output is regenerated once under a
//     corrective instruction; a guarded output is refused with ErrOutputGuarded.
//
//  9. UsageSink.Record (§6.6), then Sessions.Append (§6.10), then Cache.Set
//     (§6.11), then Tracer.Finish (§3.7).
//
//     resp, err := p.Complete(ctx, req)
func (p *Pipeline) Complete(ctx context.Context, req chat.Request) (chat.Response, error) {
	if err := ctx.Err(); err != nil {
		return chat.Response{}, err
	}

	// 1. Observability run — bracket the whole request (§3.7). A nil Tracer
	// yields a nil handle; trace() / fail() / finish() are all nil-safe.
	handle := p.trace(ctx, req)

	resp, err := p.complete(ctx, req, handle)
	if err != nil {
		p.fail(handle, err)
		return chat.Response{}, err
	}
	p.finish(handle, resp)
	return resp, nil
}

// complete runs the path between the tracer brackets, returning the response or
// a typed error. Kept separate so Complete owns only the run lifecycle.
func (p *Pipeline) complete(ctx context.Context, req chat.Request, handle any) (chat.Response, error) {
	// 2. Stateful session — recover the prior transcript for this SessionID
	// (§6.10), so the request runs with full context the caller didn't resend.
	req, err := p.loadSession(req)
	if err != nil {
		return chat.Response{}, err
	}

	// 3. Response cache — exact short-circuit, zero inference (§6.11). A hit is
	// still appended to the session so a cached turn advances the conversation.
	if p.Cache != nil {
		if hit, ok := p.Cache.Get(req); ok {
			if err := p.appendSession(req, hit); err != nil {
				return chat.Response{}, err
			}
			return hit, nil
		}
	}

	// 4. Context fit — count tokens and middle-out compress if over window
	// (§6.13, §6.11) before the request is placed.
	req, err = p.fit(req)
	if err != nil {
		return chat.Response{}, err
	}

	// 5. Routing — ordered endpoints, first preferred, rest are fallbacks (§6.2).
	endpoints, err := p.route(req)
	if err != nil {
		return chat.Response{}, err
	}

	// 6. Input safety — refuse a guarded turn before any inference (§6.18).
	if p.Guard != nil && p.Guard.CheckInput(req) == DecisionGuard {
		return chat.Response{}, core.E("pipeline", "input safety", ErrInputGuarded)
	}

	// 7 + 8. Inference (fusion or single-backend with retry + fallback), then
	// output safety with one bounded regeneration.
	resp, err := p.generate(ctx, endpoints, req)
	if err != nil {
		return chat.Response{}, err
	}

	// 9. Account, append the turn, cache, return (§6.6, §6.10, §6.11).
	if p.UsageSink != nil {
		p.UsageSink.Record(req, resp)
	}
	if err := p.appendSession(req, resp); err != nil {
		return chat.Response{}, err
	}
	if p.Cache != nil {
		p.Cache.Set(req, resp)
	}
	return resp, nil
}

// route selects the endpoint chain, mapping the router's failures onto the
// pipeline's typed errors (§6.2).
func (p *Pipeline) route(req chat.Request) ([]Endpoint, error) {
	endpoints, err := p.Router.Select(req)
	if err != nil {
		return nil, core.E("pipeline", "route request", err)
	}
	if len(endpoints) == 0 {
		return nil, core.E("pipeline", "route request", ErrNoEndpoints)
	}
	return endpoints, nil
}

// generate produces the response for an admitted request: the fusion panel when
// the request wants it (§6.9), otherwise the backend across the endpoint
// fallback chain (§6.7). It then applies output safety with one bounded
// regeneration (§6.18), returning the first acceptable response or a typed error.
func (p *Pipeline) generate(ctx context.Context, endpoints []Endpoint, req chat.Request) (chat.Response, error) {
	resp, err := p.infer(ctx, endpoints, req)
	if err != nil {
		return chat.Response{}, err
	}

	if p.Guard == nil {
		return resp, nil
	}

	switch p.Guard.CheckOutput(req, resp) {
	case DecisionGuard:
		return chat.Response{}, core.E("pipeline", "output safety", ErrOutputGuarded)
	case DecisionMediate:
		// Regenerate once under a corrective instruction (§6.18). The redo runs
		// the same inference path; its output is re-checked but not re-mediated.
		redo := withCorrective(req)
		resp, err = p.infer(ctx, endpoints, redo)
		if err != nil {
			return chat.Response{}, err
		}
		if p.Guard.CheckOutput(redo, resp) == DecisionGuard {
			return chat.Response{}, core.E("pipeline", "output safety", ErrOutputGuarded)
		}
	}
	return resp, nil
}

// infer runs the chosen inference strategy: fusion when requested (§6.9), else
// the single-backend fallback chain (§6.7).
func (p *Pipeline) infer(ctx context.Context, endpoints []Endpoint, req chat.Request) (chat.Response, error) {
	if p.Fuser != nil && p.Fuser.Wants(req) {
		resp, err := p.Fuser.Run(ctx, req)
		if err != nil {
			return chat.Response{}, core.E("pipeline", "fusion", err)
		}
		return resp, nil
	}
	return p.backendChain(ctx, endpoints, req)
}

// backendChain tries each endpoint in order, returning the first success. Each
// attempt is wrapped in the retry Policy (§6.7) when one is set; a backend error
// (after its retries are exhausted) advances to the next endpoint. Exhausting
// the chain is ErrAllEndpointsFailed with the last cause attached.
func (p *Pipeline) backendChain(ctx context.Context, endpoints []Endpoint, req chat.Request) (chat.Response, error) {
	var last error
	for _, ep := range endpoints {
		if err := ctx.Err(); err != nil {
			return chat.Response{}, err
		}
		resp, err := p.callBackend(ctx, ep, req)
		if err == nil {
			return resp, nil
		}
		last = err
	}
	// Sentinel as the immediate cause so callers branch with core.Is; the last
	// backend cause is folded into the message for diagnostics.
	msg := "all endpoints failed"
	if last != nil {
		msg = "all endpoints failed: " + last.Error()
	}
	return chat.Response{}, core.E("pipeline", msg, ErrAllEndpointsFailed)
}

// callBackend runs one endpoint, wrapping the call in the retry Policy when set
// (§6.7) so a transient failure is retried before the chain advances. With no
// Policy the backend is called exactly once.
func (p *Pipeline) callBackend(ctx context.Context, ep Endpoint, req chat.Request) (chat.Response, error) {
	if p.Policy == nil {
		return p.Backend.Complete(ctx, ep, req)
	}
	var resp chat.Response
	err := p.Policy.Do(ctx, func() error {
		var callErr error
		resp, callErr = p.Backend.Complete(ctx, ep, req)
		return callErr
	})
	if err != nil {
		return chat.Response{}, err
	}
	return resp, nil
}

// --- nil-safe optional stage helpers ---------------------------------------

// trace opens the observability run when a Tracer is set, returning the opaque
// handle (nil when no Tracer).
func (p *Pipeline) trace(ctx context.Context, req chat.Request) any {
	if p.Tracer == nil {
		return nil
	}
	return p.Tracer.Start(ctx, req)
}

// finish closes the run successfully (no-op without a Tracer).
func (p *Pipeline) finish(handle any, resp chat.Response) {
	if p.Tracer != nil {
		p.Tracer.Finish(handle, resp)
	}
}

// fail closes the run as failed (no-op without a Tracer).
func (p *Pipeline) fail(handle any, err error) {
	if p.Tracer != nil {
		p.Tracer.Fail(handle, err)
	}
}

// loadSession recovers the prior transcript for the request (no-op without a
// Sessions seam), returning the request to run.
func (p *Pipeline) loadSession(req chat.Request) (chat.Request, error) {
	if p.Sessions == nil {
		return req, nil
	}
	loaded, err := p.Sessions.Load(req)
	if err != nil {
		return chat.Request{}, core.E("pipeline", "session load", err)
	}
	return loaded, nil
}

// appendSession records the completed turn onto the session (no-op without a
// Sessions seam).
func (p *Pipeline) appendSession(req chat.Request, resp chat.Response) error {
	if p.Sessions == nil {
		return nil
	}
	if err := p.Sessions.Append(req, resp); err != nil {
		return core.E("pipeline", "session append", err)
	}
	return nil
}

// fit applies the context-fit transform (no-op without a Fitter), returning the
// request to place.
func (p *Pipeline) fit(req chat.Request) (chat.Request, error) {
	if p.Fitter == nil {
		return req, nil
	}
	fitted, err := p.Fitter.Fit(req)
	if err != nil {
		return chat.Request{}, core.E("pipeline", "context fit", err)
	}
	return fitted, nil
}

// withCorrective returns a copy of req with the corrective system instruction
// prepended (RFC §6.18) — the steer for a mediated regeneration. The caller's
// message slice is never mutated.
func withCorrective(req chat.Request) chat.Request {
	steer := chat.Message{Role: chat.System, Content: []chat.ContentBlock{chat.Text(correctiveInstruction)}}
	msgs := make([]chat.Message, 0, len(req.Messages)+1)
	msgs = append(msgs, steer)
	msgs = append(msgs, req.Messages...)
	req.Messages = msgs
	return req
}
