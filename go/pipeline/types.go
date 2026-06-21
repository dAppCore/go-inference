// SPDX-Licence-Identifier: EUPL-1.2

// Package pipeline composes the serving path for one chat request, in the
// order RFC §6 lays it out: observability run (RFC.inference-stack §3.7) →
// stateful session load (§6.10) → response cache (§6.11) → context fit /
// middle-out transform (§6.13) → provider routing + fallback (§6.2, §6.7) →
// request-path safety (§6.18) → inference, single-backend or fusion panel
// (§6.1, §6.9), each backend call wrapped in retry backoff (§6.7) → output
// safety (§6.18) → usage accounting (§6.6) → session append (§6.10) → cache set
// (§6.11). It owns the *order* of those concerns, not their implementations:
// every collaborator is an interface, so the real cache / router / guard / sink
// / backend / tracer / session / fitter / fuser adapters wire in at NewWired,
// and the path is exercised in pure Go with fakes.
//
// Every optional seam is nil-safe: a Pipeline with only the core seams set
// (Cache, Router, Guard, UsageSink, Backend) behaves exactly as it did before
// the optional stages existed — a nil Tracer / Sessions / Fitter / Fuser /
// Policy means that stage is simply skipped.
//
//	p := pipeline.New(cache, router, guard, sink, backend)
//	resp, err := p.Complete(ctx, chat.Request{
//		Model:    "gemma-4-31b",
//		Messages: []chat.Message{chat.UserText("hello")},
//	})
package pipeline

import (
	"context"

	core "dappco.re/go"
	chat "dappco.re/go/inference/chat"
)

// Endpoint identifies one place a request can run — a local device runtime
// (go-mlx Metal, a CUDA/ROCm GPU) or an external provider (RFC §6.2). The
// pipeline only needs to tell them apart and try them in order; the router owns
// budget / quant / SLO selection.
type Endpoint struct {
	ID string
}

// Decision is a guard verdict for one input or output turn (RFC §6.18):
//   - DecisionPass    — within policy, proceed.
//   - DecisionMediate — over policy but recoverable; steer and regenerate once.
//   - DecisionGuard   — over policy; refuse.
type Decision string

const (
	DecisionPass    Decision = "pass"
	DecisionMediate Decision = "mediate"
	DecisionGuard   Decision = "guard"
)

// --- Core collaborators ----------------------------------------------------
//
// Each is the minimal interface the pipeline depends on. Real adapters (the
// respcache, provider router, welfare safety gate, usage sink, and inference
// backend siblings) satisfy these without the pipeline importing them — the
// adapters live in wired.go.

// Cache is the response cache (RFC §6.11): an exact-match (or semantic) lookup
// that returns a stored completion with NO inference. A hit short-circuits the
// whole path; Set populates it after a fresh completion.
type Cache interface {
	Get(req chat.Request) (chat.Response, bool)
	Set(req chat.Request, resp chat.Response)
}

// Router selects the ordered endpoints to try for a request (RFC §6.2). The
// first is preferred; the rest are the fallback chain (§6.7). An empty result
// is a routing failure.
type Router interface {
	Select(req chat.Request) ([]Endpoint, error)
}

// Guard is the request-path safety gate (RFC §6.18). CheckInput scores the
// incoming request; CheckOutput scores a generated response. Either may pass,
// mediate (regenerate once), or guard (refuse).
type Guard interface {
	CheckInput(req chat.Request) Decision
	CheckOutput(req chat.Request, resp chat.Response) Decision
}

// UsageSink records a completed response's usage for accounting (RFC §6.6) —
// the metrics-log write. Best-effort: it returns nothing and never blocks the
// response.
type UsageSink interface {
	Record(req chat.Request, resp chat.Response)
}

// Backend runs one inference against a chosen endpoint (RFC §6.1). A non-nil
// error makes the pipeline fall through to the next endpoint (§6.7).
type Backend interface {
	Complete(ctx context.Context, endpoint Endpoint, req chat.Request) (chat.Response, error)
}

// --- Optional stage seams --------------------------------------------------
//
// Each is an interface FIELD on Pipeline; a nil field means the stage is
// skipped, so the original five-seam behaviour is preserved exactly when none
// of these are set. The wired.go adapters map the concrete obs / session /
// budget+transform / fusion / retry packages onto them.

// Tracer brackets a run around one request (RFC.inference-stack §3.7 — the
// observability run-tree). Start opens the run and returns an opaque handle the
// pipeline threads back into Finish (on success) or Fail (on any error path),
// so the durable sink lands inputs, model, decisions, and timing — the EU AI
// Act audit trail (§3.8). A nil Tracer means no run is opened.
//
// The handle is opaque (any) so this package never imports pkg/obs; the adapter
// in wired.go carries the obs run pointer through it.
type Tracer interface {
	Start(ctx context.Context, req chat.Request) any
	Finish(handle any, resp chat.Response)
	Fail(handle any, err error)
}

// Sessions is the stateful-conversation seam (RFC §6.10). Load resolves the
// prior turns for a request's SessionID and returns the request to actually run
// — the same request with the recovered transcript prepended — so a multi-turn
// chat continues without the caller resending it. Append records the completed
// turn (the user input + the assistant reply) back onto the session after a
// successful completion. A nil Sessions means the request runs stateless and no
// turn is appended.
type Sessions interface {
	Load(req chat.Request) (chat.Request, error)
	Append(req chat.Request, resp chat.Response) error
}

// Fitter is the context-fit seam (RFC §6.13 + §6.11 "Message transforms"). Fit
// counts a request's prompt against the target window and, when it overflows,
// middle-out compresses the conversation so it still fits; it returns the
// request to place (compressed when it had to be, untouched otherwise). A nil
// Fitter means the request is placed as-is — no counting, no transform.
type Fitter interface {
	Fit(req chat.Request) (chat.Request, error)
}

// Fuser is the multi-model deliberation seam (RFC §6.9). When a request asks
// for fusion (Wants reports true for it), Run executes the panel + judge in
// place of a single backend call and returns the judge's final answer as a
// chat.Response. A nil Fuser (or a request that does not want fusion) takes the
// ordinary single-backend path instead.
type Fuser interface {
	// Wants reports whether this request should be served by the fusion panel
	// rather than a single backend call (RFC §6.9 — the `fusion` alias / plugin).
	Wants(req chat.Request) bool
	// Run executes the panel + judge for the request and returns the final
	// answer. A non-nil error fails the request (no single-backend fallback —
	// fusion was explicitly requested).
	Run(ctx context.Context, req chat.Request) (chat.Response, error)
}

// Policy wraps one backend call in retry backoff (RFC §6.7). Do calls fn and,
// on a retryable failure, backs off and retries within its envelope; a
// permanent failure surfaces immediately. The pipeline wraps every endpoint
// attempt through Do, so a transient 429 / 503 / timeout is retried before the
// fallback chain advances to the next endpoint. A nil Policy means each backend
// call is made exactly once (the original behaviour).
//
// Do is injectable so tests drive the retry loop without real sleeps.
type Policy interface {
	Do(ctx context.Context, fn func() error) error
}

// --- Typed errors ----------------------------------------------------------
//
// Sentinels so callers branch on the failure class with core.Is / errors.Is.
// They are wrapped with core.E("pipeline", …) at the point of failure.

var (
	// ErrNoEndpoints — the router returned an empty endpoint set (RFC §6.2).
	ErrNoEndpoints = core.NewError("pipeline: router selected no endpoints")
	// ErrAllEndpointsFailed — every routed endpoint errored (RFC §6.7).
	ErrAllEndpointsFailed = core.NewError("pipeline: all endpoints failed")
	// ErrInputGuarded — the input guard refused the request (RFC §6.18).
	ErrInputGuarded = core.NewError("pipeline: input refused by safety guard")
	// ErrOutputGuarded — the output guard refused the response (RFC §6.18).
	ErrOutputGuarded = core.NewError("pipeline: output refused by safety guard")
)
