// SPDX-Licence-Identifier: EUPL-1.2

package pipeline

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/ai"
	"dappco.re/go/inference/kv/budget"
	chat "dappco.re/go/inference/chat"
	"dappco.re/go/inference/fusion"
	"dappco.re/go/inference/obs"
	"dappco.re/go/inference/kv/respcache"
	"dappco.re/go/inference/safety"
	"dappco.re/go/inference/session"
	"dappco.re/go/inference/transform"
	"dappco.re/go/inference/usage"
	"dappco.re/go/inference/welfare"
)

// Wiring carries the real packages NewWired adapts onto the pipeline seams. The
// only required piece is Backend — the actual inference call (the pieces below
// it have working defaults / are optional). Each non-nil field opts its stage in:
//
//	p := pipeline.NewWired(pipeline.Wiring{
//		Backend:  myInferenceClient,
//		Pool:     endpoints,                 // ai routing over this pool
//		Cache:    respcache.New(nil),        // response cache (§6.11)
//		Welfare:  welfare.New(welfare.Config{}), // safety detect (§6.18)
//		Tree:     obs.NewRunTree(obs.MintIDs(), time.Now), // run-tree (§3.7)
//		Sessions: session.NewManager(session.NewMemoryStore()), // sessions (§6.10)
//		Counter:  myTokenCounter, Window: 8192,                  // context fit (§6.13)
//	})
//	resp, err := p.Complete(ctx, req)
type Wiring struct {
	// Backend is the inference call (RFC §6.1) — the one piece the pipeline
	// cannot synthesise. Required.
	Backend Backend

	// Pool is the routable endpoint set the ai router selects over (§6.2). When
	// empty, the router adapter routes every request to a single synthetic
	// "primary" endpoint so a bare wiring still serves.
	Pool []ai.Endpoint
	// SelectTemplate seeds the per-request ai.SelectRequest (price ceiling, ZDR,
	// quant constraints, provider preferences); the request's model + fallback
	// chain are filled in per call.
	SelectTemplate ai.SelectRequest

	// Cache is the response cache (§6.11); nil skips the cache stage.
	Cache *respcache.Cache
	// CacheTTL is the entry lifetime for cache writes; 0 means no expiry.
	CacheTTL time.Duration

	// Welfare + SafetyPolicy form the safety guard (§6.18); a nil Welfare skips
	// the guard stage. A zero SafetyPolicy uses safety.DefaultPolicy().
	Welfare      *welfare.Service
	SafetyPolicy safety.Policy

	// Pricing + RecordUsage form the usage sink (§6.6); a nil RecordUsage skips
	// accounting. RecordUsage receives the accounted cost for each completion.
	Pricing     usage.Pricing
	RecordUsage func(req chat.Request, resp chat.Response, cost float64)

	// Tree is the observability run-tree (§3.7); nil skips tracing.
	Tree *obs.RunTree

	// Sessions is the conversation registry (§6.10); nil runs stateless.
	Sessions *session.Manager

	// Counter + Window form the context-fit transform (§6.13); a nil Counter or
	// non-positive Window skips fitting.
	Counter transform.Counter
	Window  int

	// Fusion + WantsFusion form the deliberation seam (§6.9); a nil WantsFusion
	// (or one that returns false) takes the single-backend path. Fusion is the
	// panel + judge config.
	Fusion      fusion.Config
	WantsFusion func(req chat.Request) bool
}

// NewWired builds a *Pipeline from the real packages, mapping each concrete
// package onto a seam interface (RFC §6 — the assembled serving path). The thin
// adapters live in this file so the core pipeline (pipeline.go / types.go) stays
// interface-only and import-light. A stage whose wiring is absent is simply not
// set, so the pipeline skips it exactly as for a hand-built Pipeline.
//
//	p := pipeline.NewWired(pipeline.Wiring{Backend: client, Pool: pool})
//	resp, err := p.Complete(ctx, chat.Request{Model: "gemma-4-e4b", Messages: msgs})
func NewWired(w Wiring) *Pipeline {
	p := &Pipeline{
		Router:  &routerAdapter{pool: w.Pool, template: w.SelectTemplate},
		Backend: w.Backend,
	}

	if w.Cache != nil {
		p.Cache = &cacheAdapter{cache: w.Cache, ttl: w.CacheTTL}
	}
	if w.Welfare != nil {
		policy := w.SafetyPolicy
		if policy == (safety.Policy{}) {
			policy = safety.DefaultPolicy()
		}
		p.Guard = &guardAdapter{welfare: w.Welfare, policy: policy}
	}
	if w.RecordUsage != nil {
		p.UsageSink = &usageAdapter{pricing: w.Pricing, record: w.RecordUsage}
	}
	if w.Tree != nil {
		p.Tracer = &tracerAdapter{tree: w.Tree}
	}
	if w.Sessions != nil {
		p.Sessions = &sessionAdapter{manager: w.Sessions}
	}
	if w.Counter != nil && w.Window > 0 {
		p.Fitter = &fitterAdapter{counter: w.Counter, window: w.Window}
	}
	if w.WantsFusion != nil {
		p.Fuser = &fuserAdapter{cfg: w.Fusion, wants: w.WantsFusion}
	}
	return p
}

// --- cacheAdapter: respcache → Cache (§6.11) -------------------------------

type cacheAdapter struct {
	cache *respcache.Cache
	ttl   time.Duration
}

func (a *cacheAdapter) Get(req chat.Request) (chat.Response, bool) {
	out, ok := a.cache.Get(toCacheRequest(req))
	if !ok {
		return chat.Response{}, false
	}
	return fromCompletion(out), true
}

func (a *cacheAdapter) Set(req chat.Request, resp chat.Response) {
	a.cache.Set(toCacheRequest(req), respcache.Completion{
		Text:         resp.Text,
		Model:        req.PrimaryModel(),
		FinishReason: resp.FinishReason,
	}, a.ttl)
}

// toCacheRequest projects the canonical request onto the cache's key view — the
// subset of §6.1 fields that determine the completion. Multimodal content is
// flattened to its text (the cache keys on text, not media bytes).
func toCacheRequest(req chat.Request) respcache.Request {
	msgs := make([]respcache.Message, len(req.Messages))
	for i, m := range req.Messages {
		msgs[i] = respcache.Message{Role: m.Role.String(), Content: m.Text()}
	}
	return respcache.Request{
		Model:       req.PrimaryModel(),
		Messages:    msgs,
		Temperature: req.Temperature,
		TopP:        req.TopP,
		MaxTokens:   req.MaxTokens,
		Seed:        req.Seed,
		Stop:        req.Stop,
	}
}

// fromCompletion lifts a stored completion back into a canonical response.
func fromCompletion(c respcache.Completion) chat.Response {
	return chat.Response{
		Messages:     []chat.Message{{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text(c.Text)}}},
		Text:         c.Text,
		FinishReason: c.FinishReason,
	}
}

// --- routerAdapter: ai.SelectEndpoints → Router (§6.2) ---------------------

type routerAdapter struct {
	pool     []ai.Endpoint
	template ai.SelectRequest
}

func (a *routerAdapter) Select(req chat.Request) ([]Endpoint, error) {
	// An empty pool means "no routing data" — serve a single synthetic primary
	// endpoint so a bare wiring still completes (the backend ignores the id, or
	// keys on the primary model).
	if len(a.pool) == 0 {
		return []Endpoint{{ID: req.PrimaryModel()}}, nil
	}

	sel := a.template
	sel.Model = req.Model
	sel.Models = req.Models

	result := ai.SelectEndpoints(sel, a.pool)
	if !result.OK {
		return nil, core.E("pipeline", "select endpoints", result.Value.(error))
	}
	chosen := result.Value.([]ai.Endpoint)
	out := make([]Endpoint, len(chosen))
	for i, ep := range chosen {
		// Provider + model uniquely names a routed endpoint; the backend keys on it.
		out[i] = Endpoint{ID: core.Concat(ep.Provider, "|", ep.Model)}
	}
	return out, nil
}

// --- guardAdapter: welfare + safety → Guard (§6.18) ------------------------

type guardAdapter struct {
	welfare *welfare.Service
	policy  safety.Policy
}

func (a *guardAdapter) CheckInput(req chat.Request) Decision {
	latest, priors := userTurns(req)
	in := a.welfare.Detect(latest, priors)
	// Judge input alone: a clean output read can't lift an over-policy input.
	return toDecision(safety.Decide(in, welfare.DetectResult{}, a.policy))
}

func (a *guardAdapter) CheckOutput(req chat.Request, resp chat.Response) Decision {
	_, priors := userTurns(req)
	out := a.welfare.Detect(resp.Text, priors)
	// Judge output alone: the input already passed CheckInput.
	return toDecision(safety.Decide(welfare.DetectResult{}, out, a.policy))
}

// toDecision maps safety's verdict onto the pipeline's guard decision.
func toDecision(d safety.Decision) Decision {
	switch d {
	case safety.Guard:
		return DecisionGuard
	case safety.Mediate:
		return DecisionMediate
	default:
		return DecisionPass
	}
}

// userTurns returns the latest user message's text and the prior user turns
// (oldest→newest), the shape welfare.Detect reads.
func userTurns(req chat.Request) (latest string, priors []string) {
	// Count the user turns first so priors is sized exactly once: it holds every
	// user turn except the last (which becomes latest). The single-turn case
	// keeps priors nil — no allocation — and the multi-turn case presizes rather
	// than growing geometrically (and re-slicing off a seeded empty head).
	users := 0
	for _, m := range req.Messages {
		if m.Role == chat.User {
			users++
		}
	}
	if users > 1 {
		priors = make([]string, 0, users-1)
	}
	seen := 0
	for _, m := range req.Messages {
		if m.Role != chat.User {
			continue
		}
		seen++
		if seen == users {
			latest = m.Text()
		} else {
			priors = append(priors, m.Text())
		}
	}
	return latest, priors
}

// --- usageAdapter: usage (+ accounting) → UsageSink (§6.6) -----------------

type usageAdapter struct {
	pricing usage.Pricing
	record  func(req chat.Request, resp chat.Response, cost float64)
}

func (a *usageAdapter) Record(req chat.Request, resp chat.Response) {
	cost := a.pricing.AccountedCost(readUsage(resp.Usage))
	a.record(req, resp, cost)
}

// readUsage lifts a usage.Usage out of the response's opaque Usage field (the
// canonical chat.Response keeps it as any to stay import-light, §6.6). A missing
// or differently-typed value accounts as zero — a response with no token report
// costs nothing rather than erroring the path.
func readUsage(v any) usage.Usage {
	if u, ok := v.(usage.Usage); ok {
		return u
	}
	return usage.Usage{}
}

// --- tracerAdapter: obs.RunTree → Tracer (§3.7) ----------------------------

type tracerAdapter struct {
	tree *obs.RunTree
}

func (a *tracerAdapter) Start(_ context.Context, req chat.Request) any {
	return a.tree.StartRun("chat", map[string]any{
		"model":    req.PrimaryModel(),
		"messages": len(req.Messages),
	})
}

func (a *tracerAdapter) Finish(handle any, resp chat.Response) {
	run, _ := handle.(*obs.Run)
	a.tree.Finish(run, map[string]any{"text": resp.Text}, resp.Usage)
}

func (a *tracerAdapter) Fail(handle any, err error) {
	run, _ := handle.(*obs.Run)
	a.tree.Fail(run, err)
}

// --- sessionAdapter: session.Manager → Sessions (§6.10) --------------------

type sessionAdapter struct {
	manager *session.Manager
}

func (a *sessionAdapter) Load(req chat.Request) (chat.Request, error) {
	// No session id → run stateless (a one-shot completion).
	if req.SessionID == "" {
		return req, nil
	}
	sess, err := a.manager.Get(req.SessionID)
	if err != nil {
		return chat.Request{}, err
	}
	// Prepend the stored transcript before the caller's new turns (0% replay,
	// §6.10): the caller sends only what is new, the registry supplies the rest.
	// One presized allocation for the combined transcript — the nested-append
	// form grew twice (the stored turns, then again to fit the new ones).
	if len(sess.Turns) > 0 {
		msgs := make([]chat.Message, 0, len(sess.Turns)+len(req.Messages))
		msgs = append(msgs, sess.Turns...)
		msgs = append(msgs, req.Messages...)
		req.Messages = msgs
	}
	return req, nil
}

func (a *sessionAdapter) Append(req chat.Request, resp chat.Response) error {
	if req.SessionID == "" {
		return nil
	}
	// Record the live turn: the most-recent user message, then the assistant
	// reply, so the next request continues from here.
	if latest, ok := lastUser(req); ok {
		if _, err := a.manager.Append(req.SessionID, latest); err != nil {
			return err
		}
	}
	reply := chat.Message{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text(resp.Text)}}
	_, err := a.manager.Append(req.SessionID, reply)
	return err
}

// lastUser returns the most-recent user message of the request, if any.
func lastUser(req chat.Request) (chat.Message, bool) {
	for i := len(req.Messages) - 1; i >= 0; i-- {
		if req.Messages[i].Role == chat.User {
			return req.Messages[i], true
		}
	}
	return chat.Message{}, false
}

// --- fitterAdapter: budget + transform → Fitter (§6.13, §6.11) -------------

type fitterAdapter struct {
	counter transform.Counter
	window  int
}

func (a *fitterAdapter) Fit(req chat.Request) (chat.Request, error) {
	out, _, err := transform.MiddleOut(req.Messages, a.counter, a.window)
	if err != nil {
		// ErrCannotFit returns the best-effort compressed set: place it and let
		// routing fall out to a roomier endpoint (§6.2) rather than failing here.
		if core.Is(err, transform.ErrCannotFit) {
			req.Messages = out
			return req, nil
		}
		return chat.Request{}, err
	}
	req.Messages = out
	return req, nil
}

// budgetFits is the placement predicate the host pairs with the fitter when it
// wants the §6.13 grade (fits / needs-transform / needs-larger / overflows)
// before placing — exposed so a caller routing on memory budget can reuse the
// same budget.Budget the fitter is sized from.
//
//	if pipeline.BudgetFits(b, msgs, "gemma-4-31b", 512, ep) { place(ep) }
func budgetFits(b *budget.Budget, msgs []chat.Message, model string, expected int, ep budget.Endpoint) bool {
	return b.Decide(msgs, model, expected, ep).Decision == budget.DecisionFits
}

// BudgetFits reports whether a request fits an endpoint's window and memory
// budget (§6.13) — the placement check a host runs alongside the fitter.
func BudgetFits(b *budget.Budget, msgs []chat.Message, model string, expected int, ep budget.Endpoint) bool {
	return budgetFits(b, msgs, model, expected, ep)
}

// --- fuserAdapter: fusion → Fuser (§6.9) -----------------------------------

type fuserAdapter struct {
	cfg   fusion.Config
	wants func(req chat.Request) bool
}

func (a *fuserAdapter) Wants(req chat.Request) bool { return a.wants(req) }

func (a *fuserAdapter) Run(ctx context.Context, req chat.Request) (chat.Response, error) {
	prompt := fusionPrompt(req)
	res, err := fusion.Run(ctx, prompt, a.cfg)
	if err != nil {
		return chat.Response{}, err
	}
	return chat.Response{
		Messages:     []chat.Message{{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text(res.Answer)}}},
		Text:         res.Answer,
		FinishReason: "stop",
	}, nil
}

// fusionPrompt builds the deliberation prompt from the request — the latest user
// turn (fusion deliberates over one prompt, §6.9). Falls back to the whole
// flattened conversation when there is no user turn.
func fusionPrompt(req chat.Request) string {
	if m, ok := lastUser(req); ok {
		return m.Text()
	}
	parts := make([]string, 0, len(req.Messages))
	for _, m := range req.Messages {
		parts = append(parts, m.Text())
	}
	return core.Join("\n", parts...)
}
