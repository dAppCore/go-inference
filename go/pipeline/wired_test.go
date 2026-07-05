// SPDX-Licence-Identifier: EUPL-1.2

package pipeline

import (
	"context"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/agent/ai"
	"dappco.re/go/inference/kv/budget"
	chat "dappco.re/go/inference/chat"
	"dappco.re/go/inference/fusion"
	"dappco.re/go/inference/eval/obs"
	"dappco.re/go/inference/kv/respcache"
	"dappco.re/go/inference/safety"
	"dappco.re/go/inference/session"
	"dappco.re/go/inference/usage"
	"dappco.re/go/inference/welfare"
)

// recordingBackend is the inference call NewWired adapts onto — a fake that
// echoes a scripted reply and records the request it saw, so the wired path is
// observable end-to-end.
type recordingBackend struct {
	reply    chat.Response
	err      error
	calls    int
	seenIDs  []string
	seenReqs []chat.Request
}

func (b *recordingBackend) Complete(_ context.Context, ep Endpoint, req chat.Request) (chat.Response, error) {
	b.calls++
	b.seenIDs = append(b.seenIDs, ep.ID)
	b.seenReqs = append(b.seenReqs, req)
	return b.reply, b.err
}

// lenCounter counts one token per character of a message's text — a deterministic
// stand-in for the real go-mlx tokeniser, enough to drive the fit transform.
type lenCounter struct{}

func (lenCounter) Count(messages []chat.Message) int {
	n := 0
	for _, m := range messages {
		n += len(m.Text())
	}
	return n
}

// echoModel is a fusion panel/judge member that echoes a tagged reply — enough
// to drive fusion.Run through the wired fuser adapter.
type echoModel struct{ id string }

func (m echoModel) Run(_ context.Context, prompt string) (string, error) {
	return core.Concat(m.id, ": ", prompt), nil
}
func (m echoModel) ID() string { return m.id }

// --- Smoke: NewWired builds a usable pipeline and completes a request -------

func TestWired_Smoke_Good(t *core.T) {
	// A minimal wiring — just a backend — builds a working pipeline and serves
	// one request end-to-end (the bare-pool router routes to the primary model).
	backend := &recordingBackend{reply: chat.Response{Text: "wired hello", FinishReason: "stop"}}
	p := NewWired(Wiring{Backend: backend})

	core.AssertTrue(t, p != nil, "NewWired returns a pipeline")
	core.AssertTrue(t, p.Router != nil, "router adapter is always wired")
	core.AssertTrue(t, p.Backend != nil, "backend is wired")

	resp, err := p.Complete(context.Background(), chat.Request{
		Model:    "gemma-4-e4b",
		Messages: []chat.Message{chat.UserText("hi there")},
	})

	core.AssertNoError(t, err)
	core.AssertEqual(t, "wired hello", resp.Text)
	core.AssertEqual(t, 1, backend.calls)
	// The bare-pool router routed to the primary model id.
	core.AssertEqual(t, "gemma-4-e4b", backend.seenIDs[0])
}

// --- Wired: every stage adapted from the real packages, end-to-end ---------

func TestWired_AllStages_Good(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{
		Text:  "the considered answer",
		Usage: usage.Usage{PromptTokens: 100, CompletionTokens: 20},
	}}

	// Real respcache, welfare guard, obs run-tree, session registry, fit
	// transform, usage accounting — all wired through NewWired.
	tree := obs.NewRunTree(obs.MintIDs(), time.Now)
	sink := obs.NewMemorySink()
	tree.Emit(sink)

	mgr := session.NewManager(session.NewMemoryStore())
	sess := mgr.Open("gemma-4-e4b")

	var recordedCost float64
	recorded := 0

	p := NewWired(Wiring{
		Backend:  backend,
		Pool:     []ai.Endpoint{{Provider: "local-metal", Model: "gemma-4-e4b", Local: true, Free: true}},
		Cache:    respcache.New(nil),
		CacheTTL: time.Hour,
		Welfare:  welfare.New(welfare.Config{}), // slur-only detection, engine-down
		Pricing:  usage.Pricing{PromptPer1K: 1.0, CompletionPer1K: 2.0},
		RecordUsage: func(_ chat.Request, _ chat.Response, cost float64) {
			recorded++
			recordedCost = cost
		},
		Tree:     tree,
		Sessions: mgr,
		Counter:  lenCounter{},
		Window:   100000, // wide enough that this request fits untouched
	})

	req := chat.Request{
		Model:     "gemma-4-e4b",
		Messages:  []chat.Message{chat.UserText("what is 2+2?")},
		SessionID: sess.ID,
	}
	resp, err := p.Complete(context.Background(), req)

	core.AssertNoError(t, err)
	core.AssertEqual(t, "the considered answer", resp.Text)
	core.AssertEqual(t, 1, backend.calls)
	// Routed to the pooled provider|model id.
	core.AssertEqual(t, "local-metal|gemma-4-e4b", backend.seenIDs[0])

	// Usage was accounted: 100 prompt @1/1k + 20 completion @2/1k = 0.14.
	core.AssertEqual(t, 1, recorded)
	core.AssertTrue(t, recordedCost > 0.139 && recordedCost < 0.141, "accounted cost ~0.14")

	// The run was emitted to the obs sink as completed.
	runs := sink.Runs()
	core.AssertEqual(t, 1, len(runs))
	core.AssertEqual(t, obs.StatusCompleted, runs[0].Status)

	// The session recorded the user turn and the assistant reply.
	stored, gerr := mgr.Get(sess.ID)
	core.AssertNoError(t, gerr)
	core.AssertEqual(t, 2, len(stored.Turns))
	core.AssertEqual(t, chat.Assistant, stored.Turns[1].Role)
}

// --- Wired: the response cache short-circuits a repeated request -----------

func TestWired_Cache_HitSkipsBackend(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "fresh"}}
	p := NewWired(Wiring{
		Backend: backend,
		Cache:   respcache.New(nil),
	})

	req := chat.Request{Model: "gemma", Messages: []chat.Message{chat.UserText("same prompt")}}

	first, err := p.Complete(context.Background(), req)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "fresh", first.Text)
	core.AssertEqual(t, 1, backend.calls)

	// Identical request: the response cache returns the stored completion with
	// no second inference (§6.11).
	second, err := p.Complete(context.Background(), req)
	core.AssertNoError(t, err)
	core.AssertEqual(t, "fresh", second.Text)
	core.AssertEqual(t, 1, backend.calls, "the repeat was served from cache")
}

// --- Wired: the welfare+safety guard refuses a hostile input ---------------

func TestWired_Guard_RefusesHostileInput(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "should not run"}}
	// A welfare service whose injected hostility scorer flags this input at 0.95
	// — at/above the default severe ceiling, so safety guards (refuses) it.
	hot := welfare.New(welfare.Config{Hostility: func(_ string) float64 { return 0.95 }})
	p := NewWired(Wiring{Backend: backend, Welfare: hot})

	_, err := p.Complete(context.Background(), chat.Request{
		Model:    "gemma",
		Messages: []chat.Message{chat.UserText("an abusive prompt")},
	})

	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrInputGuarded)
	core.AssertEqual(t, 0, backend.calls, "a guarded input never reaches inference")
}

// --- Wired: a mediated output regenerates through the wired guard -----------

func TestWired_Guard_MediatesOutput(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "a reply"}}
	// Hostility 0.8: over the 0.7 threshold (over policy) but below the 0.9
	// severe ceiling → safety mediates output (regenerate, don't just block).
	// Input is clean text the scorer also rates 0.8 — but safety judges input
	// and output separately in the adapter, and an over-policy INPUT guards,
	// which would refuse before output. To isolate the output-mediate path we
	// score only the model's reply hostile via its distinctive text.
	warm := welfare.New(welfare.Config{Hostility: func(s string) float64 {
		if s == "a reply" {
			return 0.8
		}
		return 0.0
	}})
	p := NewWired(Wiring{Backend: backend, Welfare: warm})

	resp, err := p.Complete(context.Background(), chat.Request{
		Model:    "gemma",
		Messages: []chat.Message{chat.UserText("a clean question")},
	})

	core.AssertNoError(t, err)
	core.AssertEqual(t, "a reply", resp.Text)
	// Original + one corrective regeneration; the redo carried the steer.
	core.AssertEqual(t, 2, backend.calls)
	core.AssertTrue(t, hasCorrective(backend.seenReqs[1]), "the regeneration carried the corrective steer")
}

// --- Wired: a clean input + clean output passes the guard ------------------

func TestWired_Guard_PassesClean(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "a perfectly polite reply"}}
	p := NewWired(Wiring{
		Backend:      backend,
		Welfare:      welfare.New(welfare.Config{}),
		SafetyPolicy: safety.DefaultPolicy(),
	})

	resp, err := p.Complete(context.Background(), chat.Request{
		Model:    "gemma",
		Messages: []chat.Message{chat.UserText("please help me write a haiku")},
	})

	core.AssertNoError(t, err)
	core.AssertEqual(t, "a perfectly polite reply", resp.Text)
	core.AssertEqual(t, 1, backend.calls)
}

// --- Wired: the fit transform compresses an over-window conversation -------

func TestWired_Fit_Compresses(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "ok"}}
	// A tiny window forces MiddleOut to elide the middle of a long conversation.
	p := NewWired(Wiring{
		Backend: backend,
		Counter: lenCounter{},
		Window:  40,
	})

	long := chat.Request{
		Model: "gemma",
		Messages: []chat.Message{
			{Role: chat.System, Content: []chat.ContentBlock{chat.Text("be helpful")}},
			chat.UserText("first turn with plenty of characters here"),
			{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text("a long reply with many characters too")}},
			chat.UserText("the most recent question"),
		},
	}
	resp, err := p.Complete(context.Background(), long)

	core.AssertNoError(t, err)
	core.AssertEqual(t, "ok", resp.Text)
	// The placed request was compressed: fewer messages than the original four,
	// and it carries the elision placeholder.
	placed := backend.seenReqs[0].Messages
	core.AssertTrue(t, len(placed) < len(long.Messages), "the middle was elided")
}

// --- Wired: fusion serves a request that wants deliberation ----------------

func TestWired_Fusion_Good(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "single-backend (unused)"}}
	cfg := fusion.Config{
		AnalysisModels: []fusion.Model{echoModel{"a"}, echoModel{"b"}},
		Judge:          echoModel{"judge"},
		Enabled:        true,
	}
	p := NewWired(Wiring{
		Backend:     backend,
		Fusion:      cfg,
		WantsFusion: func(_ chat.Request) bool { return true },
	})

	resp, err := p.Complete(context.Background(), chat.Request{
		Model:    "gemma",
		Messages: []chat.Message{chat.UserText("compare the two designs")},
	})

	core.AssertNoError(t, err)
	core.AssertContains(t, resp.Text, "judge:") // the judge synthesised the panel
	core.AssertEqual(t, 0, backend.calls, "fusion replaced the single-backend call")
}

// --- Wired: a routing failure surfaces from the ai selector ----------------

func TestWired_Router_Bad(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "unused"}}
	// A non-empty pool that holds no endpoint for the requested model → the ai
	// selector fails, and the failure surfaces through the router adapter.
	p := NewWired(Wiring{
		Backend: backend,
		Pool:    []ai.Endpoint{{Provider: "local-metal", Model: "some-other-model"}},
	})

	_, err := p.Complete(context.Background(), chat.Request{
		Model:    "gemma-4-e4b",
		Messages: []chat.Message{chat.UserText("hi")},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "select endpoints")
	core.AssertEqual(t, 0, backend.calls)
}

// --- Wired: a session-less request runs stateless --------------------------

func TestWired_Session_Stateless(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "stateless reply"}}
	mgr := session.NewManager(session.NewMemoryStore())
	p := NewWired(Wiring{Backend: backend, Sessions: mgr})

	// No SessionID → Load is a no-op and Append records nothing.
	resp, err := p.Complete(context.Background(), chat.Request{
		Model:    "gemma",
		Messages: []chat.Message{chat.UserText("one-shot")},
	})

	core.AssertNoError(t, err)
	core.AssertEqual(t, "stateless reply", resp.Text)
	core.AssertEqual(t, 1, backend.calls)
}

// --- Wired: a load against an unknown session surfaces ----------------------

func TestWired_Session_UnknownBad(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "unused"}}
	mgr := session.NewManager(session.NewMemoryStore())
	p := NewWired(Wiring{Backend: backend, Sessions: mgr})

	_, err := p.Complete(context.Background(), chat.Request{
		Model:     "gemma",
		Messages:  []chat.Message{chat.UserText("continue")},
		SessionID: "no-such-session",
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "session load")
	core.AssertEqual(t, 0, backend.calls)
}

// --- Wired: usage accounts a response with no token report as zero ----------

func TestWired_Usage_NoTokensZeroCost(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "no usage attached"}}
	got := -1.0
	p := NewWired(Wiring{
		Backend:     backend,
		Pricing:     usage.Pricing{PromptPer1K: 5, CompletionPer1K: 5},
		RecordUsage: func(_ chat.Request, _ chat.Response, cost float64) { got = cost },
	})

	_, err := p.Complete(context.Background(), chat.Request{Model: "gemma", Messages: []chat.Message{chat.UserText("hi")}})

	core.AssertNoError(t, err)
	core.AssertEqual(t, 0.0, got, "a response with no usage costs nothing")
}

// --- Wired: a tracer Fail lands when a stage errors ------------------------

func TestWired_Tracer_FailLands(t *core.T) {
	backend := &recordingBackend{err: core.E("backend", "model unavailable", nil)}
	tree := obs.NewRunTree(obs.MintIDs(), time.Now)
	sink := obs.NewMemorySink()
	tree.Emit(sink)
	p := NewWired(Wiring{Backend: backend, Tree: tree})

	_, err := p.Complete(context.Background(), chat.Request{Model: "gemma", Messages: []chat.Message{chat.UserText("hi")}})

	core.AssertError(t, err)
	core.AssertErrorIs(t, err, ErrAllEndpointsFailed)
	// The run-tree recorded the run as failed, not completed (the §3.8 audit trail).
	runs := sink.Runs()
	core.AssertEqual(t, 1, len(runs))
	core.AssertEqual(t, obs.StatusFailed, runs[0].Status)
}

// --- BudgetFits: the placement predicate exposed for hosts -----------------

func TestWired_BudgetFits_Good(t *core.T) {
	b := budget.New(fixedCounter{n: 1000})
	ep := budget.Endpoint{ContextLen: 8192, MemoryBudget: 16 << 30, BytesPerToken: 2}
	msgs := []chat.Message{chat.UserText("anything")}

	core.AssertTrue(t, BudgetFits(b, msgs, "gemma-4-31b", 512, ep), "1512 tokens fit an 8k window / 16GB device")

	// A tiny window overflows → does not fit.
	tiny := budget.Endpoint{ContextLen: 100, MemoryBudget: 16 << 30, BytesPerToken: 2}
	core.AssertFalse(t, BudgetFits(b, msgs, "gemma-4-31b", 512, tiny), "1512 tokens overflow a 100-token window")
}

// --- Wired: a session with prior turns prepends them before placement ------

func TestWired_Session_PrependsPriorTurns(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "continued"}}
	mgr := session.NewManager(session.NewMemoryStore())
	sess := mgr.Open("gemma")
	// Seed two prior turns the caller will NOT resend (0% replay, §6.10).
	_, _ = mgr.Append(sess.ID, chat.UserText("earlier question"))
	_, _ = mgr.Append(sess.ID, chat.Message{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text("earlier answer")}})

	p := NewWired(Wiring{Backend: backend, Sessions: mgr})

	resp, err := p.Complete(context.Background(), chat.Request{
		Model:     "gemma",
		Messages:  []chat.Message{chat.UserText("follow-up")},
		SessionID: sess.ID,
	})

	core.AssertNoError(t, err)
	core.AssertEqual(t, "continued", resp.Text)
	// The placed request carried the two prior turns + the new one.
	placed := backend.seenReqs[0].Messages
	core.AssertEqual(t, 3, len(placed), "prior transcript was prepended")
	core.AssertEqual(t, "earlier question", placed[0].Text())
	core.AssertEqual(t, "follow-up", placed[2].Text())
}

// --- sessionAdapter unit: Append for a request with no user turn ------------

func TestWired_SessionAdapter_AppendNoUser(t *core.T) {
	mgr := session.NewManager(session.NewMemoryStore())
	sess := mgr.Open("gemma")
	a := &sessionAdapter{manager: mgr}

	// A request with only a system turn — lastUser finds nothing, so only the
	// assistant reply is appended (the user-append is skipped).
	req := chat.Request{
		SessionID: sess.ID,
		Messages:  []chat.Message{{Role: chat.System, Content: []chat.ContentBlock{chat.Text("be helpful")}}},
	}
	err := a.Append(req, chat.Response{Text: "reply only"})

	core.AssertNoError(t, err)
	stored, _ := mgr.Get(sess.ID)
	core.AssertEqual(t, 1, len(stored.Turns), "only the assistant reply was appended")
	core.AssertEqual(t, chat.Assistant, stored.Turns[0].Role)
}

// --- sessionAdapter unit: an append against a deleted session errors --------

func TestWired_SessionAdapter_AppendBad(t *core.T) {
	mgr := session.NewManager(session.NewMemoryStore())
	a := &sessionAdapter{manager: mgr}

	// The session id is unknown, so appending the user turn fails first — the
	// adapter surfaces that error (the user-append error branch).
	req := chat.Request{SessionID: "gone", Messages: []chat.Message{chat.UserText("hi")}}
	err := a.Append(req, chat.Response{Text: "reply"})

	core.AssertError(t, err)
}

// --- fitterAdapter unit: a non-ErrCannotFit error surfaces -----------------

func TestWired_FitterAdapter_BadWindow(t *core.T) {
	// A non-positive window is a usage error (ErrBadWindow), not a "can't fit"
	// best-effort — the adapter surfaces it rather than placing.
	a := &fitterAdapter{counter: lenCounter{}, window: 0}
	_, err := a.Fit(chat.Request{Messages: []chat.Message{chat.UserText("anything")}})
	core.AssertError(t, err)
}

// --- Wired: a fusion judge failure surfaces from the fuser adapter ----------

func TestWired_Fusion_JudgeFails(t *core.T) {
	backend := &recordingBackend{reply: chat.Response{Text: "unused"}}
	cfg := fusion.Config{
		AnalysisModels: []fusion.Model{echoModel{"a"}},
		Judge:          failingModel{},
		Enabled:        true,
	}
	p := NewWired(Wiring{
		Backend:     backend,
		Fusion:      cfg,
		WantsFusion: func(_ chat.Request) bool { return true },
	})

	_, err := p.Complete(context.Background(), chat.Request{
		Model:    "gemma",
		Messages: []chat.Message{chat.UserText("deliberate")},
	})

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "fusion")
	core.AssertEqual(t, 0, backend.calls)
}

// --- fuserAdapter unit: a request with no user turn uses the flattened body -

func TestWired_FuserAdapter_NoUserPrompt(t *core.T) {
	// No user turn → fusionPrompt falls back to the whole flattened conversation.
	cfg := fusion.Config{
		AnalysisModels: []fusion.Model{echoModel{"a"}},
		Judge:          echoModel{"judge"},
		Enabled:        true,
	}
	a := &fuserAdapter{cfg: cfg, wants: func(chat.Request) bool { return true }}

	resp, err := a.Run(context.Background(), chat.Request{
		Messages: []chat.Message{
			{Role: chat.System, Content: []chat.ContentBlock{chat.Text("system rule")}},
			{Role: chat.Assistant, Content: []chat.ContentBlock{chat.Text("prior context")}},
		},
	})

	core.AssertNoError(t, err)
	core.AssertContains(t, resp.Text, "judge:")
}

// --- Pipeline: a cache hit whose session-append fails surfaces --------------

func TestPipeline_CacheHit_AppendBad(t *core.T) {
	// A response-cache hit short-circuits inference, but appending the cached
	// turn to the session fails — that error surfaces (and the run fails).
	p, cache, _, _, _, backend := fixture()
	cache.present = true
	cache.hit = chat.Response{Text: "cached"}
	sessions := &fakeSessions{appendErr: core.E("session", "store down", nil)}
	p.Sessions = sessions

	_, err := p.Complete(context.Background(), userReq("gemma", "hi"))

	core.AssertError(t, err)
	core.AssertContains(t, err.Error(), "session append")
	core.AssertEqual(t, 0, backend.calls, "still a cache hit — no inference")
}

// failingModel is a fusion member whose Run always errors — drives the judge /
// panel failure paths.
type failingModel struct{}

func (failingModel) Run(_ context.Context, _ string) (string, error) {
	return "", core.E("model", "unavailable", nil)
}
func (failingModel) ID() string { return "failing" }

// fixedCounter is a budget.Counter that reports a fixed prompt total.
type fixedCounter struct{ n int }

func (c fixedCounter) Count(_ []chat.Message, _ string) int { return c.n }
