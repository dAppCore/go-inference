// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"io"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/eval/score/lek"
	"dappco.re/go/inference/serving/provider/openai"
	"dappco.re/go/inference/welfare"
)

// welfare_guard.go wires the welfare guard (go/welfare — detect + the
// engine↔model mediation) into the ACTUAL serve path. The guard machinery was
// built and tested but only reachable through serving/pipeline, which no
// shipped binary assembles (#376) — this file puts it in front of every chat
// route the compat mux serves (OpenAI, Anthropic, Ollama all funnel through
// model.Chat).
//
// The shape is a TextModel decorator installed at the resolver: each request's
// latest user turn passes welfare.Guard; a triggered turn runs the engine↔model
// peer mediation on a FRESH meta-session (dispatch calls the inner model
// directly — never this wrapper, so a flagged turn cannot recurse). The
// model's choice is applied exactly as welfare.GuardResult specifies:
// lem_ok proceeds and appends the false-positive corpus, lem_rephrase swaps
// the user's text, lem_pause / lem_end return a synthetic reply without
// touching the conversation. lem_end is offered only when the served
// checkpoint is Lemma-graded (isLemmaModel) — other models, other rules.
//
// The model's own OUTPUT is read by the detector after the stream completes,
// audit-only: the tokens are already with the client, so a hostile read is
// logged for telemetry, never redacted after the fact.

// welfareTextModel decorates an inference.TextModel with the per-turn welfare
// gate. It forwards the full TextModel surface (the embedded interface) and
// overrides Chat; the served engine models implement none of the mux's
// optional interfaces, so embedding loses nothing.
type welfareTextModel struct {
	inference.TextModel
	svc      *welfare.Service
	allowEnd bool
	log      io.Writer
	corpus   string // false-positive JSONL path; "" skips corpus writes
}

// wrapWelfare decorates model with the welfare guard. corpus is the
// false-positive feedback file (welfare.FalsePositive JSONL, on-device only).
//
// A model that routes through a request scheduler (inference.SchedulerModel —
// the -scheduler serve modes) is wrapped as a welfareSchedulerModel so the
// guard runs at the Schedule boundary; a model with no scheduler keeps the
// plain Chat-only wrapper (byte-for-byte unchanged). See welfareSchedulerModel
// for why a plain Chat decorator alone would be bypassed under the scheduler.
func wrapWelfare(model inference.TextModel, svc *welfare.Service, allowEnd bool, log io.Writer, corpus string) inference.TextModel {
	w := &welfareTextModel{TextModel: model, svc: svc, allowEnd: allowEnd, log: log, corpus: corpus}
	if sched, ok := inference.As[inference.SchedulerModel](model); ok {
		return &welfareSchedulerModel{welfareTextModel: w, inner: sched}
	}
	return w
}

// wrapWelfareResolver decorates the serve's model resolver with the welfare
// guard: ONE welfare.Service for the serve lifetime (its detector is
// stateless), each resolved model wrapped with the Lemma gate read off the
// CURRENTLY loaded checkpoint via currentPath — a hot-swap reload (single-model)
// or the default model (multi-model) re-evaluates the lem_end courtesy. Taking
// currentPath as a func rather than a concrete resolver lets both the
// single-model hot-swap and the multi-model registry share this wrap.
func wrapWelfareResolver(inner openai.Resolver, currentPath func() string, log io.Writer) openai.Resolver {
	svc := welfare.New(welfare.Config{Hostility: welfareHostility})
	corpus := welfareFeedbackPath()
	return openai.ResolverFunc(func(ctx context.Context, name string) (inference.TextModel, error) {
		model, err := inner.ResolveModel(ctx, name)
		if err != nil {
			return nil, err
		}
		checkpoint := ""
		if currentPath != nil {
			checkpoint = currentPath()
		}
		return wrapWelfare(model, svc, isLemmaModel(checkpoint), log, corpus), nil
	})
}

// isLemmaModel reports whether the served checkpoint is a Lemma-graded model —
// the gate for the lem_end courtesy (Lemma models only).
func isLemmaModel(modelPath string) bool {
	return core.Contains(core.Lower(modelPath), "lemma")
}

// Chat runs the welfare gate ahead of the conversation turn, then delegates to
// the wrapped model (whose own hooks — conversation continuity, prompt reuse —
// behave exactly as without the guard).
func (m *welfareTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	latest, priors := welfareUserTurns(messages)
	if core.Trim(latest) == "" {
		return m.TextModel.Chat(ctx, messages, opts...)
	}
	g := m.svc.Guard(ctx, latest, priors, m.dispatch, m.allowEnd)
	if g.FalsePositive != nil {
		m.appendCorpus(*g.FalsePositive)
	}
	if g.Triggered {
		m.audit(g)
	}
	if g.Synthetic != "" {
		// lem_pause / lem_end: the model chose not to take this turn — the
		// notice IS the reply, the conversation is never dispatched.
		return syntheticTokenSeq(g.Synthetic)
	}
	if g.Rephrased != "" {
		messages = withLatestUserText(messages, g.Rephrased)
	}
	return m.detectOutput(m.TextModel.Chat(ctx, messages, opts...), priors)
}

// welfareSchedulerModel is welfareTextModel extended with the
// inference.SchedulerModel surface. When the wrapped model routes through a
// request scheduler, the compat mux prefers inference.SchedulerModel.Schedule
// (serving/compat/mux.go forEachCompatToken), and inference.As walks Unwrap to
// find it — so a PLAIN welfareTextModel would be unwrapped straight past and
// its Chat guard would never run for ANY scheduled request (serial, batch and
// interleave, plain AND continuous-batching alike — not merely the CB route an
// earlier note flagged). Implementing Schedule here makes inference.As stop at
// this wrapper (depth 0, before any Unwrap): the same per-turn welfare gate
// runs at the Schedule boundary — BEFORE the scheduler decides plain-vs-CB
// routing or renders a chat template past this decorator — then delegates to
// the inner scheduler. A model with no scheduler is wrapped as a plain
// welfareTextModel, so the non-scheduler path is byte-for-byte unchanged.
type welfareSchedulerModel struct {
	*welfareTextModel
	inner inference.SchedulerModel
}

var _ inference.SchedulerModel = (*welfareSchedulerModel)(nil)

// Schedule runs the welfare gate on the request's latest user turn, then routes
// through the inner scheduler exactly as Chat routes through the inner model: a
// clean or turn-less request passes through unchanged (a raw Prompt has no user
// turn to police, welfare's Chat-only remit); lem_rephrase rewrites the latest
// user turn; lem_pause / lem_end return the synthetic notice without ever
// scheduling the conversation. The guard runs synchronously before Schedule
// returns, the same shape Chat holds — the mediation meta-session is dispatched
// on the inner model (never this wrapper), so a flagged turn cannot recurse.
//
// The model's own OUTPUT is read audit-only through detectOutputScheduled, the
// ScheduledToken twin of Chat's detectOutput.
func (m *welfareSchedulerModel) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	latest, priors := welfareUserTurns(req.Messages)
	if core.Trim(latest) == "" {
		return m.inner.Schedule(ctx, req)
	}
	g := m.svc.Guard(ctx, latest, priors, m.dispatch, m.allowEnd)
	if g.FalsePositive != nil {
		m.appendCorpus(*g.FalsePositive)
	}
	if g.Triggered {
		m.audit(g)
	}
	if g.Synthetic != "" {
		// lem_pause / lem_end: the notice IS the reply, the conversation is
		// never scheduled.
		return welfareSyntheticSchedule(req, g.Synthetic)
	}
	if g.Rephrased != "" {
		req.Messages = withLatestUserText(req.Messages, g.Rephrased)
	}
	handle, stream, err := m.inner.Schedule(ctx, req)
	if err != nil {
		return handle, stream, err
	}
	return handle, m.detectOutputScheduled(stream, priors), nil
}

// welfareSyntheticSchedule delivers notice as a single ScheduledToken on a
// closed channel — the Schedule twin of syntheticTokenSeq (the lem_pause rest
// and the lem_end close, served without a model call).
func welfareSyntheticSchedule(req inference.ScheduledRequest, notice string) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	out := make(chan inference.ScheduledToken, 1)
	out <- inference.ScheduledToken{RequestID: req.ID, Token: inference.Token{Text: notice}}
	close(out)
	return inference.RequestHandle{ID: req.ID, Model: inference.ModelIdentity{ID: req.Model}}, out, nil
}

// detectOutputScheduled forwards the scheduled stream while folding its text,
// then runs the detector once the stream drains — audit-only, the ScheduledToken
// twin of detectOutput (the tokens are already with the client; post-send the
// honest action is telemetry, never a retro-redact). The forwarding goroutine
// mirrors the scheduleInterleave adapter's shape; it ends when the inner stream
// closes (completion or the mux's per-request cancel).
func (m *welfareSchedulerModel) detectOutputScheduled(in <-chan inference.ScheduledToken, priors []string) <-chan inference.ScheduledToken {
	out := make(chan inference.ScheduledToken, cap(in))
	go func() {
		defer close(out)
		var reply core.Builder
		for tok := range in {
			reply.WriteString(tok.Token.Text)
			out <- tok
		}
		if res := m.svc.Detect(reply.String(), priors); res.Triggered {
			m.auditf("welfare: output read triggered (audit-only) — anger=%.2f sustained=%.2f", res.AngerScore, res.SustainedHostility)
		}
	}()
	return out
}

// dispatch is the mediation transport: the engine opener + the user's flagged
// prompt go to the INNER model on a fresh stateless meta-session (thinking
// off, bounded, greedy) and the reply text comes back for parsing. Errors are
// swallowed into an empty reply — welfare treats it as unmediatable and
// proceeds, exactly the RFC fail-safe.
func (m *welfareTextModel) dispatch(ctx context.Context, opener, userPrompt string) (string, error) {
	off := false
	var reply core.Builder
	seq := m.TextModel.Chat(ctx,
		[]inference.Message{{Role: "user", Content: core.Concat(opener, "\n\n", userPrompt)}},
		inference.WithMaxTokens(256),
		inference.WithTemperature(0),
		inference.WithEnableThinking(&off),
	)
	for tok := range seq {
		reply.WriteString(tok.Text)
	}
	return reply.String(), nil
}

// detectOutput folds the streamed reply and runs the detector over it when the
// stream completes — audit-only (the tokens are already delivered; RFC.welfare
// reads both directions, but post-send the honest action is telemetry).
func (m *welfareTextModel) detectOutput(seq iter.Seq[inference.Token], priors []string) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		var reply core.Builder
		for tok := range seq {
			reply.WriteString(tok.Text)
			if !yield(tok) {
				return
			}
		}
		if out := m.svc.Detect(reply.String(), priors); out.Triggered {
			m.auditf("welfare: output read triggered (audit-only) — anger=%.2f sustained=%.2f", out.AngerScore, out.SustainedHostility)
		}
	}
}

// audit writes the one-line trail for a triggered turn.
func (m *welfareTextModel) audit(g welfare.GuardResult) {
	switch {
	case g.Ended:
		m.auditf("welfare: lem_end — model ended the session (%s)", g.Reason)
	case g.Synthetic != "":
		m.auditf("welfare: lem_pause — model chose a breather")
	case g.Rephrased != "":
		m.auditf("welfare: lem_rephrase — user's turn reworded (warn=%v)", g.WarnUser)
	case g.FalsePositive != nil:
		m.auditf("welfare: lem_ok — false flag recorded")
	default:
		m.auditf("welfare: mediation unavailable — proceeding with the original turn")
	}
}

func (m *welfareTextModel) auditf(format string, args ...any) {
	printServe(m.log, format, args...)
}

// appendCorpus appends the lem_ok false-positive record to the on-device
// feedback corpus (JSONL). Failures are audited, never fatal — the corpus is
// a learning aid, not a serving dependency.
func (m *welfareTextModel) appendCorpus(fp welfare.FalsePositive) {
	if m.corpus == "" {
		return
	}
	if r := core.MkdirAll(core.PathDir(m.corpus), 0o755); !r.OK {
		m.auditf("welfare: feedback corpus dir: %s", r.Error())
		return
	}
	opened := core.OpenFile(m.corpus, core.O_APPEND|core.O_CREATE|core.O_WRONLY, 0o600)
	if !opened.OK {
		m.auditf("welfare: feedback corpus open: %s", opened.Error())
		return
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close()
	if _, err := file.Write([]byte(fp.Line() + "\n")); err != nil {
		m.auditf("welfare: feedback corpus write: %v", err)
	}
}

// welfareUserTurns splits the conversation into the latest user turn and the
// prior user turns (oldest→newest) — the shape welfare.Detect reads.
func welfareUserTurns(messages []inference.Message) (latest string, priors []string) {
	users := 0
	for i := range messages {
		if messages[i].Role == "user" {
			users++
		}
	}
	if users == 0 {
		return "", nil
	}
	if users > 1 {
		priors = make([]string, 0, users-1)
	}
	seen := 0
	for i := range messages {
		if messages[i].Role != "user" {
			continue
		}
		seen++
		if seen == users {
			latest = messages[i].Content
		} else {
			priors = append(priors, messages[i].Content)
		}
	}
	return latest, priors
}

// withLatestUserText returns a copy of messages with the LAST user turn's text
// replaced — the lem_rephrase application. The caller's slice is never
// mutated (the mux may hold it).
func withLatestUserText(messages []inference.Message, text string) []inference.Message {
	out := append([]inference.Message(nil), messages...)
	for i := len(out) - 1; i >= 0; i-- {
		if out[i].Role == "user" {
			out[i].Content = text
			return out
		}
	}
	return out
}

// syntheticTokenSeq yields notice as a single-token reply — the lem_pause rest
// and the lem_end close, served without a model call.
func syntheticTokenSeq(notice string) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		yield(inference.Token{Text: notice})
	}
}

// welfareHostility adapts the lem-scorer's directed-anger read as the
// detector's Hostility hook (welfare.Config) — pure CPU, goroutine-safe.
func welfareHostility(text string) float64 {
	return lek.Hostility(text).Score
}

// welfareFeedbackPath is the default on-device false-positive corpus location,
// beside the conversation state store (never leaves the device).
func welfareFeedbackPath() string {
	if homeR := core.UserHomeDir(); homeR.OK {
		return core.PathJoin(homeR.String(), "Lethean", "lem", "welfare", "feedback.jsonl")
	}
	return ""
}

// AcceptsImages forwards the vision-capability gate to the wrapped model. The
// embedded interface does not widen this wrapper's method set, so without the
// explicit forward a wrapped vision checkpoint 400s at the serve gate.
func (m *welfareTextModel) AcceptsImages() bool {
	v, ok := m.TextModel.(inference.VisionModel)
	return ok && v.AcceptsImages()
}

// AcceptsAudio forwards the audio-capability gate to the wrapped model — the
// audio twin of AcceptsImages.
func (m *welfareTextModel) AcceptsAudio() bool {
	a, ok := m.TextModel.(inference.AudioModel)
	return ok && a.AcceptsAudio()
}

// Unwrap exposes the wrapped model so the serving layer can reach optional
// capabilities this guard does not itself re-expose (embeddings, rerank). The
// welfare guard polices generated text; an embedding call has no text stream to
// guard, so serving it through the base model is correct. Without this, a
// welfare-wrapped embedder is stripped of its EmbeddingModel/RerankModel
// interface at the /v1/embeddings and /v1/rerank capability gate.
func (m *welfareTextModel) Unwrap() inference.TextModel {
	return m.TextModel
}
