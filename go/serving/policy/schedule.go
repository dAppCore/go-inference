// SPDX-Licence-Identifier: EUPL-1.2

package policy

import (
	"context"

	"dappco.re/go/inference"
)

// schedule.go closes the scheduler bypass of the outbound policy. policyTextModel
// (wrap.go) decorates Chat and exposes Unwrap, but carries NO Schedule method —
// so when the deployment also runs -scheduler, the compat mux prefers
// inference.SchedulerModel.Schedule (serving/compat/mux.go forEachCompatToken)
// and inference.As walks Unwrap straight PAST the policy guard to the scheduler
// beneath. The result was that -policy + --scheduler (any mode: serial, batch,
// interleave — plain OR continuous-batching) served model output with the
// outbound policy (redact / refuse / rewrite) entirely unenforced. This mirrors
// the welfare guard's identical hole, fixed at the same Schedule seam in
// serving/welfare_guard.go's welfareSchedulerModel.

// policySchedulerModel is policyTextModel extended with the
// inference.SchedulerModel surface. When the wrapped model routes through a
// request scheduler, implementing Schedule here makes inference.As STOP at this
// wrapper (depth 0, before any Unwrap), so the mux runs the outbound policy over
// the scheduled output stream instead of unwrapping past it. A model with no
// scheduler keeps the plain Chat-only policyTextModel (byte-for-byte unchanged).
//
// Unlike welfare's Schedule guard — whose output read is audit-only — the policy
// enforcer is ACTIVE on the output: every scheduled request's token stream runs
// through the SAME streaming Enforcer fold the Chat path uses (enforce in
// wrap.go), so redact spans are replaced mid-stream, refuse terminates that
// request's stream with the configured message, rewrite spans are mediated in
// place, hold-back withholds an incomplete match across token boundaries, and
// audit Events fire exactly as on the plain path. Cancellation and backpressure
// pass through: sends select on ctx so a client disconnect cannot block the fold
// goroutine, and a refuse cancels the inner request rather than leaking it.
type policySchedulerModel struct {
	*policyTextModel
	inner inference.SchedulerModel
}

var _ inference.SchedulerModel = (*policySchedulerModel)(nil)

// Schedule routes the request through the inner scheduler unchanged (the outbound
// policy polices OUTPUT, never the request), then wraps the returned per-request
// token stream so every token's Text feeds the same Enforcer the Chat path drives.
// The handle is passed through verbatim so the mux's cancel-by-ID (client abort)
// still reaches the inner scheduler through this wrapper's Unwrap chain.
func (m *policySchedulerModel) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	handle, stream, err := m.inner.Schedule(ctx, req)
	if err != nil {
		return handle, stream, err
	}
	return handle, m.enforceScheduled(ctx, handle, stream), nil
}

// enforceScheduled runs the outbound policy over one scheduled request's token
// stream — the ScheduledToken twin of policyTextModel.enforce. It reads the inner
// stream, folds each token's Text through a fresh Enforcer, emits the settled
// (enforced) bytes as their own ScheduledToken, withholds an incomplete match
// until the next chunk resolves it, and flushes the held-back tail when the inner
// stream drains. On a refuse it emits the refusal message, cancels the inner
// request, and stops — terminating THIS request's stream cleanly without
// touching any other. The forwarding goroutine cannot leak: every send selects
// on ctx (a client disconnect ends it, and the inner — sharing ctx — stops
// itself), and a refuse cancels the inner producer the same best-effort way the
// mux cancels on consumer abort.
func (m *policySchedulerModel) enforceScheduled(ctx context.Context, handle inference.RequestHandle, in <-chan inference.ScheduledToken) <-chan inference.ScheduledToken {
	out := make(chan inference.ScheduledToken, cap(in))
	go func() {
		defer close(out)
		enf := m.newEnforcer(ctx)
		last := inference.ScheduledToken{RequestID: handle.ID}
		for tok := range in {
			last = tok
			settled, events, stop := enf.Feed(tok.Token.Text)
			m.audit(events)
			if settled != "" && !m.sendScheduled(ctx, out, tok, settled) {
				return // ctx cancelled: the inner shares ctx and stops itself
			}
			if stop {
				// refuse: the message is already emitted; stop the inner producer
				// (mux-style best-effort — the scheduler's per-request send respects
				// this cancel), then end THIS request's stream by closing out.
				m.cancelInner(ctx, handle.ID)
				return
			}
		}
		// End of the inner stream: flush the Enforcer's held-back tail (a match
		// still wholly present is applied; leftover disputable bytes emit as-is).
		settled, events, _ := enf.Close()
		m.audit(events)
		if settled != "" {
			m.sendScheduled(ctx, out, last, settled)
		}
	}()
	return out
}

// sendScheduled delivers the enforced text as a ScheduledToken carrying the
// source token's request-local envelope (RequestID / Metrics / Labels) with the
// Token reduced to the settled Text — the ScheduledToken analogue of the Chat
// path emitting inference.Token{Text: out}. It selects on ctx so a consumer that
// has gone away (client disconnect) cannot block the fold goroutine; it reports
// false when ctx ended so the caller returns and closes the stream.
func (m *policySchedulerModel) sendScheduled(ctx context.Context, out chan<- inference.ScheduledToken, src inference.ScheduledToken, text string) bool {
	enforced := inference.ScheduledToken{
		RequestID: src.RequestID,
		Token:     inference.Token{Text: text},
		Metrics:   src.Metrics,
		Labels:    src.Labels,
	}
	select {
	case out <- enforced:
		return true
	case <-ctx.Done():
		return false
	}
}

// cancelInner asks the inner scheduler to stop the identified request — the
// scheduled analogue of the Chat path breaking its range to signal the model to
// stop generating (enforce in wrap.go). Best-effort: a scheduler that exposes
// CancelRequest (every serving/scheduler mode does) stops the in-flight request
// so its producer goroutine retires; a scheduler that does not simply runs the
// request to its natural end, exactly as the mux's own cancel-on-abort tolerates.
func (m *policySchedulerModel) cancelInner(ctx context.Context, id string) {
	if c, ok := m.inner.(inference.CancellableModel); ok {
		_, _ = c.CancelRequest(ctx, id)
	}
}
