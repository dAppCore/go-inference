// SPDX-Licence-Identifier: EUPL-1.2

package policy

import (
	"context"
	"io"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// policyFakeScheduler is policyFakeModel plus the inference.SchedulerModel and
// inference.CancellableModel surfaces — the scheduler-route twin of the fake's
// Chat path. It mirrors the real scheduler's per-request discipline: each
// Schedule derives a per-request cancel context from the caller's ctx, the
// producer goroutine delivers tokens through a select that respects that ctx
// (so a cancel — via ctx or CancelRequest — stops it and closes the stream),
// and CancelRequest cancels the identified request. hang keeps the producer
// alive after its tokens drain (modelling an ongoing generation) so a refuse or
// a mid-stream cancel has something live to stop, proving the fold does not leak.
type policyFakeScheduler struct {
	policyFakeModel
	hang bool // after the tokens drain, block until the request's ctx is cancelled

	mu            sync.Mutex
	cancels       map[string]context.CancelFunc
	scheduleCalls int
	cancelledIDs  []string
	wg            sync.WaitGroup // tracks producer goroutines so a test can prove none leak
}

func (f *policyFakeScheduler) Schedule(ctx context.Context, req inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	f.mu.Lock()
	f.scheduleCalls++
	if f.cancels == nil {
		f.cancels = map[string]context.CancelFunc{}
	}
	rctx, cancel := context.WithCancel(ctx)
	id := req.ID
	f.cancels[id] = cancel
	f.mu.Unlock()

	out := make(chan inference.ScheduledToken) // unbuffered: exercises real backpressure
	f.wg.Add(1)
	go func() {
		defer f.wg.Done()
		defer close(out)
		defer cancel()
		for i, t := range f.tokens {
			select {
			case out <- inference.ScheduledToken{RequestID: id, Token: inference.Token{ID: int32(i + 1), Text: t}}:
			case <-rctx.Done():
				return
			}
		}
		if f.hang {
			<-rctx.Done()
		}
	}()
	return inference.RequestHandle{ID: id}, out, nil
}

func (f *policyFakeScheduler) CancelRequest(_ context.Context, id string) (inference.RequestCancelResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cancelledIDs = append(f.cancelledIDs, id)
	if c, ok := f.cancels[id]; ok {
		c()
		return inference.RequestCancelResult{ID: id, Cancelled: true}, nil
	}
	return inference.RequestCancelResult{ID: id}, nil
}

var (
	_ inference.SchedulerModel   = (*policyFakeScheduler)(nil)
	_ inference.CancellableModel = (*policyFakeScheduler)(nil)
)

// drainScheduleText resolves the wrapper's scheduler surface (inference.As MUST
// stop at the policy wrapper — the whole point of the fix) and drains one
// scheduled stream to its concatenated Text, plus the collected tokens so a test
// can inspect the per-token envelope.
func drainScheduleText(t *testing.T, m inference.TextModel, id string, msgs []inference.Message) (string, []inference.ScheduledToken) {
	t.Helper()
	sched, ok := inference.As[inference.SchedulerModel](m)
	if !ok {
		t.Fatal("policy-wrapped scheduler model does not present Schedule — inference.As unwrapped past the guard (the bypass this fix closes)")
	}
	_, stream, err := sched.Schedule(context.Background(), inference.ScheduledRequest{ID: id, Messages: msgs})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	var b core.Builder
	var toks []inference.ScheduledToken
	for tok := range stream {
		b.WriteString(tok.Token.Text)
		toks = append(toks, tok)
	}
	return b.String(), toks
}

// wrapSched builds the policy wrapper over a fake scheduler exactly as
// WrapResolver does at serve time, asserting it wrapped as the scheduler guard.
func wrapSched(t *testing.T, jsonSrc string, fake *policyFakeScheduler, log io.Writer) *policySchedulerModel {
	t.Helper()
	pol := mustCompile(t, jsonSrc)
	m, err := WrapResolver(resolverOf(fake), pol, log).ResolveModel(context.Background(), "any")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	ps, ok := m.(*policySchedulerModel)
	if !ok {
		t.Fatalf("a scheduler model must wrap as *policySchedulerModel, got %T", m)
	}
	return ps
}

// --- the seam: inference.As stops at the policy wrapper ----------------------

// TestPolicy_Schedule_NonScheduler_Unchanged_Good pins the other half: a model
// WITHOUT a scheduler stays the plain Chat-only policyTextModel, so the
// non-scheduler serve path (mux calls Chat) is byte-for-byte unchanged and the
// mux correctly falls through to Chat.
func TestPolicy_Schedule_NonScheduler_Unchanged_Good(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"SECRET","action":"redact"}]}`)
	m, err := WrapResolver(resolverOf(&policyFakeModel{}), pol, nil).ResolveModel(context.Background(), "any")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	if _, ok := m.(*policySchedulerModel); ok {
		t.Fatal("a non-scheduler model must NOT wrap as *policySchedulerModel")
	}
	if _, ok := inference.As[inference.SchedulerModel](m); ok {
		t.Fatal("a non-scheduler policy wrapper must not present Schedule (the mux then correctly falls to Chat)")
	}
}

// --- per-grade parity with the plain Chat path -------------------------------

// TestPolicy_Schedule_Passthrough_Good pins transparent pass-through: a clean
// scheduled stream is byte-identical to the inner scheduler's tokens (this is the
// policy-ON, no-match case; policy OFF builds no wrapper at all, see serve.go).
// It also pins that inference.As stops at the wrapper rather than the inner.
func TestPolicy_Schedule_Passthrough_Good(t *testing.T) {
	fake := &policyFakeScheduler{policyFakeModel: policyFakeModel{tokens: []string{"a perfectly ", "ordinary ", "answer here"}}}
	m := wrapSched(t, `{"rules":[{"match":"term","value":"SECRET","action":"redact"}]}`, fake, nil)
	got, _ := drainScheduleText(t, m, "r1", []inference.Message{{Role: "user", Content: "hi"}})
	if got != "a perfectly ordinary answer here" {
		t.Fatalf("clean scheduled reply = %q, want the inner tokens byte-identical", got)
	}
	if fake.scheduleCalls != 1 {
		t.Fatalf("inner Schedule calls = %d, want 1", fake.scheduleCalls)
	}
}

// TestPolicy_Schedule_Redact_Good pins redaction over the scheduler route with
// the matched span split across two ScheduledTokens — the same observable output
// the plain Chat path produces on the same input.
func TestPolicy_Schedule_Redact_Good(t *testing.T) {
	fake := &policyFakeScheduler{policyFakeModel: policyFakeModel{tokens: []string{"the PROJ", "ECT-X sh", "ips soon"}}}
	m := wrapSched(t, `{"rules":[{"match":"term","value":"PROJECT-X","action":"redact"}]}`, fake, nil)
	got, _ := drainScheduleText(t, m, "r1", nil)
	if got != "the [redacted] ships soon" {
		t.Fatalf("redacted scheduled reply = %q", got)
	}
}

// TestPolicy_Schedule_Refuse_Good pins that a refuse ends THIS request's stream
// with the configured message (the same refusal shape the plain path produces),
// and that the inner request is cancelled so its producer retires rather than
// leaking — proven by hang: the producer only ends when CancelRequest fires.
func TestPolicy_Schedule_Refuse_Good(t *testing.T) {
	fake := &policyFakeScheduler{
		policyFakeModel: policyFakeModel{tokens: []string{"here is the ", "SEC", "RET value and more"}},
		hang:            true,
	}
	m := wrapSched(t, `{"rules":[{"match":"term","value":"SECRET","action":"refuse","message":"Not in this deployment."}]}`, fake, nil)
	got, _ := drainScheduleText(t, m, "r1", nil)
	if got != "here is the Not in this deployment." {
		t.Fatalf("refused scheduled reply = %q, want the plain path's refusal shape", got)
	}
	// The refuse must have cancelled the inner request; wait for its producer to
	// retire (proves no goroutine leak on the wrapper-initiated stop).
	waitProducers(t, &fake.wg)
	fake.mu.Lock()
	cancelled := len(fake.cancelledIDs) == 1 && fake.cancelledIDs[0] == "r1"
	fake.mu.Unlock()
	if !cancelled {
		t.Fatalf("refuse did not cancel the inner request; cancelledIDs = %v", fake.cancelledIDs)
	}
}

// TestPolicy_Schedule_Rewrite_Good pins grade-G2 rewrite over the scheduler
// route: the mediator is invoked once with the COMPLETE span (split across
// tokens) and its transform replaces the span in-stream — same output as the
// plain mediated Chat path.
func TestPolicy_Schedule_Rewrite_Good(t *testing.T) {
	fake := &policyFakeScheduler{policyFakeModel: policyFakeModel{tokens: []string{"the PROJ", "ECT-X sh", "ips soon"}}}
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"PROJECT-X","action":"rewrite"}]}`)
	var calls int
	mediate := func(_ context.Context, _ int, span string) (string, error) {
		calls++
		if span != "PROJECT-X" {
			t.Fatalf("mediator got partial span %q, want the complete span", span)
		}
		return "our flagship", nil
	}
	m, err := WrapResolverMediated(resolverOf(fake), pol, nil, mediate).ResolveModel(context.Background(), "any")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	got, _ := drainScheduleText(t, m, "r1", nil)
	if got != "the our flagship ships soon" {
		t.Fatalf("mediated scheduled reply = %q", got)
	}
	if calls != 1 {
		t.Fatalf("mediator invoked %d times, want exactly 1", calls)
	}
}

// TestPolicy_Schedule_RewriteDegraded_Ugly pins the degraded rewrite over the
// scheduler route: an echoing mediator (its output still hits the rule) is
// re-enforced to redact and audited as degraded — the same fail-safe the plain
// path holds, never leaking the span.
func TestPolicy_Schedule_RewriteDegraded_Ugly(t *testing.T) {
	fake := &policyFakeScheduler{policyFakeModel: policyFakeModel{tokens: []string{"the PROJ", "ECT-X here"}}}
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"PROJECT-X","action":"rewrite"}]}`)
	echo := func(_ context.Context, _ int, span string) (string, error) { return span, nil }
	var log core.Builder
	m, err := WrapResolverMediated(resolverOf(fake), pol, &log, echo).ResolveModel(context.Background(), "any")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	got, _ := drainScheduleText(t, m, "r1", nil)
	if got != "the [redacted] here" {
		t.Fatalf("degraded scheduled reply = %q, want the residual term redacted", got)
	}
	audit := log.String()
	if !core.Contains(audit, "rule #0 rewrite degraded on output") {
		t.Fatalf("audit = %q, want a degraded rewrite line", audit)
	}
	if core.Contains(audit, "PROJECT-X") {
		t.Fatalf("audit leaked matched content: %q", audit)
	}
}

// TestPolicy_Schedule_HoldBackAcrossTokens_Good pins that the fold sees TEXT, not
// token edges: a term split so its final byte lands only in the LAST token is
// still caught — the enforcer withholds the incomplete prefix across the
// ScheduledToken boundary until the next token resolves it.
func TestPolicy_Schedule_HoldBackAcrossTokens_Good(t *testing.T) {
	// "SECRET" arrives one byte per token — every boundary bisects the term.
	fake := &policyFakeScheduler{policyFakeModel: policyFakeModel{tokens: []string{"S", "E", "C", "R", "E", "T", "!"}}}
	m := wrapSched(t, `{"rules":[{"match":"term","value":"SECRET","action":"redact"}]}`, fake, nil)
	got, _ := drainScheduleText(t, m, "r1", nil)
	if got != "[redacted]!" {
		t.Fatalf("split-term scheduled reply = %q, want the term redacted despite the token boundaries", got)
	}
}

// TestPolicy_Schedule_Audit_Good pins the audit receipt over the scheduler
// route: one line per enforcement, carrying the rule index + action and NEVER
// the matched content — identical to the plain path's audit.
func TestPolicy_Schedule_Audit_Good(t *testing.T) {
	fake := &policyFakeScheduler{policyFakeModel: policyFakeModel{tokens: []string{"the client and PROJECT-X"}}}
	var log core.Builder
	m := wrapSched(t, `{"rules":[
		{"match":"term","value":"client","action":"redact"},
		{"match":"term","value":"PROJECT-X","action":"redact"}
	]}`, fake, &log)
	drainScheduleText(t, m, "r1", nil)
	audit := log.String()
	if !core.Contains(audit, "rule #0 redact") || !core.Contains(audit, "rule #1 redact") {
		t.Fatalf("audit = %q, want both rule enforcements logged", audit)
	}
	if core.Contains(audit, "client") || core.Contains(audit, "PROJECT-X") {
		t.Fatalf("audit leaked matched content: %q", audit)
	}
}

// TestPolicy_Schedule_Cancel_Bad pins cancellation mid-stream: the consumer stops
// reading and the request's ctx is cancelled, so the fold goroutine ends without
// deadlock and the inner producer (sharing the ctx) retires — no goroutine leak.
// Run under -race, it also pins the fold's channel handshake is data-race free.
func TestPolicy_Schedule_Cancel_Bad(t *testing.T) {
	fake := &policyFakeScheduler{
		policyFakeModel: policyFakeModel{tokens: []string{"one ", "two ", "three ", "four ", "five "}},
		hang:            true,
	}
	m := wrapSched(t, `{"rules":[{"match":"term","value":"SECRET","action":"redact"}]}`, fake, nil)

	ctx, cancel := context.WithCancel(context.Background())
	_, stream, err := m.Schedule(ctx, inference.ScheduledRequest{ID: "r1"})
	if err != nil {
		t.Fatalf("Schedule: %v", err)
	}
	// Read one token, then cancel and stop reading — the fold must not block on
	// its next send and must not leave the inner producer stuck.
	<-stream
	cancel()

	// The fold goroutine must close the stream once ctx ends (it may deliver a
	// few already-in-flight tokens first); draining to close must terminate.
	done := make(chan struct{})
	go func() {
		defer close(done)
		for range stream {
		}
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("fold goroutine did not close the stream after cancel — deadlock/leak")
	}
	// The inner producer (sharing ctx) must retire too.
	waitProducers(t, &fake.wg)
}

// TestPolicy_Schedule_ScheduleError_Bad pins fail-through: an inner Schedule
// error propagates unwrapped, so the mux surfaces the real dispatch failure
// rather than a policy-wrapped empty stream.
func TestPolicy_Schedule_ScheduleError_Bad(t *testing.T) {
	inner := &erroringScheduler{}
	pol := mustCompile(t, `{"rules":[]}`)
	m := &policySchedulerModel{
		policyTextModel: &policyTextModel{TextModel: inner, pol: pol},
		inner:           inner,
	}
	if _, _, err := m.Schedule(context.Background(), inference.ScheduledRequest{ID: "r1"}); err == nil {
		t.Fatal("inner Schedule error must propagate")
	}
}

// erroringScheduler is a scheduler whose Schedule always fails.
type erroringScheduler struct{ policyFakeModel }

func (e *erroringScheduler) Schedule(context.Context, inference.ScheduledRequest) (inference.RequestHandle, <-chan inference.ScheduledToken, error) {
	return inference.RequestHandle{}, nil, core.NewError("scheduler dispatch failed")
}

var _ inference.SchedulerModel = (*erroringScheduler)(nil)

// waitProducers fails if the fake's producer goroutines have not all retired
// within a short window — the goroutine-leak assertion.
func waitProducers(t *testing.T, wg *sync.WaitGroup) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		wg.Wait()
	}()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("inner producer goroutine did not retire — leak")
	}
}
