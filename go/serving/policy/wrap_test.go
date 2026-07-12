// SPDX-Licence-Identifier: EUPL-1.2

package policy

import (
	"context"
	"iter"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving/provider/openai"
)

// policyFakeModel is an inference.TextModel double whose Chat streams a fixed
// slice of token texts — the caller controls the exact chunk boundaries so a
// match can be forced to span them.
type policyFakeModel struct{ tokens []string }

func fakeSeq(texts ...string) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		for i, t := range texts {
			if !yield(inference.Token{ID: int32(i + 1), Text: t}) {
				return
			}
		}
	}
}

func (f *policyFakeModel) Chat(context.Context, []inference.Message, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return fakeSeq(f.tokens...)
}
func (f *policyFakeModel) Generate(context.Context, string, ...inference.GenerateOption) iter.Seq[inference.Token] {
	return fakeSeq(f.tokens...)
}
func (f *policyFakeModel) Classify(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok(nil)
}
func (f *policyFakeModel) BatchGenerate(context.Context, []string, ...inference.GenerateOption) core.Result {
	return core.Ok(nil)
}
func (f *policyFakeModel) ModelType() string                  { return "policy-fake" }
func (f *policyFakeModel) Info() inference.ModelInfo          { return inference.ModelInfo{} }
func (f *policyFakeModel) Metrics() inference.GenerateMetrics { return inference.GenerateMetrics{} }
func (f *policyFakeModel) Err() core.Result                   { return core.Ok(nil) }
func (f *policyFakeModel) Close() core.Result                 { return core.Ok(nil) }

var _ inference.TextModel = (*policyFakeModel)(nil)

// resolverOf returns a resolver that always yields model.
func resolverOf(model inference.TextModel) openai.Resolver {
	return openai.ResolverFunc(func(context.Context, string) (inference.TextModel, error) {
		return model, nil
	})
}

// drain resolves and streams a Chat reply through the wrapped resolver.
func drain(t *testing.T, r openai.Resolver) string {
	t.Helper()
	model, err := r.ResolveModel(context.Background(), "any")
	if err != nil {
		t.Fatalf("resolve: %v", err)
	}
	var b core.Builder
	for tok := range model.Chat(context.Background(), []inference.Message{{Role: "user", Content: "hi"}}) {
		b.WriteString(tok.Text)
	}
	return b.String()
}

// TestPolicy_WrapResolver_Redact_Good pins redaction through the full wrapper —
// tokens arrive split across the matched span and are still redacted.
func TestPolicy_WrapResolver_Redact_Good(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"PROJECT-X","action":"redact"}]}`)
	fake := &policyFakeModel{tokens: []string{"the PROJ", "ECT-X sh", "ips soon"}}
	got := drain(t, WrapResolver(resolverOf(fake), pol, nil))
	if got != "the [redacted] ships soon" {
		t.Fatalf("wrapped reply = %q", got)
	}
}

// TestPolicy_WrapResolver_Refuse_Good pins that a refuse ends the reply at the
// match with the configured message.
func TestPolicy_WrapResolver_Refuse_Good(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"SECRET","action":"refuse","message":"Not in this deployment."}]}`)
	fake := &policyFakeModel{tokens: []string{"here is the ", "SEC", "RET value and more"}}
	got := drain(t, WrapResolver(resolverOf(fake), pol, nil))
	if got != "here is the Not in this deployment." {
		t.Fatalf("wrapped refuse reply = %q", got)
	}
}

// TestPolicy_WrapResolver_Passthrough pins byte-exactness through the wrapper on
// a clean stream: the reply is the model's tokens concatenated, unchanged.
func TestPolicy_WrapResolver_Passthrough(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"SECRET","action":"redact"}]}`)
	fake := &policyFakeModel{tokens: []string{"a perfectly ", "ordinary ", "answer here"}}
	got := drain(t, WrapResolver(resolverOf(fake), pol, nil))
	if got != "a perfectly ordinary answer here" {
		t.Fatalf("clean wrapped reply = %q", got)
	}
}

// TestPolicy_WrapResolver_ResolveError pins that an inner resolve failure
// propagates unwrapped — the policy layer never masks a load error.
func TestPolicy_WrapResolver_ResolveError(t *testing.T) {
	pol := mustCompile(t, `{"rules":[]}`)
	inner := openai.ResolverFunc(func(context.Context, string) (inference.TextModel, error) {
		return nil, core.E("test", "model not found", nil)
	})
	_, err := WrapResolver(inner, pol, nil).ResolveModel(context.Background(), "missing")
	if err == nil || !core.Contains(err.Error(), "model not found") {
		t.Fatalf("resolve error = %v, want the inner failure to propagate", err)
	}
}

// TestPolicy_WrapResolver_Audit pins the audit line: one entry per enforcement,
// carrying the rule index + action and NEVER the matched content.
func TestPolicy_WrapResolver_Audit(t *testing.T) {
	pol := mustCompile(t, `{"rules":[
		{"match":"term","value":"client","action":"redact"},
		{"match":"term","value":"PROJECT-X","action":"redact"}
	]}`)
	fake := &policyFakeModel{tokens: []string{"the client and PROJECT-X"}}
	var log core.Builder
	drain(t, WrapResolver(resolverOf(fake), pol, &log))
	audit := log.String()
	if !core.Contains(audit, "rule #0 redact") || !core.Contains(audit, "rule #1 redact") {
		t.Fatalf("audit = %q, want both rule enforcements logged", audit)
	}
	if core.Contains(audit, "client") || core.Contains(audit, "PROJECT-X") {
		t.Fatalf("audit leaked matched content: %q", audit)
	}
}

// TestPolicy_WrapResolverMediated_Rewrite_Good pins grade-G2 rewrite through the
// full wrapper — tokens arrive split across the span and the mediator's transform
// replaces the whole span.
func TestPolicy_WrapResolverMediated_Rewrite_Good(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"PROJECT-X","action":"rewrite"}]}`)
	fake := &policyFakeModel{tokens: []string{"the PROJ", "ECT-X sh", "ips soon"}}
	mediate := func(_ context.Context, _ int, _ string) (string, error) {
		return "our flagship", nil
	}
	got := drain(t, WrapResolverMediated(resolverOf(fake), pol, nil, mediate))
	if got != "the our flagship ships soon" {
		t.Fatalf("mediated wrapped reply = %q", got)
	}
}

// TestPolicy_WrapResolverMediated_MediatorError pins the wrapper's fail-safe: a
// mediator error degrades to redact — the reply survives, the span never leaks.
func TestPolicy_WrapResolverMediated_MediatorError(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"SECRET","action":"rewrite","replacement":"[gone]"}]}`)
	fake := &policyFakeModel{tokens: []string{"the SEC", "RET here"}}
	boom := func(context.Context, int, string) (string, error) {
		return "SECRET", core.E("test", "mediator down", nil)
	}
	got := drain(t, WrapResolverMediated(resolverOf(fake), pol, nil, boom))
	if got != "the [gone] here" {
		t.Fatalf("degraded wrapped reply = %q, want the redact fallback", got)
	}
}

// TestPolicy_WrapResolverMediated_AuditDegraded pins that a rewrite which had to
// degrade (here an echoing mediator whose output is redacted by re-enforcement)
// is audited as "degraded", never "enforced", and still never leaks the matched
// content through the audit line.
func TestPolicy_WrapResolverMediated_AuditDegraded(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"PROJECT-X","action":"rewrite"}]}`)
	fake := &policyFakeModel{tokens: []string{"the PROJ", "ECT-X here"}}
	echo := func(_ context.Context, _ int, span string) (string, error) { return span, nil }
	var log core.Builder
	got := drain(t, WrapResolverMediated(resolverOf(fake), pol, &log, echo))
	if got != "the [redacted] here" {
		t.Fatalf("degraded wrapped reply = %q, want the residual term redacted", got)
	}
	audit := log.String()
	if !core.Contains(audit, "rule #0 rewrite degraded on output") {
		t.Fatalf("audit = %q, want a degraded rewrite line", audit)
	}
	if core.Contains(audit, "PROJECT-X") {
		t.Fatalf("audit leaked matched content: %q", audit)
	}
}

// TestPolicy_WrapResolver_RewriteDegrades pins that the G1 wrapper (no mediator)
// on a rewrite policy degrades every rewrite to redact rather than leaking — the
// serving layer boots fatal to avoid this in production.
func TestPolicy_WrapResolver_RewriteDegrades(t *testing.T) {
	pol := mustCompile(t, `{"rules":[{"match":"term","value":"SECRET","action":"rewrite","replacement":"[x]"}]}`)
	fake := &policyFakeModel{tokens: []string{"the SEC", "RET here"}}
	got := drain(t, WrapResolver(resolverOf(fake), pol, nil))
	if got != "the [x] here" {
		t.Fatalf("no-mediator wrapped reply = %q, want the redact fallback", got)
	}
}

// policyCapableFake is a TextModel carrying both media capabilities, so the
// wrapper's forwarding of the serve gates is observable.
type policyCapableFake struct{ inference.TextModel }

func (policyCapableFake) AcceptsImages() bool { return true }
func (policyCapableFake) AcceptsAudio() bool  { return true }

// TestPolicyTextModel_ForwardsCapabilityGates_Good: the policy wrap must not
// hide the wrapped checkpoint's media capabilities — the serve handler gates
// input_audio/image_url on these assertions.
func TestPolicyTextModel_ForwardsCapabilityGates_Good(t *testing.T) {
	inner := policyCapableFake{}
	wrapped := inference.TextModel(&policyTextModel{TextModel: inner})
	v, ok := wrapped.(inference.VisionModel)
	if !ok || !v.AcceptsImages() {
		t.Fatalf("policy wrap hides AcceptsImages (ok=%v) — image serve gate 400s", ok)
	}
	a, ok := wrapped.(inference.AudioModel)
	if !ok || !a.AcceptsAudio() {
		t.Fatalf("policy wrap hides AcceptsAudio (ok=%v) — audio serve gate 400s", ok)
	}
}
