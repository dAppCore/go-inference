// SPDX-Licence-Identifier: EUPL-1.2

package policy

import (
	"context"
	"io"
	"iter"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving/provider/openai"
)

// WrapResolver decorates inner so every resolved model's OUTPUT passes the
// outbound policy before it reaches the client. It is composed OUTERMOST on the
// serve's resolver stack — after the welfare guard — so the policy enforces on
// the final tokens the deployment would otherwise emit. log receives one audit
// line per enforcement (rule index + action, never the matched content); a nil
// log silences the audit.
//
//	resolver = policy.WrapResolver(resolver, pol, os.Stderr)
//
// It wires no mediator, so a rewrite rule (grade G2) degrades to redact — use
// WrapResolverMediated for a policy that NeedsMediator.
func WrapResolver(inner openai.Resolver, pol *Policy, log io.Writer) openai.Resolver {
	return wrapResolver(inner, pol, log, nil)
}

// WrapResolverMediated is WrapResolver with a grade-G2 mediator wired in: a
// rewrite rule routes its matched span through mediate (see NewMediatingEnforcer)
// so the reply survives with the span transformed. Use it whenever the policy
// NeedsMediator; the serving layer refuses to boot a rewrite policy without one.
//
//	resolver = policy.WrapResolverMediated(resolver, pol, os.Stderr, mediate)
func WrapResolverMediated(inner openai.Resolver, pol *Policy, log io.Writer, mediate Mediator) openai.Resolver {
	return wrapResolver(inner, pol, log, mediate)
}

// wrapResolver is the shared decorator: it resolves via inner, then wraps the
// model so its Chat stream runs through the outbound policy, carrying mediate
// (nil for the redact/refuse-only path).
func wrapResolver(inner openai.Resolver, pol *Policy, log io.Writer, mediate Mediator) openai.Resolver {
	return openai.ResolverFunc(func(ctx context.Context, name string) (inference.TextModel, error) {
		model, err := inner.ResolveModel(ctx, name)
		if err != nil {
			return nil, err
		}
		return &policyTextModel{TextModel: model, pol: pol, log: log, mediate: mediate}, nil
	})
}

// policyTextModel decorates an inference.TextModel, enforcing the outbound
// policy over the Chat token stream. Every other TextModel method forwards
// unchanged via the embedded interface — Chat is the surface the compat mux
// serves for OpenAI, Anthropic, and Ollama alike, so guarding it guards them
// all.
type policyTextModel struct {
	inference.TextModel
	pol     *Policy
	log     io.Writer
	mediate Mediator // grade-G2 rewrite hook; nil for the redact/refuse-only path
}

// Chat streams the model's reply through a fresh policy Enforcer: matched spans
// are redacted or (grade G2) mediated, a refuse rule ends the reply with its
// configured message, and everything else streams through byte-for-byte.
func (m *policyTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.enforce(ctx, m.TextModel.Chat(ctx, messages, opts...))
}

// enforce wraps seq with a per-stream Enforcer. On a refuse it stops consuming
// the inner stream (breaking the range signals the model to stop generating);
// at end of stream it flushes the Enforcer's held-back tail. ctx is the parent
// for any grade-G2 mediator call over this stream.
func (m *policyTextModel) enforce(ctx context.Context, seq iter.Seq[inference.Token]) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		enf := m.newEnforcer(ctx)
		for tok := range seq {
			out, events, stop := enf.Feed(tok.Text)
			m.audit(events)
			if out != "" && !yield(inference.Token{Text: out}) {
				return
			}
			if stop {
				return
			}
		}
		out, events, _ := enf.Close()
		m.audit(events)
		if out != "" {
			yield(inference.Token{Text: out})
		}
	}
}

// newEnforcer builds the per-stream Enforcer, wiring the grade-G2 mediator when
// one is present so rewrite rules can mediate; otherwise a plain Enforcer (a
// rewrite would degrade to redact — the serving layer boots fatal instead).
func (m *policyTextModel) newEnforcer(ctx context.Context) *Enforcer {
	if m.mediate != nil {
		return m.pol.NewMediatingEnforcer(ctx, m.mediate)
	}
	return m.pol.NewEnforcer()
}

// audit writes one serve-log line per enforcement — the rule index, action, and
// (for a rewrite that fell back to redact) a degraded marker. The matched text is
// deliberately never logged: the deployment that configured the rule may consider
// the match itself sensitive, and the degraded flag is content-free — it says a
// rewrite degraded, never what matched.
func (m *policyTextModel) audit(events []Event) {
	if m.log == nil {
		return
	}
	for _, ev := range events {
		verb := "enforced"
		if ev.Degraded {
			verb = "degraded"
		}
		core.Print(m.log, "policy: rule #%d %s %s on output", ev.RuleIndex, ev.Action, verb)
	}
}

// AcceptsImages forwards the vision-capability gate to the wrapped model. The
// embedded interface does not widen this wrapper's method set, so without the
// explicit forward a wrapped vision checkpoint 400s at the serve gate.
func (m *policyTextModel) AcceptsImages() bool {
	v, ok := m.TextModel.(inference.VisionModel)
	return ok && v.AcceptsImages()
}

// AcceptsAudio forwards the audio-capability gate to the wrapped model — the
// audio twin of AcceptsImages.
func (m *policyTextModel) AcceptsAudio() bool {
	a, ok := m.TextModel.(inference.AudioModel)
	return ok && a.AcceptsAudio()
}

// Unwrap exposes the wrapped model so the serving layer can reach optional
// capabilities this guard does not itself re-expose (embeddings, rerank). The
// policy guard enforces the outbound text stream; an embedding call produces no
// text to police, so serving it through the base model is correct. Without
// this, a policy-wrapped embedder is stripped of its EmbeddingModel/RerankModel
// interface at the /v1/embeddings and /v1/rerank capability gate.
func (m *policyTextModel) Unwrap() inference.TextModel {
	return m.TextModel
}
