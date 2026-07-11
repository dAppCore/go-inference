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
func WrapResolver(inner openai.Resolver, pol *Policy, log io.Writer) openai.Resolver {
	return openai.ResolverFunc(func(ctx context.Context, name string) (inference.TextModel, error) {
		model, err := inner.ResolveModel(ctx, name)
		if err != nil {
			return nil, err
		}
		return &policyTextModel{TextModel: model, pol: pol, log: log}, nil
	})
}

// policyTextModel decorates an inference.TextModel, enforcing the outbound
// policy over the Chat token stream. Every other TextModel method forwards
// unchanged via the embedded interface — Chat is the surface the compat mux
// serves for OpenAI, Anthropic, and Ollama alike, so guarding it guards them
// all.
type policyTextModel struct {
	inference.TextModel
	pol *Policy
	log io.Writer
}

// Chat streams the model's reply through a fresh policy Enforcer: matched spans
// are redacted, a refuse rule ends the reply with its configured message, and
// everything else streams through byte-for-byte.
func (m *policyTextModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.enforce(m.TextModel.Chat(ctx, messages, opts...))
}

// enforce wraps seq with a per-stream Enforcer. On a refuse it stops consuming
// the inner stream (breaking the range signals the model to stop generating);
// at end of stream it flushes the Enforcer's held-back tail.
func (m *policyTextModel) enforce(seq iter.Seq[inference.Token]) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		enf := m.pol.NewEnforcer()
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

// audit writes one serve-log line per enforcement — the rule index and action
// only. The matched text is deliberately never logged: the deployment that
// configured the rule may consider the match itself sensitive.
func (m *policyTextModel) audit(events []Event) {
	if m.log == nil {
		return
	}
	for _, ev := range events {
		core.Print(m.log, "policy: rule #%d %s enforced on output", ev.RuleIndex, ev.Action)
	}
}
