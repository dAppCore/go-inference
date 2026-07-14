// SPDX-Licence-Identifier: EUPL-1.2

// serial_model.go serialises the generation lane of a single-session base model.
// Every scheduler mode drives requests in parallel — interleave runs each
// admitted request's source on its own goroutine, batch runs a lockstep lane
// set, and serial mode itself can run MaxConcurrent workers — so a base model
// that declares inference.SerialModel (its KV cache and drafter command buffer
// are singletons: the MTP speculative pair) would have its shared GPU scratch
// raced by two concurrent Generate/Chat calls and crash (the loadKV nil
// drafter-KV SIGSEGV under -scheduler interleave, #1842). The scheduler wraps
// such a model once at New so exactly one generation runs at a time; the rest
// queue on the gate.
package scheduler

import (
	"context"
	"iter"

	"dappco.re/go/inference"
)

// serialModel gates a single-session base model so only one Generate/Chat
// iteration runs at a time. It is an inference.WrappedModel: embedding the
// TextModel interface forwards every non-overridden method to the base, and
// Unwrap lets inference.As reach the base model's other optional capabilities
// (SpeculativeMetricsProvider, the CB renderer probes, …) through the wrapper —
// the capability-stripping answer WrappedModel exists for.
type serialModel struct {
	inference.TextModel
	// gate is a capacity-1 semaphore: the single in-flight generation slot. A
	// generation holds it for its whole token iteration (acquired when the
	// caller starts pulling, released when the iterator drains or the caller
	// stops early), so concurrent lanes serialise instead of racing the base
	// model's shared session.
	gate chan struct{}
}

var _ inference.WrappedModel = (*serialModel)(nil)

// newSerialModel wraps base so its generation lane runs one request at a time.
func newSerialModel(base inference.TextModel) *serialModel {
	return &serialModel{TextModel: base, gate: make(chan struct{}, 1)}
}

// Unwrap exposes the base model to inference.As / inference.BaseTextModel so a
// capability the wrapper does not itself re-declare stays reachable.
func (m *serialModel) Unwrap() inference.TextModel { return m.TextModel }

// Generate serialises the base model's raw-prompt generation lane.
func (m *serialModel) Generate(ctx context.Context, prompt string, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.serialise(ctx, func() iter.Seq[inference.Token] {
		return m.TextModel.Generate(ctx, prompt, opts...)
	})
}

// Chat serialises the base model's chat generation lane.
func (m *serialModel) Chat(ctx context.Context, messages []inference.Message, opts ...inference.GenerateOption) iter.Seq[inference.Token] {
	return m.serialise(ctx, func() iter.Seq[inference.Token] {
		return m.TextModel.Chat(ctx, messages, opts...)
	})
}

// serialise returns an iterator that holds the single generation slot for the
// whole run of gen's tokens. The slot is taken when the caller begins pulling
// and released when the iterator drains or the caller stops early (defer). A
// caller whose ctx is cancelled while queued yields nothing rather than waiting
// out the in-flight generation, so cancellation stays responsive under load.
func (m *serialModel) serialise(ctx context.Context, gen func() iter.Seq[inference.Token]) iter.Seq[inference.Token] {
	return func(yield func(inference.Token) bool) {
		if ctx == nil {
			ctx = context.Background()
		}
		select {
		case m.gate <- struct{}{}:
			defer func() { <-m.gate }()
		case <-ctx.Done():
			return
		}
		for tok := range gen() {
			if !yield(tok) {
				return
			}
		}
	}
}
