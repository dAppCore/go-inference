// SPDX-Licence-Identifier: EUPL-1.2

package inference

// WrappedModel is implemented by TextModel decorators — the serving welfare and
// policy guards — so the serving layer can reach optional capabilities the
// decorator does not itself re-expose. A decorator guards the text-output
// stream (Chat); capabilities like embeddings and rerank belong to the base
// model and produce no text to guard, so unwrapping to serve them is correct,
// not a bypass of the guard.
//
// This exists because a Go decorator that embeds the TextModel interface only
// carries TextModel's method set — an optional interface such as EmbeddingModel
// asserted against the wrapper fails even when the base model implements it
// (the capability-stripping bug class). BaseTextModel walks past the wrappers so
// the /v1/embeddings and /v1/rerank capability gate sees the real model.
//
//	if embedder, ok := inference.BaseTextModel(model).(inference.EmbeddingModel); ok { ... }
type WrappedModel interface {
	Unwrap() TextModel
}

// BaseTextModel unwraps WrappedModel decorators to the innermost TextModel;
// an undecorated model returns unchanged. The walk is bounded so a pathological
// self- or cycle-referential wrapper cannot spin — it returns the last model it
// reached rather than looping.
//
//	base := inference.BaseTextModel(resolved)
func BaseTextModel(model TextModel) TextModel {
	const maxUnwrapDepth = 16
	for depth := 0; depth < maxUnwrapDepth && model != nil; depth++ {
		wrapped, ok := model.(WrappedModel)
		if !ok {
			return model
		}
		inner := wrapped.Unwrap()
		if inner == nil || inner == model {
			return model
		}
		model = inner
	}
	return model
}

// As reports whether model — or a model reached by walking its Unwrap() chain —
// implements T, returning the first implementation found, outermost in first.
// It mirrors errors.As's semantics for the decorator chain: embedding the
// TextModel interface widens a struct's fields but never its method set, so an
// optional capability the wrapper does not explicitly re-declare is invisible to
// a direct type assertion against the wrapper, EVEN THOUGH the base model
// implements it (the capability-stripping bug class BaseTextModel was built to
// answer for embeddings/rerank — see the doc on WrappedModel).
//
// As generalises that answer to every optional capability probe (VisionModel,
// AudioModel, EmbeddingModel, RerankModel, SchedulerModel, CancellableModel,
// ToolParser, …), not only the ones a decorator's author remembered to forward
// by hand — so a NEW wrapper is safe by default even if it forwards nothing.
// Existing explicit forwards (welfareTextModel/policyTextModel/profileModel's
// AcceptsImages/AcceptsAudio) keep working unchanged: As finds them at the
// outer level on the first iteration, before it ever reaches for Unwrap.
//
//	if embedder, ok := inference.As[inference.EmbeddingModel](model); ok {
//	    result, err := embedder.Embed(ctx, req)
//	}
func As[T any](model TextModel) (T, bool) {
	const maxUnwrapDepth = 16
	for depth := 0; depth < maxUnwrapDepth && model != nil; depth++ {
		if typed, ok := model.(T); ok {
			return typed, true
		}
		wrapped, ok := model.(WrappedModel)
		if !ok {
			break
		}
		inner := wrapped.Unwrap()
		if inner == nil || inner == model {
			break
		}
		model = inner
	}
	var zero T
	return zero, false
}
