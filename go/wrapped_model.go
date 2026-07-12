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
