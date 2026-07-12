// SPDX-Licence-Identifier: EUPL-1.2

package inference

// PromptTokenizer is the optional model capability that exposes the model's own
// text→token encoding. A [SessionHandle.Prefill] tokenises its prompt
// INTERNALLY, so nothing outside the engine can compute the token-prefix key two
// conversations would share; this capability is the honest seam that lets the
// serving layer compute that key without a second tokeniser of its own (which
// would drift from the engine's).
//
// It is probed exactly like every other optional capability — and, because a
// serving decorator (welfare / policy guard) embeds [TextModel] and so widens
// its fields but never its method set, it MUST be reached with [As] rather than
// a direct type assertion, or a wrapped model hides the base model's tokeniser
// (the capability-stripping bug class [As] exists to answer):
//
//	if tk, ok := inference.As[inference.PromptTokenizer](model); ok {
//	    tokens, err := tk.Tokenize("You are a helpful assistant.")
//	    _ = tokens
//	    _ = err
//	}
//
// Tokenize returns the model-native token IDs for text — the same IDs a
// Prefill of the same text would load — or an error when the model carries no
// usable tokeniser.
type PromptTokenizer interface {
	Tokenize(text string) ([]int32, error)
}
