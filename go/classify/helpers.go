// SPDX-Licence-Identifier: EUPL-1.2

package classify

import "dappco.re/go"

// failResult coerces a value — either an error or an already-failed core.Result —
// into a failed core.Result. Mirrors the helper used across the core packages.
func failResult(v any) core.Result {
	if r, ok := v.(core.Result); ok {
		if !r.OK {
			return r
		}
		if err, ok := r.Value.(error); ok {
			return core.Fail(err)
		}
		return core.Fail(core.NewError(r.Error()))
	}
	if err, ok := v.(error); ok {
		return core.Fail(err)
	}
	return core.Fail(core.NewError(core.Sprintf("%v", v)))
}

// isFrenchLanguage reports whether lang is French (fr or fr-*). Article prompts
// branch on this to offer the correct determiner set.
func isFrenchLanguage(lang string) bool {
	lang = core.Lower(lang)
	return lang == "fr" || core.HasPrefix(lang, "fr-")
}
