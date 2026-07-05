// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	core "dappco.re/go"
	"dappco.re/go/inference/model/bundle"
)

// firstNonEmpty returns the first non-empty string after trimming whitespace.
//
//	value := firstNonEmpty(primary, fallback)
func firstNonEmpty(values ...string) string {
	for _, v := range values {
		if v != "" && core.Trim(v) != "" {
			return v
		}
	}
	return ""
}

// firstNonEmptyString is the legacy alias used through the agent_memory
// code path; behaves identically to firstNonEmpty.
//
//	value := firstNonEmptyString(a, b)
func firstNonEmptyString(values ...string) string {
	return firstNonEmpty(values...)
}

// stateHash returns the SHA-256 hex of value via the bundle package
// (canonical hashing helper for state-bundle metadata).
//
//	h := stateHash(value)
func stateHash(value string) string {
	return bundle.HashString(value)
}

// stateBundleTokenizer normalises a bundle.Tokenizer so missing hashes
// are filled. Forwards to bundle.NormaliseTokenizer; retained as a
// helper for the legacy agent index code path.
//
//	t := stateBundleTokenizer(t)
func stateBundleTokenizer(t bundle.Tokenizer) bundle.Tokenizer {
	return bundle.NormaliseTokenizer(t)
}

// cloneStringMap deep-copies a string-keyed string map.
//
//	cloned := cloneStringMap(src)
func cloneStringMap(src map[string]string) map[string]string {
	if len(src) == 0 {
		return nil
	}
	return core.MapClone(src)
}
