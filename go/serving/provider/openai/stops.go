// SPDX-Licence-Identifier: EUPL-1.2

// Stop-sequence truncation for generated chat content.
package openai

import core "dappco.re/go"

// TruncateAtStopSequence removes the first matching stop sequence and anything
// after it.
func TruncateAtStopSequence(content string, stops []string) string {
	cut, ok := firstStopSequenceCut(content, stops)
	if !ok {
		return content
	}
	return content[:cut]
}

func firstStopSequenceCut(content string, stops []string) (int, bool) {
	if content == "" || len(stops) == 0 {
		return 0, false
	}
	best := -1
	for _, stop := range stops {
		idx := indexString(content, stop)
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best {
			best = idx
		}
	}
	if best < 0 {
		return 0, false
	}
	return best, true
}

// indexString delegates to core.Index (strings.Index — Rabin-Karp +
// SIMD byte search). The earlier hand-rolled loop was O(N×M) per call
// and fired multiple times per chat-completion (stop-sequence cut +
// thinking-extractor per streaming chunk + channel-marker detection
// on every delta).
//
// Returns -1 on empty needle to preserve the caller contract — the
// stop-sequence + extractor paths treat empty as "no match" rather
// than the strings.Index "match at 0" semantics.
func indexString(s, needle string) int {
	if needle == "" {
		return -1
	}
	return core.Index(s, needle)
}
