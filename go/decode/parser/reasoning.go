// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func parseReasoningText(text string, markers []reasoningMarker, leads string) inference.ReasoningParseResult {
	// Fuse first findReasoningStart with the short-circuit probe — if
	// it misses, return text verbatim with no builder alloc + no
	// .String() copy. The previous shape always built the builder +
	// wrote len(text) bytes + paid the .String() copy on every call;
	// per-response cost on every non-reasoning response.
	idx, marker, ok := findReasoningStart(text, markers, leads)
	if !ok {
		return inference.ReasoningParseResult{VisibleText: text}
	}
	// Probe the closing marker BEFORE allocating the builder. The
	// unclosed-first-marker case (model emitted `<think>...` then
	// streaming cut off, or the partial-flush hit before the close
	// tag landed) wants visible == text[:idx] — a direct slice into
	// the input — and a single reasoning segment for the open span.
	// The previous shape always allocated the builder + wrote
	// text[:idx] into it + paid String() to extract the same bytes;
	// the slice path drops two heap allocations on this hot edge.
	afterStart := text[idx+len(marker.start):]
	end, endSize := firstReasoningEnd(afterStart, marker.ends)
	if end < 0 {
		result := inference.ReasoningParseResult{VisibleText: text[:idx]}
		if reasoning := trimReasoningText(afterStart); reasoning != "" {
			result.Reasoning = []inference.ReasoningSegment{{Kind: marker.kind, Text: reasoning, StartToken: idx}}
		}
		return result
	}
	// Pre-grow the visible builder to the first span's visible bound:
	// text before the open marker (idx) plus everything after this
	// span's close marker (len(text) - idx - len(marker.start) - end -
	// endSize). For the dominant single-span shape that's exact; for
	// multi-span it's a tight lower-ish estimate that still collapses
	// the buffer-doubling cascade WriteString would otherwise pay
	// (memprofile attributed ~65% of allocated bytes to that doubling)
	// down to one backing-buffer alloc. A whole-len(text) grow would
	// over-allocate ~10x when the reasoning span dominates the stream.
	visible := core.NewBuilder()
	visible.Grow(len(text) - len(marker.start) - end - endSize)
	// Single span is the dominant shape (one `<think>…</think>` block
	// then content); pre-size segments to cap 1 so the common case takes
	// exactly one slice alloc rather than append's grow-from-zero.
	segments := make([]inference.ReasoningSegment, 0, 1)
	pending := text
	tokenOffset := 0
	for {
		visible.WriteString(pending[:idx])
		tokenOffset += idx
		reasoning := trimReasoningText(afterStart[:end])
		if reasoning != "" {
			segments = append(segments, inference.ReasoningSegment{Kind: marker.kind, Text: reasoning, StartToken: tokenOffset, EndToken: tokenOffset + end})
		}
		pending = afterStart[end+endSize:]
		tokenOffset += len(marker.start) + end + endSize
		if pending == "" {
			break
		}
		idx, marker, ok = findReasoningStart(pending, markers, leads)
		if !ok {
			visible.WriteString(pending)
			break
		}
		afterStart = pending[idx+len(marker.start):]
		end, endSize = firstReasoningEnd(afterStart, marker.ends)
		if end < 0 {
			visible.WriteString(pending[:idx])
			if reasoning := trimReasoningText(afterStart); reasoning != "" {
				segments = append(segments, inference.ReasoningSegment{Kind: marker.kind, Text: reasoning, StartToken: tokenOffset + idx})
			}
			break
		}
	}
	return inference.ReasoningParseResult{VisibleText: visible.String(), Reasoning: segments}
}

// findReasoningStart returns the earliest position at which any marker's start
// occurs in text (longest start winning a tie), or (-1, zero, false) if none
// does. leads is markerLeadBytes(markers' starts): a marker can only begin at
// one of those bytes, so IndexAny jumps straight to each candidate position and
// the per-marker HasPrefix check runs only there — replacing one full
// indexString scan of the whole text per marker (O(markers × text)) with
// O(text + hits × markers). Byte-identical to the min-index / longest-tie scan
// it replaces (the earliest start position equals the min over markers of first
// occurrence, and among starts at that position the longest wins both ways).
func findReasoningStart(text string, markers []reasoningMarker, leads string) (int, reasoningMarker, bool) {
	var marker reasoningMarker
	for i := 0; i < len(text); {
		rel := core.IndexAny(text[i:], leads)
		if rel < 0 {
			break
		}
		pos := i + rel
		best := -1
		for _, candidate := range markers {
			if core.HasPrefix(text[pos:], candidate.start) &&
				(best < 0 || len(candidate.start) > len(marker.start)) {
				best = pos
				marker = candidate
			}
		}
		if best >= 0 {
			return best, marker, true
		}
		i = pos + 1
	}
	return -1, marker, false
}

func firstReasoningEnd(text string, ends []string) (int, int) {
	best := -1
	bestSize := 0
	for _, end := range ends {
		idx := indexString(text, end)
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best {
			best = idx
			bestSize = len(end)
		}
	}
	return best, bestSize
}

func trimReasoningText(text string) string {
	return core.Trim(text)
}
