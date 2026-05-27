// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	core "dappco.re/go"
	"dappco.re/go/inference"
)

func parseReasoningText(text string, markers []reasoningMarker) inference.ReasoningParseResult {
	// Fuse first findReasoningStart with the short-circuit probe — if
	// it misses, return text verbatim with no builder alloc + no
	// .String() copy. The previous shape always built the builder +
	// wrote len(text) bytes + paid the .String() copy on every call;
	// per-response cost on every non-reasoning response.
	idx, marker, ok := findReasoningStart(text, markers)
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
	visible := core.NewBuilder()
	segments := []inference.ReasoningSegment{}
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
		idx, marker, ok = findReasoningStart(pending, markers)
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

func findReasoningStart(text string, markers []reasoningMarker) (int, reasoningMarker, bool) {
	best := -1
	var marker reasoningMarker
	for _, candidate := range markers {
		idx := indexString(text, candidate.start)
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best || idx == best && len(candidate.start) > len(marker.start) {
			best = idx
			marker = candidate
		}
	}
	return best, marker, best >= 0
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
