// SPDX-Licence-Identifier: EUPL-1.2

// Reasoning-channel extraction: separating model-internal thinking from
// assistant content in the streamed token sequence.
package openai

import (
	"unicode"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/parser"
)

// The marker grammar — channel open/close, the gemma turn terminator, and the
// explicit paired reasoning spans — is owned by decode/parser (grammar.go) and
// consumed here, so a grammar fix lands once for both streaming engines. The
// local aliases keep this file's hot paths reading as before.
const (
	channelMarker = parser.ChannelOpenMarker
	// channelCloseMarker terminates a reasoning channel in Gemma4's output
	// (`<|channel>thought…<channel|>answer`). Unlike the gpt-oss style — where
	// the next `<|channel>` OPEN implicitly ends the prior channel — Gemma4
	// emits an explicit close, after which the remaining tokens are the
	// visible answer.
	channelCloseMarker = parser.ChannelCloseMarker
	// turnTerminator is gemma4's turn-end token — a PLAIN vocab token in MLX
	// snapshots, so its literal text reaches the extractor and the assistant
	// lane swallows it (the id still stops generation upstream).
	turnTerminator = parser.GemmaTurnTerminator
)

var reasoningMarkers = parser.PairedReasoningMarkers()

// reasoningMarkerStarts is the per-package cached list of marker starts
// passed to splitSafeSuffix from drain. Built once at package init so
// every per-token Process call shares the same slice header instead of
// re-allocating len(reasoningMarkers)+1 entries on every miss path.
var reasoningMarkerStarts = func() []string {
	out := make([]string, 0, len(reasoningMarkers)+2)
	out = append(out, channelMarker, turnTerminator)
	for _, marker := range reasoningMarkers {
		out = append(out, marker.Start)
	}
	return out
}()

// channelMarkers is the cached pair handed to splitSafeSuffix from the
// in-thought-channel drain branch: a partial OPEN (<|channel>) or CLOSE
// (<channel|>) straddling a token boundary must be held back, not
// mis-emitted as thinking. Built once so the per-token path shares the
// slice header instead of re-allocating on every miss.
var channelMarkers = []string{channelMarker, channelCloseMarker}

// ThinkingExtractor separates model-internal reasoning text from assistant
// content.
type ThinkingExtractor struct {
	pending        string
	content        string
	thinking       string
	inPaired       bool
	pairedEnd      string
	currentChannel string
}

func NewThinkingExtractor() *ThinkingExtractor {
	return &ThinkingExtractor{currentChannel: "assistant"}
}

func (e *ThinkingExtractor) Process(token inference.Token) (contentDelta, thoughtDelta string) {
	if e == nil {
		return "", ""
	}
	e.pending += token.Text
	return e.drain(false)
}

func (e *ThinkingExtractor) Flush() (contentDelta, thoughtDelta string) {
	if e == nil {
		return "", ""
	}
	contentDelta, thoughtDelta = e.drain(true)
	if e.pending == "" {
		return contentDelta, thoughtDelta
	}
	if e.inPaired || parser.IsReasoningChannel(e.currentChannel) {
		thoughtDelta += e.pending
		e.thinking += e.pending
	} else {
		contentDelta += e.pending
		e.content += e.pending
	}
	e.pending = ""
	e.inPaired = false
	return contentDelta, thoughtDelta
}

func (e *ThinkingExtractor) Content() string {
	if e == nil {
		return ""
	}
	return e.content
}

func (e *ThinkingExtractor) Thinking() string {
	if e == nil {
		return ""
	}
	return e.thinking
}

func (e *ThinkingExtractor) drain(final bool) (string, string) {
	// Lazy-allocate the deltas. Per-token streaming on plain (non-
	// reasoning) tokens only ever writes to contentDelta; the prior
	// shape paid for both builders up front on every Process call.
	var contentDelta, thoughtDelta *core.Builder
	for e.pending != "" {
		if e.inPaired {
			idx := indexString(e.pending, e.pairedEnd)
			if idx >= 0 {
				if idx > 0 {
					thoughtDelta = ensureBuilder(thoughtDelta)
					writeThought(e, thoughtDelta, e.pending[:idx])
				}
				e.pending = e.pending[idx+len(e.pairedEnd):]
				e.inPaired = false
				e.pairedEnd = ""
				continue
			}
			emit, keep := splitSafeSuffixOne(e.pending, e.pairedEnd, final)
			if emit != "" {
				thoughtDelta = ensureBuilder(thoughtDelta)
				writeThought(e, thoughtDelta, emit)
			}
			e.pending = keep
			if keep != "" && !final {
				break
			}
			continue
		}

		if ok := e.consumeMarkerAtStart(); ok {
			continue
		}

		if parser.IsReasoningChannel(e.currentChannel) {
			// A reasoning channel ends one of two ways: gpt-oss opens the
			// next channel (<|channel>name), Gemma4 emits an explicit close
			// (<channel|>). Honour whichever marker appears first.
			openIdx := indexString(e.pending, channelMarker)
			closeIdx := indexString(e.pending, channelCloseMarker)
			marker, idx := "", -1
			if closeIdx >= 0 && (openIdx < 0 || closeIdx < openIdx) {
				marker, idx = channelCloseMarker, closeIdx
			} else if openIdx >= 0 {
				marker, idx = channelMarker, openIdx
			}
			if idx >= 0 {
				if idx > 0 {
					thoughtDelta = ensureBuilder(thoughtDelta)
					writeThought(e, thoughtDelta, e.pending[:idx])
				}
				e.pending = e.pending[idx:]
				if marker == channelCloseMarker {
					// Gemma4 close: drop it and treat the rest as the answer.
					e.pending = e.pending[len(channelCloseMarker):]
					e.currentChannel = "assistant"
					continue
				}
				if e.consumeMarkerAtStart() {
					continue
				}
				if !final {
					break
				}
				thoughtDelta = ensureBuilder(thoughtDelta)
				writeThought(e, thoughtDelta, channelMarker)
				e.pending = e.pending[len(channelMarker):]
				continue
			}
			emit, keep := splitSafeSuffix(e.pending, channelMarkers, final)
			if emit != "" {
				thoughtDelta = ensureBuilder(thoughtDelta)
				writeThought(e, thoughtDelta, emit)
			}
			e.pending = keep
			if keep != "" && !final {
				break
			}
			continue
		}

		start, idx := earliestReasoningStart(e.pending)
		channelIdx := indexString(e.pending, channelMarker)
		if channelIdx >= 0 && (idx < 0 || channelIdx < idx) {
			idx = channelIdx
			start = channelMarker
		}
		if termIdx := indexString(e.pending, turnTerminator); termIdx >= 0 && (idx < 0 || termIdx < idx) {
			// Bare turn terminator in the assistant lane: emit the visible
			// prefix, swallow the terminator — its text is never content.
			if termIdx > 0 {
				contentDelta = ensureBuilder(contentDelta)
				writeContent(e, contentDelta, e.pending[:termIdx])
			}
			e.pending = e.pending[termIdx+len(turnTerminator):]
			continue
		}
		if idx >= 0 {
			if idx > 0 {
				contentDelta = ensureBuilder(contentDelta)
				writeContent(e, contentDelta, e.pending[:idx])
			}
			e.pending = e.pending[idx:]
			if start == channelMarker {
				if e.consumeMarkerAtStart() {
					continue
				}
				if !final {
					break
				}
				contentDelta = ensureBuilder(contentDelta)
				writeContent(e, contentDelta, channelMarker)
				e.pending = e.pending[len(channelMarker):]
				continue
			}
			e.inPaired = true
			e.pairedEnd = pairedEndFor(start)
			e.pending = e.pending[len(start):]
			continue
		}
		emit, keep := splitSafeSuffix(e.pending, markerStarts(), final)
		if emit != "" {
			contentDelta = ensureBuilder(contentDelta)
			writeContent(e, contentDelta, emit)
		}
		e.pending = keep
		if keep != "" && !final {
			break
		}
	}
	return builderString(contentDelta), builderString(thoughtDelta)
}

// ensureBuilder lazy-allocates a strings.Builder on first write. The
// drain hot loop's plain-token path emits everything via contentDelta;
// thoughtDelta only ever exists if a reasoning marker is in flight.
func ensureBuilder(b *core.Builder) *core.Builder {
	if b != nil {
		return b
	}
	return core.NewBuilder()
}

// builderString returns the builder contents or "" if the builder was
// never lazy-allocated (i.e. no writes to that channel this drain).
func builderString(b *core.Builder) string {
	if b == nil {
		return ""
	}
	return b.String()
}

// splitSafeSuffixOne is the single-marker fast path of splitSafeSuffix.
// Avoids the per-call []string{marker} slice alloc paid by the drain
// loop's per-token hot-path branches.
func splitSafeSuffixOne(s, marker string, final bool) (emit, keep string) {
	if final {
		return s, ""
	}
	maxN := min(len(s), len(marker)-1)
	keepLen := 0
	for n := 1; n <= maxN; n++ {
		if s[len(s)-n:] == marker[:n] && n > keepLen {
			keepLen = n
		}
	}
	if keepLen == 0 {
		return s, ""
	}
	return s[:len(s)-keepLen], s[len(s)-keepLen:]
}

func (e *ThinkingExtractor) consumeMarkerAtStart() bool {
	if !core.HasPrefix(e.pending, channelMarker) {
		for _, marker := range reasoningMarkers {
			if core.HasPrefix(e.pending, marker.Start) {
				e.inPaired = true
				e.pairedEnd = marker.End
				e.pending = e.pending[len(marker.Start):]
				return true
			}
		}
		return false
	}
	remaining := e.pending[len(channelMarker):]
	consumedSpace := 0
	for consumedSpace < len(remaining) {
		r, size := rune(remaining[consumedSpace]), 1
		if r >= 0x80 {
			r, size = utf8Rune(remaining[consumedSpace:])
		}
		if !unicode.IsSpace(r) {
			break
		}
		consumedSpace += size
	}
	nameLen := 0
	for consumedSpace+nameLen < len(remaining) {
		c := remaining[consumedSpace+nameLen]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			nameLen++
			continue
		}
		break
	}
	if nameLen == 0 {
		return false
	}
	e.currentChannel = core.Lower(remaining[consumedSpace : consumedSpace+nameLen])
	e.pending = remaining[consumedSpace+nameLen:]
	return true
}

func utf8Rune(s string) (rune, int) {
	for _, r := range s {
		return r, len(string(r))
	}
	return 0, 0
}

func writeContent(e *ThinkingExtractor, builder interface{ WriteString(string) (int, error) }, text string) {
	if text == "" {
		return
	}
	builder.WriteString(text)
	e.content += text
}

func writeThought(e *ThinkingExtractor, builder interface{ WriteString(string) (int, error) }, text string) {
	if text == "" {
		return
	}
	builder.WriteString(text)
	e.thinking += text
}

func earliestReasoningStart(s string) (string, int) {
	best := -1
	bestStart := ""
	for _, marker := range reasoningMarkers {
		idx := indexString(s, marker.Start)
		if idx < 0 {
			continue
		}
		if best < 0 || idx < best {
			best = idx
			bestStart = marker.Start
		}
	}
	return bestStart, best
}

func pairedEndFor(start string) string {
	for _, marker := range reasoningMarkers {
		if marker.Start == start {
			return marker.End
		}
	}
	return ""
}

// markerStarts returns the cached slice header — read-only after init.
// Sharing the header across calls avoids the per-token alloc that the
// previous shape paid on every miss path of drain.
func markerStarts() []string {
	return reasoningMarkerStarts
}

func splitSafeSuffix(s string, markers []string, final bool) (emit, keep string) {
	if final {
		return s, ""
	}
	keepLen := 0
	for _, marker := range markers {
		max := min(len(s), len(marker)-1)
		for n := 1; n <= max; n++ {
			if s[len(s)-n:] == marker[:n] && n > keepLen {
				keepLen = n
			}
		}
	}
	if keepLen == 0 {
		return s, ""
	}
	return s[:len(s)-keepLen], s[len(s)-keepLen:]
}
