// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	"strings"

	core "dappco.re/go"
)

// result := parser.Filter(text, parser.Config{Mode: parser.Capture}, hint)
// visible := result.Text
func Filter(text string, cfg Config, hint Hint) Result {
	processor := NewProcessor(cfg, hint)
	builder := core.NewBuilder()
	builder.WriteString(processor.Process(text))
	builder.WriteString(processor.Flush())
	return Result{
		Text:      builder.String(),
		Reasoning: processor.Reasoning(),
		Chunks:    processor.Chunks(),
	}
}

// p := parser.NewProcessor(cfg, hint)
// visible := p.Process(piece) + p.Flush()
type Processor struct {
	cfg             Config
	mode            Mode
	markers         []thinkingMarker
	startSet        []string // cached marker.start values — invariant once markers is set
	startLeads      string   // distinct lead bytes of startSet — the findStart anchor set
	terminators     []string // bare turn-end tokens swallowed from visible output (gemma <end_of_turn>)
	terminatorLeads string   // distinct lead bytes of terminators — the findTerminator anchor set
	holdbackSet     []string // startSet + terminators — the streaming partial-suffix set
	pending         string
	inReasoning     bool
	current         thinkingMarker
	reasoning       strings.Builder // reasoning text, folded per token as it arrives
	// blockStart marks where the current reasoning block begins as a BYTE
	// offset into the reasoning builder. The block's text is
	// reasoning.String()[blockStart:] — emitReasoningBlock takes that window
	// as a zero-copy view and advances blockStart to the builder's new
	// length. The previous shape kept reasoning as a []string appended per
	// token then joined; folding into a builder trades the per-token slice
	// headers (16 B each, more than the short reasoning fragments they point
	// at) for a compact byte buffer, and turns Reasoning() into a zero-copy
	// String() instead of a full re-join. The block window stays byte-stable
	// because a builder only appends — earlier views keep their length and
	// the old backing array if a later write reallocs.
	blockStart int
	chunks     []Chunk
}

// p := parser.NewProcessor(parser.Config{Mode: parser.Capture}, hint)
func NewProcessor(cfg Config, hint Hint) *Processor {
	// markersForHint + thinkingStartsForHint return cached views
	// owned by the registry's builtinOutputParser. They are read-only
	// after construction; sharing the headers avoids per-stream alloc
	// of both the marker slice and the start-set slice (the previous
	// shape paid both per NewProcessor call).
	markers, startSet, terminators, holdback, startLeads, terminatorLeads := markersStartsTerminatorsForHint(hint)
	return &Processor{
		cfg:             cfg,
		mode:            NormaliseMode(cfg.Mode),
		markers:         markers,
		startSet:        startSet,
		startLeads:      startLeads,
		terminators:     terminators,
		terminatorLeads: terminatorLeads,
		holdbackSet:     holdback,
	}
}

// mode := parser.NormaliseMode("")  // returns parser.Show
func NormaliseMode(mode Mode) Mode {
	switch mode {
	case "", Show:
		return Show
	case Hide, Capture:
		return mode
	default:
		return Show
	}
}

func markersForHint(hint Hint) []thinkingMarker {
	markers, _ := markersAndStartsForHint(hint)
	return markers
}

// markersAndStartsForHint returns the flattened thinkingMarker view and
// the parallel start-set view for the resolved parser. Both slices are
// owned by the parser instance held in the registry — callers must treat
// them as read-only. Non-builtin parsers (custom registrations) fall back
// to allocating fresh views, preserving the legacy shape for those paths.
func markersAndStartsForHint(hint Hint) ([]thinkingMarker, []string) {
	markers, starts, _, _, _, _ := markersStartsTerminatorsForHint(hint)
	return markers, starts
}

// markersStartsTerminatorsForHint is markersAndStartsForHint plus the family's
// bare turn terminators and the combined streaming holdback set (all four are
// registry-owned read-only views).
func markersStartsTerminatorsForHint(hint Hint) ([]thinkingMarker, []string, []string, []string, string, string) {
	p, ok := ForHint(hint).(*builtinOutputParser)
	if !ok || p == nil {
		p = newBuiltinOutputParser("generic", genericMarkers())
	}
	return p.thinkingMarkers, p.thinkingStarts, p.terminators, p.thinkingHoldback, p.startLeads, p.terminatorLeads
}

// visible := p.Process(piece)
func (p *Processor) Process(text string) string {
	if p.mode == Show || text == "" {
		return text
	}
	p.pending += text
	return p.drain(false)
}

// tail := p.Flush()
func (p *Processor) Flush() string {
	if p.mode == Show {
		return ""
	}
	out := p.drain(true)
	// drain(final=true) consumes pending completely on every path — the keep
	// holdback that retains a partial marker only arms mid-stream (!final) —
	// so there is never a residue to splice here; only an open reasoning
	// block remains to close. TestProcessorDrainFinalConsumesPending pins
	// the postcondition so a drain change cannot silently reopen the gap.
	if p.inReasoning {
		p.emitReasoningBlock()
		p.inReasoning = false
	}
	return out
}

// reasoning := p.Reasoning()
func (p *Processor) Reasoning() string {
	return p.reasoning.String()
}

// chunks := p.Chunks()
func (p *Processor) Chunks() []Chunk {
	if len(p.chunks) == 0 {
		return nil
	}
	return append([]Chunk(nil), p.chunks...)
}

func (p *Processor) drain(final bool) string {
	if p.pending == "" {
		return ""
	}
	// Lazy-init the builder. Per-token streaming hits drain on every
	// token; the common no-marker path writes a single slice that can
	// be returned directly without ever touching a builder. The builder
	// only allocates when we cross a marker boundary mid-string and
	// need to splice a visible prefix with a suffix later in the loop.
	var out *strings.Builder
	for p.pending != "" {
		if p.inReasoning {
			idx := indexString(p.pending, p.current.end)
			if idx >= 0 {
				p.addReasoning(p.pending[:idx])
				p.pending = p.pending[idx+len(p.current.end):]
				p.emitReasoningBlock()
				p.inReasoning = false
				continue
			}
			keep := 0
			if !final {
				keep = longestSuffixPrefix(p.pending, []string{p.current.end})
			}
			consume := len(p.pending) - keep
			if consume > 0 {
				p.addReasoning(p.pending[:consume])
				p.pending = p.pending[consume:]
			}
			break
		}

		idx, marker, ok := p.findStart(p.pending)
		tidx, tlen := p.findTerminator(p.pending)
		if tlen > 0 && (!ok || tidx < idx) {
			// A bare turn terminator outside any span: emit the visible prefix,
			// swallow the terminator itself — its text is never content.
			if tidx > 0 {
				if out == nil {
					out = core.NewBuilder()
				}
				out.WriteString(p.pending[:tidx])
			}
			p.pending = p.pending[tidx+tlen:]
			continue
		}
		if ok {
			if idx > 0 {
				if out == nil {
					out = core.NewBuilder()
				}
				out.WriteString(p.pending[:idx])
			}
			p.pending = p.pending[idx+len(marker.start):]
			p.current = marker
			p.inReasoning = true
			continue
		}
		keep := 0
		if !final {
			keep = longestSuffixPrefix(p.pending, p.holdbackSet)
		}
		consume := len(p.pending) - keep
		if consume == 0 {
			break
		}
		if out == nil {
			// Single-write path — return the slice directly without
			// paying for a builder alloc. This is the streaming hot
			// path: per-token Process call, no marker in pending,
			// consume the visible bytes and return.
			output := p.pending[:consume]
			p.pending = p.pending[consume:]
			return output
		}
		out.WriteString(p.pending[:consume])
		p.pending = p.pending[consume:]
		break
	}
	if out == nil {
		return ""
	}
	return out.String()
}

// findTerminator returns the earliest bare turn-terminator occurrence in text
// as (index, length), or (-1, 0) when none of the family's terminators appear.
// terminatorLeads (their distinct lead bytes) anchors the scan: IndexAny hops to
// each candidate position and the HasPrefix check runs only there, replacing one
// full indexString per terminator. On a tie the first terminator in iteration
// order wins its length — identical to the prior strict-less min-index scan
// (which also never overrode an equal index).
func (p *Processor) findTerminator(text string) (int, int) {
	for i := 0; i < len(text); {
		rel := core.IndexAny(text[i:], p.terminatorLeads)
		if rel < 0 {
			break
		}
		pos := i + rel
		for _, term := range p.terminators {
			if core.HasPrefix(text[pos:], term) {
				return pos, len(term)
			}
		}
		i = pos + 1
	}
	return -1, 0
}

// findStart returns the earliest position at which any thinking marker's start
// occurs in text (longest start winning a tie), or (-1, zero, false) if none
// does. startLeads (the markers' distinct lead bytes) anchors the scan so a full
// indexString runs once per lead-byte hit rather than once per marker over the
// whole text — the per-token streaming cost drops from O(markers × pending) to
// O(pending + hits × markers). Byte-identical to the min-index / longest-tie
// scan it replaces.
func (p *Processor) findStart(text string) (int, thinkingMarker, bool) {
	var marker thinkingMarker
	for i := 0; i < len(text); {
		rel := core.IndexAny(text[i:], p.startLeads)
		if rel < 0 {
			break
		}
		pos := i + rel
		best := -1
		for _, candidate := range p.markers {
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

func (p *Processor) addReasoning(text string) {
	if text == "" {
		return
	}
	p.reasoning.WriteString(text)
}

func (p *Processor) emitReasoningBlock() {
	text := p.reasoning.String()[p.blockStart:]
	p.blockStart = p.reasoning.Len()
	if text == "" {
		return
	}
	chunk := Chunk{
		Text:    text,
		Channel: p.current.channel,
		Model:   p.current.model,
	}
	p.chunks = append(p.chunks, chunk)
	if p.mode == Capture && p.cfg.Capture != nil {
		p.cfg.Capture(chunk)
	}
}

func longestSuffixPrefix(text string, markers []string) int {
	best := 0
	for _, marker := range markers {
		max := min(len(marker)-1, len(text))
		for size := max; size > best; size-- {
			if core.HasPrefix(marker, text[len(text)-size:]) {
				best = size
				break
			}
		}
	}
	return best
}
