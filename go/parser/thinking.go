// SPDX-Licence-Identifier: EUPL-1.2

package parser

import (
	core "dappco.re/go"
)

//	result := parser.Filter(text, parser.Config{Mode: parser.Capture}, hint)
//	visible := result.Text
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

//	p := parser.NewProcessor(cfg, hint)
//	visible := p.Process(piece) + p.Flush()
type Processor struct {
	cfg            Config
	mode           Mode
	markers        []thinkingMarker
	startSet       []string // cached marker.start values — invariant once markers is set
	pending        string
	inReasoning    bool
	current        thinkingMarker
	reasoningParts []string
	blockParts     []string
	chunks         []Chunk
}

//	p := parser.NewProcessor(parser.Config{Mode: parser.Capture}, hint)
func NewProcessor(cfg Config, hint Hint) *Processor {
	markers := markersForHint(hint)
	startSet := make([]string, len(markers))
	for i, m := range markers {
		startSet[i] = m.start
	}
	return &Processor{
		cfg:      cfg,
		mode:     NormaliseMode(cfg.Mode),
		markers:  markers,
		startSet: startSet,
	}
}

//	mode := parser.NormaliseMode("")  // returns parser.Show
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
	p, ok := ForHint(hint).(*builtinOutputParser)
	if !ok || p == nil {
		p = newBuiltinOutputParser("generic", genericMarkers())
	}
	markers := make([]thinkingMarker, 0, len(p.markers))
	for _, m := range p.markers {
		for _, end := range m.ends {
			if m.start == "" || end == "" {
				continue
			}
			markers = append(markers, thinkingMarker{
				start:   m.start,
				end:     end,
				channel: m.kind,
				model:   p.ParserID(),
			})
		}
	}
	return markers
}

//	visible := p.Process(piece)
func (p *Processor) Process(text string) string {
	if p.mode == Show || text == "" {
		return text
	}
	p.pending += text
	return p.drain(false)
}

//	tail := p.Flush()
func (p *Processor) Flush() string {
	if p.mode == Show {
		return ""
	}
	out := p.drain(true)
	if p.pending == "" {
		if p.inReasoning {
			p.emitReasoningBlock()
			p.inReasoning = false
		}
		return out
	}
	if p.inReasoning {
		p.addReasoning(p.pending)
		p.pending = ""
		p.emitReasoningBlock()
		p.inReasoning = false
		return out
	}
	out += p.pending
	p.pending = ""
	return out
}

//	reasoning := p.Reasoning()
func (p *Processor) Reasoning() string {
	return core.Join("", p.reasoningParts...)
}

//	chunks := p.Chunks()
func (p *Processor) Chunks() []Chunk {
	if len(p.chunks) == 0 {
		return nil
	}
	return append([]Chunk(nil), p.chunks...)
}

func (p *Processor) drain(final bool) string {
	out := core.NewBuilder()
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
		if ok {
			out.WriteString(p.pending[:idx])
			p.pending = p.pending[idx+len(marker.start):]
			p.current = marker
			p.inReasoning = true
			continue
		}
		keep := 0
		if !final {
			keep = longestSuffixPrefix(p.pending, p.startSet)
		}
		consume := len(p.pending) - keep
		if consume > 0 {
			out.WriteString(p.pending[:consume])
			p.pending = p.pending[consume:]
		}
		break
	}
	return out.String()
}

func (p *Processor) findStart(text string) (int, thinkingMarker, bool) {
	best := -1
	var marker thinkingMarker
	for _, candidate := range p.markers {
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

func (p *Processor) addReasoning(text string) {
	if text == "" {
		return
	}
	p.reasoningParts = append(p.reasoningParts, text)
	p.blockParts = append(p.blockParts, text)
}

func (p *Processor) emitReasoningBlock() {
	text := core.Join("", p.blockParts...)
	p.blockParts = nil
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
		max := len(marker) - 1
		if max > len(text) {
			max = len(text)
		}
		for size := max; size > best; size-- {
			if core.HasPrefix(marker, text[len(text)-size:]) {
				best = size
				break
			}
		}
	}
	return best
}
