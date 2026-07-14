// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"context"

	tea "github.com/charmbracelet/bubbletea"

	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/parser"
)

// The generation goroutine streams events into a channel; waitEvent re-arms as
// a tea.Cmd after every message — the canonical Bubble Tea bridge from an
// iterator to the update loop.

type streamEvent struct {
	visible string // answer text delta (thinking stripped)
	thought string // reasoning text delta
	done    bool
	err     error
	metrics *inference.GenerateMetrics
}

type streamMsg streamEvent

// generation is one in-flight turn: its cancel func and event channel.
type generation struct {
	cancel context.CancelFunc
	events chan streamEvent
}

// startGeneration launches m.Chat on its own goroutine, splitting the raw
// stream through the family-aware reasoning parser (thinking → thought deltas,
// the rest → visible deltas). Fully greedy is NOT forced — the TUI serves the
// model's declared sampling defaults; thinkingOff opts the turn out of the
// gemma4 thinking default (#1847).
func startGeneration(model inference.TextModel, history []inference.Message, maxTokens int, thinkingOff bool) *generation {
	ctx, cancel := context.WithCancel(context.Background())
	g := &generation{cancel: cancel, events: make(chan streamEvent, 64)}

	proc := parser.NewProcessor(
		parser.Config{Mode: inference.ThinkingCapture, Capture: func(c inference.ThinkingChunk) {
			g.events <- streamEvent{thought: c.Text}
		}},
		parser.HintFromInference(model.Info()),
	)

	opts := []inference.GenerateOption{
		inference.WithMaxTokens(maxTokens),
		inference.WithMetricsSink(func(gm inference.GenerateMetrics) {
			m := gm
			g.events <- streamEvent{metrics: &m}
		}),
	}
	if thinkingOff {
		off := false
		opts = append(opts, inference.WithEnableThinking(&off))
	}

	go func() {
		defer close(g.events)
		for tok := range model.Chat(ctx, history, opts...) {
			if visible := proc.Process(tok.Text); visible != "" {
				g.events <- streamEvent{visible: visible}
			}
		}
		if tail := proc.Flush(); tail != "" {
			g.events <- streamEvent{visible: tail}
		}
		ev := streamEvent{done: true}
		if r := model.Err(); !r.OK {
			if err, ok := r.Value.(error); ok {
				ev.err = err
			}
		}
		g.events <- ev
	}()
	return g
}

// waitEvent yields the next stream event to the update loop; a closed channel
// surfaces as a final done message.
func waitEvent(g *generation) tea.Cmd {
	return func() tea.Msg {
		ev, ok := <-g.events
		if !ok {
			return streamMsg{done: true}
		}
		return streamMsg(ev)
	}
}
