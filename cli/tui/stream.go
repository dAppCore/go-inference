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
	SessionID string
	JobID     string
	visible   string // answer text delta (thinking stripped)
	thought   string // reasoning text delta
	done      bool
	err       error
	metrics   *inference.GenerateMetrics
}

type streamMsg streamEvent

// streamGeneration runs m.Chat, splitting the raw stream through the
// family-aware reasoning parser (thinking → thought deltas, the rest →
// visible deltas). The caller owns the context and registration lifecycle.
func streamGeneration(ctx context.Context, g *generation, model inference.TextModel, history []inference.Message, genOpts []inference.GenerateOption, finish func()) {
	var requestErr error
	errorSinkCalled := false
	tag := func(event streamEvent) streamEvent {
		event.SessionID = g.SessionID
		event.JobID = g.JobID
		return event
	}
	emit := func(event streamEvent) {
		select {
		case g.events <- tag(event):
		case <-ctx.Done():
		}
	}
	proc := parser.NewProcessor(
		parser.Config{Mode: inference.ThinkingCapture, Capture: func(c inference.ThinkingChunk) {
			emit(streamEvent{thought: c.Text})
		}},
		parser.HintFromInference(model.Info()),
	)

	opts := append([]inference.GenerateOption{}, genOpts...)
	opts = append(opts, inference.WithMetricsSink(func(gm inference.GenerateMetrics) {
		m := gm
		emit(streamEvent{metrics: &m})
	}))

	defer func() {
		if finish != nil {
			finish()
		}
		close(g.events)
	}()
	stream := model.Chat(ctx, history, opts...)
	if scoped, ok := model.(scopedErrorChatModel); ok {
		stream = scoped.chatWithErrorSink(ctx, history, func(err error) {
			requestErr = err
			errorSinkCalled = true
		}, opts...)
	}
	for tok := range stream {
		if visible := proc.Process(tok.Text); visible != "" {
			emit(streamEvent{visible: visible})
		}
	}
	if tail := proc.Flush(); tail != "" {
		emit(streamEvent{visible: tail})
	}
	ev := streamEvent{done: true}
	if errorSinkCalled {
		ev.err = requestErr
	} else if r := model.Err(); !r.OK {
		if err, ok := r.Value.(error); ok {
			ev.err = err
		}
	}
	// A cancelled stream may decline the send above; use a final non-blocking
	// attempt so an attentive UI sees done immediately. Closing the channel is
	// still the fallback completion signal when its buffer is full.
	select {
	case g.events <- tag(ev):
	default:
	}
}

// waitEvent yields the next stream event to the update loop; a closed channel
// surfaces as a final done message.
func waitEvent(g *generation) tea.Cmd {
	return func() tea.Msg {
		if g == nil {
			return streamMsg{done: true}
		}
		ev, ok := <-g.events
		if !ok {
			return streamMsg{SessionID: g.SessionID, JobID: g.JobID, done: true}
		}
		return streamMsg(ev)
	}
}
