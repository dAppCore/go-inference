// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"context"
	"strings"
	"sync/atomic"
	"time"

	tea "github.com/charmbracelet/bubbletea"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/serving"
)

// The Service tab hosts the OpenAI/Anthropic/Ollama-compatible HTTP API for
// the model already loaded in the TUI — the same weights, not a second load.
// The application supplies its already-scheduled model lane, so a coding agent
// hitting the API and a turn typed in Chat queue behind each other instead of
// racing the engine. The service owns only its listener and context.

// serviceAddrs are the listen presets cycled with ←/→ while stopped.
var serviceAddrs = []struct{ addr, hint string }{
	{":36911", "Lethean's own port — the default the client hints below use"},
	{":11434", "Ollama's port — drop-in for clients hard-wired to an Ollama install"},
	{":8080", "plain local HTTP port"},
	{"0.0.0.0:36911", "every interface — reachable from other machines on your network"},
}

type serviceState struct {
	running  bool
	stopping bool // cancel sent, waiting for the listener to drain
	addrIdx  int
	custom   string // overrides the preset when non-empty (tests, future flag)

	requests *atomic.Int64
	cancel   context.CancelFunc
	events   chan serviceEvent
	note     string // status / error line for the tab
}

type serviceEvent struct{ err error } // Serve's return: nil = clean shutdown

type serviceMsg struct{ ev serviceEvent }

type serviceTickMsg struct{}

func newService() serviceState {
	return serviceState{requests: &atomic.Int64{}}
}

func (s serviceState) addr() string {
	if s.custom != "" {
		return s.custom
	}
	return serviceAddrs[s.addrIdx].addr
}

// baseURL renders the address as the URL clients dial. A bare ":port" listen
// is reachable as localhost; the any-interface preset needs the machine's own
// address, which only the operator knows.
func (s serviceState) baseURL() string {
	addr := s.addr()
	if strings.HasPrefix(addr, ":") {
		return "http://localhost" + addr
	}
	if after, ok := strings.CutPrefix(addr, "0.0.0.0"); ok {
		return "http://<this-machine>" + after
	}
	return "http://" + addr
}

// serviceResolver answers every model name with the supplied model lane — the
// request's `model` field is cosmetic, exactly like `lem serve`. The counter
// is the tab's requests-served receipt.
type serviceResolver struct {
	model inference.TextModel
	n     *atomic.Int64
}

func (r serviceResolver) ResolveModel(context.Context, string) (inference.TextModel, error) {
	r.n.Add(1)
	return r.model, nil
}

// start boots the HTTP listener on its own goroutine. Listener failures (port
// in use) arrive as a serviceMsg. The model's lifecycle remains with app.
func (s *serviceState) start(model inference.TextModel) tea.Cmd {
	if s.running {
		return nil
	}
	if model == nil {
		s.note = "load a model first — pick one in Models"
		return nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	events := make(chan serviceEvent, 1)
	s.requests.Store(0)
	resolver := serviceResolver{model: model, n: s.requests}
	addr := s.addr()
	go func() {
		events <- serviceEvent{err: serving.Serve(ctx, addr, resolver)}
	}()
	s.cancel, s.events = cancel, events
	s.running, s.stopping, s.note = true, false, ""
	return tea.Batch(waitService(events), serviceTick())
}

// stop asks the listener to drain; finish() lands when Serve returns.
func (s *serviceState) stop() {
	if !s.running || s.stopping {
		return
	}
	s.stopping = true
	s.cancel()
}

// finish folds Serve's return into the tab — idempotent with teardown, so a
// late serviceMsg after a synchronous teardown is absorbed cleanly.
func (s *serviceState) finish(err error) {
	s.cancel = nil
	s.running, s.stopping = false, false
	if err != nil {
		s.note = "service: " + err.Error()
	} else if s.note == "" {
		s.note = "stopped"
	}
}

// teardown stops the service synchronously — the quit and model-swap path.
// The Serve goroutine still delivers its event; finish absorbs it.
func (s *serviceState) teardown(reason string) {
	if !s.running {
		return
	}
	if s.cancel != nil {
		s.cancel()
	}
	s.cancel = nil
	s.running, s.stopping = false, false
	s.note = reason
}

func waitService(ch chan serviceEvent) tea.Cmd {
	return func() tea.Msg { return serviceMsg{ev: <-ch} }
}

// serviceTick refreshes the render once a second while serving, so the
// requests counter moves without a keypress.
func serviceTick() tea.Cmd {
	return tea.Tick(time.Second, func(time.Time) tea.Msg { return serviceTickMsg{} })
}

func (s serviceState) view(modelName string, width int) string {
	var b strings.Builder
	b.WriteString(styleTitle.Render("service") + "  " +
		styleThought.Render("OpenAI · Anthropic · Ollama HTTP API for the loaded model") + "\n\n")

	state := styleStatus.Render("○ stopped")
	if s.running {
		label := "● serving on " + s.addr()
		if modelName != "" {
			label = "● serving " + modelName + " on " + s.addr()
		}
		state = styleAccent.Render(label)
	}
	b.WriteString("  " + state + "\n\n")

	addrLabel := styleAnswer.Render("address")
	value := "‹ " + s.addr() + " ›"
	hint := serviceAddrs[s.addrIdx].hint
	if s.running {
		value = s.addr()
		hint = "locked while serving — stop first to change it"
	}
	b.WriteString("  " + addrLabel + "  " + styleTitle.Render(value) + "\n")
	b.WriteString("    " + styleThought.Render(hint) + "\n\n")

	if s.running || s.requests.Load() > 0 {
		b.WriteString("  " + styleAnswer.Render("requests") + "  " +
			styleTitle.Render(core.Sprintf("%d", s.requests.Load())) + "\n\n")
	}

	base := s.baseURL()
	b.WriteString(styleTitle.Render("point a client here") + "  " +
		styleThought.Render("the request's model name is cosmetic — the loaded model answers") + "\n")
	b.WriteString("  " + styleAnswer.Render("opencode / codex / OpenAI SDKs") + "   " + styleAccent.Render(base+"/v1") + "\n")
	b.WriteString("  " + styleAnswer.Render("Claude Code / Anthropic SDKs  ") + "   " + styleAccent.Render(base) + "\n")
	b.WriteString("  " + styleAnswer.Render("Ollama clients                ") + "   " + styleAccent.Render(base) + "\n\n")

	b.WriteString(styleTitle.Render("smoke") + "\n")
	b.WriteString("  " + styleThought.Render("curl -s "+base+"/v1/chat/completions \\") + "\n")
	b.WriteString("  " + styleThought.Render(`    -d '{"model":"lem","messages":[{"role":"user","content":"hello"}]}'`) + "\n\n")

	b.WriteString("  " + styleThought.Render("TUI chat and API requests share the model through one serial lane —") + "\n")
	b.WriteString("  " + styleThought.Render("turns queue behind each other, nothing races the engine.") + "\n\n")

	if s.note != "" {
		b.WriteString("  " + styleErr.Render(s.note) + "\n\n")
	}
	b.WriteString(styleStatus.Render("enter start/stop · ←/→ address (while stopped)"))
	return b.String()
}
