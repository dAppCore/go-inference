// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"

	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/decode/parser"
)

// turn is one rendered element in the transcript.
type turn struct {
	role    string // "user" | "assistant" | "tool"
	thought string
	text    string
	calls   []string // rendered tool-call receipts on an assistant turn
}

type app struct {
	activeTab tabID

	picker  list.Model
	spin    spinner.Model
	input   textarea.Model
	view    viewport.Model
	width   int
	height  int
	ready   bool
	loading string // model path mid-load ("" = idle)

	model     inference.TextModel
	modelName string

	cfg   settings
	modes modeState
	tools toolState
	svc   serviceState

	turns      []turn
	gen        *generation
	generating bool
	toolHops   int // auto-continuations this turn chain (bounded)
	lastTokS   float64
	errText    string
}

type loadedMsg struct {
	model inference.TextModel
	name  string
}
type loadErrMsg struct{ err error }

// loadModel loads the checkpoint through the registered engine as a tea.Cmd.
func loadModel(path string, ctxLen int) tea.Cmd {
	return func() tea.Msg {
		var opts []inference.LoadOption
		if ctxLen > 0 {
			opts = append(opts, inference.WithContextLen(ctxLen))
		}
		r := inference.LoadModel(path, opts...)
		if !r.OK {
			if err, ok := r.Value.(error); ok {
				return loadErrMsg{err: err}
			}
			return loadErrMsg{err: core.NewError("load failed")}
		}
		return loadedMsg{model: r.Value.(inference.TextModel), name: displayName(path)}
	}
}

func newApp(modelPath string, ctxLen, maxTokens int) app {
	sp := spinner.New()
	sp.Spinner = spinner.MiniDot

	in := textarea.New()
	in.Placeholder = "ask… (enter sends · esc cancels a reply · tab switches tabs · ctrl+c quits)"
	in.SetHeight(3)
	in.ShowLineNumbers = false
	in.Focus()

	cfg := newSettings()
	for i, v := range ctxSteps {
		if v == ctxLen {
			cfg.ctxIdx = i
		}
	}
	for i, v := range maxTokSteps {
		if v == maxTokens {
			cfg.maxTokIdx = i
		}
	}

	a := app{
		activeTab: tabModels,
		picker:    newPicker(),
		spin:      sp,
		input:     in,
		cfg:       cfg,
		modes:     modeState{},
		tools:     newTools(),
		svc:       newService(),
	}
	if modelPath != "" {
		a.activeTab = tabChat
		a.loading = modelPath
	}
	return a
}

func (a app) Init() tea.Cmd {
	cmds := []tea.Cmd{a.spin.Tick, discoverModels}
	if a.loading != "" {
		cmds = append(cmds, loadModel(a.loading, a.cfg.contextLen()))
	}
	return tea.Batch(cmds...)
}

func (a app) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		a.width, a.height = msg.Width, msg.Height
		a.picker.SetSize(msg.Width-2, a.contentHeight())
		a.input.SetWidth(msg.Width - 6)
		a.view = viewport.New(msg.Width-2, a.transcriptHeight())
		a.view.SetContent(a.renderTranscript())
		a.view.GotoBottom()
		a.ready = true
		return a, nil

	case discoveredMsg:
		return a, a.picker.SetItems(msg.items)

	case loadedMsg:
		if a.model != nil {
			_ = a.model.Close()
		}
		a.model, a.modelName = msg.model, msg.name
		a.loading = ""
		a.errText = ""
		a.activeTab = tabChat
		a.refreshTranscript()
		return a, nil

	case loadErrMsg:
		a.errText = msg.err.Error()
		a.loading = ""
		a.activeTab = tabModels
		return a, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		a.spin, cmd = a.spin.Update(msg)
		return a, cmd

	case streamMsg:
		return a.onStream(msg)

	case serviceMsg:
		a.svc.finish(msg.ev.err)
		return a, nil

	case serviceTickMsg:
		if a.svc.running {
			return a, serviceTick() // re-arm: keeps the requests counter live
		}
		return a, nil

	case tea.KeyMsg:
		return a.onKey(msg)
	}
	return a.route(msg)
}

// onStream folds one generation event into the live assistant turn; on done it
// runs the tool loop when armed.
func (a app) onStream(ev streamMsg) (tea.Model, tea.Cmd) {
	if len(a.turns) > 0 {
		last := &a.turns[len(a.turns)-1]
		if last.role == "assistant" {
			last.thought += ev.thought
			last.text += ev.visible
		}
	}
	if ev.metrics != nil {
		a.lastTokS = ev.metrics.DecodeTokensPerSec
	}
	if ev.err != nil {
		a.errText = ev.err.Error()
	}
	if !ev.done {
		a.refreshTranscript()
		return a, waitEvent(a.gen)
	}
	a.generating = false
	a.gen = nil
	if cmd := a.runToolLoop(); cmd != nil {
		a.refreshTranscript()
		return a, cmd
	}
	a.toolHops = 0
	a.refreshTranscript()
	return a, nil
}

// runToolLoop parses the finished assistant turn for tool calls; when the
// Tools tab armed them it executes each locally, appends the wrapped tool
// results, and auto-continues the conversation (bounded hops).
func (a *app) runToolLoop() tea.Cmd {
	if !a.tools.enabled || a.toolHops >= 2 || len(a.turns) == 0 {
		return nil
	}
	last := &a.turns[len(a.turns)-1]
	if last.role != "assistant" {
		return nil
	}
	calls, visible := parser.ParseGemmaToolCalls(last.text)
	if len(calls) == 0 {
		return nil
	}
	last.text = strings.TrimSpace(visible)
	for _, call := range calls {
		result := a.tools.execute(call)
		last.calls = append(last.calls, call.Name+" → "+result)
		a.turns = append(a.turns, turn{role: "tool", text: parser.RenderGemmaToolResponse(result)})
	}
	a.turns = append(a.turns, turn{role: "assistant"})
	a.toolHops++
	a.generating = true
	a.gen = startGeneration(a.chatModel(), a.history(), a.generateOpts())
	return waitEvent(a.gen)
}

func (a app) onKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c":
		if a.gen != nil {
			a.gen.cancel()
		}
		a.svc.teardown("stopped (quit)")
		if a.model != nil {
			_ = a.model.Close()
		}
		return a, tea.Quit
	case "tab", "shift+tab":
		if msg.String() == "tab" {
			a.activeTab = a.activeTab.next()
		} else {
			a.activeTab = a.activeTab.prev()
		}
		if a.activeTab == tabModels && len(a.picker.Items()) == 0 {
			return a, discoverModels
		}
		return a, nil
	case "esc":
		if a.generating {
			a.gen.cancel() // stops the turn; the stream drains to done
			return a, nil
		}
	case "ctrl+t":
		// quick thinking toggle: flips between explicit on and off
		if a.cfg.thinkIdx == 2 {
			a.cfg.thinkIdx = 1
		} else {
			a.cfg.thinkIdx = 2
		}
		return a, nil
	}

	switch a.activeTab {
	case tabModels:
		if msg.String() == "enter" && a.loading == "" {
			if item, ok := a.picker.SelectedItem().(modelItem); ok {
				// the service serves THIS model's weights — stop it before they change
				a.svc.teardown("stopped — model changing")
				a.loading = item.path
				return a, tea.Batch(a.spin.Tick, loadModel(item.path, a.cfg.contextLen()))
			}
			return a, nil
		}
	case tabService:
		switch msg.String() {
		case "enter":
			if a.svc.running {
				if a.generating {
					a.svc.note = "a reply is streaming — esc it or let it finish, then stop"
					return a, nil
				}
				a.svc.stop()
				return a, nil
			}
			return a, a.svc.start(a.model)
		case "left", "h":
			if !a.svc.running {
				a.svc.addrIdx = (a.svc.addrIdx + len(serviceAddrs) - 1) % len(serviceAddrs)
			}
			return a, nil
		case "right", "l":
			if !a.svc.running {
				a.svc.addrIdx = (a.svc.addrIdx + 1) % len(serviceAddrs)
			}
			return a, nil
		}
	case tabSettings:
		switch msg.String() {
		case "up", "k":
			a.cfg = a.cfg.move(-1)
			return a, nil
		case "down", "j":
			a.cfg = a.cfg.move(1)
			return a, nil
		case "left", "h":
			a.cfg = a.cfg.adjust(-1)
			return a, nil
		case "right", "l", "enter":
			a.cfg = a.cfg.adjust(1)
			return a, nil
		}
	case tabModes:
		switch msg.String() {
		case "up", "k":
			a.modes = a.modes.move(-1)
			return a, nil
		case "down", "j":
			a.modes = a.modes.move(1)
			return a, nil
		}
	case tabTools:
		if msg.String() == "enter" {
			a.tools.enabled = !a.tools.enabled
			return a, nil
		}
	case tabChat:
		if msg.String() == "enter" && !a.generating && a.model != nil {
			prompt := strings.TrimSpace(a.input.Value())
			if prompt == "" {
				return a, nil
			}
			a.input.Reset()
			a.turns = append(a.turns, turn{role: "user", text: prompt}, turn{role: "assistant"})
			a.generating = true
			a.toolHops = 0
			a.errText = ""
			a.gen = startGeneration(a.chatModel(), a.history(), a.generateOpts())
			a.refreshTranscript()
			return a, waitEvent(a.gen)
		}
	}
	return a.route(msg)
}

// route hands the message to the focused component for the active tab.
func (a app) route(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	switch a.activeTab {
	case tabModels:
		a.picker, cmd = a.picker.Update(msg)
	case tabChat:
		if a.generating {
			a.view, cmd = a.view.Update(msg)
		} else {
			a.input, cmd = a.input.Update(msg)
		}
	}
	return a, cmd
}

// chatModel is the model TUI turns generate on: the service's serial lane
// while the API is up (so terminal turns and HTTP requests queue, never race),
// the raw model otherwise.
func (a app) chatModel() inference.TextModel {
	if a.svc.running && a.svc.sched != nil {
		return a.svc.sched
	}
	return a.model
}

// generateOpts folds the Modes preset with the Settings overrides — an
// explicit Settings thinking choice wins over the preset's.
func (a app) generateOpts() []inference.GenerateOption {
	opts := append([]inference.GenerateOption{}, a.modes.current().opts()...)
	opts = append(opts, inference.WithMaxTokens(a.cfg.maxTokens()))
	if th := a.cfg.thinking(); th != nil {
		opts = append(opts, inference.WithEnableThinking(th))
	}
	return opts
}

// history rebuilds the inference messages from the transcript — the message
// slice IS the memory, exactly like a stateless API client. Tool declarations
// ride the system turn when the Tools tab armed them (the serve convention).
func (a app) history() []inference.Message {
	msgs := make([]inference.Message, 0, len(a.turns)+1)
	if decl := a.tools.declarations(); decl != "" {
		msgs = append(msgs, inference.Message{Role: "system", Content: decl})
	}
	for _, t := range a.turns {
		if t.role == "assistant" && t.text == "" && t.thought == "" {
			continue // the live, still-empty turn
		}
		msgs = append(msgs, inference.Message{Role: t.role, Content: t.text})
	}
	return msgs
}

func (a *app) refreshTranscript() {
	if !a.ready {
		return
	}
	a.view.Height = a.transcriptHeight()
	a.view.SetContent(a.renderTranscript())
	a.view.GotoBottom()
}

func (a app) contentHeight() int {
	h := a.height - 4 // tab bar + status
	if h < 3 {
		h = 3
	}
	return h
}

func (a app) transcriptHeight() int {
	h := a.height - a.input.Height() - 8 // tab bar + input border + status
	if h < 3 {
		h = 3
	}
	return h
}

func (a app) renderTranscript() string {
	var b strings.Builder
	for i, t := range a.turns {
		if i > 0 {
			b.WriteString("\n\n")
		}
		switch t.role {
		case "user":
			b.WriteString(styleUser.Render("you ") + styleAnswer.Render(t.text))
		case "tool":
			b.WriteString(styleThought.Render("tool result fed back"))
		default:
			if t.thought != "" {
				b.WriteString(styleThought.Render("· thinking · "+strings.TrimSpace(t.thought)) + "\n")
			}
			b.WriteString(styleAccent.Render(a.modelName+" ") + styleAnswer.Render(t.text))
			for _, c := range t.calls {
				b.WriteString("\n" + styleThought.Render("→ "+c))
			}
			if t.text == "" && t.thought == "" && a.generating && i == len(a.turns)-1 {
				b.WriteString(styleThought.Render(a.spin.View() + " …"))
			}
		}
	}
	return lipgloss.NewStyle().Width(a.view.Width).Render(b.String())
}

func (a app) statusLine() string {
	parts := []string{}
	if a.modelName != "" {
		parts = append(parts, a.modelName)
	} else {
		parts = append(parts, "no model — pick one in Models")
	}
	parts = append(parts, "mode "+a.modes.current().name, "thinking "+thinkNames[a.cfg.thinkIdx])
	if a.tools.enabled {
		parts = append(parts, "tools on")
	}
	if a.svc.running {
		parts = append(parts, core.Sprintf("api %s · %d req", a.svc.addr(), a.svc.requests.Load()))
	}
	if a.lastTokS > 0 {
		parts = append(parts, core.Sprintf("%.1f tok/s", a.lastTokS))
	}
	if a.loading != "" {
		parts = append(parts, a.spin.View()+" loading "+displayName(a.loading))
	}
	if a.generating {
		parts = append(parts, a.spin.View()+" generating (esc cancels)")
	}
	if a.errText != "" {
		parts = append(parts, styleErr.Render(a.errText))
	}
	return styleStatus.Render(strings.Join(parts, "  ·  "))
}

func (a app) View() string {
	bar := renderTabBar(a.activeTab, a.width)
	var content string
	switch a.activeTab {
	case tabModels:
		content = a.picker.View()
	case tabService:
		content = a.svc.view(a.modelName, a.width)
	case tabSettings:
		content = a.cfg.view(a.width)
	case tabTools:
		content = a.tools.view(a.width)
	case tabModes:
		content = a.modes.view(a.width)
	default: // chat
		if a.model == nil && a.loading == "" {
			content = "\n  " + styleStatus.Render("no model loaded — press tab to reach Models and pick one")
		} else if a.model == nil {
			content = "\n  " + a.spin.View() + styleStatus.Render(" loading "+displayName(a.loading)+" …")
		} else {
			content = lipgloss.JoinVertical(lipgloss.Left,
				a.view.View(),
				styleInputBorder.Render(a.input.View()),
			)
		}
	}
	return lipgloss.JoinVertical(lipgloss.Left, bar, content, a.statusLine())
}
