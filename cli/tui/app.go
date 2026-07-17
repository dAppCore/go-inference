// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"context"
	"strings"

	"github.com/charmbracelet/bubbles/key"
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
	activePanel   panelID
	inspectorOpen bool
	styles        uiStyles
	keys          keyMap

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
	lane      *modelLane
	jobs      *jobManager
	cancel    context.CancelFunc
	sessionID string

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
	ctx, cancel := context.WithCancel(context.Background())
	styles := newUIStyles(midnightTheme())
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
		activePanel: panelModels,
		styles:      styles,
		keys:        newKeyMap(),
		picker:      newPicker(styles),
		spin:        sp,
		input:       in,
		cfg:         cfg,
		modes:       modeState{},
		tools:       newTools(),
		svc:         newService(),
		jobs:        newJobManager(ctx),
		cancel:      cancel,
		sessionID:   newRecordID(),
	}
	if modelPath != "" {
		a.activePanel = panelChat
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
		metrics := measureFrame(msg.Width, msg.Height, a.inspectorOpen)
		a.picker.SetSize(max(1, metrics.mainWidth), max(1, metrics.mainHeight))
		a.input.SetWidth(max(1, metrics.mainWidth-4))
		a.view = viewport.New(max(1, metrics.mainWidth), a.transcriptHeight())
		a.view.SetContent(a.renderTranscript())
		a.view.GotoBottom()
		a.ready = true
		return a, nil

	case discoveredMsg:
		return a, a.picker.SetItems(msg.items)

	case loadedMsg:
		laneResult := newModelLane(msg.model, msg.name)
		if !laneResult.OK {
			if msg.model != nil {
				_ = msg.model.Close()
			}
			a.errText = laneResult.Error()
			a.loading = ""
			a.activePanel = panelModels
			return a, nil
		}
		_ = a.jobs.CancelAll()
		if a.lane != nil {
			_ = a.lane.Close()
		}
		a.lane = laneResult.Value.(*modelLane)
		a.model, a.modelName = a.lane.Model(), msg.name
		a.loading = ""
		a.errText = ""
		a.activePanel = panelChat
		a.refreshTranscript()
		return a, nil

	case loadErrMsg:
		a.errText = msg.err.Error()
		a.loading = ""
		a.activePanel = panelModels
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
	if a.gen == nil || ev.SessionID != a.gen.SessionID || ev.JobID != a.gen.JobID {
		return a, nil
	}
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
	return a.beginGeneration()
}

func (a app) onKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if key.Matches(msg, a.keys.ToggleInspector) {
		a.inspectorOpen = !a.inspectorOpen
		if a.ready {
			metrics := measureFrame(a.width, a.height, a.inspectorOpen)
			a.picker.SetSize(max(1, metrics.mainWidth), max(1, metrics.mainHeight))
			a.input.SetWidth(max(1, metrics.mainWidth-4))
			a.view.Width = max(1, metrics.mainWidth)
			a.view.Height = a.transcriptHeight()
		}
		return a, nil
	}
	switch msg.String() {
	case "ctrl+c":
		_ = a.jobs.CancelAll()
		a.cancel()
		a.svc.teardown("stopped (quit)")
		if a.lane != nil {
			_ = a.lane.Close()
		}
		return a, tea.Quit
	case "tab", "shift+tab":
		if msg.String() == "tab" {
			a.activePanel = a.activePanel.next()
		} else {
			a.activePanel = a.activePanel.prev()
		}
		if a.activePanel == panelModels && len(a.picker.Items()) == 0 {
			return a, discoverModels
		}
		return a, nil
	case "esc":
		if a.generating {
			_ = a.jobs.Cancel(a.gen.SessionID) // stream drains to tagged done
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

	switch a.activePanel {
	case panelModels:
		if msg.String() == "enter" && a.loading == "" {
			if item, ok := a.picker.SelectedItem().(modelItem); ok {
				// the service serves THIS model's weights — stop it before they change
				a.svc.teardown("stopped — model changing")
				a.loading = item.path
				return a, tea.Batch(a.spin.Tick, loadModel(item.path, a.cfg.contextLen()))
			}
			return a, nil
		}
	case panelService:
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
	case panelChat:
		if msg.String() == "enter" && !a.generating && a.model != nil {
			prompt := strings.TrimSpace(a.input.Value())
			if prompt == "" {
				return a, nil
			}
			a.input.Reset()
			a.turns = append(a.turns, turn{role: "user", text: prompt}, turn{role: "assistant"})
			a.toolHops = 0
			a.errText = ""
			cmd := a.beginGeneration()
			a.refreshTranscript()
			return a, cmd
		}
	}
	return a.route(msg)
}

// route hands the message to the focused component for the active tab.
func (a app) route(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	switch a.activePanel {
	case panelModels:
		a.picker, cmd = a.picker.Update(msg)
	case panelChat:
		if a.generating {
			a.view, cmd = a.view.Update(msg)
		} else {
			a.input, cmd = a.input.Update(msg)
		}
	}
	return a, cmd
}

// chatModel is always the application's shared serial lane. Starting or
// stopping the HTTP service cannot change generation ownership or ordering.
func (a app) chatModel() inference.TextModel {
	return a.model
}

// beginGeneration registers the active session turn with the job manager and
// returns Bubble Tea's first wait command.
func (a *app) beginGeneration() tea.Cmd {
	result := a.jobs.Start(a.sessionID, newRecordID(), a.chatModel(), a.history(), a.generateOpts())
	if !result.OK {
		a.errText = result.Error()
		a.generating = false
		a.gen = nil
		return nil
	}
	a.gen = result.Value.(*generation)
	a.generating = true
	return waitEvent(a.gen)
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
	return max(1, measureFrame(a.width, a.height, a.inspectorOpen).mainHeight)
}

func (a app) transcriptHeight() int {
	h := a.contentHeight() - a.input.Height() - 2 // composer border
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
			b.WriteString(a.styles.user.Render("you ") + a.styles.answer.Render(t.text))
		case "tool":
			b.WriteString(a.styles.thought.Render("tool result fed back"))
		default:
			if t.thought != "" {
				b.WriteString(a.styles.thought.Render("· thinking · "+strings.TrimSpace(t.thought)) + "\n")
			}
			b.WriteString(a.styles.assistant.Render(a.modelName+" ") + a.styles.answer.Render(t.text))
			for _, c := range t.calls {
				b.WriteString("\n" + a.styles.thought.Render("→ "+c))
			}
			if t.text == "" && t.thought == "" && a.generating && i == len(a.turns)-1 {
				b.WriteString(a.styles.thought.Render(a.spin.View() + " …"))
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
		parts = append(parts, a.styles.err.Render("error: "+a.errText))
	}
	return a.styles.status.Render(strings.Join(parts, "  ·  "))
}

func (a app) View() string {
	content := a.panelView()
	return renderFrame(frameSpec{
		Width:         a.width,
		Height:        a.height,
		Active:        a.activePanel,
		SessionStrip:  a.sessionStrip(),
		Main:          content,
		Inspector:     a.inspectorView(),
		Footer:        a.footerLine(),
		InspectorOpen: a.inspectorOpen,
	}, a.styles)
}

func (a app) panelView() string {
	switch a.activePanel {
	case panelModels:
		return a.picker.View()
	case panelService:
		return a.svc.view(a.modelName, a.width, a.styles)
	case panelWork:
		return lipgloss.JoinVertical(lipgloss.Left,
			a.styles.title.Render("Work"),
			"",
			a.styles.status.Render("○ no active work"),
			a.styles.thought.Render("Local work tracking and agent capabilities arrive in the next workspace slice."),
		)
	default: // chat
		if a.model == nil && a.loading == "" {
			return "\n  " + a.styles.status.Render("○ no model loaded — open Models and choose one")
		}
		if a.model == nil {
			return "\n  " + a.spin.View() + a.styles.status.Render(" loading "+displayName(a.loading)+" …")
		}
		return lipgloss.JoinVertical(lipgloss.Left,
			a.view.View(),
			a.styles.inputBorder.Render(a.input.View()),
		)
	}
}

func (a app) sessionStrip() string {
	title := "New session"
	for _, turn := range a.turns {
		if turn.role == "user" && core.Trim(turn.text) != "" {
			title = core.Trim(turn.text)
			break
		}
	}
	marker := "●"
	state := "idle"
	if a.generating {
		marker, state = "◉", "generating"
	}
	return marker + " " + title + "  ·  " + state
}

func (a app) inspectorView() string {
	model := "○ none loaded"
	if a.modelName != "" {
		model = "● " + a.modelName
	}
	generation := "○ idle"
	if a.generating {
		generation = "◉ generating"
	}
	tools := "○ off"
	if a.tools.enabled {
		tools = "● on"
	}
	return strings.Join([]string{
		a.styles.title.Render("INSPECTOR"),
		"",
		a.styles.accent.Render("SESSION") + "  ● active",
		a.styles.accent.Render("MODEL") + "  " + model,
		a.styles.accent.Render("GENERATION") + "  " + generation,
		a.styles.accent.Render("MODE") + "  " + a.modes.current().name,
		a.styles.accent.Render("TOOLS") + "  " + tools,
		"",
		a.styles.thought.Render("ctrl+o toggles this inspector"),
	}, "\n")
}

func (a app) footerLine() string {
	keys := "tab panels  ·  ctrl+k commands  ·  ctrl+o inspector  ·  f1 help"
	if chooseLayout(a.width) == layoutNarrow {
		keys = "tab panels  ·  ^K commands  ·  ^O info  ·  F1 help"
	}
	status := a.statusLine()
	if status == "" {
		return keys
	}
	return status + "  │  " + keys
}
