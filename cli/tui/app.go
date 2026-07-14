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
)

// appState is the screen the app is on: picking a model, loading it, chatting.
type appState int

const (
	statePick appState = iota
	stateLoad
	stateChat
)

// turn is one rendered exchange element in the transcript.
type turn struct {
	role    string // "user" | "assistant"
	thought string
	text    string
}

type app struct {
	state appState

	picker  list.Model
	spin    spinner.Model
	input   textarea.Model
	view    viewport.Model
	width   int
	height  int
	ready   bool
	loading string // model path being loaded

	model     inference.TextModel
	modelName string
	ctxLen    int
	maxTokens int

	turns       []turn
	gen         *generation
	generating  bool
	thinkingOff bool
	lastTokS    float64
	errText     string
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
	in.Placeholder = "ask… (enter sends, esc cancels a reply, ctrl+t toggles thinking, ctrl+c quits)"
	in.SetHeight(3)
	in.ShowLineNumbers = false
	in.Focus()

	a := app{
		state:     statePick,
		picker:    newPicker(),
		spin:      sp,
		input:     in,
		ctxLen:    ctxLen,
		maxTokens: maxTokens,
	}
	if modelPath != "" {
		a.state = stateLoad
		a.loading = modelPath
	}
	return a
}

func (a app) Init() tea.Cmd {
	if a.state == stateLoad {
		return tea.Batch(a.spin.Tick, loadModel(a.loading, a.ctxLen))
	}
	return tea.Batch(a.spin.Tick, discoverModels)
}

func (a app) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		a.width, a.height = msg.Width, msg.Height
		a.picker.SetSize(msg.Width-2, msg.Height-2)
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
		a.state = stateChat
		a.errText = ""
		a.refreshTranscript()
		return a, nil

	case loadErrMsg:
		a.errText = msg.err.Error()
		a.state = statePick
		return a, nil

	case spinner.TickMsg:
		var cmd tea.Cmd
		a.spin, cmd = a.spin.Update(msg)
		return a, cmd

	case streamMsg:
		return a.onStream(msg)

	case tea.KeyMsg:
		return a.onKey(msg)
	}
	return a.route(msg)
}

// onStream folds one generation event into the live assistant turn.
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
	a.refreshTranscript()
	if ev.done {
		a.generating = false
		a.gen = nil
		return a, nil
	}
	return a, waitEvent(a.gen)
}

func (a app) onKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch msg.String() {
	case "ctrl+c":
		if a.gen != nil {
			a.gen.cancel()
		}
		if a.model != nil {
			_ = a.model.Close()
		}
		return a, tea.Quit
	case "esc":
		if a.state == stateChat && a.generating {
			a.gen.cancel() // stops the turn; the stream drains to done
			return a, nil
		}
		if a.state == stateChat && !a.generating {
			// back to the picker (keeps the model loaded until a new pick)
			a.state = statePick
			return a, discoverModels
		}
	case "ctrl+t":
		if a.state == stateChat {
			a.thinkingOff = !a.thinkingOff
			return a, nil
		}
	case "enter":
		if a.state == statePick {
			if item, ok := a.picker.SelectedItem().(modelItem); ok {
				a.state = stateLoad
				a.loading = item.path
				return a, tea.Batch(a.spin.Tick, loadModel(item.path, a.ctxLen))
			}
			return a, nil
		}
		if a.state == stateChat && !a.generating {
			prompt := strings.TrimSpace(a.input.Value())
			if prompt == "" {
				return a, nil
			}
			a.input.Reset()
			a.turns = append(a.turns, turn{role: "user", text: prompt}, turn{role: "assistant"})
			a.generating = true
			a.errText = ""
			a.gen = startGeneration(a.model, a.history(), a.maxTokens, a.thinkingOff)
			a.refreshTranscript()
			return a, waitEvent(a.gen)
		}
	}
	return a.route(msg)
}

// route hands the message to the focused component for the current state.
func (a app) route(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	switch a.state {
	case statePick:
		a.picker, cmd = a.picker.Update(msg)
	case stateChat:
		if a.generating {
			a.view, cmd = a.view.Update(msg)
		} else {
			a.input, cmd = a.input.Update(msg)
		}
	}
	return a, cmd
}

// history rebuilds the inference messages from the transcript — the message
// slice IS the memory, exactly like a stateless API client.
func (a app) history() []inference.Message {
	msgs := make([]inference.Message, 0, len(a.turns))
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

func (a app) transcriptHeight() int {
	h := a.height - a.input.Height() - 5 // input border + status + padding
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
		if t.role == "user" {
			b.WriteString(styleUser.Render("you ") + styleAnswer.Render(t.text))
			continue
		}
		if t.thought != "" {
			b.WriteString(styleThought.Render("· thinking · "+strings.TrimSpace(t.thought)) + "\n")
		}
		b.WriteString(styleAccent.Render(a.modelName+" ") + styleAnswer.Render(t.text))
		if t.text == "" && t.thought == "" && a.generating && i == len(a.turns)-1 {
			b.WriteString(styleThought.Render(a.spin.View() + " …"))
		}
	}
	return lipgloss.NewStyle().Width(a.view.Width).Render(b.String())
}

func (a app) statusLine() string {
	think := "thinking on"
	if a.thinkingOff {
		think = "thinking off"
	}
	parts := []string{a.modelName, think}
	if a.lastTokS > 0 {
		parts = append(parts, core.Sprintf("%.1f tok/s", a.lastTokS))
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
	switch a.state {
	case statePick:
		s := a.picker.View()
		if a.errText != "" {
			s += "\n" + styleErr.Render(a.errText)
		}
		return s
	case stateLoad:
		return "\n  " + a.spin.View() + styleStatus.Render(" loading "+displayName(a.loading)+" …")
	default:
		return lipgloss.JoinVertical(lipgloss.Left,
			a.view.View(),
			styleInputBorder.Render(a.input.View()),
			a.statusLine(),
		)
	}
}
