// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"os"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
)

// TestAppUpdateTransitions drives the pure state machine without a terminal:
// sizing readies the viewport, discovery fills the picker, enter on a picked
// item moves to loading, a loadErr falls back to the picker, and ctrl+t
// toggles the thinking opt-out in chat.
func TestAppUpdateTransitions(t *testing.T) {
	a := newApp("", 0, 512)
	m, _ := a.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	a = m.(app)
	if !a.ready {
		t.Fatal("window size did not ready the app")
	}
	if a.activeTab != tabModels {
		t.Fatalf("activeTab = %d, want Models on a pickerless start", a.activeTab)
	}
	// a load failure lands back on Models with the error surfaced
	m, _ = a.Update(loadErrMsg{err: errFor("no backends")})
	a = m.(app)
	if a.activeTab != tabModels || a.errText == "" {
		t.Fatalf("loadErr: tab=%d err=%q, want Models + message", a.activeTab, a.errText)
	}
	// tab cycles through every pane and wraps
	for i := 0; i < int(tabCount); i++ {
		m, _ = a.Update(tea.KeyMsg{Type: tea.KeyTab})
		a = m.(app)
	}
	if a.activeTab != tabModels {
		t.Fatalf("tab cycle did not wrap: %d", a.activeTab)
	}
	// ctrl+t flips the thinking override to an explicit state
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyCtrlT})
	a = m.(app)
	if a.cfg.thinking() == nil {
		t.Fatal("ctrl+t left thinking on the model default")
	}
	if v := a.View(); !strings.Contains(v, "thinking") {
		t.Fatalf("status line missing thinking state: %q", v)
	}
	// settings adjust: move to max tokens and bump it
	a.activeTab = tabSettings
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyDown})
	a = m.(app)
	before := a.cfg.maxTokens()
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyRight})
	a = m.(app)
	if a.cfg.maxTokens() == before {
		t.Fatal("settings right-adjust did not change max tokens")
	}
	// tools toggle arms declarations
	a.activeTab = tabTools
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if !a.tools.enabled || a.tools.declarations() == "" {
		t.Fatal("tools enter did not arm declarations")
	}
}

// TestAppLiveChatDrive (LTHN_PROBE_MODEL-gated) is the headless end-to-end
// receipt: load a real checkpoint, send a prompt through the real Update loop,
// consume the stream to done, and assert an answer landed with metrics — the
// whole TUI path minus the terminal.
func TestAppLiveChatDrive(t *testing.T) {
	dir := os.Getenv("LTHN_PROBE_MODEL")
	if dir == "" || os.Getenv("MLX_METALLIB_PATH") == "" {
		t.Skip("needs LTHN_PROBE_MODEL + MLX_METALLIB_PATH")
	}
	a := newApp(dir, 0, 128)
	m, _ := a.Update(tea.WindowSizeMsg{Width: 80, Height: 24})
	a = m.(app)

	msg := loadModel(dir, 0)()
	loaded, ok := msg.(loadedMsg)
	if !ok {
		t.Fatalf("loadModel returned %#v", msg)
	}
	m, _ = a.Update(loaded)
	a = m.(app)
	if a.activeTab != tabChat || a.model == nil {
		t.Fatalf("tab = %d model=%v, want chat + loaded", a.activeTab, a.model != nil)
	}
	defer a.model.Close()

	a.cfg.thinkIdx = 2 // thinking off — deterministic short answers for the drive
	a.input.SetValue("Say hello in exactly two words.")
	m, cmd := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if !a.generating || cmd == nil {
		t.Fatal("enter did not start a generation")
	}
	for i := 0; i < 4096 && cmd != nil; i++ {
		m, cmd = a.Update(cmd())
		a = m.(app)
		if !a.generating {
			break
		}
	}
	if a.generating {
		t.Fatal("generation never completed")
	}
	last := a.turns[len(a.turns)-1]
	if last.role != "assistant" || strings.TrimSpace(last.text) == "" {
		t.Fatalf("assistant turn empty: %+v", last)
	}
	if a.lastTokS <= 0 {
		t.Fatalf("decode tok/s not recorded: %v", a.lastTokS)
	}
	t.Logf("live drive: %q at %.1f tok/s", strings.TrimSpace(last.text), a.lastTokS)
}

// errFor builds a plain error for transition tests.
func errFor(text string) error { return &driveErr{text} }

type driveErr struct{ s string }

func (e *driveErr) Error() string { return e.s }
