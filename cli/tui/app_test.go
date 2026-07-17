// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"io"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

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
	// service: enter with no model declines with a note, never starts
	a.activeTab = tabService
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if a.svc.running || a.svc.note == "" {
		t.Fatalf("service start without a model: running=%v note=%q", a.svc.running, a.svc.note)
	}
	// address presets cycle while stopped and render in the tab
	before = a.svc.addrIdx
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyRight})
	a = m.(app)
	if a.svc.addrIdx == before {
		t.Fatal("service right-adjust did not change the address preset")
	}
	if v := a.View(); !strings.Contains(v, a.svc.addr()) {
		t.Fatalf("service tab does not render the listen address %q", a.svc.addr())
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
	defer a.lane.Close()

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

// TestAppLiveServiceAPI (LTHN_PROBE_MODEL-gated) is the Service tab's
// end-to-end receipt: load a real checkpoint through the Update loop, start
// the API from the Service tab, drive a real OpenAI chat completion at it
// over HTTP (through the shared serial lane), then stop it cleanly.
func TestAppLiveServiceAPI(t *testing.T) {
	dir := os.Getenv("LTHN_PROBE_MODEL")
	if dir == "" || os.Getenv("MLX_METALLIB_PATH") == "" {
		t.Skip("needs LTHN_PROBE_MODEL + MLX_METALLIB_PATH")
	}
	a := newApp(dir, 0, 128)
	m, _ := a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a = m.(app)

	msg := loadModel(dir, 0)()
	loaded, ok := msg.(loadedMsg)
	if !ok {
		t.Fatalf("loadModel returned %#v", msg)
	}
	m, _ = a.Update(loaded)
	a = m.(app)
	defer a.lane.Close()

	const addr = "127.0.0.1:36917"
	a.svc.custom = addr
	a.activeTab = tabService
	m, cmd := a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if !a.svc.running || cmd == nil {
		t.Fatalf("service did not start: running=%v note=%q", a.svc.running, a.svc.note)
	}
	if v := a.View(); !strings.Contains(v, addr) {
		t.Fatal("service tab does not render the live address")
	}

	// wait for the listener, then a real OpenAI request through the lane
	client := &http.Client{Timeout: 120 * time.Second}
	body := `{"model":"lem","max_tokens":48,"chat_template_kwargs":{"enable_thinking":false},` +
		`"messages":[{"role":"user","content":"Reply with the single word OK."}]}`
	var resp *http.Response
	var err error
	for i := 0; i < 100; i++ {
		resp, err = client.Post("http://"+addr+"/v1/chat/completions", "application/json", strings.NewReader(body))
		if err == nil {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	if err != nil {
		t.Fatalf("API never answered: %v", err)
	}
	defer resp.Body.Close()
	payload, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("chat completion status %d: %s", resp.StatusCode, payload)
	}
	if !strings.Contains(string(payload), `"content"`) {
		t.Fatalf("no content in completion: %s", payload)
	}
	if a.svc.requests.Load() == 0 {
		t.Fatal("request counter did not move")
	}
	t.Logf("live API: %d req · %s", a.svc.requests.Load(), payload)

	// stop from the tab and drain Serve's return through the update loop
	m, _ = a.Update(tea.KeyMsg{Type: tea.KeyEnter})
	a = m.(app)
	if !a.svc.stopping {
		t.Fatal("enter while running did not begin the stop")
	}
	m, _ = a.Update(waitService(a.svc.events)())
	a = m.(app)
	if a.svc.running {
		t.Fatal("service did not finish cleanly")
	}
}

func TestAppServiceUsesLoadedLaneWithoutOwningIt(t *testing.T) {
	base := newFakeTextModel(map[string][]string{"hello": {"world"}})
	a := newApp("", 0, 32)
	m, _ := a.Update(loadedMsg{model: base, name: "fake"})
	a = m.(app)
	if a.lane == nil || a.model != a.lane.Model() {
		t.Fatal("loaded model was not wrapped in the application lane")
	}

	a.svc.custom = "127.0.0.1:0"
	cmd := a.svc.start(a.model)
	if cmd == nil || !a.svc.running {
		t.Fatalf("service did not start: running=%v note=%q", a.svc.running, a.svc.note)
	}
	a.svc.teardown("test stop")
	select {
	case event := <-a.svc.events:
		a.svc.finish(event.err)
	case <-time.After(2 * time.Second):
		t.Fatal("service listener did not stop")
	}
	if got := base.closes.Load(); got != 0 {
		t.Fatalf("service stop closed the loaded model %d times", got)
	}
	if r := a.lane.Close(); !r.OK {
		t.Fatalf("lane Close error = %s", r.Error())
	}
	if got := base.closes.Load(); got != 1 {
		t.Fatalf("application lane closed base %d times, want 1", got)
	}
}

// errFor builds a plain error for transition tests.
func errFor(text string) error { return &driveErr{text} }

type driveErr struct{ s string }

func (e *driveErr) Error() string { return e.s }
