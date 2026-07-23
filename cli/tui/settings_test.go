// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	coreio "dappco.re/go/io"
	tea "dappco.re/go/render/display/tui"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

func TestRenderSettings_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	view := renderSettings(newSettings(), 100, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "settings" {
		t.Fatalf("form must open with its title line: %q", lines[0])
	}
	if strings.TrimSpace(lines[1]) != "" {
		t.Fatalf("title must be followed by a blank separator: %q", lines[1])
	}
	if !strings.Contains(plain, "› context length") {
		t.Fatalf("cursor row must carry the active marker: %q", plain)
	}
	for _, text := range []string{"○ max tokens", "○ thinking"} {
		if !strings.Contains(plain, text) {
			t.Fatalf("unselected row missing the idle marker %q: %q", text, plain)
		}
	}
	for _, text := range []string{"‹ model default ›", "‹ 4096 ›"} {
		if !strings.Contains(plain, text) {
			t.Fatalf("form missing the value %q: %q", text, plain)
		}
	}
	for _, text := range []string{"KV cache size", "per-reply budget", "reasoning channel"} {
		if !strings.Contains(plain, text) {
			t.Fatalf("form missing the hint %q: %q", text, plain)
		}
	}
	if !strings.Contains(plain, "↑/↓ select · ←/→ change · ctrl+s saves · esc closes") {
		t.Fatalf("form missing the key footer: %q", plain)
	}

	row := -1
	for index, line := range lines {
		if strings.Contains(line, "› context length") {
			row = index
			break
		}
	}
	if row < 0 || row+2 >= len(lines) {
		t.Fatalf("cursor row not found in the rendered lines: %q", plain)
	}
	if !strings.Contains(lines[row+1], "KV cache size") {
		t.Fatalf("hint must sit directly beneath its value line: %q", lines[row+1])
	}
	if lines[row+2] != "" {
		t.Fatalf("rows must be separated by a blank line: %q", lines[row+2])
	}

	for index, line := range lines {
		if got := lipgloss.Width(line); got > 100 {
			t.Fatalf("line %d width = %d, exceeds 100: %q", index, got, line)
		}
	}
}

func TestRenderSettings_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	for _, width := range []int{0, -4} {
		if got := renderSettings(newSettings(), width, styles); got != "" {
			t.Fatalf("renderSettings(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderSettings_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	form := newSettings().move(-1) // wraps the cursor to the last row
	form = form.adjust(1)          // thinking: model default → on
	view := renderSettings(form, 100, styles)
	plain := ansi.Strip(view)

	if !strings.Contains(plain, "› thinking") {
		t.Fatalf("wrapped cursor must select the last row: %q", plain)
	}
	if !strings.Contains(plain, "‹ on ›") {
		t.Fatalf("adjusted value must flow through the bindings: %q", plain)
	}
	if !strings.Contains(plain, "○ context length") || strings.Contains(plain, "› context length") {
		t.Fatalf("unselected rows must carry the idle marker: %q", plain)
	}

	first := strings.Index(plain, "context length")
	second := strings.Index(plain, "max tokens")
	third := strings.Index(plain, "thinking")
	if first < 0 || second < first || third < second {
		t.Fatalf("the sequence split must preserve row order: %q", plain)
	}
}

// TestSettingsOverlay_Good drives the wired Settings screen end to end: F2
// opens it in the overlay layer, a knob adjust edits a.cfg live and persists
// to the real preference store on ctrl+s, and esc closes it cleanly.
func TestSettingsOverlay_Good(t *testing.T) {
	medium := coreio.NewMockMedium()
	opened := openPreferences(medium, appConfigPath)
	if !opened.OK {
		t.Fatalf("open preferences: %v", opened.Value)
	}
	preferences := opened.Value.(preferenceStore)
	a := newApp("", 0, 4096)
	a.attachPreferences(preferences)
	m, _ := a.Update(tea.WindowSizeMsg{Width: 100, Height: 30})
	a = m.(app)

	// F2 opens the Settings overlay.
	m, _ = a.Update(testKeyPress(tea.KeyF2))
	a = m.(app)
	if a.activeOverlay != overlaySettings {
		t.Fatalf("F2 did not open the settings overlay: overlay=%d", a.activeOverlay)
	}

	// The form renders in the overlay layer — the KV-cache hint is unique to
	// the settings form (it is not in the footer or any other screen).
	if plain := ansi.Strip(a.View().Content); !strings.Contains(plain, "KV cache size") {
		t.Fatalf("settings form not rendered in the overlay layer:\n%s", plain)
	}

	// Navigate to the max-tokens row and bump it one step; the edit lands on
	// a.cfg immediately, exactly as the value hint promises.
	m, _ = a.Update(testKeyPress(tea.KeyDown))
	a = m.(app)
	m, _ = a.Update(testKeyPress(tea.KeyRight))
	a = m.(app)
	if a.cfg.maxTokens() != 8192 {
		t.Fatalf("adjust did not raise max tokens live: %d", a.cfg.maxTokens())
	}

	// Ctrl+S commits the generation knobs through the same store the
	// inspector writes; reopening the store proves the round-trip.
	m, _ = a.Update(testModifiedKeyPress('s', tea.ModCtrl))
	a = m.(app)
	if a.errText != "" {
		t.Fatalf("ctrl+s reported an error: %q", a.errText)
	}
	reopened := openPreferences(medium, appConfigPath)
	if !reopened.OK {
		t.Fatalf("reopen preferences: %v", reopened.Value)
	}
	if values := reopened.Value.(preferenceStore).Values(); values.MaxTokens != 8192 {
		t.Fatalf("adjust did not round-trip through the store: %#v", values)
	}

	// Esc closes the overlay without disturbing the live edit.
	m, _ = a.Update(testKeyPress(tea.KeyEsc))
	a = m.(app)
	if a.activeOverlay != overlayNone {
		t.Fatalf("esc did not close the settings overlay: overlay=%d", a.activeOverlay)
	}
	if a.cfg.maxTokens() != 8192 {
		t.Fatalf("esc reverted the live edit: %d", a.cfg.maxTokens())
	}
}

// TestSettingsOverlay_Bad proves the commit path fails loudly when no
// preference store is connected — the overlay stays open and surfaces the
// reason rather than silently discarding the change.
func TestSettingsOverlay_Bad(t *testing.T) {
	a := newApp("", 0, 4096)
	m, _ := a.Update(testKeyPress(tea.KeyF2))
	a = m.(app)
	if a.activeOverlay != overlaySettings {
		t.Fatalf("F2 did not open the settings overlay: overlay=%d", a.activeOverlay)
	}
	m, _ = a.Update(testModifiedKeyPress('s', tea.ModCtrl))
	a = m.(app)
	if a.errText == "" {
		t.Fatal("ctrl+s without a store did not surface an error")
	}
	if a.activeOverlay != overlaySettings {
		t.Fatalf("a failed commit closed the overlay: overlay=%d", a.activeOverlay)
	}
}
