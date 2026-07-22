// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

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
	if !strings.Contains(plain, "↑/↓ select · ←/→ change · values apply as hinted") {
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
