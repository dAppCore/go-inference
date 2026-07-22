// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"
)

func TestRenderTools_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	view := renderTools(newTools(), 100, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "tools" {
		t.Fatalf("tab must open with its title line: %q", lines[0])
	}
	if strings.TrimSpace(lines[1]) != "" {
		t.Fatalf("title must be followed by a blank separator: %q", lines[1])
	}
	if !strings.Contains(plain, "function calling: disabled — replies are plain chat") {
		t.Fatalf("fresh state must render as disabled: %q", plain)
	}
	for _, text := range []string{
		"get_time  Get the current local date and time.",
		"word_count  Count the words in the given text.",
	} {
		if !strings.Contains(plain, text) {
			t.Fatalf("tool row missing %q: %q", text, plain)
		}
	}

	first := -1
	for index, line := range lines {
		if strings.Contains(line, "get_time") {
			first = index
			break
		}
	}
	if first < 0 || first+1 >= len(lines) {
		t.Fatalf("tool rows not found in the rendered lines: %q", plain)
	}
	if !strings.Contains(lines[first+1], "word_count") {
		t.Fatalf("tool rows must sit on adjacent lines: %q then %q", lines[first], lines[first+1])
	}

	if strings.Contains(plain, "recent calls") {
		t.Fatalf("a run-free state must not render the recent-calls section: %q", plain)
	}
	if !strings.Contains(plain, "enter toggles · calls appear dim in the chat transcript") {
		t.Fatalf("tab missing the key footer: %q", plain)
	}

	for index, line := range lines {
		if got := lipgloss.Width(line); got > 100 {
			t.Fatalf("line %d width = %d, exceeds 100: %q", index, got, line)
		}
	}
}

func TestRenderTools_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	for _, width := range []int{0, -4} {
		if got := renderTools(newTools(), width, styles); got != "" {
			t.Fatalf("renderTools(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderTools_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	state := newTools()
	state.setEnabled(true)
	state.lastRun = []string{
		"get_time → one", "get_time → two", "get_time → three",
		"get_time → four", "get_time → five", "word_count → 3 words",
	}
	view := renderTools(state, 100, styles)
	plain := ansi.Strip(view)

	if !strings.Contains(plain, "function calling: enabled — declarations ride the system turn; calls run locally and feed back") {
		t.Fatalf("enabled state must render its full description: %q", plain)
	}
	if !strings.Contains(plain, "recent calls") {
		t.Fatalf("receipts must surface the recent-calls section: %q", plain)
	}
	if strings.Contains(plain, "get_time → one") {
		t.Fatalf("only the last five receipts may render: %q", plain)
	}
	for _, receipt := range []string{"get_time → two", "get_time → five", "word_count → 3 words"} {
		if !strings.Contains(plain, receipt) {
			t.Fatalf("receipt %q missing from the recent-calls section: %q", receipt, plain)
		}
	}
	if strings.Index(plain, "recent calls") > strings.Index(plain, "get_time → two") {
		t.Fatalf("receipts must follow their section heading: %q", plain)
	}
}
