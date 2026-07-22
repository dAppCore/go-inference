// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	core "dappco.re/go"
)

func TestWorkEditor_KeyboardAndLayout(t *testing.T) {
	editor := newWorkEditor(workItemRecord{Title: "Initial", Task: "Full task", Repo: "/tmp/repo with spaces"})
	if editor.title.Value() != "Initial" || editor.task.Value() != "Full task" || editor.repository.Value() != "/tmp/repo with spaces" {
		t.Fatalf("editor values = %#v", editor)
	}
	for _, message := range []tea.KeyMsg{{Type: tea.KeyTab}, {Type: tea.KeyTab}, {Type: tea.KeyShiftTab}} {
		editor.Update(message)
	}
	for _, width := range []int{48, 120} {
		view := editor.View(width, 16, newUIStyles(midnightTheme()))
		for line, text := range strings.Split(view, "\n") {
			if got := lipgloss.Width(text); got > width {
				t.Fatalf("width %d line %d overflows at %d", width, line, got)
			}
		}
		if !strings.Contains(view, "Work title") || !strings.Contains(view, "Repository") || !strings.Contains(view, "ctrl+s saves") {
			t.Fatalf("width %d editor view:\n%s", width, view)
		}
	}
}

func TestWorkEditor_MultilineValidation(t *testing.T) {
	editor := newWorkEditor(workItemRecord{Title: "Title", Task: "First line", Repo: "/tmp/repo"})
	editor.focus = 1
	editor.applyFocus()
	editor.Update(tea.KeyMsg{Type: tea.KeyEnter})
	if !strings.Contains(editor.task.Value(), "\n") {
		t.Fatalf("task enter did not create a newline: %q", editor.task.Value())
	}
	app := newApp("", 0, 64)
	repository := openTestDuckRepository(t)
	defer closeTestDuckRepository(t, repository)
	if result := app.attachWork(repository, newUnavailableAgentProvider(defaultAgentUnavailableReason)); !result.OK {
		t.Fatalf("attachWork: %v", result.Value)
	}
	app.workEditor = newWorkEditor(workItemRecord{})
	if result := app.saveWorkEditor(); result.OK || app.workEditor.validation == "" {
		t.Fatalf("empty editor result=%#v validation=%q", result, app.workEditor.validation)
	}
}

func TestRenderWorkEditor_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	editor := newWorkEditor(workItemRecord{Title: "Initial", Task: "Full task", Repo: "/tmp/repo"})
	view := editor.View(48, 16, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "Create Work" {
		t.Fatalf("fresh editor must open with the create title: %q", lines[0])
	}
	if strings.TrimSpace(lines[1]) != "" {
		t.Fatalf("title must be followed by a blank separator: %q", lines[1])
	}
	if strings.TrimSpace(lines[2]) != "Work title" {
		t.Fatalf("the first caption must sit beneath the separator: %q", lines[2])
	}
	if !strings.Contains(lines[3], "Initial") {
		t.Fatalf("the title input must sit directly beneath its caption: %q", lines[3])
	}
	for _, text := range []string{"Full task", "Repository", "tab changes field · ctrl+s saves · esc cancels"} {
		if !strings.Contains(plain, text) {
			t.Fatalf("editor missing %q: %q", text, plain)
		}
	}
	for index, line := range lines {
		if got := lipgloss.Width(line); got > 48 {
			t.Fatalf("line %d width = %d, exceeds 48: %q", index, got, line)
		}
	}
}

func TestRenderWorkEditor_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	editor := newWorkEditor(workItemRecord{})
	for _, width := range []int{0, -4} {
		if got := editor.View(width, 16, styles); got != "" {
			t.Fatalf("View(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderWorkEditor_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	editor := newWorkEditor(workItemRecord{ID: "work-1", Title: "Existing", Task: "T", Repo: "/tmp/repo"})
	editor.validation = "title is required"
	plain := ansi.Strip(editor.View(48, 18, styles))

	if !strings.Contains(plain, "Edit Work") || strings.Contains(plain, "Create Work") {
		t.Fatalf("an editing record must swap in the edit title: %q", plain)
	}
	if !strings.Contains(plain, "title is required") {
		t.Fatalf("a rejected save must render its validation line: %q", plain)
	}
	if strings.Index(plain, "ctrl+s saves") > strings.Index(plain, "title is required") {
		t.Fatalf("validation must follow the key footer: %q", plain)
	}
}

func TestLaunchReview_ConfirmCancelAndRedaction(t *testing.T) {
	overlay := newLaunchReviewOverlay(agentReview{
		Feature: agentFeatureDispatch, Title: "Review native agent launch",
		Body:    "Command: codex exec --token [REDACTED] --model gpt-5\nSource: /tmp/repo",
		Warning: "Native agent execution has host access.", ConfirmRequired: true,
	}, "codex", "gpt-5")
	if view := overlay.View(72, 18, newUIStyles(midnightTheme())); !strings.Contains(view, "Command: codex exec --token [REDACTED] --model gpt-5") || !strings.Contains(view, "Native agent execution has host access.") || strings.Contains(view, "token secret") {
		t.Fatalf("launch review redaction:\n%s", view)
	}
	if accepted := overlay.Update(tea.KeyMsg{Type: tea.KeyEnter}); !accepted {
		t.Fatal("enter did not confirm launch review")
	}
	if !overlay.confirmed {
		t.Fatal("launch review confirmation was not retained")
	}
	if cancelled := overlay.Update(tea.KeyMsg{Type: tea.KeyEsc}); cancelled {
		t.Fatal("escape should cancel rather than confirm")
	}
}

func TestLaunchReview_ProviderAndModelSelection(t *testing.T) {
	overlay := newAgentSelectionOverlay("codex", "gpt-5")
	overlay.Update(tea.KeyMsg{Type: tea.KeyTab})
	overlay.Update(tea.KeyMsg{Type: tea.KeyBackspace})
	overlay.Update(tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{'x'}})
	provider, model := overlay.selection()
	if provider != "codex" || model != "gpt-x" {
		t.Fatalf("launch selection = %q/%q", provider, model)
	}
}

func TestAgentAnswerAndAcceptanceOverlays_RequireExplicitConfirmation(t *testing.T) {
	answer := newAgentAnswerOverlay("run-9", "question-2", "Which target?")
	answer.input.SetValue("release")
	if accepted := answer.Update(tea.KeyMsg{Type: tea.KeyEnter}); !accepted || answer.answer() != "release" {
		t.Fatalf("answer confirmation = %v/%q", accepted, answer.answer())
	}
	review := agentReview{Feature: agentFeatureChangesReview, Title: "Review agent changes", Body: "Diff:\n+change", Warning: "No validation command is configured; acknowledge this explicitly.", ConfirmRequired: true, NeedsAcknowledgement: true, AcceptanceAllowed: true}
	confirm := newChangeAcceptanceOverlay(review)
	if confirm.Update(tea.KeyMsg{Type: tea.KeyEnter}) {
		t.Fatal("accepted without acknowledgement")
	}
	confirm.acknowledged = true
	if confirm.Update(tea.KeyMsg{Type: tea.KeyEnter}) || !confirm.final {
		t.Fatal("acknowledged review did not open final confirmation")
	}
	if !confirm.Update(tea.KeyMsg{Type: tea.KeyEnter}) {
		t.Fatal("final confirmation was not explicit")
	}
}

func TestAgentOverlay_ChangeReviewViewportScrollsWithoutDroppingContent(t *testing.T) {
	lines := make([]string, 80)
	for index := range lines {
		lines[index] = core.Sprintf("review line %03d", index+1)
	}
	overlay := newChangeAcceptanceOverlay(agentReview{
		Feature: agentFeatureChangesReview, Title: "Review agent changes",
		Body: core.Join("\n", lines...), AcceptanceAllowed: true,
	})
	overlay.View(48, 12, newUIStyles(midnightTheme()))
	if got := overlay.viewport.TotalLineCount(); got != len(lines) {
		t.Fatalf("viewport line count = %d, want %d", got, len(lines))
	}
	lineOffset := overlay.viewport.YOffset
	overlay.Update(tea.KeyMsg{Type: tea.KeyDown})
	if overlay.viewport.YOffset <= lineOffset {
		t.Fatalf("line down offset = %d, want > %d", overlay.viewport.YOffset, lineOffset)
	}
	pageOffset := overlay.viewport.YOffset
	overlay.Update(tea.KeyMsg{Type: tea.KeyPgDown})
	if overlay.viewport.YOffset <= pageOffset {
		t.Fatalf("page down offset = %d, want > %d", overlay.viewport.YOffset, pageOffset)
	}
	if got := overlay.viewport.TotalLineCount(); got != len(lines) {
		t.Fatalf("scrolled viewport line count = %d, want %d", got, len(lines))
	}
}

func TestRenderAgentAnswer_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newAgentAnswerOverlay("run-9", "question-2", "Which target?")
	view := overlay.View(48, 14, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "Answer agent question" {
		t.Fatalf("overlay must open with its title line: %q", lines[0])
	}
	if strings.TrimSpace(lines[1]) != "" {
		t.Fatalf("title must be followed by a blank separator: %q", lines[1])
	}
	if strings.TrimSpace(lines[2]) != "Which target?" {
		t.Fatalf("the question must sit beneath the separator: %q", lines[2])
	}
	if !strings.Contains(plain, "Type the answer for this native run") {
		t.Fatalf("overlay missing the textarea placeholder: %q", plain)
	}
	if !strings.Contains(plain, "enter submits · esc cancels") {
		t.Fatalf("overlay missing the key footer: %q", plain)
	}
	for index, line := range lines {
		if got := lipgloss.Width(line); got > 48 {
			t.Fatalf("line %d width = %d, exceeds 48: %q", index, got, line)
		}
	}
}

func TestRenderAgentAnswer_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newAgentAnswerOverlay("run-9", "question-2", "Which target?")
	for _, width := range []int{0, -4} {
		if got := overlay.View(width, 14, styles); got != "" {
			t.Fatalf("View(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderAgentAnswer_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	// A blank question (everything trims away) drops the question row
	// rather than rendering an empty paragraph.
	overlay := newAgentAnswerOverlay("run-9", "question-2", "   ")
	plain := ansi.Strip(overlay.View(48, 14, styles))
	lines := strings.Split(plain, "\n")
	if strings.TrimSpace(lines[0]) != "Answer agent question" {
		t.Fatalf("overlay must open with its title line: %q", lines[0])
	}
	if !strings.Contains(plain, "enter submits · esc cancels") {
		t.Fatalf("question-free overlay must keep the key footer: %q", plain)
	}
}

func TestRenderChangeReview_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newChangeAcceptanceOverlay(agentReview{
		Feature: agentFeatureChangesReview, Title: "Review agent changes",
		Body: "Diff:\n+change", Warning: "No validation command is configured; acknowledge this explicitly.",
		ConfirmRequired: true, NeedsAcknowledgement: true, AcceptanceAllowed: true,
	})
	view := overlay.View(70, 16, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "Review agent changes" {
		t.Fatalf("overlay must open with its title line: %q", lines[0])
	}
	if !strings.Contains(plain, "No validation command is configured; acknowledge this explicitly.") {
		t.Fatalf("overlay missing the warning: %q", plain)
	}
	for _, text := range []string{"Diff:", "+change"} {
		if !strings.Contains(plain, text) {
			t.Fatalf("viewport body missing %q: %q", text, plain)
		}
	}
	if !strings.Contains(plain, "a acknowledges no validation · enter continues · esc cancels") {
		t.Fatalf("unacknowledged review must show the acknowledge prompt: %q", plain)
	}
	for index, line := range lines {
		if got := lipgloss.Width(line); got > 70 {
			t.Fatalf("line %d width = %d, exceeds 70: %q", index, got, line)
		}
	}
}

func TestRenderChangeReview_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newChangeAcceptanceOverlay(agentReview{Title: "Review agent changes"})
	for _, width := range []int{0, -4} {
		if got := overlay.View(width, 16, styles); got != "" {
			t.Fatalf("View(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderChangeReview_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newChangeAcceptanceOverlay(agentReview{
		Feature: agentFeatureChangesReview, Title: "Review agent changes",
		Body: "Diff:\n+change", AcceptanceAllowed: true,
	})
	overlay.Update(tea.KeyMsg{Type: tea.KeyEnter}) // arm the final confirmation
	plain := ansi.Strip(overlay.View(70, 16, styles))

	if !strings.Contains(plain, "enter applies this exact reviewed receipt · esc cancels") {
		t.Fatalf("armed review must show the apply prompt: %q", plain)
	}
	if strings.Contains(plain, "a acknowledges") || strings.Contains(plain, "enter continues") {
		t.Fatalf("armed review must show only the apply prompt: %q", plain)
	}
	if lines := strings.Split(plain, "\n"); !strings.Contains(lines[1], "Diff:") {
		t.Fatalf("a warning-free review must put the viewport directly beneath the title: %q", lines[1])
	}
}

func TestRenderLaunchReview_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newLaunchReviewOverlay(agentReview{
		Feature: agentFeatureDispatch, Title: "Review native agent launch",
		Body:    "Command: codex exec --token [REDACTED] --model gpt-5\nSource: /tmp/repo",
		Warning: "Native agent execution has host access.", ConfirmRequired: true,
	}, "codex", "gpt-5")
	view := overlay.View(72, 18, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "Review native agent launch" {
		t.Fatalf("overlay must open with its title line: %q", lines[0])
	}
	for _, text := range []string{
		"Provider: codex", "Model: gpt-5",
		"Native agent execution has host access.",
		"Command: codex exec --token [REDACTED] --model gpt-5",
		"Source: /tmp/repo",
		"enter confirms · esc cancels",
	} {
		if !strings.Contains(plain, text) {
			t.Fatalf("read-only review missing %q: %q", text, plain)
		}
	}
	if strings.Index(plain, "Model: gpt-5") > strings.Index(plain, "host access") {
		t.Fatalf("the provider/model band must render above the warning: %q", plain)
	}
	for index, line := range lines {
		if got := lipgloss.Width(line); got > 72 {
			t.Fatalf("line %d width = %d, exceeds 72: %q", index, got, line)
		}
	}
}

func TestRenderLaunchReview_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newLaunchReviewOverlay(agentReview{Title: "Review native agent launch"}, "codex", "gpt-5")
	for _, width := range []int{0, -4} {
		if got := overlay.View(width, 18, styles); got != "" {
			t.Fatalf("View(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderLaunchReview_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	// The editable selection shape: captions around live inputs, the body
	// (with its blank line preserved between rows) and warning beneath.
	overlay := newAgentSelectionOverlay("codex", "gpt-5")
	overlay.review.Body = "Registration: pending\n\nQueue: ready"
	view := overlay.View(60, 18, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "Select native agent" {
		t.Fatalf("selection overlay must open with its title: %q", lines[0])
	}
	if strings.TrimSpace(lines[2]) != "Provider" {
		t.Fatalf("the provider caption must sit beneath the separator: %q", lines[2])
	}
	if !strings.Contains(lines[3], "codex") {
		t.Fatalf("the provider input must sit directly beneath its caption: %q", lines[3])
	}
	if strings.TrimSpace(lines[4]) != "Model" {
		t.Fatalf("the model caption must sit between the inputs: %q", lines[4])
	}
	if !strings.Contains(lines[5], "gpt-5") {
		t.Fatalf("the model input must sit directly beneath its caption: %q", lines[5])
	}
	row := -1
	for index, line := range lines {
		if strings.Contains(line, "Registration: pending") {
			row = index
			break
		}
	}
	if row < 0 || row+2 >= len(lines) {
		t.Fatalf("selection overlay missing the body rows: %q", plain)
	}
	if strings.TrimSpace(lines[row+1]) != "" || !strings.Contains(lines[row+2], "Queue: ready") {
		t.Fatalf("a blank body line must survive between rows: %q then %q", lines[row+1], lines[row+2])
	}
	if !strings.Contains(plain, "Select the provider and model before reviewing project") {
		t.Fatalf("selection overlay missing its warning: %q", plain)
	}
	if !strings.Contains(plain, "tab selects provider/model · enter confirms · esc cancels") {
		t.Fatalf("selection overlay missing the key footer: %q", plain)
	}
}

type launchReviewProvider struct {
	caps           []agentCapability
	reviews        []agentReview
	reviewRequests []agentReviewRequest
	runs           []agentRequest
}

func (provider *launchReviewProvider) Capabilities() []agentCapability {
	return append([]agentCapability(nil), provider.caps...)
}
func (*launchReviewProvider) Snapshot(context.Context) core.Result { return core.Ok(agentSnapshot{}) }
func (provider *launchReviewProvider) Review(_ context.Context, request agentReviewRequest) core.Result {
	provider.reviewRequests = append(provider.reviewRequests, request)
	if len(provider.reviews) == 0 {
		return core.Fail(core.E("test.launchReviewProvider.Review", "no review fixture remains", nil))
	}
	review := provider.reviews[0]
	provider.reviews = provider.reviews[1:]
	return core.Ok(review)
}
func (provider *launchReviewProvider) Run(_ context.Context, request agentRequest) core.Result {
	provider.runs = append(provider.runs, request)
	if len(provider.runs) == 1 {
		return core.Ok(agentReview{Feature: agentFeatureDispatch, Title: "Review native agent launch", Body: "Command: codex exec --api-key [REDACTED]", Warning: "Native agent execution has host access.", ConfirmRequired: true})
	}
	return core.Ok(agentActionReceipt{Feature: agentFeatureDispatch, WorkID: request.WorkID, Status: "queued"})
}
func (*launchReviewProvider) Close() core.Result { return core.Ok(nil) }
