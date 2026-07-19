// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"strings"
	"testing"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

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
