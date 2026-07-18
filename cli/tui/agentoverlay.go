// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	core "dappco.re/go"
)

// workEditor keeps Work creation and editing local to the TUI. It deliberately
// has no provider dependency: recording a Work must never start a Git service.
type workEditor struct {
	title      textinput.Model
	task       textarea.Model
	repository textinput.Model
	focus      int
	editingID  string
	validation string
}

func newWorkEditor(record workItemRecord) *workEditor {
	title := textinput.New()
	title.Prompt = ""
	title.Placeholder = "Work title"
	title.SetValue(record.Title)
	title.Focus()
	task := textarea.New()
	task.Prompt = ""
	task.Placeholder = "Describe the complete task"
	task.ShowLineNumbers = false
	task.SetHeight(4)
	task.SetValue(record.Task)
	repository := textinput.New()
	repository.Prompt = ""
	repository.Placeholder = "/path/to/repository"
	repository.SetValue(record.Repo)
	return &workEditor{title: title, task: task, repository: repository, editingID: record.ID}
}

func (editor *workEditor) Update(message tea.Msg) tea.Cmd {
	if editor == nil {
		return nil
	}
	if key, ok := message.(tea.KeyMsg); ok {
		switch key.String() {
		case "tab":
			editor.focus = (editor.focus + 1) % 3
		case "shift+tab":
			editor.focus = (editor.focus + 2) % 3
		default:
			return editor.updateFocused(message)
		}
		editor.applyFocus()
		return nil
	}
	return editor.updateFocused(message)
}

func (editor *workEditor) updateFocused(message tea.Msg) tea.Cmd {
	var command tea.Cmd
	switch editor.focus {
	case 0:
		editor.title, command = editor.title.Update(message)
	case 1:
		editor.task, command = editor.task.Update(message)
	default:
		editor.repository, command = editor.repository.Update(message)
	}
	return command
}

func (editor *workEditor) applyFocus() {
	if editor == nil {
		return
	}
	editor.title.Blur()
	editor.task.Blur()
	editor.repository.Blur()
	switch editor.focus {
	case 0:
		editor.title.Focus()
	case 1:
		editor.task.Focus()
	default:
		editor.repository.Focus()
	}
}

func (editor *workEditor) values() (string, string, string) {
	if editor == nil {
		return "", "", ""
	}
	return core.Trim(editor.title.Value()), core.Trim(editor.task.Value()), core.Trim(editor.repository.Value())
}

func (editor *workEditor) View(width, height int, styles uiStyles) string {
	if editor == nil {
		return ""
	}
	fieldWidth := max(12, width-6)
	editor.title.Width = fieldWidth
	editor.repository.Width = fieldWidth
	editor.task.SetWidth(fieldWidth)
	title := "Create Work"
	if editor.editingID != "" {
		title = "Edit Work"
	}
	return fitPane(core.Join("\n", title, "", "Work title", editor.title.View(), "", "Full task", editor.task.View(), "", "Repository", editor.repository.View(), "", "tab changes field · ctrl+s saves · esc cancels", editor.validation), width, height, styles.panel)
}

type launchReviewOverlay struct {
	review        agentReview
	provider      string
	model         string
	providerInput textinput.Model
	modelInput    textinput.Model
	focus         int
	editable      bool
	confirmed     bool
}

func newLaunchReviewOverlay(review agentReview, provider, model string) *launchReviewOverlay {
	provider = core.Trim(provider)
	model = core.Trim(model)
	providerInput := textinput.New()
	providerInput.Prompt = ""
	providerInput.Placeholder = "default provider"
	providerInput.SetValue(provider)
	providerInput.Focus()
	modelInput := textinput.New()
	modelInput.Prompt = ""
	modelInput.Placeholder = "default model"
	modelInput.SetValue(model)
	return &launchReviewOverlay{review: review, provider: provider, model: model, providerInput: providerInput, modelInput: modelInput}
}

func newAgentSelectionOverlay(provider, model string) *launchReviewOverlay {
	overlay := newLaunchReviewOverlay(agentReview{
		Feature: agentFeatureDispatch, Title: "Select native agent", ConfirmRequired: true,
		Warning: "Select the provider and model before reviewing project registration.",
	}, provider, model)
	overlay.editable = true
	return overlay
}

// Update returns true only for an explicit confirmation; Escape is always a
// cancellation and never starts a process.
func (overlay *launchReviewOverlay) Update(message tea.KeyMsg) bool {
	if overlay == nil {
		return false
	}
	switch message.String() {
	case "enter":
		overlay.provider, overlay.model = overlay.selection()
		overlay.confirmed = true
		return true
	case "esc":
		overlay.confirmed = false
		return false
	case "tab":
		if !overlay.editable {
			return false
		}
		overlay.focus = (overlay.focus + 1) % 2
		overlay.applyFocus()
		return false
	case "shift+tab":
		if !overlay.editable {
			return false
		}
		overlay.focus = (overlay.focus + 1) % 2
		overlay.applyFocus()
		return false
	}
	if !overlay.editable {
		return false
	}
	var command tea.Cmd
	if overlay.focus == 0 {
		overlay.providerInput, command = overlay.providerInput.Update(message)
	} else {
		overlay.modelInput, command = overlay.modelInput.Update(message)
	}
	_ = command
	overlay.provider, overlay.model = overlay.selection()
	return false
}

func (overlay *launchReviewOverlay) applyFocus() {
	if overlay == nil {
		return
	}
	overlay.providerInput.Blur()
	overlay.modelInput.Blur()
	if overlay.focus == 0 {
		overlay.providerInput.Focus()
	} else {
		overlay.modelInput.Focus()
	}
}

func (overlay *launchReviewOverlay) selection() (string, string) {
	if overlay == nil {
		return "", ""
	}
	return core.Trim(overlay.providerInput.Value()), core.Trim(overlay.modelInput.Value())
}

func (overlay *launchReviewOverlay) View(width, height int, styles uiStyles) string {
	if overlay == nil {
		return ""
	}
	provider, model := overlay.selection()
	if provider == "" {
		provider = "default provider"
	}
	if model == "" {
		model = "default model"
	}
	if !overlay.editable {
		return fitPane(core.Join("\n", overlay.review.Title, "", "Provider: "+provider, "Model: "+model, "", overlay.review.Warning, "", overlay.review.Body, "", "enter confirms · esc cancels"), width, height, styles.panel)
	}
	fieldWidth := max(12, width-6)
	overlay.providerInput.Width = fieldWidth
	overlay.modelInput.Width = fieldWidth
	return fitPane(core.Join("\n", overlay.review.Title, "", "Provider", overlay.providerInput.View(), "Model", overlay.modelInput.View(), "", overlay.review.Body, "", overlay.review.Warning, "", "tab selects provider/model · enter confirms · esc cancels"), width, height, styles.panel)
}

type agentActionMsg struct {
	operationID uint64
	feature     agentFeature
	stage       agentReviewStage
	request     agentRequest
	result      core.Result
}
