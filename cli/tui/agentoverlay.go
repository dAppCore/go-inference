// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	_ "embed"

	tea "dappco.re/go/html/tui"
	"dappco.re/go/html/tui/textarea"
	"dappco.re/go/html/tui/textinput"
	"dappco.re/go/html/tui/viewport"

	core "dappco.re/go"
	"dappco.re/go/html/ctml"
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

// workEditorCTML is the Work editor overlay's markup — see workeditor.ctml
// for the seams it exposes (the create/edit title split, the conditional
// validation line, class tokens).
//
//go:embed workeditor.ctml
var workEditorCTML []byte

// workEditorBindings selects the create-vs-edit title — both texts are
// static markup, so the mode is carried by which zero-or-one-row sequence
// holds the row, the selection-as-sequence-split idiom — and binds the
// save-validation error (zero-or-one row; an empty sequence renders
// nothing).
func workEditorBindings(editor *workEditor) ctml.Bindings {
	sequences := map[string][]map[string]any{
		"createTitle": {},
		"editTitle":   {},
		"validation":  {},
	}
	if editor != nil && editor.editingID != "" {
		sequences["editTitle"] = append(sequences["editTitle"], map[string]any{})
	} else {
		sequences["createTitle"] = append(sequences["createTitle"], map[string]any{})
	}
	if editor != nil && editor.validation != "" {
		sequences["validation"] = append(sequences["validation"], map[string]any{"text": editor.validation})
	}
	return ctml.Bindings{Sequences: sequences}
}

func (editor *workEditor) View(width, height int, styles uiStyles) string {
	if editor == nil {
		return ""
	}
	fieldWidth := max(12, width-6)
	editor.title.Width = fieldWidth
	editor.repository.Width = fieldWidth
	editor.task.SetWidth(fieldWidth)
	head, foot := renderOverlayFrame(workEditorCTML, width, styles, workEditorBindings(editor))
	return fitPane(core.Join("\n", head, editor.title.View(), "", "Full task", editor.task.View(), "", "Repository", editor.repository.View(), foot), width, height, styles.panel)
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

type agentAnswerOverlay struct {
	runID      string
	questionID string
	question   string
	input      textarea.Model
}

func newAgentAnswerOverlay(runID, questionID, question string) *agentAnswerOverlay {
	input := textarea.New()
	input.Prompt = ""
	input.Placeholder = "Type the answer for this native run"
	input.SetHeight(4)
	input.ShowLineNumbers = false
	input.Focus()
	return &agentAnswerOverlay{runID: core.Trim(runID), questionID: core.Trim(questionID), question: core.Trim(question), input: input}
}

func (overlay *agentAnswerOverlay) Update(message tea.KeyMsg) bool {
	if overlay == nil {
		return false
	}
	if message.String() == "enter" {
		return overlay.answer() != ""
	}
	var command tea.Cmd
	overlay.input, command = overlay.input.Update(message)
	_ = command
	return false
}

func (overlay *agentAnswerOverlay) answer() string {
	if overlay == nil {
		return ""
	}
	return core.Trim(overlay.input.Value())
}

// agentAnswerCTML is the answer overlay's markup — see agentanswer.ctml
// for the seams it exposes (the question row, class tokens).
//
//go:embed agentanswer.ctml
var agentAnswerCTML []byte

// agentAnswerBindings binds the pending question — a lone dynamic value
// riding a one-row sequence.
func agentAnswerBindings(overlay *agentAnswerOverlay) ctml.Bindings {
	question := ""
	if overlay != nil {
		question = overlay.question
	}
	return ctml.Bindings{Sequences: map[string][]map[string]any{
		"question": {{"text": question}},
	}}
}

func (overlay *agentAnswerOverlay) View(width, height int, styles uiStyles) string {
	if overlay == nil {
		return ""
	}
	overlay.input.SetWidth(max(12, width-6))
	head, foot := renderOverlayFrame(agentAnswerCTML, width, styles, agentAnswerBindings(overlay))
	return fitPane(core.Join("\n", head, "", overlay.input.View(), foot), width, height, styles.panel)
}

type changeAcceptanceOverlay struct {
	review       agentReview
	acknowledged bool
	final        bool
	viewport     viewport.Model
}

func newChangeAcceptanceOverlay(review agentReview) *changeAcceptanceOverlay {
	return &changeAcceptanceOverlay{review: review, viewport: viewport.New(1, 1)}
}

func (overlay *changeAcceptanceOverlay) Update(message tea.KeyMsg) bool {
	if overlay == nil {
		return false
	}
	if message.String() == "a" {
		if overlay.review.NeedsAcknowledgement {
			overlay.acknowledged = true
		}
		return false
	}
	if message.String() != "enter" {
		var command tea.Cmd
		overlay.viewport, command = overlay.viewport.Update(message)
		_ = command
		return false
	}
	if !overlay.review.AcceptanceAllowed {
		return false
	}
	if overlay.review.NeedsAcknowledgement && !overlay.acknowledged {
		return false
	}
	if !overlay.final {
		overlay.final = true
		return false
	}
	return true
}

// changeReviewCTML is the change-acceptance overlay's markup — see
// changereview.ctml for the seams it exposes (the title row, the
// conditional warning, the three-way prompt split, class tokens).
//
//go:embed changereview.ctml
var changeReviewCTML []byte

// changeAcceptanceBindings binds the review title and warning, and selects
// the gate prompt: all three prompt texts are static markup, so the gate
// stage is carried by which zero-or-one-row sequence holds the row, the
// selection-as-sequence-split idiom.
func changeAcceptanceBindings(overlay *changeAcceptanceOverlay) ctml.Bindings {
	sequences := map[string][]map[string]any{
		"review":            {},
		"warn":              {},
		"promptContinue":    {},
		"promptAcknowledge": {},
		"promptApply":       {},
	}
	if overlay != nil {
		sequences["review"] = append(sequences["review"], map[string]any{"title": overlay.review.Title})
		if overlay.review.Warning != "" {
			sequences["warn"] = append(sequences["warn"], map[string]any{"text": overlay.review.Warning})
		}
		switch {
		case overlay.final:
			sequences["promptApply"] = append(sequences["promptApply"], map[string]any{})
		case overlay.review.NeedsAcknowledgement && !overlay.acknowledged:
			sequences["promptAcknowledge"] = append(sequences["promptAcknowledge"], map[string]any{})
		default:
			sequences["promptContinue"] = append(sequences["promptContinue"], map[string]any{})
		}
	}
	return ctml.Bindings{Sequences: sequences}
}

func (overlay *changeAcceptanceOverlay) View(width, height int, styles uiStyles) string {
	if overlay == nil {
		return ""
	}
	overlay.viewport.Width, overlay.viewport.Height = max(1, width-4), max(1, height-6)
	overlay.viewport.SetContent(overlay.review.Body)
	head, foot := renderOverlayFrame(changeReviewCTML, width, styles, changeAcceptanceBindings(overlay))
	return fitPane(core.Join("\n", head, overlay.viewport.View(), foot), width, height, styles.panel)
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

// launchReviewCTML and agentSelectCTML are the launch-review overlay's two
// markup shapes — launchreview.ctml (read-only receipt review, full HCF)
// and agentselect.ctml (editable provider/model selection, HF around the
// inputs); see each file for the seams it exposes.
//
//go:embed launchreview.ctml
var launchReviewCTML []byte

//go:embed agentselect.ctml
var agentSelectCTML []byte

// reviewBodyRows splits a multi-line receipt body into one row per line —
// a bound value cannot carry a line break through an inline run, so each
// row closes with <br> in the markup; empty lines ride as empty rows.
func reviewBodyRows(body string) []map[string]any {
	rows := []map[string]any{}
	for _, line := range core.Split(body, "\n") {
		rows = append(rows, map[string]any{"line": line})
	}
	return rows
}

// launchReviewBindings binds the read-only shape: the title with the
// defaulted provider/model pair (one row), the conditional warning, and
// the receipt body lines.
func launchReviewBindings(overlay *launchReviewOverlay, provider, model string) ctml.Bindings {
	sequences := map[string][]map[string]any{
		"review": {},
		"warn":   {},
		"body":   {},
	}
	if overlay != nil {
		sequences["review"] = append(sequences["review"], map[string]any{
			"title": overlay.review.Title, "provider": provider, "model": model,
		})
		if overlay.review.Warning != "" {
			sequences["warn"] = append(sequences["warn"], map[string]any{"text": overlay.review.Warning})
		}
		sequences["body"] = reviewBodyRows(overlay.review.Body)
	}
	return ctml.Bindings{Sequences: sequences}
}

// agentSelectionBindings binds the editable shape: the title (one row),
// the receipt body lines, and the conditional warning.
func agentSelectionBindings(overlay *launchReviewOverlay) ctml.Bindings {
	sequences := map[string][]map[string]any{
		"review": {},
		"body":   {},
		"warn":   {},
	}
	if overlay != nil {
		sequences["review"] = append(sequences["review"], map[string]any{"title": overlay.review.Title})
		sequences["body"] = reviewBodyRows(overlay.review.Body)
		if overlay.review.Warning != "" {
			sequences["warn"] = append(sequences["warn"], map[string]any{"text": overlay.review.Warning})
		}
	}
	return ctml.Bindings{Sequences: sequences}
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
		return fitPane(renderOverlayLayout(launchReviewCTML, width, styles, launchReviewBindings(overlay, provider, model)), width, height, styles.panel)
	}
	fieldWidth := max(12, width-6)
	overlay.providerInput.Width = fieldWidth
	overlay.modelInput.Width = fieldWidth
	head, foot := renderOverlayFrame(agentSelectCTML, width, styles, agentSelectionBindings(overlay))
	return fitPane(core.Join("\n", head, overlay.providerInput.View(), "Model", overlay.modelInput.View(), foot), width, height, styles.panel)
}

type agentActionMsg struct {
	operationID uint64
	feature     agentFeature
	stage       agentReviewStage
	request     agentRequest
	result      core.Result
}
