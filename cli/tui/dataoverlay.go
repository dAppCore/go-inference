// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
)

// dataItemEditor is the edit-as-derived editor seam — two text areas
// (Prompt/Response) pre-filled from the selected item, Tab cycles focus,
// Ctrl+S saves as a new derived item. Mirrors workEditor's shape exactly
// (agentoverlay.go): no store dependency of its own — saving goes through
// dataPanel.EditAsDerived via app.saveDataEditor, exactly as workEditor's
// values() feed app.saveWorkEditor.
type dataItemEditor struct {
	original dataset.Item
	prompt   textarea.Model
	response textarea.Model
	focus    int
}

func newDataItemEditor(item dataset.Item) *dataItemEditor {
	initialPrompt, initialResponse, _ := dataItemExchange(item)
	prompt := textarea.New()
	prompt.Prompt = ""
	prompt.Placeholder = "Prompt"
	prompt.ShowLineNumbers = false
	prompt.SetHeight(5)
	prompt.SetValue(initialPrompt)
	prompt.Focus()
	response := textarea.New()
	response.Prompt = ""
	response.Placeholder = "Response"
	response.ShowLineNumbers = false
	response.SetHeight(8)
	response.SetValue(initialResponse)
	return &dataItemEditor{original: item, prompt: prompt, response: response}
}

func (editor *dataItemEditor) Update(message tea.Msg) tea.Cmd {
	if editor == nil {
		return nil
	}
	if key, ok := message.(tea.KeyMsg); ok {
		switch key.String() {
		case "tab", "shift+tab":
			editor.focus = (editor.focus + 1) % 2
			editor.applyFocus()
			return nil
		}
	}
	return editor.updateFocused(message)
}

func (editor *dataItemEditor) updateFocused(message tea.Msg) tea.Cmd {
	var command tea.Cmd
	if editor.focus == 0 {
		editor.prompt, command = editor.prompt.Update(message)
	} else {
		editor.response, command = editor.response.Update(message)
	}
	return command
}

func (editor *dataItemEditor) applyFocus() {
	if editor == nil {
		return
	}
	editor.prompt.Blur()
	editor.response.Blur()
	if editor.focus == 0 {
		editor.prompt.Focus()
	} else {
		editor.response.Focus()
	}
}

func (editor *dataItemEditor) values() (string, string) {
	if editor == nil {
		return "", ""
	}
	return editor.prompt.Value(), editor.response.Value()
}

func (editor *dataItemEditor) View(width, height int, styles uiStyles) string {
	if editor == nil {
		return ""
	}
	fieldWidth := max(12, width-6)
	editor.prompt.SetWidth(fieldWidth)
	editor.response.SetWidth(fieldWidth)
	promptLabel := "Prompt"
	if editor.original.Kind == dataset.KindMessages {
		promptLabel = "Prompt (context only — earlier turns are kept as-is)"
	}
	return fitPane(core.Join("\n",
		"Edit as derived", "",
		promptLabel, editor.prompt.View(), "",
		"Response", editor.response.View(), "",
		"tab changes field · ctrl+s saves as a new derived item · esc cancels",
	), width, height, styles.panel)
}

// dataNoteOverlay collects one freeform line — a quarantine-clear
// justification or a tag label — for a single item (itemID set) or as
// the shared note/label a bulk action's confirm overlay applies to every
// filtered item (itemID empty). Update only reports a submit on Enter
// with a non-empty trimmed value — the note-required guard lives here,
// not in the caller, so an empty Enter is silently ignored rather than
// producing an empty-note write.
type dataNoteOverlay struct {
	action dataAction
	itemID string
	title  string
	prompt string
	input  textinput.Model
}

func newDataNoteOverlay(action dataAction, itemID, title, prompt, placeholder string) *dataNoteOverlay {
	input := textinput.New()
	input.Prompt = ""
	input.Placeholder = placeholder
	input.Focus()
	return &dataNoteOverlay{action: action, itemID: itemID, title: title, prompt: prompt, input: input}
}

// Update reports whether key completed a submit — Enter with a non-empty
// trimmed value. Any other key is forwarded to the text input.
func (overlay *dataNoteOverlay) Update(message tea.KeyMsg) bool {
	if overlay == nil {
		return false
	}
	if message.String() == "enter" {
		return core.Trim(overlay.input.Value()) != ""
	}
	var command tea.Cmd
	overlay.input, command = overlay.input.Update(message)
	_ = command
	return false
}

func (overlay *dataNoteOverlay) Value() string {
	if overlay == nil {
		return ""
	}
	return core.Trim(overlay.input.Value())
}

// Bulk reports whether this note is the shared note for a bulk action
// (itemID empty) rather than a single item's.
func (overlay *dataNoteOverlay) Bulk() bool {
	return overlay != nil && overlay.itemID == ""
}

func (overlay *dataNoteOverlay) View(width, height int, styles uiStyles) string {
	if overlay == nil {
		return ""
	}
	overlay.input.Width = max(12, width-6)
	return fitPane(core.Join("\n", overlay.title, "", overlay.prompt, overlay.input.View(), "", "enter submits · esc cancels"), width, height, styles.panel)
}

// dataBulkOverlay gates a bulk-apply-to-current-filter action behind an
// explicit two-phase confirm (mirrors changeAcceptanceOverlay's arm/
// confirm shape, agentoverlay.go) showing the exact item count about to
// be written — "no confirm, no writes": Confirm only returns true on the
// SECOND Enter, and every path that dismisses the overlay without
// reaching that (Escape, switching panels, quitting) never calls
// dataPanel.BulkApply at all.
type dataBulkOverlay struct {
	action dataAction
	count  int
	note   string
	armed  bool
}

func newDataBulkOverlay(action dataAction, count int, note string) *dataBulkOverlay {
	return &dataBulkOverlay{action: action, count: count, note: core.Trim(note)}
}

// Confirm reports whether key is the second, confirming Enter. Any other
// key is ignored — the overlay deliberately consumes input until an
// explicit confirm or cancel, matching changeAcceptanceOverlay's
// precedent.
func (overlay *dataBulkOverlay) Confirm(key string) bool {
	if overlay == nil || key != "enter" {
		return false
	}
	if !overlay.armed {
		overlay.armed = true
		return false
	}
	return true
}

func (overlay *dataBulkOverlay) View(width, height int, styles uiStyles) string {
	if overlay == nil {
		return ""
	}
	prompt := "enter continues · esc cancels"
	if overlay.armed {
		prompt = "enter applies this action to every listed item · esc cancels"
	}
	lines := []string{
		"Bulk " + overlay.action.title(),
		"",
		core.Sprintf("This will apply to %d item(s) matching the current filter.", overlay.count),
	}
	if overlay.note != "" {
		lines = append(lines, "", "Note: "+overlay.note)
	}
	lines = append(lines, "", prompt)
	return fitPane(core.Join("\n", lines...), width, height, styles.panel)
}

// dataFilterOverlay edits the Data panel's structural filter as one line
// of text in parseDataFilterExpr's grammar, pre-filled from the panel's
// current filter (FilterExpr) — unlike dataNoteOverlay, an empty value is
// a valid submit (it clears every filter dimension).
type dataFilterOverlay struct {
	input textinput.Model
}

func newDataFilterOverlay(current string) *dataFilterOverlay {
	input := textinput.New()
	input.Prompt = ""
	input.Placeholder = "dataset=slug,status=pending,kind=pair,source=capture:serve,lek>=80"
	input.SetValue(current)
	input.Focus()
	return &dataFilterOverlay{input: input}
}

func (overlay *dataFilterOverlay) Update(message tea.KeyMsg) bool {
	if overlay == nil {
		return false
	}
	if message.String() == "enter" {
		return true
	}
	var command tea.Cmd
	overlay.input, command = overlay.input.Update(message)
	_ = command
	return false
}

func (overlay *dataFilterOverlay) Value() string {
	if overlay == nil {
		return ""
	}
	return overlay.input.Value()
}

func (overlay *dataFilterOverlay) View(width, height int, styles uiStyles) string {
	if overlay == nil {
		return ""
	}
	overlay.input.Width = max(12, width-6)
	return fitPane(core.Join("\n",
		"Filter", "",
		"dataset= status= kind= source= <score expr>, comma-separated",
		overlay.input.View(), "",
		"enter applies · esc cancels",
	), width, height, styles.panel)
}
