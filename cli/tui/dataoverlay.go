// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	_ "embed"

	tea "dappco.re/go/render/display/tui"
	"dappco.re/go/render/display/tui/textarea"
	"dappco.re/go/render/display/tui/textinput"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
	"dappco.re/go/render/engine/ctml"
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
	if key, ok := message.(tea.KeyPressMsg); ok {
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

// dataEditorCTML is the edit-as-derived overlay's markup — see
// dataeditor.ctml for the seams it exposes (the prompt-caption split,
// class tokens).
//
//go:embed dataeditor.ctml
var dataEditorCTML []byte

// dataItemEditorBindings selects the prompt caption by item kind: both
// captions are static markup text, so the messages-kind note is carried by
// which zero-or-one-row sequence holds the row, the selection-as-
// sequence-split idiom.
func dataItemEditorBindings(editor *dataItemEditor) ctml.Bindings {
	sequences := map[string][]map[string]any{
		"pairLabel":     {},
		"messagesLabel": {},
	}
	if editor != nil && editor.original.Kind == dataset.KindMessages {
		sequences["messagesLabel"] = append(sequences["messagesLabel"], map[string]any{})
	} else {
		sequences["pairLabel"] = append(sequences["pairLabel"], map[string]any{})
	}
	return ctml.Bindings{Sequences: sequences}
}

func (editor *dataItemEditor) View(width, height int, styles uiStyles) string {
	if editor == nil {
		return ""
	}
	fieldWidth := max(12, width-6)
	editor.prompt.SetWidth(fieldWidth)
	editor.response.SetWidth(fieldWidth)
	head, foot := renderOverlayFrame(dataEditorCTML, width, styles, dataItemEditorBindings(editor))
	return fitPane(core.Join("\n",
		head, editor.prompt.View(), "",
		"Response", editor.response.View(),
		foot,
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
func (overlay *dataNoteOverlay) Update(message tea.KeyPressMsg) bool {
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

// dataNoteCTML is the note/label overlay's markup — see datanote.ctml for
// the seams it exposes (the caller-supplied title/prompt row, class tokens).
//
//go:embed datanote.ctml
var dataNoteCTML []byte

// dataNoteBindings binds the caller-supplied title and prompt — one row,
// because a lone pair of dynamic values rides a one-row sequence.
func dataNoteBindings(overlay *dataNoteOverlay) ctml.Bindings {
	title, prompt := "", ""
	if overlay != nil {
		title, prompt = overlay.title, overlay.prompt
	}
	return ctml.Bindings{Sequences: map[string][]map[string]any{
		"note": {{"title": title, "prompt": prompt}},
	}}
}

func (overlay *dataNoteOverlay) View(width, height int, styles uiStyles) string {
	if overlay == nil {
		return ""
	}
	overlay.input.SetWidth(max(12, width-6))
	head, foot := renderOverlayFrame(dataNoteCTML, width, styles, dataNoteBindings(overlay))
	return fitPane(core.Join("\n", head, overlay.input.View(), foot), width, height, styles.panel)
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

// dataBulkCTML is the bulk-confirm overlay's markup — see databulk.ctml
// for the seams it exposes (the title/count row, the conditional note, the
// armed-prompt split, class tokens).
//
//go:embed databulk.ctml
var dataBulkCTML []byte

// dataBulkBindings binds the action title and item count, the optional
// shared note (zero-or-one row), and the phase prompt: both prompt texts
// are static markup, so the armed state is carried by which zero-or-one-row
// sequence holds the row, the selection-as-sequence-split idiom.
func dataBulkBindings(overlay *dataBulkOverlay) ctml.Bindings {
	sequences := map[string][]map[string]any{
		"bulk":    {},
		"note":    {},
		"arm":     {},
		"confirm": {},
	}
	if overlay != nil {
		sequences["bulk"] = append(sequences["bulk"], map[string]any{
			"title": overlay.action.title(),
			"count": core.Sprintf("%d", overlay.count),
		})
		if overlay.note != "" {
			sequences["note"] = append(sequences["note"], map[string]any{"text": overlay.note})
		}
		if overlay.armed {
			sequences["confirm"] = append(sequences["confirm"], map[string]any{})
		} else {
			sequences["arm"] = append(sequences["arm"], map[string]any{})
		}
	}
	return ctml.Bindings{Sequences: sequences}
}

func (overlay *dataBulkOverlay) View(width, height int, styles uiStyles) string {
	if overlay == nil {
		return ""
	}
	return fitPane(renderOverlayLayout(dataBulkCTML, width, styles, dataBulkBindings(overlay)), width, height, styles.panel)
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

func (overlay *dataFilterOverlay) Update(message tea.KeyPressMsg) bool {
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

// dataFilterCTML is the filter overlay's markup — see datafilter.ctml for
// the seams it exposes (class tokens; every line is static, so it binds
// nothing).
//
//go:embed datafilter.ctml
var dataFilterCTML []byte

func (overlay *dataFilterOverlay) View(width, height int, styles uiStyles) string {
	if overlay == nil {
		return ""
	}
	overlay.input.SetWidth(max(12, width-6))
	head, foot := renderOverlayFrame(dataFilterCTML, width, styles)
	return fitPane(core.Join("\n", head, overlay.input.View(), foot), width, height, styles.panel)
}
