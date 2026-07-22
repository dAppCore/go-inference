// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
)

func keyMsg(s string) tea.KeyMsg {
	switch s {
	case "enter":
		return tea.KeyMsg{Type: tea.KeyEnter}
	case "tab":
		return tea.KeyMsg{Type: tea.KeyTab}
	case "esc":
		return tea.KeyMsg{Type: tea.KeyEsc}
	default:
		return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune(s)}
	}
}

func testTime() time.Time {
	return time.Date(2026, time.July, 19, 9, 0, 0, 0, time.UTC)
}

func dataMessagesJSON(t *testing.T, turns []dataset.MessageTurn) []byte {
	t.Helper()
	marshalled := core.JSONMarshal(dataset.MessagesContent{Messages: turns})
	if !marshalled.OK {
		t.Fatalf("marshal messages content: %v", marshalled.Value)
	}
	return marshalled.Value.([]byte)
}

// ---- dataItemEditor ----

func TestNewDataItemEditor_Good(t *testing.T) {
	item := conformancePairItem("ds", "original prompt", "original response", testTime())
	editor := newDataItemEditor(item)
	prompt, response := editor.values()
	if prompt != "original prompt" || response != "original response" {
		t.Fatalf("values = %q / %q", prompt, response)
	}
}

func TestDataItemEditor_FocusCyclesAndRoutesInput(t *testing.T) {
	item := conformancePairItem("ds", "p", "r", testTime())
	editor := newDataItemEditor(item)
	if editor.focus != 0 {
		t.Fatalf("initial focus = %d, want 0 (prompt)", editor.focus)
	}
	editor.Update(keyMsg("tab"))
	if editor.focus != 1 {
		t.Fatalf("focus after tab = %d, want 1 (response)", editor.focus)
	}
	editor.Update(keyMsg("x"))
	_, response := editor.values()
	if !strings.Contains(response, "x") {
		t.Fatalf("response after typing on focus 1 = %q, want it to contain the typed rune", response)
	}
	editor.Update(keyMsg("tab"))
	if editor.focus != 0 {
		t.Fatalf("focus after second tab = %d, want 0", editor.focus)
	}
}

func TestDataItemEditor_View_MessagesKindNotesContextOnly(t *testing.T) {
	content := dataMessagesJSON(t, []dataset.MessageTurn{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}})
	item := dataset.Item{ID: "m1", DatasetID: "ds", Kind: dataset.KindMessages, Content: content}
	editor := newDataItemEditor(item)
	view := editor.View(80, 24, newUIStyles(midnightTheme()))
	if !strings.Contains(view, "context only") {
		t.Fatalf("messages-kind editor view does not flag the prompt field as context-only:\n%s", view)
	}
}

func TestDataItemEditor_Nil(t *testing.T) {
	var editor *dataItemEditor
	if got := editor.Update(keyMsg("a")); got != nil {
		t.Fatalf("nil editor Update returned a non-nil cmd")
	}
	if prompt, response := editor.values(); prompt != "" || response != "" {
		t.Fatalf("nil editor values = %q / %q", prompt, response)
	}
	if view := editor.View(80, 24, newUIStyles(midnightTheme())); view != "" {
		t.Fatalf("nil editor view = %q", view)
	}
}

func TestRenderDataEditor_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	editor := newDataItemEditor(conformancePairItem("ds", "the prompt", "the response", testTime()))
	view := editor.View(60, 24, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "Edit as derived" {
		t.Fatalf("overlay must open with its title line: %q", lines[0])
	}
	if strings.TrimSpace(lines[1]) != "" {
		t.Fatalf("title must be followed by a blank separator: %q", lines[1])
	}
	if strings.TrimSpace(lines[2]) != "Prompt" {
		t.Fatalf("pair-kind editor must caption the first field Prompt: %q", lines[2])
	}
	if !strings.Contains(lines[3], "the prompt") {
		t.Fatalf("prompt textarea must sit directly beneath its caption: %q", lines[3])
	}
	if !strings.Contains(plain, "Response") {
		t.Fatalf("overlay missing the Response caption: %q", plain)
	}
	if !strings.Contains(plain, "tab changes field · ctrl+s saves as a new derived item ·") {
		t.Fatalf("overlay missing the key footer: %q", plain)
	}
	for index, line := range lines {
		if got := lipgloss.Width(line); got > 60 {
			t.Fatalf("line %d width = %d, exceeds 60: %q", index, got, line)
		}
	}
}

func TestRenderDataEditor_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	editor := newDataItemEditor(conformancePairItem("ds", "p", "r", testTime()))
	for _, width := range []int{0, -4} {
		if got := editor.View(width, 24, styles); got != "" {
			t.Fatalf("View(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderDataEditor_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	content := dataMessagesJSON(t, []dataset.MessageTurn{{Role: "user", Content: "hi"}, {Role: "assistant", Content: "hello"}})
	editor := newDataItemEditor(dataset.Item{ID: "m1", DatasetID: "ds", Kind: dataset.KindMessages, Content: content})
	plain := ansi.Strip(editor.View(70, 24, styles))

	if !strings.Contains(plain, "Prompt (context only — earlier turns are kept as-is)") {
		t.Fatalf("messages-kind editor must swap in the context-only caption: %q", plain)
	}
	if strings.Index(plain, "context only") > strings.Index(plain, "Response") {
		t.Fatalf("the caption split must keep the prompt caption above Response: %q", plain)
	}
}

// ---- dataNoteOverlay ----

func TestDataNoteOverlay_RequiresNonEmptyValueToSubmit(t *testing.T) {
	overlay := newDataNoteOverlay(dataActionTag, "item-1", "Tag", "Tag label", "label")
	if overlay.Update(keyMsg("enter")) {
		t.Fatal("empty note overlay reported a submit on Enter")
	}
	for _, r := range "favourite" {
		overlay.Update(keyMsg(string(r)))
	}
	if !overlay.Update(keyMsg("enter")) {
		t.Fatal("non-empty note overlay did not report a submit on Enter")
	}
	if got := overlay.Value(); got != "favourite" {
		t.Fatalf("Value() = %q, want %q", got, "favourite")
	}
}

func TestDataNoteOverlay_WhitespaceOnlyDoesNotSubmit(t *testing.T) {
	overlay := newDataNoteOverlay(dataActionQuarantineClear, "item-1", "Clear quarantine", "Why?", "note")
	for _, r := range "   " {
		overlay.Update(keyMsg(string(r)))
	}
	if overlay.Update(keyMsg("enter")) {
		t.Fatal("whitespace-only note overlay reported a submit on Enter")
	}
}

func TestDataNoteOverlay_Bulk(t *testing.T) {
	single := newDataNoteOverlay(dataActionTag, "item-1", "Tag", "p", "ph")
	if single.Bulk() {
		t.Fatal("single-item note overlay reported Bulk() = true")
	}
	bulk := newDataNoteOverlay(dataActionTag, "", "Bulk tag", "p", "ph")
	if !bulk.Bulk() {
		t.Fatal("bulk note overlay (empty itemID) reported Bulk() = false")
	}
}

func TestDataNoteOverlay_Nil(t *testing.T) {
	var overlay *dataNoteOverlay
	if overlay.Update(keyMsg("enter")) {
		t.Fatal("nil overlay reported a submit")
	}
	if overlay.Value() != "" {
		t.Fatalf("nil overlay Value() = %q", overlay.Value())
	}
	if overlay.Bulk() {
		t.Fatal("nil overlay Bulk() = true")
	}
	if view := overlay.View(80, 24, newUIStyles(midnightTheme())); view != "" {
		t.Fatalf("nil overlay view = %q", view)
	}
}

func TestRenderDataNote_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newDataNoteOverlay(dataActionTag, "item-1", "Tag item", "Tag label", "label")
	view := overlay.View(48, 12, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "Tag item" {
		t.Fatalf("overlay must open with its title line: %q", lines[0])
	}
	if strings.TrimSpace(lines[1]) != "" {
		t.Fatalf("title must be followed by a blank separator: %q", lines[1])
	}
	if strings.TrimSpace(lines[2]) != "Tag label" {
		t.Fatalf("prompt must sit beneath the separator: %q", lines[2])
	}
	if !strings.Contains(lines[3], "label") {
		t.Fatalf("the input must sit directly beneath the prompt: %q", lines[3])
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

func TestRenderDataNote_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newDataNoteOverlay(dataActionTag, "item-1", "Tag", "p", "ph")
	for _, width := range []int{0, -4} {
		if got := overlay.View(width, 12, styles); got != "" {
			t.Fatalf("View(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderDataNote_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	// The bulk flavour (empty itemID) renders the same frame — the shared
	// note is a caller concern, not a markup one.
	overlay := newDataNoteOverlay(dataActionQuarantineClear, "", "Bulk clear quarantine", "Why should these clear?", "note")
	plain := ansi.Strip(overlay.View(48, 12, styles))
	if strings.Index(plain, "Bulk clear quarantine") > strings.Index(plain, "Why should these clear?") {
		t.Fatalf("title must render above the prompt: %q", plain)
	}
	if !strings.Contains(plain, "enter submits · esc cancels") {
		t.Fatalf("bulk-flavoured note overlay missing the key footer: %q", plain)
	}
}

// ---- dataBulkOverlay: the count-confirmation gate ----

// TestDataBulkOverlay_TwoPhaseConfirm proves the exact gate the task brief
// requires: "no confirm, no writes". Confirm() is the ONLY signal
// app.confirmDataBulk ever acts on (see onOverlayKey's overlayDataBulk
// case), so this test is a direct proof that a single Enter — or any
// other key, including Escape's caller-side handling — never reports a
// confirm; only a SECOND Enter does.
func TestDataBulkOverlay_TwoPhaseConfirm(t *testing.T) {
	overlay := newDataBulkOverlay(dataActionApprove, 42, "")
	for _, key := range []string{"a", "j", "k", "tab", "q"} {
		if overlay.Confirm(key) {
			t.Fatalf("key %q reported a confirm before any Enter", key)
		}
	}
	if overlay.armed {
		t.Fatal("overlay armed itself without an Enter")
	}
	if overlay.Confirm("enter") {
		t.Fatal("the FIRST enter reported a confirm — it must only arm")
	}
	if !overlay.armed {
		t.Fatal("the first enter did not arm the overlay")
	}
	// Any non-enter key between the two enters must not consume the arm
	// or falsely confirm.
	if overlay.Confirm("j") {
		t.Fatal("a non-enter key after arming reported a confirm")
	}
	if !overlay.Confirm("enter") {
		t.Fatal("the SECOND enter did not report a confirm")
	}
}

func TestDataBulkOverlay_ViewShowsCountAndNote(t *testing.T) {
	overlay := newDataBulkOverlay(dataActionQuarantineClear, 7, "false positive batch")
	unarmed := overlay.View(80, 24, newUIStyles(midnightTheme()))
	if !strings.Contains(unarmed, "7") || !strings.Contains(unarmed, "false positive batch") || !strings.Contains(unarmed, "continues") {
		t.Fatalf("unarmed bulk view:\n%s", unarmed)
	}
	overlay.Confirm("enter")
	armed := overlay.View(80, 24, newUIStyles(midnightTheme()))
	if !strings.Contains(armed, "applies this action") {
		t.Fatalf("armed bulk view did not change its prompt:\n%s", armed)
	}
}

func TestDataBulkOverlay_Nil(t *testing.T) {
	var overlay *dataBulkOverlay
	if overlay.Confirm("enter") {
		t.Fatal("nil overlay reported a confirm")
	}
	if view := overlay.View(80, 24, newUIStyles(midnightTheme())); view != "" {
		t.Fatalf("nil overlay view = %q", view)
	}
}

func TestRenderDataBulk_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newDataBulkOverlay(dataActionApprove, 42, "")
	view := overlay.View(48, 12, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "Bulk Approve" {
		t.Fatalf("overlay must open with its action title: %q", lines[0])
	}
	if strings.TrimSpace(lines[1]) != "" {
		t.Fatalf("title must be followed by a blank separator: %q", lines[1])
	}
	if !strings.Contains(plain, "This will apply to 42 item(s) matching the") {
		t.Fatalf("overlay missing the count sentence: %q", plain)
	}
	if strings.Contains(plain, "Note:") {
		t.Fatalf("a note-free bulk action must not render a note row: %q", plain)
	}
	if !strings.Contains(plain, "enter continues · esc cancels") {
		t.Fatalf("unarmed overlay missing the continue prompt: %q", plain)
	}
	for index, line := range lines {
		if got := lipgloss.Width(line); got > 48 {
			t.Fatalf("line %d width = %d, exceeds 48: %q", index, got, line)
		}
	}
}

func TestRenderDataBulk_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newDataBulkOverlay(dataActionApprove, 1, "")
	for _, width := range []int{0, -4} {
		if got := overlay.View(width, 12, styles); got != "" {
			t.Fatalf("View(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderDataBulk_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newDataBulkOverlay(dataActionQuarantineClear, 7, "false positive batch")
	overlay.Confirm("enter") // arm
	plain := ansi.Strip(overlay.View(48, 12, styles))

	if !strings.Contains(plain, "Note: false positive batch") {
		t.Fatalf("collected note must render its row: %q", plain)
	}
	if strings.Index(plain, "item(s)") > strings.Index(plain, "Note:") {
		t.Fatalf("the note must follow the count sentence: %q", plain)
	}
	if !strings.Contains(plain, "enter applies this action to every listed item") {
		t.Fatalf("armed overlay must swap to the apply prompt: %q", plain)
	}
	if strings.Contains(plain, "enter continues") {
		t.Fatalf("armed overlay must not keep the continue prompt: %q", plain)
	}
}

// ---- dataFilterOverlay ----

func TestDataFilterOverlay_EnterAlwaysSubmitsEvenEmpty(t *testing.T) {
	overlay := newDataFilterOverlay("status=pending")
	if got := overlay.Value(); got != "status=pending" {
		t.Fatalf("pre-filled value = %q", got)
	}
	if !overlay.Update(keyMsg("enter")) {
		t.Fatal("Enter did not report a submit")
	}
	// Clearing the field and submitting an empty value must still count
	// as a submit — an empty filter is "show everything", a valid state.
	cleared := newDataFilterOverlay("")
	if !cleared.Update(keyMsg("enter")) {
		t.Fatal("Enter on an empty filter overlay did not report a submit")
	}
}

func TestDataFilterOverlay_Nil(t *testing.T) {
	var overlay *dataFilterOverlay
	if overlay.Update(keyMsg("enter")) {
		t.Fatal("nil overlay reported a submit")
	}
	if overlay.Value() != "" {
		t.Fatalf("nil overlay Value() = %q", overlay.Value())
	}
	if view := overlay.View(80, 24, newUIStyles(midnightTheme())); view != "" {
		t.Fatalf("nil overlay view = %q", view)
	}
}

func TestRenderDataFilter_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newDataFilterOverlay("status=pending")
	view := overlay.View(80, 12, styles)
	plain := ansi.Strip(view)
	lines := strings.Split(plain, "\n")

	if strings.TrimSpace(lines[0]) != "Filter" {
		t.Fatalf("overlay must open with its title line: %q", lines[0])
	}
	if strings.TrimSpace(lines[1]) != "" {
		t.Fatalf("title must be followed by a blank separator: %q", lines[1])
	}
	if !strings.Contains(plain, "dataset= status= kind= source= <score expr>, comma-separated") {
		t.Fatalf("overlay missing the grammar hint: %q", plain)
	}
	if !strings.Contains(plain, "status=pending") {
		t.Fatalf("overlay missing the pre-filled input: %q", plain)
	}
	if !strings.Contains(plain, "enter applies · esc cancels") {
		t.Fatalf("overlay missing the key footer: %q", plain)
	}
	for index, line := range lines {
		if got := lipgloss.Width(line); got > 80 {
			t.Fatalf("line %d width = %d, exceeds 80: %q", index, got, line)
		}
	}
}

func TestRenderDataFilter_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newDataFilterOverlay("")
	for _, width := range []int{0, -4} {
		if got := overlay.View(width, 12, styles); got != "" {
			t.Fatalf("View(width=%d) = %q, want empty", width, got)
		}
	}
}

func TestRenderDataFilter_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	// A narrow pane wraps the grammar hint; the escaped <score expr> token
	// must survive the entity round trip and no line may overflow.
	overlay := newDataFilterOverlay("")
	plain := ansi.Strip(overlay.View(24, 12, styles))
	if !strings.Contains(plain, "<score") {
		t.Fatalf("escaped grammar token missing from the narrow render: %q", plain)
	}
	for index, line := range strings.Split(plain, "\n") {
		if got := lipgloss.Width(line); got > 24 {
			t.Fatalf("line %d width = %d, exceeds 24: %q", index, got, line)
		}
	}
}
