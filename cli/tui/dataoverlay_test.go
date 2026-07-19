// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"

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
