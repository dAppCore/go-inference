// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"testing"

	termlipgloss "charm.land/lipgloss/v2"
	"github.com/charmbracelet/colorprofile"
	"github.com/charmbracelet/lipgloss"
)

func TestTermStyle_Good(t *testing.T) {
	border := lipgloss.Border{
		Top:         "─",
		Bottom:      "─",
		Left:        "│",
		Right:       "│",
		TopLeft:     "┌",
		TopRight:    "┐",
		BottomLeft:  "└",
		BottomRight: "┘",
	}
	source := lipgloss.NewStyle().
		Foreground(lipgloss.Color("#112233")).
		Bold(true).
		Italic(true).
		Underline(true).
		UnderlineSpaces(false).
		Padding(1, 2, 3, 4).
		Border(border, false, true, true, false).
		BorderForeground(lipgloss.Color("#778899"))

	got := termStyle(source)
	if !got.GetBold() || !got.GetItalic() || !got.GetUnderline() {
		t.Fatalf("termStyle emphasis = bold:%t italic:%t underline:%t, want all true",
			got.GetBold(), got.GetItalic(), got.GetUnderline())
	}
	if got.GetUnderlineSpaces() {
		t.Fatal("termStyle underline spaces = true, want false")
	}
	if top, right, bottom, left := got.GetPadding(); top != 1 || right != 2 || bottom != 3 || left != 4 {
		t.Fatalf("termStyle padding = (%d, %d, %d, %d), want (1, 2, 3, 4)", top, right, bottom, left)
	}
	gotBorder, top, right, bottom, left := got.GetBorder()
	if gotBorder.Top != border.Top || gotBorder.Right != border.Right ||
		gotBorder.Bottom != border.Bottom || gotBorder.Left != border.Left ||
		top || !right || !bottom || left {
		t.Fatalf("termStyle border = %#v (%t, %t, %t, %t), want %#v (false, true, true, false)",
			gotBorder, top, right, bottom, left, border)
	}
	assertSameTermColor(t, "foreground", source.GetForeground(), got.GetForeground())
	assertSameTermColor(t, "border foreground", source.GetBorderRightForeground(), got.GetBorderRightForeground())
}

func TestTermStyles_Good(t *testing.T) {
	got := termStyles(map[string]lipgloss.Style{
		"title": lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("#abcdef")),
	})
	if len(got) != 1 || !got["title"].GetBold() {
		t.Fatalf("termStyles = %#v, want one bold title style", got)
	}
}

func TestTermOutput_Good(t *testing.T) {
	profile := termlipgloss.Writer.Profile
	termlipgloss.Writer.Profile = colorprofile.NoTTY
	t.Cleanup(func() {
		termlipgloss.Writer.Profile = profile
	})

	rendered := termlipgloss.NewStyle().
		Bold(true).
		Underline(true).
		Foreground(termlipgloss.Color("#112233")).
		Render("Chat")
	if got := termOutput(rendered); got != "Chat" {
		t.Fatalf("termOutput = %q, want %q", got, "Chat")
	}
}

type rgbaColor interface {
	RGBA() (r, g, b, a uint32)
}

func assertSameTermColor(t *testing.T, name string, want, got rgbaColor) {
	t.Helper()
	wantR, wantG, wantB, wantA := want.RGBA()
	gotR, gotG, gotB, gotA := got.RGBA()
	if gotR != wantR || gotG != wantG || gotB != wantB || gotA != wantA {
		t.Fatalf("%s = rgba(%d, %d, %d, %d), want rgba(%d, %d, %d, %d)",
			name, gotR, gotG, gotB, gotA, wantR, wantG, wantB, wantA)
	}
}
