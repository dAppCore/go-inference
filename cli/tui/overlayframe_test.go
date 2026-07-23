// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"
	"testing"

	"github.com/charmbracelet/x/ansi"
)

func TestRenderOverlayFrame_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	head, foot := renderOverlayFrame(dataFilterCTML, 48, styles)
	headPlain := ansi.Strip(head)
	footPlain := ansi.Strip(foot)

	headLines := strings.Split(headPlain, "\n")
	if strings.TrimSpace(headLines[0]) != "Filter" {
		t.Fatalf("header band must open with the title line: %q", headLines[0])
	}
	if !strings.Contains(headPlain, "comma-separated") {
		t.Fatalf("header band missing the hint: %q", headPlain)
	}
	if strings.Contains(headPlain, "enter applies") {
		t.Fatalf("footer text leaked into the header band: %q", headPlain)
	}

	footLines := strings.Split(footPlain, "\n")
	if strings.TrimSpace(footLines[0]) != "" {
		t.Fatalf("footer band must open with its padded blank row: %q", footLines[0])
	}
	if !strings.Contains(footPlain, "enter applies · esc cancels") {
		t.Fatalf("footer band missing the key hints: %q", footPlain)
	}
	if strings.Contains(footPlain, "Filter") {
		t.Fatalf("header text leaked into the footer band: %q", footPlain)
	}
}

func TestRenderOverlayFrame_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	// Not a <layout> root — ParseLayout must refuse it and the frame must
	// come back empty rather than half-rendered.
	head, foot := renderOverlayFrame([]byte("<p>not a layout</p>"), 48, styles)
	if head != "" || foot != "" {
		t.Fatalf("unparseable markup returned a frame: head=%q foot=%q", head, foot)
	}
}

func TestRenderOverlayFrame_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	// Every conditional H sequence empty: the header band shrinks to the
	// title alone and the split still lands between the bands.
	overlay := newChangeAcceptanceOverlay(agentReview{Title: "Review agent changes"})
	head, foot := renderOverlayFrame(changeReviewCTML, 60, styles, changeAcceptanceBindings(overlay))
	if got := strings.TrimSpace(ansi.Strip(head)); got != "Review agent changes" {
		t.Fatalf("warning-free header band = %q, want the title alone", got)
	}
	if !strings.Contains(ansi.Strip(foot), "enter continues · esc cancels") {
		t.Fatalf("stage-free review must fall back to the continue prompt: %q", foot)
	}
}

func TestRenderOverlayLayout_Good(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	overlay := newDataBulkOverlay(dataActionApprove, 3, "")
	plain := ansi.Strip(renderOverlayLayout(dataBulkCTML, 48, styles, dataBulkBindings(overlay)))
	first := strings.Index(plain, "Bulk Approve")
	second := strings.Index(plain, "This will apply to 3 item(s)")
	third := strings.Index(plain, "enter continues · esc cancels")
	if first < 0 || second < first || third < second {
		t.Fatalf("HCF regions must render in H, C, F order: %q", plain)
	}
}

func TestRenderOverlayLayout_Bad(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	if got := renderOverlayLayout([]byte("<section>not a layout</section>"), 48, styles); got != "" {
		t.Fatalf("unparseable markup rendered: %q", got)
	}
}

func TestRenderOverlayLayout_Ugly(t *testing.T) {
	styles := newUIStyles(midnightTheme())
	// A nil overlay binds every sequence empty: the layout must render its
	// static skeleton (the footer band) without a title or content row.
	plain := ansi.Strip(renderOverlayLayout(dataBulkCTML, 48, styles, dataBulkBindings(nil)))
	if strings.Contains(plain, "Bulk") || strings.Contains(plain, "item(s)") {
		t.Fatalf("empty bindings must render no bound rows: %q", plain)
	}
}
