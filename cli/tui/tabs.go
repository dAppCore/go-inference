// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	_ "embed"

	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	core "dappco.re/go"
	"dappco.re/go/html"
	"dappco.re/go/html/ctml"
	"dappco.re/go/html/teabox"
)

type panelID uint8

const (
	panelChat panelID = iota
	panelWork
	panelModels
	panelService
	panelData
	panelCount
)

var panelNames = [panelCount]string{"Chat", "Work", "Models", "Service", "Data"}
var compactPanelNames = [panelCount]string{"Chat", "Work", "Models", "API", "Data"}

func (panel panelID) next() panelID { return (panel + 1) % panelCount }
func (panel panelID) prev() panelID { return (panel + panelCount - 1) % panelCount }

// tabsCTML is the tab strip's markup — see tabs.ctml for the seams it
// exposes (row sequences, class tokens, the panel-bar box id).
//
//go:embed tabs.ctml
var tabsCTML []byte

// panelBarBlockID is the id attribute on the strip's <nav> — the block the
// go-html terminal renderer records in the box map.
const panelBarBlockID = "panel-bar"

// panelTabBlockID names the derived per-tab box for one panel, e.g. "tab-chat".
func panelTabBlockID(panel panelID) string { return "tab-" + panel.String() }

// panelBarNames picks the label set for the layout. Compact labels preserve
// all destinations on narrow terminals without squeezing or clipping.
func panelBarNames(kind layoutKind) [panelCount]string {
	if kind == layoutNarrow {
		return compactPanelNames
	}
	return panelNames
}

// panelBarBindings splits the panel labels around the active tab. Three
// sequences (before / active / after) because .ctml class attributes are
// static strings — an <each> row cannot vary its own class — so the active
// styling is carried by which sequence a row lands in. A strip this size
// re-binds on every change for free.
func panelBarBindings(active panelID, kind layoutKind) ctml.Bindings {
	names := panelBarNames(kind)
	sequences := map[string][]map[string]any{
		"tabsBefore": {},
		"tabsActive": {},
		"tabsAfter":  {},
	}
	for panel := panelID(0); panel < panelCount; panel++ {
		row := map[string]any{"label": names[panel]}
		switch {
		case panel < active:
			sequences["tabsBefore"] = append(sequences["tabsBefore"], row)
		case panel == active:
			sequences["tabsActive"] = append(sequences["tabsActive"], row)
		default:
			sequences["tabsAfter"] = append(sequences["tabsAfter"], row)
		}
	}
	return ctml.Bindings{Sequences: sequences}
}

// panelBarTheme maps the markup's class tokens onto the existing palette, so
// the .ctml render reuses uiStyles paint exactly — no colours of its own.
func panelBarTheme(styles uiStyles) *html.TermTheme {
	theme := html.DefaultTermTheme()
	theme.Text = styles.header
	theme.Classes = map[string]lipgloss.Style{
		"brand":        styles.brand,
		"nav-active":   styles.navActive,
		"nav-inactive": styles.navInactive,
	}
	return theme
}

// renderPanelBar draws brand and navigation as one stable header line — the
// box-less View path over renderPanelBarBoxes.
func renderPanelBar(active panelID, width int, kind layoutKind, styles uiStyles) string {
	line, _ := renderPanelBarBoxes(active, width, kind, styles)
	return line
}

// renderPanelBarBoxes parses tabs.ctml with the current tab bindings, renders
// it through the go-html terminal renderer, and returns the fitted header
// line plus the box map mouse resolution needs: the <nav>'s own box from the
// renderer and one derived box per visible tab from mergePanelTabBoxes.
func renderPanelBarBoxes(active panelID, width int, kind layoutKind, styles uiStyles) (string, html.BoxMap) {
	if width <= 0 {
		return "", html.BoxMap{}
	}
	tree, err := ctml.Parse(tabsCTML, panelBarBindings(active, kind))
	if err != nil {
		// tabs.ctml is embedded and static, so a parse failure is a build
		// defect; TestRenderPanelBar_Good pins the markup as parseable.
		return "", html.BoxMap{}
	}
	line, boxes := html.RenderTermBoxes(tree, html.NewContext(), html.TermOptions{Width: width, Theme: panelBarTheme(styles)})
	mergePanelTabBoxes(boxes, line, active, kind)
	return fitLine(line, width, styles.header), boxes
}

// mergePanelTabBoxes derives one box per visible tab and records it beside
// the renderer's own <nav> box. go-html's box map identifies block-level
// elements only — the inline tab spans inside this single-row strip never
// record — so each tab's rectangle is measured from the render itself: its
// "● Label"/"○ Label" segment is located in the ANSI-stripped first row and
// mapped to cell coordinates. A tab pushed off the visible row gets no box,
// and therefore no hit.
func mergePanelTabBoxes(boxes html.BoxMap, rendered string, active panelID, kind layoutKind) {
	bar, ok := boxes[panelBarBlockID]
	if !ok {
		return
	}
	names := panelBarNames(kind)
	plain := core.Split(ansi.Strip(rendered), "\n")[0]
	cursor := 0
	for panel := panelID(0); panel < panelCount; panel++ {
		marker := "○ "
		if panel == active {
			marker = "● "
		}
		segment := marker + names[panel]
		offset := core.Index(plain[cursor:], segment)
		if offset < 0 {
			continue
		}
		start := cursor + offset
		boxes[panelTabBlockID(panel)] = html.Box{
			Row:    bar.Row,
			Col:    bar.Col + lipgloss.Width(plain[:start]),
			Width:  lipgloss.Width(segment),
			Height: 1,
			Node:   bar.Node,
		}
		cursor = start + len(segment)
	}
}

// panelBarHit resolves a strip-local coordinate to the tab that painted
// there. teabox picks the smallest box containing the point, so a click on a
// tab resolves to its derived box rather than the whole strip, and a click
// on the brand or an inter-tab gap resolves to the strip and reports no tab.
func panelBarHit(boxes html.BoxMap, x, y int) (panelID, bool) {
	hit, ok := teabox.Resolve(boxes, x, y)
	if !ok {
		return 0, false
	}
	for panel := panelID(0); panel < panelCount; panel++ {
		if hit.BlockID == panelTabBlockID(panel) {
			return panel, true
		}
	}
	return 0, false
}

func (panel panelID) String() string {
	if panel >= panelCount {
		return "Unknown"
	}
	return core.Lower(panelNames[panel])
}
