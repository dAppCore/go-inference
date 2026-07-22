// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	_ "embed"

	"github.com/charmbracelet/lipgloss"

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
// exposes (the tabs sequence, class tokens, the native slot boxes).
//
//go:embed tabs.ctml
var tabsCTML []byte

// brandSlotID and tabsSlotID are the strip's native box keys: the renderer
// records the FitSlots layout's L slot (the brand cell) and C slot (the tab
// region) under its own slot ids.
const (
	brandSlotID = "L"
	tabsSlotID  = "C"
)

// panelBarNames picks the label set for the layout. Compact labels preserve
// all destinations on narrow terminals without squeezing or clipping.
func panelBarNames(kind layoutKind) [panelCount]string {
	if kind == layoutNarrow {
		return compactPanelNames
	}
	return panelNames
}

// panelBarCell is one tab's bound cell: the rendered text (leading gap +
// marker + label), the class carrying its active styling, and the width of
// the leading gap — the inter-tab spacing that stays a non-hit when
// panelBarHit walks the cells.
type panelBarCell struct {
	panel panelID
	class string
	cell  string
	gap   int
}

// panelBarCells builds the strip's cells in panel order — the ONE source
// both the markup bindings and mouse resolution read, so the rendered strip
// and the hit walk can never disagree. The first tab carries no leading gap
// (the L slot's chrome and the C content gutter already separate it from
// the brand); every later tab leads with the two-cell gap.
func panelBarCells(active panelID, kind layoutKind) []panelBarCell {
	names := panelBarNames(kind)
	cells := make([]panelBarCell, 0, panelCount)
	for panel := panelID(0); panel < panelCount; panel++ {
		marker, class := "○ ", "nav-inactive"
		if panel == active {
			marker, class = "● ", "nav-active"
		}
		gap := 2
		if panel == 0 {
			gap = 0
		}
		cells = append(cells, panelBarCell{
			panel: panel,
			class: class,
			cell:  core.Repeat(" ", gap) + marker + names[panel],
			gap:   gap,
		})
	}
	return cells
}

// panelBarBindings binds one row per tab — active styling rides the
// row-scoped class bind (class="{{tab.class}}"), so a single sequence
// serves the whole strip; the host re-binds on every active-tab change.
func panelBarBindings(active panelID, kind layoutKind) ctml.Bindings {
	cells := panelBarCells(active, kind)
	rows := make([]map[string]any, 0, len(cells))
	for _, cell := range cells {
		rows = append(rows, map[string]any{"class": cell.class, "cell": cell.cell})
	}
	return ctml.Bindings{Sequences: map[string][]map[string]any{"tabs": rows}}
}

// panelBarTheme maps the markup's class tokens onto the existing palette,
// and shapes the strip's slot chrome: the L slot keeps the fit geometry's
// fixed 4-column chrome as a space-glyph left/right border with no
// top/bottom — invisible, and one row tall — so the recorded slot boxes
// tile the rendered strip exactly.
func panelBarTheme(styles uiStyles) *html.TermTheme {
	theme := html.DefaultTermTheme()
	theme.Text = styles.header
	theme.Sidebar = lipgloss.NewStyle().
		Border(lipgloss.Border{Left: " ", Right: " "}).
		BorderTop(false).BorderBottom(false)
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

// renderPanelBarBoxes parses tabs.ctml with the current tab bindings and
// renders it as a FitSlots layout through RenderTermBoxes: the brand and
// tab slots size to their content, pack edge-to-edge on one row, and record
// native slot boxes that tile the strip. The boxes are clamped to the
// fitted width, because fitLine truncates the rendered line the same way.
func renderPanelBarBoxes(active panelID, width int, kind layoutKind, styles uiStyles) (string, html.BoxMap) {
	if width <= 0 {
		return "", html.BoxMap{}
	}
	layout, err := ctml.ParseLayout(tabsCTML, panelBarBindings(active, kind))
	if err != nil {
		// tabs.ctml is embedded and static, so a parse failure is a build
		// defect; TestRenderPanelBar_Good pins the markup as parseable.
		return "", html.BoxMap{}
	}
	line, boxes := layout.RenderTermBoxes(html.NewContext(), html.TermOptions{
		Width:    width,
		Theme:    panelBarTheme(styles),
		FitSlots: true,
	})
	clampBoxesToWidth(boxes, width)
	return fitLine(line, width, styles.header), boxes
}

// clampBoxesToWidth trims the box map to the fitted render: fitLine
// truncates the strip to width, so a box past the edge is dropped and a box
// crossing it is narrowed — the map always describes the visible cells.
func clampBoxesToWidth(boxes html.BoxMap, width int) {
	for id, box := range boxes {
		if box.Col >= width {
			delete(boxes, id)
			continue
		}
		if box.Col+box.Width > width {
			box.Width = width - box.Col
			boxes[id] = box
		}
	}
}

// panelBarHit resolves a strip-local coordinate to the tab that painted
// there. teabox picks the smallest recorded box, so the brand cell resolves
// to the L slot and reports no tab; a hit on the C slot walks the same
// cells the bindings rendered — from the slot's one-column content gutter,
// each cell spans its leading gap (a non-hit) then its marker+label — so
// the render and the resolution can never disagree.
func panelBarHit(boxes html.BoxMap, x, y int, active panelID, kind layoutKind) (panelID, bool) {
	hit, ok := teabox.Resolve(boxes, x, y)
	if !ok || hit.BlockID != tabsSlotID {
		return 0, false
	}
	tabs := boxes[tabsSlotID]
	edge := tabs.Col + tabs.Width
	pos := tabs.Col + 1 // the C content's one-column alignment gutter
	for _, cell := range panelBarCells(active, kind) {
		start := pos + cell.gap
		end := min(pos+lipgloss.Width(cell.cell), edge)
		if start >= edge {
			break
		}
		if x >= start && x < end {
			return cell.panel, true
		}
		pos += lipgloss.Width(cell.cell)
	}
	return 0, false
}

func (panel panelID) String() string {
	if panel >= panelCount {
		return "Unknown"
	}
	return core.Lower(panelNames[panel])
}
