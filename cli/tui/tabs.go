// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"github.com/charmbracelet/lipgloss"

	core "dappco.re/go"
)

type panelID uint8

const (
	panelChat panelID = iota
	panelWork
	panelModels
	panelService
	panelCount
)

var panelNames = [panelCount]string{"Chat", "Work", "Models", "Service"}
var compactPanelNames = [panelCount]string{"Chat", "Work", "Models", "API"}

func (panel panelID) next() panelID { return (panel + 1) % panelCount }
func (panel panelID) prev() panelID { return (panel + panelCount - 1) % panelCount }

// renderPanelBar draws brand and navigation as one stable header. Compact
// labels preserve all four destinations without squeezing or clipping.
func renderPanelBar(active panelID, width int, kind layoutKind, styles uiStyles) string {
	if width <= 0 {
		return ""
	}
	names := panelNames
	if kind == layoutNarrow {
		names = compactPanelNames
	}
	items := make([]string, 0, panelCount+1)
	items = append(items, styles.brand.Render("LEM"))
	for panel := panelID(0); panel < panelCount; panel++ {
		label := names[panel]
		if panel == active {
			items = append(items, styles.navActive.Render("● "+label))
		} else {
			items = append(items, styles.navInactive.Render("○ "+label))
		}
	}
	return fitLine(lipgloss.JoinHorizontal(lipgloss.Center, intersperse(items, "  ")...), width, styles.header)
}

func intersperse(items []string, separator string) []string {
	if len(items) < 2 {
		return items
	}
	result := make([]string, 0, len(items)*2-1)
	for i, item := range items {
		if i > 0 {
			result = append(result, separator)
		}
		result = append(result, item)
	}
	return result
}

func (panel panelID) String() string {
	if panel >= panelCount {
		return "Unknown"
	}
	return core.Lower(panelNames[panel])
}
