// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// The tab bar follows the bubbletea tabs example: joined bordered boxes with
// the active tab's bottom border opened into the content pane.

type tabID int

const (
	tabChat tabID = iota
	tabModels
	tabService
	tabSettings
	tabTools
	tabModes
	tabCount
)

var tabNames = [tabCount]string{"Chat", "Models", "Service", "Settings", "Tools", "Modes"}

func tabBorder(left, middle, right string) lipgloss.Border {
	b := lipgloss.RoundedBorder()
	b.BottomLeft, b.Bottom, b.BottomRight = left, middle, right
	return b
}

var (
	inactiveTabBorder = tabBorder("┴", "─", "┴")
	activeTabBorder   = tabBorder("┘", " ", "└")
	inactiveTabStyle  = lipgloss.NewStyle().Border(inactiveTabBorder, true).
				BorderForeground(lipgloss.Color("238")).
				Foreground(lipgloss.Color("245")).Padding(0, 1)
	activeTabStyle = inactiveTabStyle.
			Border(activeTabBorder, true).
			BorderForeground(lipgloss.Color("109")).
			Foreground(lipgloss.Color("252")).Bold(true)
	tabGapStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("238"))
)

// renderTabBar draws the header row, padding the gap to width with the open
// bottom rule so the bar reads as the top of the content frame.
func renderTabBar(active tabID, width int) string {
	var rendered []string
	for i := tabID(0); i < tabCount; i++ {
		if i == active {
			rendered = append(rendered, activeTabStyle.Render(tabNames[i]))
		} else {
			rendered = append(rendered, inactiveTabStyle.Render(tabNames[i]))
		}
	}
	row := lipgloss.JoinHorizontal(lipgloss.Bottom, rendered...)
	if gap := width - lipgloss.Width(row); gap > 0 {
		rule := strings.Repeat("─", gap)
		pad := strings.Repeat("\n", max(0, lipgloss.Height(row)-1))
		row = lipgloss.JoinHorizontal(lipgloss.Bottom, row, tabGapStyle.Render(pad+rule))
	}
	return row
}

func (t tabID) next() tabID { return (t + 1) % tabCount }
func (t tabID) prev() tabID { return (t + tabCount - 1) % tabCount }

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
