// SPDX-Licence-Identifier: EUPL-1.2

// Package tui is the lem terminal UI (`lem tui`): a model picker over
// inference.Discover and a streaming chat over inference.TextModel.Chat,
// built on Bubble Tea + Bubbles + Lip Gloss. Dark by default.
package tui

import "github.com/charmbracelet/lipgloss"

// The palette stays deliberately dim — dark background, low-brightness
// accents — with colour reserved for state (thinking vs answer vs status).
var (
	styleTitle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("246"))

	styleUser    = lipgloss.NewStyle().Foreground(lipgloss.Color("110")).Bold(true)
	styleAnswer  = lipgloss.NewStyle().Foreground(lipgloss.Color("252"))
	styleThought = lipgloss.NewStyle().Foreground(lipgloss.Color("240")).Italic(true)
	styleErr     = lipgloss.NewStyle().Foreground(lipgloss.Color("167"))

	styleStatus = lipgloss.NewStyle().Foreground(lipgloss.Color("244"))
	styleAccent = lipgloss.NewStyle().Foreground(lipgloss.Color("109"))

	styleInputBorder = lipgloss.NewStyle().
				Border(lipgloss.RoundedBorder()).
				BorderForeground(lipgloss.Color("238")).
				Padding(0, 1)
)
