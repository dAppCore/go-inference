// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"strings"

	"github.com/charmbracelet/lipgloss"
)

type layoutKind uint8

const (
	layoutNarrow layoutKind = iota
	layoutOverlay
	layoutWide
)

const wideInspectorWidth = 32

func chooseLayout(width int) layoutKind {
	switch {
	case width >= 120:
		return layoutWide
	case width >= 80:
		return layoutOverlay
	default:
		return layoutNarrow
	}
}

type frameSpec struct {
	Width         int
	Height        int
	Active        panelID
	SessionStrip  string
	Main          string
	Inspector     string
	Footer        string
	InspectorOpen bool
}

type frameMetrics struct {
	kind            layoutKind
	innerWidth      int
	innerHeight     int
	regionHeight    int
	mainWidth       int
	mainHeight      int
	inspectorWidth  int
	inspectorHeight int
}

func measureFrame(width, height int, inspectorOpen bool) frameMetrics {
	metrics := frameMetrics{
		kind:         chooseLayout(width),
		innerWidth:   max(0, width-2),
		innerHeight:  max(0, height-2),
		regionHeight: max(1, height-5), // border + header + sessions + footer
	}
	metrics.mainWidth = metrics.innerWidth
	metrics.mainHeight = metrics.regionHeight
	switch metrics.kind {
	case layoutWide:
		metrics.inspectorWidth = wideInspectorWidth
		metrics.inspectorHeight = metrics.regionHeight
		metrics.mainWidth = max(1, metrics.innerWidth-wideInspectorWidth-1)
	case layoutOverlay:
		if inspectorOpen {
			metrics.inspectorWidth = metrics.innerWidth
			metrics.inspectorHeight = min(7, max(3, metrics.regionHeight/3))
			metrics.mainHeight = max(1, metrics.regionHeight-metrics.inspectorHeight-1)
		}
	case layoutNarrow:
		if inspectorOpen {
			metrics.inspectorWidth = metrics.innerWidth
			metrics.inspectorHeight = metrics.regionHeight
		}
	}
	return metrics
}

// renderFrame composes the permanent workspace shell. All truncation is done
// by Lip Gloss on rendered cell widths; no ANSI string is sliced manually.
func renderFrame(spec frameSpec, styles uiStyles) string {
	if spec.Width <= 0 || spec.Height <= 0 {
		return ""
	}
	if spec.Width < 3 || spec.Height < 4 {
		return minimalFrame(spec.Width, spec.Height)
	}
	metrics := measureFrame(spec.Width, spec.Height, spec.InspectorOpen)
	header := renderPanelBar(spec.Active, metrics.innerWidth, metrics.kind, styles)
	sessions := fitLine(styles.title.Render("SESSIONS")+"  "+styles.session.Render(spec.SessionStrip), metrics.innerWidth, styles.session)
	region := renderWorkspaceRegion(spec, metrics, styles)
	footer := fitLine(spec.Footer, metrics.innerWidth, styles.footer)
	inside := lipgloss.JoinVertical(lipgloss.Left, header, sessions, region, footer)
	return styles.outerFrame.
		Width(metrics.innerWidth).
		Height(metrics.innerHeight).
		MaxWidth(spec.Width).
		MaxHeight(spec.Height).
		Render(inside)
}

func renderWorkspaceRegion(spec frameSpec, metrics frameMetrics, styles uiStyles) string {
	switch metrics.kind {
	case layoutWide:
		main := fitPane(spec.Main, metrics.mainWidth, metrics.mainHeight, styles.panel)
		separator := fitPane(strings.Repeat("│\n", max(0, metrics.regionHeight-1))+"│", 1, metrics.regionHeight, styles.separator)
		inspector := fitPane(spec.Inspector, metrics.inspectorWidth, metrics.inspectorHeight, styles.inspector)
		return lipgloss.JoinHorizontal(lipgloss.Top, main, separator, inspector)
	case layoutOverlay:
		if spec.InspectorOpen {
			inspector := fitPane(spec.Inspector, metrics.inspectorWidth, metrics.inspectorHeight, styles.inspector)
			separator := fitLine(strings.Repeat("─", metrics.innerWidth), metrics.innerWidth, styles.separator)
			main := fitPane(spec.Main, metrics.mainWidth, metrics.mainHeight, styles.panel)
			return lipgloss.JoinVertical(lipgloss.Left, inspector, separator, main)
		}
	case layoutNarrow:
		if spec.InspectorOpen {
			return fitPane(spec.Inspector, metrics.innerWidth, metrics.regionHeight, styles.inspector)
		}
	}
	return fitPane(spec.Main, metrics.mainWidth, metrics.mainHeight, styles.panel)
}

func fitLine(content string, width int, style lipgloss.Style) string {
	if width <= 0 {
		return ""
	}
	return style.Width(width).MaxWidth(width).MaxHeight(1).Render(content)
}

func fitPane(content string, width, height int, style lipgloss.Style) string {
	if width <= 0 || height <= 0 {
		return ""
	}
	return style.Width(width).Height(height).MaxWidth(width).MaxHeight(height).Render(content)
}

func minimalFrame(width, height int) string {
	if width <= 0 || height <= 0 {
		return ""
	}
	line := strings.Repeat("─", width)
	lines := make([]string, height)
	for i := range lines {
		lines[i] = line
	}
	return strings.Join(lines, "\n")
}
