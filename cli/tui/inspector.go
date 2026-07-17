// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"strings"

	core "dappco.re/go"
)

type inspectorControl uint8

const (
	inspectorControlContext inspectorControl = iota
	inspectorControlMaxTokens
	inspectorControlThinking
	inspectorControlTheme
	inspectorControlMode
	inspectorControlTools
	inspectorControlCount
)

var inspectorThemes = []string{"midnight", "aurora", "daylight"}

type inspectorState struct {
	cursor    int
	dirty     bool
	theme     string
	runtime   runtimeInspection
	knowledge knowledgeInspection
}

func newInspector() inspectorState {
	return inspectorState{theme: "midnight"}
}

func (inspector inspectorState) Dirty() bool { return inspector.dirty }
func (inspector inspectorState) Theme() string {
	if inspector.theme == "" {
		return "midnight"
	}
	return inspector.theme
}

func (inspector *inspectorState) Select(control inspectorControl) bool {
	if inspector == nil || control >= inspectorControlCount {
		return false
	}
	inspector.cursor = int(control)
	return true
}

func (inspector *inspectorState) Move(delta int) {
	if inspector == nil {
		return
	}
	inspector.cursor = ((inspector.cursor+delta)%int(inspectorControlCount) + int(inspectorControlCount)) % int(inspectorControlCount)
}

func (inspector *inspectorState) ApplyRuntime(result core.Result) {
	if inspector != nil {
		inspector.runtime = runtimeInspectionFrom(result)
	}
}

func (inspector *inspectorState) ApplyKnowledge(result core.Result) {
	if inspector != nil {
		inspector.knowledge = knowledgeInspectionFrom(result)
	}
}

func (inspector *inspectorState) Adjust(target *app, delta int) core.Result {
	if inspector == nil || target == nil {
		return core.Fail(core.E("tui.inspector.Adjust", "inspector target is unavailable", nil))
	}
	switch inspectorControl(inspector.cursor) {
	case inspectorControlContext:
		target.cfg.ctxIdx = wrapIndex(target.cfg.ctxIdx, delta, len(ctxSteps))
	case inspectorControlMaxTokens:
		target.cfg.maxTokIdx = wrapIndex(target.cfg.maxTokIdx, delta, len(maxTokSteps))
	case inspectorControlThinking:
		target.cfg.thinkIdx = wrapIndex(target.cfg.thinkIdx, delta, len(thinkNames))
	case inspectorControlTheme:
		index := themeIndex(inspector.Theme())
		inspector.theme = inspectorThemes[wrapIndex(index, delta, len(inspectorThemes))]
	case inspectorControlMode:
		target.modes = target.modes.move(delta)
	case inspectorControlTools:
		switch {
		case delta < 0:
			target.tools.setEnabled(false)
		case delta > 0:
			target.tools.setEnabled(true)
		default:
			target.tools.toggle()
		}
	default:
		return core.Fail(core.E("tui.inspector.Adjust", "unknown inspector control", nil))
	}
	inspector.dirty = true
	return core.Ok(nil)
}

func (inspector *inspectorState) Save(target *app) core.Result {
	if inspector == nil || target == nil || target.preferences == nil {
		return core.Fail(core.E("tui.inspector.Save", "preference store is unavailable", nil))
	}
	thinking := []string{"model", "on", "off"}[target.cfg.thinkIdx]
	values := []struct {
		key   string
		value any
	}{
		{preferenceContextLength, target.cfg.contextLen()},
		{preferenceMaxTokens, target.cfg.maxTokens()},
		{preferenceThinking, thinking},
		{preferenceTheme, inspector.Theme()},
	}
	for _, value := range values {
		if result := target.preferences.Set(value.key, value.value); !result.OK {
			return result
		}
	}
	if result := target.preferences.Commit(); !result.OK {
		return result
	}
	target.rebuildTheme(themeForName(inspector.Theme()))
	inspector.dirty = false
	return core.Ok(nil)
}

func (inspector inspectorState) View(target app, width, height int) string {
	if width <= 0 || height <= 0 {
		return ""
	}
	var builder strings.Builder
	builder.WriteString(target.styles.title.Render("INSPECTOR") + "\n\n")
	switch target.activePanel {
	case panelWork:
		inspector.renderWork(&builder, target)
	case panelModels:
		builder.WriteString(target.styles.accent.Render("MODEL DETAIL") + "\n")
		if selected, ok := target.picker.SelectedItem().(modelItem); ok {
			builder.WriteString(target.styles.title.Render(selected.name) + "\n")
			builder.WriteString(target.styles.status.Render(selected.modelType) + "\n")
			builder.WriteString(target.styles.thought.Render(selected.path) + "\n\n")
		} else {
			builder.WriteString(target.styles.status.Render("○ select a discovered model") + "\n\n")
		}
		builder.WriteString(target.styles.accent.Render("LOADED") + "  ")
		if target.modelName == "" {
			builder.WriteString(target.styles.status.Render("○ none"))
		} else {
			builder.WriteString(target.styles.success.Render("● " + target.modelName))
		}
	case panelService:
		builder.WriteString(target.styles.accent.Render("ADDRESS") + "\n")
		builder.WriteString(target.styles.title.Render(target.svc.addr()) + "\n\n")
		builder.WriteString(target.styles.accent.Render("REQUESTS") + "  ")
		builder.WriteString(target.styles.title.Render(core.Sprintf("%d", target.svc.requests.Load())) + "\n\n")
		builder.WriteString(target.styles.accent.Render("STATE") + "  ")
		if target.svc.running {
			builder.WriteString(target.styles.success.Render("● serving"))
		} else {
			builder.WriteString(target.styles.status.Render("○ stopped"))
		}
	default:
		inspector.renderChat(&builder, target)
	}
	return fitPane(builder.String(), width, height, target.styles.inspector)
}

func (inspector inspectorState) renderWork(builder *strings.Builder, target app) {
	builder.WriteString(target.styles.accent.Render("WORK DETAIL") + "\n")
	if target.work == nil {
		builder.WriteString(target.styles.status.Render("○ no work item selected") + "\n\n")
	} else if selected, ok := target.work.Selected(); ok {
		builder.WriteString(target.styles.title.Render(selected.Title) + "\n")
		builder.WriteString(target.styles.status.Render(workGlyph(selected.Status)+" "+core.Upper(selected.Status)) + "\n")
		if selected.Repo != "" {
			builder.WriteString(target.styles.thought.Render(selected.Repo+"  "+selected.Branch) + "\n")
		}
		if selected.Question != "" {
			builder.WriteString(target.styles.attention.Render("? "+selected.Question) + "\n")
		}
		builder.WriteString("\n")
	} else {
		builder.WriteString(target.styles.status.Render("○ no work item selected") + "\n\n")
	}

	inspector.renderRuntime(builder, target)

	capabilities := []agentCapability{}
	if target.work != nil {
		capabilities = target.work.Capabilities()
	} else if target.agent != nil {
		capabilities = target.agent.Capabilities()
	}
	byFeature := make(map[agentFeature]agentCapability, len(capabilities))
	reason := defaultAgentUnavailableReason
	for _, capability := range capabilities {
		byFeature[capability.Feature] = capability
		if !capability.Available && capability.Reason != "" {
			reason = capability.Reason
		}
	}
	builder.WriteString(target.styles.accent.Render("AGENT CAPABILITY") + "\n")
	available := 0
	for _, capability := range capabilities {
		if capability.Available {
			available++
		}
	}
	if available == 0 {
		builder.WriteString(target.styles.attention.Render("○ not installed") + "\n")
		builder.WriteString(target.styles.thought.Render(reason) + "\n\n")
	} else {
		builder.WriteString(target.styles.success.Render(core.Sprintf("● %d actions available", available)) + "\n\n")
	}

	selectedFeature := agentFeature("")
	if target.work != nil {
		selectedFeature = target.work.SelectedAction().Feature
	}
	for _, group := range agentFeatureGroups {
		builder.WriteString(target.styles.accent.Render(group.Title) + "\n")
		for _, feature := range group.Features {
			capability, exists := byFeature[feature]
			if !exists {
				capability = agentCapability{Feature: feature, Reason: reason}
			}
			cursor := "  "
			if feature == selectedFeature {
				cursor = "› "
			}
			glyph := "○"
			style := target.styles.status
			if capability.Available {
				glyph = "●"
				style = target.styles.success
			}
			builder.WriteString(cursor + style.Render(glyph+" "+agentFeatureTitle(feature)) + "\n")
		}
		builder.WriteString("\n")
	}
}

func (inspector inspectorState) renderRuntime(builder *strings.Builder, target app) {
	builder.WriteString(target.styles.accent.Render("RUNTIME") + "\n")
	if target.work != nil {
		if selected, ok := target.work.Selected(); ok && selected.Runtime != "" {
			builder.WriteString(target.styles.title.Render("assigned  "+selected.Runtime) + "\n")
		}
	}
	switch {
	case !inspector.runtime.ready:
		builder.WriteString(target.styles.status.Render("○ detection pending") + "\n\n")
	case inspector.runtime.reason != "":
		builder.WriteString(target.styles.attention.Render("○ unavailable") + "\n")
		builder.WriteString(target.styles.thought.Render(inspector.runtime.reason) + "\n\n")
	case len(inspector.runtime.capabilities) == 0:
		builder.WriteString(target.styles.status.Render("○ none available") + "\n\n")
	default:
		for index, capability := range inspector.runtime.capabilities {
			marker := "○"
			if index == 0 {
				marker = "●"
			}
			name := capability.Name
			if capability.Version != "" {
				name = core.Concat(name, "  ", capability.Version)
			}
			builder.WriteString(target.styles.success.Render(marker+" "+name) + "\n")
			features := runtimeFeatureLabels(capability)
			if len(features) > 0 {
				builder.WriteString(target.styles.thought.Render("  "+core.Join(" · ", features...)) + "\n")
			}
			if capability.Path != "" {
				builder.WriteString(target.styles.thought.Render("  "+capability.Path) + "\n")
			}
		}
		builder.WriteString("\n")
	}
}

func runtimeFeatureLabels(capability runtimeCapability) []string {
	features := make([]string, 0, 6)
	if capability.GPU {
		features = append(features, "GPU")
	}
	if capability.NetworkIsolation {
		features = append(features, "network isolation")
	}
	if capability.VolumeMounts {
		features = append(features, "volumes")
	}
	if capability.Encryption {
		features = append(features, "encryption")
	}
	if capability.HardwareIsolation {
		features = append(features, "hardware isolation")
	}
	if capability.SubSecondStart {
		features = append(features, "sub-second start")
	}
	return features
}

func (inspector inspectorState) renderChat(builder *strings.Builder, target app) {
	model := "○ none"
	if target.modelName != "" {
		model = "● " + target.modelName
	}
	generation := "○ idle"
	if target.generating {
		generation = "◉ generating"
	}
	builder.WriteString(target.styles.accent.Render("SESSION") + "  ● active\n")
	builder.WriteString(target.styles.accent.Render("MODEL") + "  " + model + "\n")
	builder.WriteString(target.styles.accent.Render("GENERATION") + "  " + generation + "\n\n")
	builder.WriteString(target.styles.accent.Render("SETTINGS") + "\n")
	inspector.renderControl(builder, target, inspectorControlContext, "context", core.Sprintf("%d", target.cfg.contextLen()))
	inspector.renderControl(builder, target, inspectorControlMaxTokens, "max tokens", core.Sprintf("%d", target.cfg.maxTokens()))
	inspector.renderControl(builder, target, inspectorControlThinking, "thinking", thinkNames[target.cfg.thinkIdx])
	inspector.renderControl(builder, target, inspectorControlTheme, "theme", inspector.Theme())
	builder.WriteString("\n" + target.styles.accent.Render("MODE") + "\n")
	inspector.renderControl(builder, target, inspectorControlMode, "sampling", target.modes.current().name)
	builder.WriteString("\n" + target.styles.accent.Render("TOOLS") + "\n")
	tools := "○ disabled"
	if target.tools.enabled {
		tools = "● enabled"
	}
	inspector.renderControl(builder, target, inspectorControlTools, "function calls", tools)
	builder.WriteString("\n" + target.styles.accent.Render("KNOWLEDGE") + "\n")
	switch {
	case !inspector.knowledge.ready:
		builder.WriteString(target.styles.status.Render("  ○ discovery pending") + "\n")
	case len(inspector.knowledge.documents) == 0:
		builder.WriteString(target.styles.status.Render("  ○ no local documents") + "\n")
	default:
		builder.WriteString(target.styles.success.Render(core.Sprintf("  ● %d local documents", len(inspector.knowledge.documents))) + "\n")
		for _, document := range inspector.knowledge.documents {
			builder.WriteString(target.styles.status.Render("    "+document.Title+"  "+document.Path) + "\n")
		}
	}
	if len(target.attachments) > 0 {
		builder.WriteString(target.styles.title.Render(core.Sprintf("  %d attached snapshots", len(target.attachments))) + "\n")
	}
	for _, warning := range inspector.knowledge.warnings {
		label := warning.Path
		if label == "" {
			label = warning.Mount
		}
		builder.WriteString(target.styles.attention.Render("  ! "+label) + "\n")
		builder.WriteString(target.styles.thought.Render("    "+warning.Reason) + "\n")
	}
	if inspector.dirty {
		builder.WriteString("\n" + target.styles.attention.Render("● unsaved · ctrl+s"))
	}
}

func (inspector inspectorState) renderControl(builder *strings.Builder, target app, control inspectorControl, label, value string) {
	cursor := "  "
	style := target.styles.status
	if inspector.cursor == int(control) {
		cursor = "› "
		style = target.styles.accent
	}
	builder.WriteString(cursor + style.Render(label) + "  " + target.styles.title.Render("‹ "+value+" ›") + "\n")
}

func wrapIndex(index, delta, length int) int {
	if length <= 0 {
		return 0
	}
	return ((index+delta)%length + length) % length
}

func themeIndex(name string) int {
	for index, candidate := range inspectorThemes {
		if candidate == name {
			return index
		}
	}
	return 0
}
