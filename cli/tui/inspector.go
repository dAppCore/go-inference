// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	_ "embed"

	core "dappco.re/go"
	"dappco.re/go/render/engine/html"
	"dappco.re/go/render/engine/ctml"
	"dappco.re/go/render/display/tui/style"
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

// inspectorCTML is the inspector's markup — see inspector.ctml for the
// seams it exposes (the four body gates, the knob/line sequences, class
// tokens).
//
//go:embed inspector.ctml
var inspectorCTML []byte

// View renders the contextual inspector through inspector.ctml: exactly one
// of the four panel-shaped bodies is gated in by the active panel, and the
// render is fitted to the pane as before.
func (inspector inspectorState) View(target app, width, height int) string {
	if width <= 0 || height <= 0 {
		return ""
	}
	tree, err := ctml.Parse(inspectorCTML, inspector.bindings(target))
	if err != nil {
		// inspector.ctml is embedded and static, so a parse failure is a
		// build defect; TestInspector_Good pins the markup as parseable.
		return ""
	}
	rendered := html.RenderTerm(tree, html.NewContext(), html.TermOptions{Width: width, Theme: inspectorTheme(target.styles)})
	return fitPane(rendered, width, height, target.styles.inspector)
}

// inspectorTheme maps the inspector markup's class tokens onto the existing
// palette, so the .ctml render reuses uiStyles paint exactly — no colours
// of its own. Unclassed text (the bound status values) takes the pane's own
// inspector style, exactly as the raw builder text previously inherited it
// from fitPane.
func inspectorTheme(styles uiStyles) *html.TermTheme {
	theme := html.DefaultTermTheme()
	theme.Text = styles.inspector
	theme.Classes = map[string]style.Style{
		"inspector-title": styles.title,
		"label":           styles.accent,
		"control-active":  styles.accent,
		"control-idle":    styles.status,
		"control-value":   styles.title,
		"c-title":         styles.title,
		"c-status":        styles.status,
		"c-thought":       styles.thought,
		"c-attention":     styles.attention,
		"c-success":       styles.success,
		"c-answer":        styles.answer,
	}
	return theme
}

// bindings assembles the inspector's sequences for the active panel: the
// panel's body gate holds one row, every other gate stays empty, so the
// document renders exactly one body.
func (inspector inspectorState) bindings(target app) ctml.Bindings {
	sequences := map[string][]map[string]any{
		"chatBody": {}, "chatSettings": {}, "chatMode": {}, "chatTools": {},
		"chatKnowledge": {}, "chatDirty": {},
		"workBody": {}, "workHead": {}, "workRuntime": {}, "workCap": {}, "workFeatures": {},
		"modelsBody": {}, "modelsHead": {},
		"serviceBody": {},
	}
	switch target.activePanel {
	case panelWork:
		inspector.workBindings(target, sequences)
	case panelModels:
		modelsHead := sequences["modelsHead"]
		if selected, ok := target.picker.SelectedItem().(modelItem); ok {
			modelsHead = append(modelsHead,
				map[string]any{"class": "c-title", "text": selected.name},
				map[string]any{"class": "c-status", "text": selected.modelType},
				map[string]any{"class": "c-thought", "text": selected.path},
			)
		} else {
			modelsHead = append(modelsHead, map[string]any{"class": "c-status", "text": "○ select a discovered model"})
		}
		sequences["modelsHead"] = modelsHead
		loadedClass, loaded := "c-status", "○ none"
		if target.modelName != "" {
			loadedClass, loaded = "c-success", "● "+target.modelName
		}
		sequences["modelsBody"] = append(sequences["modelsBody"], map[string]any{"loadedClass": loadedClass, "loaded": loaded})
	case panelService:
		stateClass, state := "c-status", "○ stopped"
		if target.svc.running {
			stateClass, state = "c-success", "● serving"
		}
		sequences["serviceBody"] = append(sequences["serviceBody"], map[string]any{
			"addr":       target.svc.addr(),
			"requests":   core.Sprintf("%d", target.svc.requests.Load()),
			"stateClass": stateClass,
			"state":      state,
		})
	default:
		inspector.chatBindings(target, sequences)
	}
	return ctml.Bindings{Sequences: sequences}
}

// controlRow is one knob row: the cursor and selection class ride the row
// (class="{{row.state}}"), so a single sequence serves every knob.
func (inspector inspectorState) controlRow(control inspectorControl, label, value string) map[string]any {
	state, cursor := "control-idle", "  "
	if inspector.cursor == int(control) {
		state, cursor = "control-active", "› "
	}
	return map[string]any{"state": state, "cursor": cursor, "label": label, "value": value}
}

// chatBindings fills the Chat body: the status head, the knob rows, and the
// flattened knowledge lines.
func (inspector inspectorState) chatBindings(target app, sequences map[string][]map[string]any) {
	model := "○ none"
	if target.modelName != "" {
		model = "● " + target.modelName
	}
	generation := "○ idle"
	if target.generating {
		generation = "◉ generating"
	}
	sequences["chatBody"] = append(sequences["chatBody"], map[string]any{"model": model, "generation": generation})
	sequences["chatSettings"] = append(sequences["chatSettings"],
		inspector.controlRow(inspectorControlContext, "context", core.Sprintf("%d", target.cfg.contextLen())),
		inspector.controlRow(inspectorControlMaxTokens, "max tokens", core.Sprintf("%d", target.cfg.maxTokens())),
		inspector.controlRow(inspectorControlThinking, "thinking", thinkNames[target.cfg.thinkIdx]),
		inspector.controlRow(inspectorControlTheme, "theme", inspector.Theme()),
	)
	sequences["chatMode"] = append(sequences["chatMode"],
		inspector.controlRow(inspectorControlMode, "sampling", target.modes.current().name))
	tools := "○ disabled"
	if target.tools.enabled {
		tools = "● enabled"
	}
	sequences["chatTools"] = append(sequences["chatTools"],
		inspector.controlRow(inspectorControlTools, "function calls", tools))

	knowledge := sequences["chatKnowledge"]
	line := func(class, text string) {
		knowledge = append(knowledge, map[string]any{"class": class, "text": text})
	}
	switch {
	case !inspector.knowledge.ready:
		line("c-status", "  ○ discovery pending")
	case len(inspector.knowledge.documents) == 0:
		line("c-status", "  ○ no local documents")
	default:
		line("c-success", core.Sprintf("  ● %d local documents", len(inspector.knowledge.documents)))
		for _, document := range inspector.knowledge.documents {
			line("c-status", "    "+document.Title+"  "+document.Path)
		}
	}
	if len(target.attachments) > 0 {
		line("c-title", core.Sprintf("  %d attached snapshots", len(target.attachments)))
	}
	for _, warning := range inspector.knowledge.warnings {
		label := warning.Path
		if label == "" {
			label = warning.Mount
		}
		line("c-attention", "  ! "+label)
		line("c-thought", "    "+warning.Reason)
	}
	sequences["chatKnowledge"] = knowledge

	if inspector.dirty {
		sequences["chatDirty"] = append(sequences["chatDirty"], map[string]any{})
	}
}

// workBindings fills the Work body: the selection head, the runtime lines,
// the capability summary, and the flattened feature-group lines (group
// headings, feature rows, and the blank separators between groups all ride
// one sequence, each line choosing its own palette class).
func (inspector inspectorState) workBindings(target app, sequences map[string][]map[string]any) {
	sequences["workBody"] = append(sequences["workBody"], map[string]any{})

	head := sequences["workHead"]
	if target.work == nil {
		head = append(head, map[string]any{"class": "c-status", "text": "○ no work item selected"})
	} else if selected, ok := target.work.Selected(); ok {
		head = append(head,
			map[string]any{"class": "c-title", "text": selected.Title},
			map[string]any{"class": "c-status", "text": workGlyph(selected.Status) + " " + core.Upper(selected.Status)},
		)
		if selected.Repo != "" {
			head = append(head, map[string]any{"class": "c-thought", "text": selected.Repo + "  " + selected.Branch})
		}
		if selected.Question != "" {
			head = append(head, map[string]any{"class": "c-attention", "text": "? " + selected.Question})
		}
	} else {
		head = append(head, map[string]any{"class": "c-status", "text": "○ no work item selected"})
	}
	sequences["workHead"] = head

	runtime := sequences["workRuntime"]
	if target.work != nil {
		if selected, ok := target.work.Selected(); ok && selected.Runtime != "" {
			runtime = append(runtime, map[string]any{"class": "c-title", "text": "assigned  " + selected.Runtime})
		}
	}
	switch {
	case !inspector.runtime.ready:
		runtime = append(runtime, map[string]any{"class": "c-status", "text": "○ detection pending"})
	case inspector.runtime.reason != "":
		runtime = append(runtime,
			map[string]any{"class": "c-attention", "text": "○ unavailable"},
			map[string]any{"class": "c-thought", "text": inspector.runtime.reason},
		)
	case len(inspector.runtime.capabilities) == 0:
		runtime = append(runtime, map[string]any{"class": "c-status", "text": "○ none available"})
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
			runtime = append(runtime, map[string]any{"class": "c-success", "text": marker + " " + name})
			if features := runtimeFeatureLabels(capability); len(features) > 0 {
				runtime = append(runtime, map[string]any{"class": "c-thought", "text": "  " + core.Join(" · ", features...)})
			}
			if capability.Path != "" {
				runtime = append(runtime, map[string]any{"class": "c-thought", "text": "  " + capability.Path})
			}
		}
	}
	sequences["workRuntime"] = runtime

	capabilities := []agentCapability{}
	if target.work != nil {
		capabilities = target.work.Capabilities()
	} else if target.agent != nil {
		capabilities = target.agent.Capabilities()
	}
	byFeature := make(map[agentFeature]agentCapability, len(capabilities))
	reason := defaultAgentUnavailableReason
	available := 0
	for _, capability := range capabilities {
		byFeature[capability.Feature] = capability
		if !capability.Available && capability.Reason != "" {
			reason = capability.Reason
		}
		if capability.Available {
			available++
		}
	}
	if available == 0 {
		sequences["workCap"] = append(sequences["workCap"],
			map[string]any{"class": "c-attention", "text": "○ not installed"},
			map[string]any{"class": "c-thought", "text": reason},
		)
	} else {
		sequences["workCap"] = append(sequences["workCap"],
			map[string]any{"class": "c-success", "text": core.Sprintf("● %d actions available", available)})
	}

	selectedFeature := agentFeature("")
	if target.work != nil {
		selectedFeature = target.work.SelectedAction().Feature
	}
	features := sequences["workFeatures"]
	for groupIndex, group := range agentFeatureGroups {
		if groupIndex > 0 {
			features = append(features, map[string]any{"class": "", "text": ""})
		}
		features = append(features, map[string]any{"class": "label", "text": group.Title})
		for _, feature := range group.Features {
			capability, exists := byFeature[feature]
			if !exists {
				capability = agentCapability{Feature: feature, Reason: reason}
			}
			cursor := "  "
			if feature == selectedFeature {
				cursor = "› "
			}
			glyph, class := "○", "c-status"
			if capability.Available {
				glyph, class = "●", "c-success"
			}
			features = append(features, map[string]any{"class": class, "text": cursor + glyph + " " + agentFeatureTitle(feature)})
		}
	}
	sequences["workFeatures"] = features
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
