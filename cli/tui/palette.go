// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	_ "embed"

	"github.com/charmbracelet/bubbles/help"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/textinput"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/charmbracelet/x/ansi"

	core "dappco.re/go"
	"dappco.re/go/html"
	"dappco.re/go/html/ctml"
	coreio "dappco.re/go/io"
)

type commandID string

const (
	commandNewSession       commandID = "session.new"
	commandSwitchSession    commandID = "session.switch"
	commandSearchHistory    commandID = "history.search"
	commandToggleInspector  commandID = "inspector.toggle"
	commandShowHelp         commandID = "help.show"
	commandPanelChat        commandID = "panel.chat"
	commandPanelWork        commandID = "panel.work"
	commandPanelModels      commandID = "panel.models"
	commandPanelService     commandID = "panel.service"
	commandPanelData        commandID = "panel.data"
	commandSaveSettings     commandID = "settings.save"
	commandExportMarkdown   commandID = "session.export.markdown"
	commandExportJSON       commandID = "session.export.json"
	commandRefreshWork      commandID = "work.refresh"
	commandRefreshRuntimes  commandID = "runtimes.refresh"
	commandRefreshKnowledge commandID = "knowledge.refresh"
	commandNewWork          commandID = "work.new"
	commandEditWork         commandID = "work.edit"
)

type workspaceCommand struct {
	ID          commandID
	Title       string
	Description string
	Available   bool
	Reason      string
	run         func(*app) core.Result
}

type commandListItem struct{ command workspaceCommand }

func (item commandListItem) Title() string { return item.command.Title }
func (item commandListItem) Description() string {
	if item.command.Available || item.command.Reason == "" {
		return item.command.Description
	}
	return core.Concat(item.command.Description, " — unavailable: ", item.command.Reason)
}
func (item commandListItem) FilterValue() string {
	return core.Concat(string(item.command.ID), " ", item.command.Title, " ", item.command.Description)
}

type commandPalette struct {
	commands        []workspaceCommand
	byID            map[commandID]workspaceCommand
	list            list.Model
	exporter        sessionExporter
	exportMedium    coreio.Medium
	exportDirectory string
}

func newCommandPalette(styles uiStyles) *commandPalette {
	commands := defaultWorkspaceCommands()
	items := make([]list.Item, 0, len(commands))
	byID := make(map[commandID]workspaceCommand, len(commands))
	for _, command := range commands {
		items = append(items, commandListItem{command: command})
		byID[command.ID] = command
	}
	model := list.New(items, list.NewDefaultDelegate(), 68, 16)
	model.Title = "Commands"
	model.SetShowStatusBar(false)
	model.SetFilteringEnabled(true)
	model.SetFilterState(list.Filtering)
	return &commandPalette{commands: commands, byID: byID, list: model}
}

func (palette *commandPalette) Filter(query string) []workspaceCommand {
	if palette == nil {
		return nil
	}
	query = core.Trim(query)
	if query == "" {
		return append([]workspaceCommand(nil), palette.commands...)
	}
	targets := make([]string, len(palette.commands))
	for i, command := range palette.commands {
		targets[i] = commandListItem{command: command}.FilterValue()
	}
	ranks := list.DefaultFilter(query, targets)
	matched := make([]workspaceCommand, 0, len(ranks))
	for _, rank := range ranks {
		matched = append(matched, palette.commands[rank.Index])
	}
	return matched
}

func (palette *commandPalette) Invoke(id commandID, target *app) core.Result {
	if palette == nil {
		return core.Fail(core.E("tui.commandPalette.Invoke", "command palette is unavailable", nil))
	}
	if target != nil {
		target.refreshAgentPalette()
	}
	command, exists := palette.byID[id]
	if !exists {
		return core.Fail(core.E("tui.commandPalette.Invoke", core.Concat("unknown command: ", string(id)), nil))
	}
	if !command.Available {
		return core.Fail(core.E("tui.commandPalette.Invoke", core.Concat(command.Title, " is unavailable: ", command.Reason), nil))
	}
	if target == nil || command.run == nil {
		return core.Fail(core.E("tui.commandPalette.Invoke", "command target is unavailable", nil))
	}
	return command.run(target)
}

func (palette *commandPalette) SelectedID() commandID {
	if palette == nil {
		return ""
	}
	item, ok := palette.list.SelectedItem().(commandListItem)
	if !ok {
		return ""
	}
	return item.command.ID
}

func (palette *commandPalette) Open() {
	if palette == nil {
		return
	}
	palette.list.SetFilterText("")
	palette.list.SetFilterState(list.Filtering)
}

func (palette *commandPalette) Update(message tea.Msg) tea.Cmd {
	if palette == nil {
		return nil
	}
	var command tea.Cmd
	palette.list, command = palette.list.Update(message)
	return command
}

// paletteCTML is the command palette overlay's markup — see palette.ctml
// for the seams it exposes (row/empty/page sequences, class tokens, the
// row box ids).
//
//go:embed palette.ctml
var paletteCTML []byte

// commandPaletteBindings derives the palette's rows from the list model's
// own state: the current page of visible commands as ONE sequence —
// selection styling rides the row-scoped class bind (class="{{row.state}}",
// go-html v0.13.0) and the marker glyph rides the row, the same idiom
// picker.ctml established — plus the zero-or-one-row conditional sections
// for the empty and page states. detail already carries the
// "— unavailable: reason" suffix (commandListItem.Description), so an
// unavailable command renders with the same row-idle/row-active paint an
// available one gets — matching the original list.DefaultDelegate, which
// never styled the two states apart. Title and detail truncate to the F
// band's own row budget (width-2, docs/ctml.md S:15.5) because a wrapped
// row would overflow the page the list delegate sized.
func commandPaletteBindings(palette *commandPalette, width int) ctml.Bindings {
	sequences := map[string][]map[string]any{
		"rows":  {},
		"empty": {},
		"page":  {},
	}
	if palette == nil {
		return ctml.Bindings{Sequences: sequences}
	}
	visible := palette.list.VisibleItems()
	start, end := palette.list.Paginator.GetSliceBounds(len(visible))
	active := palette.list.Index() - start
	budget := max(1, width-2)
	for index, raw := range visible[start:end] {
		entry, ok := raw.(commandListItem)
		if !ok {
			continue
		}
		state, marker := "row-idle", "○"
		if index == active {
			state, marker = "row-active", "›"
		}
		sequences["rows"] = append(sequences["rows"], map[string]any{
			"state":  state,
			"marker": marker,
			"id":     string(entry.command.ID),
			"title":  ansi.Truncate(entry.Title(), budget, "…"),
			"detail": ansi.Truncate(entry.Description(), budget, "…"),
		})
	}
	if len(visible) == 0 && palette.list.FilterState() != list.Filtering {
		sequences["empty"] = append(sequences["empty"], map[string]any{"text": "No items."})
	}
	if palette.list.Paginator.TotalPages > 1 {
		sequences["page"] = append(sequences["page"], map[string]any{
			"label": core.Sprintf("page %d/%d", palette.list.Paginator.Page+1, palette.list.Paginator.TotalPages),
		})
	}
	return ctml.Bindings{Sequences: sequences}
}

// commandPaletteTheme maps the markup's class tokens onto the existing
// palette. It carries the same band chrome every overlay uses
// (overlayFrameTheme's blank header / one-row-padded footer, so the
// palette's key-hint line keeps the same gap every overlay draws above its
// footer) plus the row-selection classes picker.ctml established, since
// the palette is the first overlay to carry a selectable row list — a
// plain renderOverlayFrame call has no seam for those extra classes, so
// this theme is built directly and rendered through renderBandFrame, the
// same bypass datalist.ctml uses to borrow the overlay band chrome from a
// primary panel.
func commandPaletteTheme(styles uiStyles) *html.TermTheme {
	theme := html.DefaultTermTheme()
	theme.Text = styles.answer
	theme.Heading = styles.title
	theme.Header = lipgloss.NewStyle()
	theme.Footer = lipgloss.NewStyle().Padding(1, 0, 0, 0)
	theme.Classes = map[string]lipgloss.Style{
		"row-idle":      styles.answer,
		"row-active":    styles.accent,
		"row-hint":      styles.thought,
		"palette-empty": styles.status,
		"palette-page":  styles.status,
		"overlay-keys":  styles.status,
	}
	return theme
}

// View renders the palette overlay through palette.ctml: the fixed
// "Commands" header, the live Bubbles filter input composed between the
// bands while filtering (the overlays' HF chrome+widget idiom, exactly
// datalist.ctml's own borrowing of it, lent back to an actual overlay),
// and the row/empty/page/key-hint footer beneath it. Cursor movement,
// fuzzy filtering, and pagination stay in list.Model — this is the render
// swap over the old palette.list.View().
func (palette *commandPalette) View(width, height int, styles uiStyles) string {
	if palette == nil {
		return ""
	}
	palette.list.SetSize(max(1, width), max(6, height))
	head, foot := renderBandFrame(paletteCTML, width, commandPaletteTheme(styles), commandPaletteBindings(palette, width))
	parts := []string{head}
	if palette.list.SettingFilter() {
		parts = append(parts, palette.list.FilterInput.View())
	}
	parts = append(parts, foot)
	return fitPane(core.Join("\n", parts...), width, height, styles.panel)
}

func (palette *commandPalette) SetAgentCapabilities(capabilities []agentCapability) {
	palette.SetAgentContext(capabilities, nil)
}

func (palette *commandPalette) SetAgentContext(capabilities []agentCapability, selected *workItemRecord, state ...agentWorkSnapshot) {
	if palette == nil {
		return
	}
	commands := make([]workspaceCommand, 0, len(palette.commands)+len(capabilities))
	for _, command := range palette.commands {
		if !core.HasPrefix(string(command.ID), "agent.") {
			commands = append(commands, command)
		}
	}
	commands = append(commands, agentWorkspaceCommandsForContext(capabilities, selected, state...)...)
	items := make([]list.Item, 0, len(commands))
	byID := make(map[commandID]workspaceCommand, len(commands))
	for _, command := range commands {
		items = append(items, commandListItem{command: command})
		byID[command.ID] = command
	}
	palette.commands = commands
	palette.byID = byID
	palette.list.SetItems(items)
}

func (palette *commandPalette) SetWorkSelection(hasSelectedWork bool) {
	if palette == nil {
		return
	}
	for index := range palette.commands {
		if palette.commands[index].ID != commandEditWork {
			continue
		}
		palette.commands[index].Available = hasSelectedWork
		palette.commands[index].Reason = ""
		if !hasSelectedWork {
			palette.commands[index].Reason = "a selected Work item is required"
		}
		palette.byID[commandEditWork] = palette.commands[index]
	}
	items := make([]list.Item, 0, len(palette.commands))
	for _, command := range palette.commands {
		items = append(items, commandListItem{command: command})
	}
	palette.list.SetItems(items)
}

// SetDataContext rebuilds every "data."-prefixed command against a live
// dataPanel.Capabilities() snapshot — the SetAgentContext precedent
// applied to the Data panel, called after every selection or status
// change (app.refreshDataPalette).
func (palette *commandPalette) SetDataContext(capabilities []dataCapability) {
	if palette == nil {
		return
	}
	commands := make([]workspaceCommand, 0, len(palette.commands)+len(capabilities))
	for _, command := range palette.commands {
		if !core.HasPrefix(string(command.ID), "data.") {
			commands = append(commands, command)
		}
	}
	commands = append(commands, dataWorkspaceCommandsForContext(capabilities)...)
	items := make([]list.Item, 0, len(commands))
	byID := make(map[commandID]workspaceCommand, len(commands))
	for _, command := range commands {
		items = append(items, commandListItem{command: command})
		byID[command.ID] = command
	}
	palette.commands = commands
	palette.byID = byID
	palette.list.SetItems(items)
}

func (palette *commandPalette) SetExporter(exporter sessionExporter, medium coreio.Medium, directory string) {
	if palette == nil {
		return
	}
	palette.exporter = exporter
	palette.exportMedium = medium
	palette.exportDirectory = directory
	available := exporter != nil && medium != nil
	for index := range palette.commands {
		format := exportFormat("")
		switch palette.commands[index].ID {
		case commandExportMarkdown:
			format = exportMarkdown
		case commandExportJSON:
			format = exportJSON
		default:
			continue
		}
		palette.commands[index].Available = available
		palette.commands[index].Reason = ""
		if !available {
			palette.commands[index].Reason = "export adapter not connected"
		}
		palette.commands[index].run = func(target *app) core.Result {
			return palette.runExport(target, format)
		}
	}
	items := make([]list.Item, 0, len(palette.commands))
	palette.byID = make(map[commandID]workspaceCommand, len(palette.commands))
	for _, command := range palette.commands {
		items = append(items, commandListItem{command: command})
		palette.byID[command.ID] = command
	}
	palette.list.SetItems(items)
}

func (palette *commandPalette) runExport(target *app, format exportFormat) core.Result {
	if palette == nil || palette.exporter == nil || palette.exportMedium == nil {
		return core.Fail(core.E("tui.command.export", "export adapter is unavailable", nil))
	}
	if target == nil || target.repository == nil {
		return core.Fail(core.E("tui.command.export", "workspace repository is unavailable", nil))
	}
	result := palette.exporter.Export(
		palette.exportMedium,
		palette.exportDirectory,
		target.sessionID,
		format,
	)
	if !result.OK {
		return result
	}
	receipt, ok := result.Value.(exportReceipt)
	if !ok {
		return core.Fail(core.E("tui.command.export", "invalid export receipt", nil))
	}
	artifact := artifactRecord{
		ID:           newRecordID(),
		SessionID:    receipt.SessionID,
		Kind:         core.Concat("export.", string(receipt.Format)),
		Path:         receipt.Path,
		Title:        core.Concat("Session export · ", receipt.Title),
		MetadataJSON: core.JSONMarshalString(receipt),
		CreatedAt:    receipt.ExportedAt,
		ArchivedAt:   unsetRecordTime(),
	}
	if saved := target.repository.SaveArtifact(artifact); !saved.OK {
		return core.Fail(core.E("tui.command.export", core.Concat("export written but artifact persistence failed: ", receipt.Path), resultError(saved)))
	}
	return core.Ok(receipt)
}

func defaultWorkspaceCommands() []workspaceCommand {
	panelCommand := func(id commandID, title string, panel panelID) workspaceCommand {
		return workspaceCommand{ID: id, Title: title, Description: "Go to the " + panelNames[panel] + " panel", Available: true, run: func(target *app) core.Result {
			target.activePanel = panel
			return core.Ok(nil)
		}}
	}
	unavailable := func(id commandID, title, description, reason string) workspaceCommand {
		return workspaceCommand{ID: id, Title: title, Description: description, Reason: reason}
	}
	commands := []workspaceCommand{
		{ID: commandNewSession, Title: "New session", Description: "Create and open a blank chat session", Available: true, run: func(target *app) core.Result { return target.createSession() }},
		{ID: commandSwitchSession, Title: "Switch session", Description: "Open the recent-session switcher", Available: true, run: func(target *app) core.Result { return target.openSessionSwitcher() }},
		{ID: commandSearchHistory, Title: "Search history", Description: "Search durable chat titles and turns", Available: true, run: func(target *app) core.Result { return target.openHistorySearch() }},
		{ID: commandToggleInspector, Title: "Toggle inspector", Description: "Show or hide contextual session details", Available: true, run: func(target *app) core.Result {
			target.toggleInspector()
			return core.Ok(nil)
		}},
		{ID: commandShowHelp, Title: "Show help", Description: "Open the complete keyboard reference", Available: true, run: func(target *app) core.Result {
			target.activeOverlay = overlayHelp
			return core.Ok(nil)
		}},
		panelCommand(commandPanelChat, "Chat panel", panelChat),
		panelCommand(commandPanelWork, "Work panel", panelWork),
		panelCommand(commandPanelModels, "Models panel", panelModels),
		panelCommand(commandPanelService, "Service panel", panelService),
		panelCommand(commandPanelData, "Data panel", panelData),
		{ID: commandSaveSettings, Title: "Save settings", Description: "Commit generation and appearance preferences", Available: true, run: func(target *app) core.Result {
			return target.inspector.Save(target)
		}},
		{ID: commandNewWork, Title: "New Work", Description: "Create a reviewed agent Work item", Available: true, run: func(target *app) core.Result {
			return target.openWorkEditor(workItemRecord{})
		}},
		{ID: commandEditWork, Title: "Edit Work", Description: "Edit the selected Work item", Reason: "a selected Work item is required", run: func(target *app) core.Result {
			if target == nil || target.work == nil {
				return core.Fail(core.E("tui.command.work.edit", "work panel is unavailable", nil))
			}
			record, ok := target.work.Selected()
			if !ok {
				return core.Fail(core.E("tui.command.work.edit", "a selected Work item is required", nil))
			}
			return target.openWorkEditor(record)
		}},
		unavailable(commandExportMarkdown, "Export Markdown", "Export the active session as Markdown", "export adapter not connected"),
		unavailable(commandExportJSON, "Export JSON", "Export the active session as structured JSON", "export adapter not connected"),
		{ID: commandRefreshWork, Title: "Refresh work", Description: "Refresh local work and provider snapshots", Available: true, run: func(target *app) core.Result {
			if target == nil || target.work == nil {
				return core.Fail(core.E("tui.command.refreshWork", "work panel is unavailable", nil))
			}
			target.agentCommand = target.requestAgentSnapshot()
			return core.Ok(nil)
		}},
		unavailable(commandRefreshRuntimes, "Refresh runtimes", "Refresh local runtime capabilities", "manual refresh is not connected; restart LEM to rescan"),
		unavailable(commandRefreshKnowledge, "Refresh knowledge", "Refresh local knowledge packs", "manual refresh is not connected; restart LEM to rescan"),
	}
	commands = append(commands, agentWorkspaceCommands(agentFeatureCatalog(defaultAgentUnavailableReason))...)
	// The static data.* catalogue is informational, same precedent as
	// agentWorkspaceCommands above — runtime invokability is rebuilt
	// through SetDataContext once an app has a connected Data panel.
	// (*dataPanel)(nil).Capabilities() is safe: every dataPanel method is
	// nil-receiver-guarded, matching workPanel's own idiom.
	return append(commands, dataWorkspaceCommandsForContext((*dataPanel)(nil).Capabilities())...)
}

func agentCommandID(feature agentFeature) commandID {
	return commandID(core.Concat("agent.", string(feature)))
}

func agentWorkspaceCommands(capabilities []agentCapability) []workspaceCommand {
	// The static catalogue is informational. Runtime invokability is rebuilt
	// through SetAgentContext once an app has a Work selection.
	return agentWorkspaceCommandsForContext(capabilities, nil)
}

func agentWorkspaceCommandsForContext(capabilities []agentCapability, selected *workItemRecord, state ...agentWorkSnapshot) []workspaceCommand {
	commands := make([]workspaceCommand, 0, len(capabilities))
	for _, capability := range capabilities {
		capability := capability
		available, reason := agentCommandAvailability(capability, selected, state...)
		commands = append(commands, workspaceCommand{
			ID:          agentCommandID(capability.Feature),
			Title:       agentFeatureTitle(capability.Feature),
			Description: core.Concat("Agent capability · ", string(capability.Feature)),
			Available:   available,
			Reason:      reason,
			run: func(target *app) core.Result {
				if target == nil || target.work == nil {
					return core.Fail(core.E("tui.agentCommand", "work panel is unavailable", nil))
				}
				target.activePanel = panelWork
				target.inspectorOpen = true
				return target.queueAgentAction(capability.Feature)
			},
		})
	}
	return commands
}

func agentCommandAvailability(capability agentCapability, selected *workItemRecord, state ...agentWorkSnapshot) (bool, string) {
	if !capability.Available {
		return false, capability.Reason
	}
	selectedState := agentWorkSnapshot{}
	hasSnapshotState := len(state) > 0
	if hasSnapshotState {
		selectedState = state[0]
	}
	if capability.Feature == agentFeatureQueueStart || capability.Feature == agentFeatureQueueStop {
		if !hasSnapshotState {
			return true, ""
		}
		queue := core.Lower(core.Trim(selectedState.QueueStatus))
		if queue == "" {
			return true, ""
		}
		if capability.Feature == agentFeatureQueueStart {
			return queue == "frozen", "queue is not frozen; draining completes existing work before it can restart"
		}
		return queue == "accepting", "queue is not accepting; it is already frozen or draining"
	}
	if !agentFeatureNeedsWork(capability.Feature) {
		return true, ""
	}
	if selected == nil {
		return false, "a selected Work item is required"
	}
	status := core.Lower(core.Trim(selected.Status))
	allowed := false
	reason := "selected Work is not in a state that allows this action"
	switch capability.Feature {
	case agentFeatureDispatch:
		allowed = status == "" || status == workStatusActive || status == "ready"
		reason = "selected Work must be ready before it can dispatch"
	case agentFeatureCancel:
		allowed = (!hasSnapshotState || selectedState.NativeRunID != "") && (status == "queued" || status == "running")
		reason = "selected Work is not queued or running"
	case agentFeatureAnswer:
		allowed = (!hasSnapshotState || (selectedState.NativeRunID != "" && selectedState.QuestionID != "" && selectedState.AnswerID == "")) && (status == workStatusWaiting || status == "question" || status == "blocked" || status == "needs_input")
		reason = "selected Work is not waiting for an answer"
	case agentFeatureRetry:
		allowed = (!hasSnapshotState || selectedState.NativeRunID != "") && (status == workStatusFailed || status == "error" || status == "cancelled" || status == "canceled")
		reason = "selected Work is not failed or cancelled"
	case agentFeatureResume:
		waiting := status == workStatusWaiting || status == "question" || status == "blocked" || status == "needs_input"
		interrupted := status == "interrupted"
		allowed = (waiting || interrupted) && (!hasSnapshotState || (selectedState.NativeRunID != "" && ((waiting && selectedState.AnswerID != "") || interrupted)))
		reason = "answer the selected native run before resuming it"
	case agentFeatureChangesReview:
		if !hasSnapshotState {
			return false, "change review overlay is scheduled for Task 14"
		}
		if selectedState.RecoveryCount > 0 || selectedState.Recovery.EventID != "" {
			return false, "resolve the retained cleanup recovery before reviewing changes"
		}
		return selectedState.NativeRunID != "" && status == "completed", "selected native run is not completed and reviewable"
	case agentFeatureAccept:
		if !hasSnapshotState {
			return false, "no durable review-ready state is exposed yet"
		}
		return selectedState.ReviewID != "" && selectedState.ReviewStatus == "prepared" && selectedState.Review.Payload != nil && selectedState.Review.AcceptanceAllowed, "review changes first; conflicts and failed validation cannot be accepted"
	case agentFeatureReject:
		if !hasSnapshotState {
			return false, "no durable review-ready state is exposed yet"
		}
		return selectedState.ReviewID != "" && selectedState.ReviewStatus != "accepted" && selectedState.ReviewStatus != "rejected", "review changes first before rejecting a native run"
	case agentFeatureRecoveryAbandon:
		if !hasSnapshotState {
			return false, "no retained recovery is exposed for the selected Work"
		}
		return selectedState.Recovery.EventID != "", "selected Work has no retained recovery"
	}
	if !allowed {
		return false, reason
	}
	return true, ""
}

func agentFeatureNeedsWork(feature agentFeature) bool {
	switch feature {
	case agentFeatureDispatch, agentFeatureCancel, agentFeatureAnswer, agentFeatureRetry,
		agentFeatureResume, agentFeatureChangesReview, agentFeatureAccept, agentFeatureReject, agentFeatureRecoveryAbandon:
		return true
	default:
		return false
	}
}

// dataCommandID builds the palette command id for a Data panel action —
// "data.<action>" for a single-item action, "data.bulk.<action>" for its
// bulk-apply-to-filter counterpart — mirroring agentCommandID's "agent."
// namespace one level down.
func dataCommandID(action dataAction, bulk bool) commandID {
	name := dataActionSlug(action)
	if bulk {
		return commandID(core.Concat("data.bulk.", name))
	}
	return commandID(core.Concat("data.", name))
}

func dataActionSlug(action dataAction) string {
	switch action {
	case dataActionApprove:
		return "approve"
	case dataActionReject:
		return "reject"
	case dataActionQuarantineClear:
		return "quarantine-clear"
	case dataActionEditAsDerived:
		return "edit-as-derived"
	case dataActionTag:
		return "tag"
	default:
		return "unknown"
	}
}

// dataWorkspaceCommandsForContext builds the "data."-prefixed palette
// commands from capabilities (dataPanel.Capabilities(), already carrying
// live Available/Reason per action) — the agentcap pattern applied to the
// Data panel: every action always appears, unavailable ones render their
// reason rather than hiding (commandListItem.Description). Every run
// closure forces the Data panel into view before dispatching, mirroring
// agentWorkspaceCommandsForContext's own target.activePanel assignment.
func dataWorkspaceCommandsForContext(capabilities []dataCapability) []workspaceCommand {
	commands := make([]workspaceCommand, 0, len(capabilities))
	for _, capability := range capabilities {
		capability := capability
		commands = append(commands, workspaceCommand{
			ID:          dataCommandID(capability.Action, capability.Bulk),
			Title:       capability.Title,
			Description: core.Concat("Data review · ", capability.Title),
			Available:   capability.Available,
			Reason:      capability.Reason,
			run: func(target *app) core.Result {
				return target.runDataCommand(capability)
			},
		})
	}
	return commands
}

type sessionSwitcherItem struct {
	SessionID   string
	Title       string
	Description string
}

type sessionListItem struct{ value sessionSwitcherItem }

func (item sessionListItem) Title() string       { return item.value.Title }
func (item sessionListItem) Description() string { return item.value.Description }
func (item sessionListItem) FilterValue() string {
	return core.Concat(item.value.Title, " ", item.value.Description)
}

type sessionSwitcher struct {
	manager *sessionManager
	items   []sessionSwitcherItem
	list    list.Model
}

func newSessionSwitcher(manager *sessionManager, styles uiStyles, width, height int) core.Result {
	if manager == nil {
		return core.Fail(core.E("tui.newSessionSwitcher", "session manager is unavailable", nil))
	}
	model := list.New(nil, list.NewDefaultDelegate(), width, height)
	model.Title = "Recent sessions"
	model.SetShowStatusBar(false)
	model.SetFilteringEnabled(true)
	model.Styles.Title = styles.title
	switcher := &sessionSwitcher{manager: manager, list: model}
	switcher.Refresh()
	return core.Ok(switcher)
}

func (switcher *sessionSwitcher) Refresh() {
	if switcher == nil || switcher.manager == nil {
		return
	}
	sessions := switcher.manager.Recent()
	switcher.items = make([]sessionSwitcherItem, 0, len(sessions))
	items := make([]list.Item, 0, len(sessions))
	for _, session := range sessions {
		marker := sessionMarker(session)
		model := session.Record.PreferredModel
		if model == "" {
			model = "no preferred model"
		}
		item := sessionSwitcherItem{
			SessionID:   session.Record.ID,
			Title:       core.Concat(marker, " ", session.Record.Title),
			Description: core.Concat(session.Record.Status, " · ", model),
		}
		switcher.items = append(switcher.items, item)
		items = append(items, sessionListItem{value: item})
	}
	switcher.list.SetItems(items)
}

func sessionMarker(session *chatSession) string {
	if session == nil {
		return "○"
	}
	if session.Attention {
		return "!"
	}
	switch session.Record.Status {
	case "generating":
		return "◉"
	case "queued":
		return "◌"
	case "failed":
		return "×"
	default:
		return "○"
	}
}

func (switcher *sessionSwitcher) Items() []sessionSwitcherItem {
	if switcher == nil {
		return nil
	}
	return append([]sessionSwitcherItem(nil), switcher.items...)
}

func (switcher *sessionSwitcher) Update(message tea.Msg) tea.Cmd {
	if switcher == nil {
		return nil
	}
	var command tea.Cmd
	switcher.list, command = switcher.list.Update(message)
	return command
}

func (switcher *sessionSwitcher) ActivateSelected() core.Result {
	if switcher == nil || switcher.manager == nil {
		return core.Fail(core.E("tui.sessionSwitcher.ActivateSelected", "session switcher is unavailable", nil))
	}
	item, ok := switcher.list.SelectedItem().(sessionListItem)
	if !ok {
		return core.Fail(core.E("tui.sessionSwitcher.ActivateSelected", "no session is selected", nil))
	}
	return switcher.manager.Switch(item.value.SessionID)
}

func (switcher *sessionSwitcher) View(width, height int) string {
	if switcher == nil {
		return ""
	}
	switcher.list.SetSize(max(1, width), max(6, height))
	return switcher.list.View()
}

type historySearchItem struct {
	hit sessionSearchHit
}

func (item historySearchItem) Title() string       { return item.hit.Session.Title }
func (item historySearchItem) Description() string { return item.hit.Snippet }
func (item historySearchItem) FilterValue() string {
	return core.Concat(item.hit.Session.Title, " ", item.hit.Snippet)
}

type historySearch struct {
	repository workspaceRepository
	manager    *sessionManager
	input      textinput.Model
	list       list.Model
	hits       []sessionSearchHit
	query      string
	matchTurn  string
}

func newHistorySearch(repository workspaceRepository, manager *sessionManager, styles uiStyles, width, height int) core.Result {
	if repository == nil || manager == nil {
		return core.Fail(core.E("tui.newHistorySearch", "repository and session manager are required", nil))
	}
	input := textinput.New()
	input.Placeholder = "Search titles and turns…"
	input.Prompt = "› "
	input.Focus()
	model := list.New(nil, list.NewDefaultDelegate(), width, max(4, height-2))
	model.Title = "History"
	model.SetShowStatusBar(false)
	model.SetFilteringEnabled(false)
	model.Styles.Title = styles.title
	return core.Ok(&historySearch{repository: repository, manager: manager, input: input, list: model})
}

func (search *historySearch) Search(query string) core.Result {
	if search == nil || search.repository == nil {
		return core.Fail(core.E("tui.historySearch.Search", "history search is unavailable", nil))
	}
	query = core.Trim(query)
	search.query = query
	search.input.SetValue(query)
	result := search.repository.SearchSessions(query, 50)
	if !result.OK {
		return result
	}
	hits, ok := result.Value.([]sessionSearchHit)
	if !ok {
		return core.Fail(core.E("tui.historySearch.Search", "invalid search result", nil))
	}
	search.hits = append([]sessionSearchHit(nil), hits...)
	items := make([]list.Item, 0, len(hits))
	for _, hit := range hits {
		items = append(items, historySearchItem{hit: hit})
	}
	search.list.SetItems(items)
	return core.Ok(search.Hits())
}

func (search *historySearch) Hits() []sessionSearchHit {
	if search == nil {
		return nil
	}
	return append([]sessionSearchHit(nil), search.hits...)
}

func (search *historySearch) MatchTurnID() string {
	if search == nil {
		return ""
	}
	return search.matchTurn
}

func (search *historySearch) ActivateSelected() core.Result {
	if search == nil || search.manager == nil {
		return core.Fail(core.E("tui.historySearch.ActivateSelected", "history search is unavailable", nil))
	}
	item, ok := search.list.SelectedItem().(historySearchItem)
	if !ok {
		return core.Fail(core.E("tui.historySearch.ActivateSelected", "no history result is selected", nil))
	}
	if result := search.manager.Switch(item.hit.Session.ID); !result.OK {
		return result
	}
	session := search.manager.Active()
	position := 0
	search.matchTurn = ""
	needle := core.Lower(search.query)
	for index, turn := range session.Turns {
		if core.Contains(core.Lower(turn.Visible), needle) {
			position = index
			search.matchTurn = turn.ID
			break
		}
	}
	if result := search.manager.SetViewport(session.Record.ID, position, false); !result.OK {
		return result
	}
	return core.Ok(item.hit)
}

func (search *historySearch) Update(message tea.Msg) tea.Cmd {
	if search == nil {
		return nil
	}
	if keyMessage, ok := message.(tea.KeyMsg); ok {
		switch keyMessage.String() {
		case "up", "down", "pgup", "pgdown", "ctrl+u", "ctrl+d":
			var command tea.Cmd
			search.list, command = search.list.Update(message)
			return command
		}
	}
	before := search.input.Value()
	var command tea.Cmd
	search.input, command = search.input.Update(message)
	if search.input.Value() != before {
		_ = search.Search(search.input.Value())
	}
	return command
}

func (search *historySearch) View(width, height int) string {
	if search == nil {
		return ""
	}
	search.input.Width = max(12, width-4)
	search.list.SetSize(max(1, width), max(4, height-2))
	return lipgloss.JoinVertical(lipgloss.Left, search.input.View(), search.list.View())
}

type overlayKind uint8

const (
	overlayNone overlayKind = iota
	overlayCommands
	overlaySessions
	overlaySearch
	overlayHelp
	overlayWorkEditor
	overlayProjectReview
	overlayGitEnableReview
	overlayLaunchReview
	overlayAgentSelection
	overlayAgentAnswer
	overlayChangeReview
	overlayDataEditor
	overlayDataNote
	overlayDataFilter
	overlayDataBulk
	overlaySettings
)

type helpOverlay struct {
	model help.Model
	keys  keyMap
}

func newHelpOverlay(keys keyMap, styles uiStyles) *helpOverlay {
	model := help.New()
	model.ShowAll = true
	model.Styles.FullKey = styles.accent
	model.Styles.FullDesc = styles.status
	model.Styles.FullSeparator = styles.separator
	return &helpOverlay{model: model, keys: keys}
}

func (overlay *helpOverlay) View(width int) string {
	if overlay == nil {
		return ""
	}
	overlay.model.Width = max(20, width)
	return lipgloss.JoinVertical(lipgloss.Left, "Keyboard help", "", overlay.model.View(overlay.keys), "", "esc closes help")
}

func renderOverlay(body string, width, height int, styles uiStyles) string {
	if width <= 0 || height <= 0 {
		return ""
	}
	overlayWidth := min(72, max(1, width-2))
	overlayHeight := min(26, max(3, height-2))
	box := styles.outerFrame.Copy().
		BorderForeground(styles.theme.focus).
		Width(max(1, overlayWidth-2)).
		Height(max(1, overlayHeight-2)).
		MaxWidth(overlayWidth).
		MaxHeight(overlayHeight).
		Render(body)
	return lipgloss.Place(width, height, lipgloss.Center, lipgloss.Top, box)
}

// overlayEmpty keeps unavailable overlays legible without adding a component.
func overlayEmpty(title, reason string) string {
	return core.Join("\n", title, "", "○ "+reason, "", "esc closes")
}
