// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"sort"
	"time"

	"github.com/charmbracelet/bubbles/list"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	core "dappco.re/go"
)

const (
	workStatusActive    = "active"
	workStatusWaiting   = "waiting"
	workStatusCompleted = "completed"
	workStatusFailed    = "failed"
)

type workListItem struct{ record workItemRecord }

func (item workListItem) Title() string {
	return core.Concat(workGroup(item.record.Status), " · ", item.record.Title)
}

func (item workListItem) Description() string {
	detail := item.record.Repo
	if item.record.Branch != "" {
		detail = core.Concat(detail, " · ", item.record.Branch)
	}
	if detail == "" {
		detail = item.record.Source
	}
	return detail
}

func (item workListItem) FilterValue() string {
	return core.Concat(
		item.record.Title, " ", item.record.Status, " ", item.record.Agent, " ",
		item.record.Repo, " ", item.record.Branch, " ", item.record.Question,
	)
}

type workPanel struct {
	repository workspaceRepository
	provider   agentProvider
	ids        func() string
	now        func() time.Time
	items      []workItemRecord
	events     map[string][]eventRecord
	list       list.Model
	actions    []agentCapability
	action     int
}

func newWorkPanel(
	repository workspaceRepository,
	provider agentProvider,
	ids func() string,
	now func() time.Time,
) core.Result {
	if repository == nil {
		return core.Fail(core.E("tui.newWorkPanel", "workspace repository is required", nil))
	}
	if provider == nil {
		provider = newUnavailableAgentProvider(defaultAgentUnavailableReason)
	}
	if ids == nil {
		ids = newRecordID
	}
	if now == nil {
		now = time.Now
	}
	model := list.New(nil, list.NewDefaultDelegate(), 48, 18)
	model.Title = "Work"
	model.SetShowStatusBar(false)
	model.SetFilteringEnabled(true)
	model.SetShowHelp(false)
	panel := &workPanel{
		repository: repository,
		provider:   provider,
		ids:        ids,
		now:        now,
		events:     make(map[string][]eventRecord),
		list:       model,
		actions:    provider.Capabilities(),
	}
	if result := panel.RefreshLocal(); !result.OK {
		return result
	}
	return core.Ok(panel)
}

func (panel *workPanel) Refresh(ctx context.Context) core.Result {
	if panel == nil || panel.repository == nil || panel.provider == nil {
		return core.Fail(core.E("tui.workPanel.Refresh", "work panel is unavailable", nil))
	}
	snapshotResult := panel.provider.Snapshot(ctx)
	if !snapshotResult.OK {
		return snapshotResult
	}
	snapshot, ok := snapshotResult.Value.(agentSnapshot)
	if !ok {
		return core.Fail(core.E("tui.workPanel.Refresh", "invalid agent snapshot", nil))
	}
	if result := panel.mergeAgentWork(snapshot.Work); !result.OK {
		return result
	}
	if result := panel.mergeAgentEvents(snapshot.Events); !result.OK {
		return result
	}
	panel.actions = panel.provider.Capabilities()
	return panel.RefreshLocal()
}

func (panel *workPanel) RefreshLocal() core.Result {
	if panel == nil || panel.repository == nil {
		return core.Fail(core.E("tui.workPanel.RefreshLocal", "work repository is unavailable", nil))
	}
	result := panel.repository.ListWorkItems(false)
	if !result.OK {
		return result
	}
	items, ok := result.Value.([]workItemRecord)
	if !ok {
		return core.Fail(core.E("tui.workPanel.RefreshLocal", "invalid work list result", nil))
	}
	sortWorkItems(items)
	panel.items = append([]workItemRecord(nil), items...)
	panel.syncList()
	if result := panel.refreshEvents(); !result.OK {
		return result
	}
	return core.Ok(panel.Items())
}

func (panel *workPanel) Items() []workItemRecord {
	if panel == nil {
		return nil
	}
	return append([]workItemRecord(nil), panel.items...)
}

func (panel *workPanel) Events(workID string) []eventRecord {
	if panel == nil {
		return nil
	}
	return append([]eventRecord(nil), panel.events[workID]...)
}

func (panel *workPanel) Capabilities() []agentCapability {
	if panel == nil {
		return nil
	}
	return append([]agentCapability(nil), panel.actions...)
}

func (panel *workPanel) CreateWork(title, task, repository string) core.Result {
	title, task, repository = core.Trim(title), core.Trim(task), core.Trim(repository)
	if title == "" || task == "" || repository == "" {
		return core.Fail(core.E("tui.workPanel.CreateWork", "work title, task, and repository are required", nil))
	}
	return panel.create(title, task, repository)
}

func (panel *workPanel) create(title, task, repository string) core.Result {
	if panel == nil {
		return core.Fail(core.E("tui.workPanel.Create", "work panel is unavailable", nil))
	}
	title = core.Trim(title)
	if title == "" {
		return core.Fail(core.E("tui.workPanel.Create", "work title is required", nil))
	}
	id := core.Trim(panel.ids())
	if id == "" {
		return core.Fail(core.E("tui.workPanel.Create", "work ID generator returned an empty value", nil))
	}
	now := panel.now().UTC()
	record := workItemRecord{
		ID:         id,
		ExternalID: "local:" + id,
		Source:     "local",
		Title:      title,
		Task:       task,
		Repo:       repository,
		Status:     workStatusActive,
		StartedAt:  now,
		UpdatedAt:  now,
		ArchivedAt: unsetRecordTime(),
	}
	if result := panel.repository.SaveWorkItem(record); !result.OK {
		return result
	}
	if result := panel.RefreshLocal(); !result.OK {
		return result
	}
	panel.selectWork(id)
	return core.Ok(record)
}

func (panel *workPanel) EditWork(id, title, task, repository string) core.Result {
	title, task, repository = core.Trim(title), core.Trim(task), core.Trim(repository)
	if title == "" || task == "" || repository == "" {
		return core.Fail(core.E("tui.workPanel.EditWork", "work title, task, and repository are required", nil))
	}
	return panel.updateWork("EditWork", id, func(record *workItemRecord) {
		record.Title, record.Task, record.Repo = title, task, repository
	})
}

func (panel *workPanel) Rename(id, title string) core.Result {
	title = core.Trim(title)
	if title == "" {
		return core.Fail(core.E("tui.workPanel.Rename", "work title is required", nil))
	}
	return panel.updateWork("Rename", id, func(record *workItemRecord) { record.Title = title })
}

func (panel *workPanel) Complete(id string) core.Result {
	return panel.updateWork("Complete", id, func(record *workItemRecord) { record.Status = workStatusCompleted })
}

func (panel *workPanel) Reopen(id string) core.Result {
	return panel.updateWork("Reopen", id, func(record *workItemRecord) { record.Status = workStatusActive })
}

func (panel *workPanel) Link(id, sessionID string) core.Result {
	return panel.updateWork("Link", id, func(record *workItemRecord) { record.SessionID = core.Trim(sessionID) })
}

func (panel *workPanel) Archive(id string) core.Result {
	return panel.updateWork("Archive", id, func(record *workItemRecord) {
		record.Archived = true
		record.ArchivedAt = panel.now().UTC()
	})
}

func (panel *workPanel) SelectAction(feature agentFeature) bool {
	if panel == nil {
		return false
	}
	for index, capability := range panel.actions {
		if capability.Feature == feature {
			panel.action = index
			return true
		}
	}
	return false
}

func (panel *workPanel) MoveAction(delta int) {
	if panel == nil || len(panel.actions) == 0 {
		return
	}
	panel.action = wrapIndex(panel.action, delta, len(panel.actions))
}

func (panel *workPanel) SelectedAction() agentCapability {
	if panel == nil || len(panel.actions) == 0 {
		return agentCapability{}
	}
	if panel.action < 0 || panel.action >= len(panel.actions) {
		panel.action = 0
	}
	return panel.actions[panel.action]
}

func (panel *workPanel) Selected() (workItemRecord, bool) {
	if panel == nil {
		return workItemRecord{}, false
	}
	item, ok := panel.list.SelectedItem().(workListItem)
	if !ok {
		return workItemRecord{}, false
	}
	return item.record, true
}

func (panel *workPanel) Update(message tea.Msg) tea.Cmd {
	if panel == nil {
		return nil
	}
	var command tea.Cmd
	panel.list, command = panel.list.Update(message)
	return command
}

func (panel *workPanel) View(width, height int, styles uiStyles) string {
	if panel == nil || width <= 0 || height <= 0 {
		return ""
	}
	panel.list.Styles.Title = styles.title
	listWidth := width
	listHeight := height
	detailWidth := width
	detailHeight := height
	if width >= 100 {
		listWidth = min(48, max(32, width/3))
		detailWidth = max(1, width-listWidth-1)
	} else {
		listHeight = max(4, height/2)
		detailHeight = max(1, height-listHeight-1)
	}
	listView := panel.renderList(listWidth, listHeight, styles)
	detailView := panel.renderDetail(detailWidth, detailHeight, styles)
	var view string
	if width >= 100 {
		separator := fitPane("│", 1, height, styles.separator)
		view = lipgloss.JoinHorizontal(lipgloss.Top,
			fitPane(listView, listWidth, height, styles.panel),
			separator,
			fitPane(detailView, detailWidth, height, styles.panel),
		)
	} else {
		view = lipgloss.JoinVertical(lipgloss.Left,
			fitPane(listView, listWidth, listHeight, styles.panel),
			"",
			fitPane(detailView, detailWidth, detailHeight, styles.panel),
		)
	}
	return fitPane(view, width, height, styles.panel)
}

func (panel *workPanel) renderList(width, height int, styles uiStyles) string {
	counts := map[string]int{"ACTIVE": 0, "WAITING": 0, "DONE": 0}
	for _, item := range panel.items {
		counts[workGroup(item.Status)]++
	}
	builder := core.NewBuilder()
	builder.WriteString(styles.title.Render("WORK"))
	builder.WriteString("  ")
	builder.WriteString(styles.status.Render(core.Sprintf(
		"ACTIVE %d · WAITING %d · DONE %d",
		counts["ACTIVE"], counts["WAITING"], counts["DONE"],
	)))
	builder.WriteString("\n")
	if panel.list.SettingFilter() || panel.list.FilterState() == list.FilterApplied {
		builder.WriteString(panel.list.FilterInput.View())
		builder.WriteString("\n")
	}
	visible := panel.list.VisibleItems()
	if len(visible) == 0 {
		builder.WriteString("\n")
		builder.WriteString(styles.status.Render("○ No work yet"))
		builder.WriteString("\n")
		builder.WriteString(styles.thought.Render("A connected provider or workspace action will create work here."))
		return builder.String()
	}
	for index, raw := range visible {
		item, ok := raw.(workListItem)
		if !ok {
			continue
		}
		cursor := "  "
		rowStyle := styles.answer
		if index == panel.list.Index() {
			cursor = "› "
			rowStyle = styles.accent
		}
		builder.WriteString(cursor)
		builder.WriteString(styles.status.Render(workGlyph(item.record.Status) + " " + workGroup(item.record.Status)))
		builder.WriteString("  ")
		builder.WriteString(rowStyle.Render(item.record.Title))
		builder.WriteString("\n")
	}
	builder.WriteString(styles.thought.Render("/ filter · ↑/↓ select"))
	return fitPane(builder.String(), width, height, styles.panel)
}

func (panel *workPanel) renderDetail(width, height int, styles uiStyles) string {
	record, ok := panel.Selected()
	if !ok {
		return styles.status.Render("Select a work item for its timeline and context.")
	}
	builder := core.NewBuilder()
	builder.WriteString(styles.title.Render(record.Title))
	builder.WriteString("  ")
	builder.WriteString(styles.status.Render(workGlyph(record.Status) + " " + core.Upper(record.Status)))
	builder.WriteString("\n\n")
	workDetailRow(builder, styles, "source", record.Source)
	workDetailRow(builder, styles, "agent", record.Agent)
	workDetailRow(builder, styles, "task", record.Task)
	workDetailRow(builder, styles, "repository", record.Repo)
	workDetailRow(builder, styles, "branch", record.Branch)
	workDetailRow(builder, styles, "runtime", record.Runtime)
	workDetailRow(builder, styles, "session", record.SessionID)
	if record.Question != "" {
		builder.WriteString("\n")
		builder.WriteString(styles.attention.Render("? QUESTION"))
		builder.WriteString("\n")
		builder.WriteString(styles.answer.Render(record.Question))
		builder.WriteString("\n")
	}
	if record.PRURL != "" {
		workDetailRow(builder, styles, "pull request", record.PRURL)
	}
	events := panel.events[record.ID]
	builder.WriteString("\n")
	builder.WriteString(styles.accent.Render("TIMELINE"))
	builder.WriteString("\n")
	if len(events) == 0 {
		builder.WriteString(styles.status.Render("○ no recorded events"))
	} else {
		for _, event := range events {
			builder.WriteString(styles.status.Render("· " + event.Kind + "  "))
			builder.WriteString(styles.answer.Render(event.Title))
			builder.WriteString("\n")
		}
	}
	return fitPane(builder.String(), width, height, styles.panel)
}

func (panel *workPanel) updateWork(operation, id string, mutate func(*workItemRecord)) core.Result {
	if panel == nil || panel.repository == nil {
		return core.Fail(core.E(core.Concat("tui.workPanel.", operation), "work panel is unavailable", nil))
	}
	id = core.Trim(id)
	for _, item := range panel.items {
		if item.ID != id {
			continue
		}
		updated := item
		mutate(&updated)
		updated.UpdatedAt = panel.now().UTC()
		if result := panel.repository.SaveWorkItem(updated); !result.OK {
			return result
		}
		if result := panel.RefreshLocal(); !result.OK {
			return result
		}
		panel.selectWork(id)
		return core.Ok(updated)
	}
	return core.Fail(core.E(core.Concat("tui.workPanel.", operation), core.Concat("unknown work item: ", id), nil))
}

func (panel *workPanel) mergeAgentWork(work []agentWorkSnapshot) core.Result {
	listed := panel.repository.ListWorkItems(true)
	if !listed.OK {
		return listed
	}
	existing, ok := listed.Value.([]workItemRecord)
	if !ok {
		return core.Fail(core.E("tui.workPanel.mergeAgentWork", "invalid work list result", nil))
	}
	byExternal := make(map[string]workItemRecord, len(existing))
	for _, record := range existing {
		byExternal[record.ExternalID] = record
	}
	ordered := append([]agentWorkSnapshot(nil), work...)
	sort.SliceStable(ordered, func(left, right int) bool { return ordered[left].ExternalID < ordered[right].ExternalID })
	for _, snapshot := range ordered {
		externalID := core.Trim(snapshot.ExternalID)
		if externalID == "" {
			return core.Fail(core.E("tui.workPanel.mergeAgentWork", "agent work external ID is required", nil))
		}
		record, exists := byExternal[externalID]
		if !exists {
			record.ID = core.Trim(panel.ids())
			if record.ID == "" {
				return core.Fail(core.E("tui.workPanel.mergeAgentWork", "work ID generator returned an empty value", nil))
			}
			record.ExternalID = externalID
			record.StartedAt = panel.now().UTC()
			record.ArchivedAt = unsetRecordTime()
		}
		record.Source = "agent"
		record.Title = core.Trim(snapshot.Title)
		if record.Title == "" {
			record.Title = externalID
		}
		record.Status = core.Lower(core.Trim(snapshot.Status))
		if record.Status == "" {
			record.Status = workStatusActive
		}
		record.Agent = snapshot.Agent
		record.Repo = snapshot.Repo
		record.Branch = snapshot.Branch
		record.Runtime = snapshot.Runtime
		record.Question = snapshot.Question
		record.PRURL = snapshot.PRURL
		record.UpdatedAt = panel.now().UTC()
		if result := panel.repository.SaveWorkItem(record); !result.OK {
			return result
		}
		byExternal[externalID] = record
	}
	return core.Ok(nil)
}

func (panel *workPanel) mergeAgentEvents(events []agentEventSnapshot) core.Result {
	listed := panel.repository.ListWorkItems(true)
	if !listed.OK {
		return listed
	}
	work, ok := listed.Value.([]workItemRecord)
	if !ok {
		return core.Fail(core.E("tui.workPanel.mergeAgentEvents", "invalid work list result", nil))
	}
	byExternal := make(map[string]workItemRecord, len(work))
	for _, record := range work {
		byExternal[record.ExternalID] = record
	}
	ordered := append([]agentEventSnapshot(nil), events...)
	sort.SliceStable(ordered, func(left, right int) bool {
		if ordered[left].CreatedAt.Equal(ordered[right].CreatedAt) {
			return ordered[left].ExternalID < ordered[right].ExternalID
		}
		return ordered[left].CreatedAt.Before(ordered[right].CreatedAt)
	})
	for _, snapshot := range ordered {
		record, exists := byExternal[snapshot.WorkID]
		if !exists {
			return core.Fail(core.E("tui.workPanel.mergeAgentEvents", core.Concat("unknown agent work: ", snapshot.WorkID), nil))
		}
		createdAt := snapshot.CreatedAt.UTC()
		if createdAt.IsZero() {
			createdAt = panel.now().UTC()
		}
		seed := core.Trim(snapshot.ExternalID)
		if seed == "" {
			seed = core.Concat(snapshot.WorkID, "|", snapshot.Kind, "|", snapshot.Title, "|", createdAt.Format(time.RFC3339Nano))
		}
		hash := core.SHA256HexString(seed)
		event := eventRecord{
			ID:          "agent-event-" + hash[:32],
			SessionID:   workEventSessionID(record),
			WorkItemID:  record.ID,
			Kind:        snapshot.Kind,
			Status:      "recorded",
			Title:       snapshot.Title,
			Detail:      snapshot.Detail,
			PayloadJSON: core.JSONMarshalString(snapshot),
			CreatedAt:   createdAt,
		}
		if event.Kind == "" {
			event.Kind = "agent.event"
		}
		if event.Title == "" {
			event.Title = agentFeatureTitle(agentFeature(event.Kind))
		}
		if result := panel.repository.SaveEvent(event); !result.OK {
			return result
		}
	}
	return core.Ok(nil)
}

func (panel *workPanel) refreshEvents() core.Result {
	panel.events = make(map[string][]eventRecord, len(panel.items))
	for _, item := range panel.items {
		result := panel.repository.Events(workEventSessionID(item))
		if !result.OK {
			return result
		}
		events, ok := result.Value.([]eventRecord)
		if !ok {
			return core.Fail(core.E("tui.workPanel.refreshEvents", "invalid event list result", nil))
		}
		filtered := make([]eventRecord, 0, len(events))
		for _, event := range events {
			if event.WorkItemID == item.ID {
				filtered = append(filtered, event)
			}
		}
		sort.SliceStable(filtered, func(left, right int) bool {
			if filtered[left].CreatedAt.Equal(filtered[right].CreatedAt) {
				return filtered[left].ID < filtered[right].ID
			}
			return filtered[left].CreatedAt.Before(filtered[right].CreatedAt)
		})
		panel.events[item.ID] = filtered
	}
	return core.Ok(nil)
}

func (panel *workPanel) syncList() {
	selectedID := ""
	if selected, ok := panel.Selected(); ok {
		selectedID = selected.ID
	}
	items := make([]list.Item, 0, len(panel.items))
	for _, record := range panel.items {
		items = append(items, workListItem{record: record})
	}
	panel.list.SetItems(items)
	if selectedID != "" {
		panel.selectWork(selectedID)
	}
}

func (panel *workPanel) selectWork(id string) {
	for index, raw := range panel.list.VisibleItems() {
		item, ok := raw.(workListItem)
		if ok && item.record.ID == id {
			panel.list.Select(index)
			return
		}
	}
}

func sortWorkItems(items []workItemRecord) {
	sort.SliceStable(items, func(left, right int) bool {
		leftRank := workGroupRank(items[left].Status)
		rightRank := workGroupRank(items[right].Status)
		if leftRank != rightRank {
			return leftRank < rightRank
		}
		if items[left].Title != items[right].Title {
			return items[left].Title < items[right].Title
		}
		return items[left].ExternalID < items[right].ExternalID
	})
}

func normalizeWorkStatus(status string) string {
	switch core.Lower(core.Trim(status)) {
	case "waiting", "question", "blocked", "needs_input":
		return workStatusWaiting
	case "completed", "done", "success", "succeeded":
		return workStatusCompleted
	case "failed", "error", "cancelled", "canceled", "interrupted":
		return workStatusFailed
	default:
		return workStatusActive
	}
}

func workGroup(status string) string {
	switch normalizeWorkStatus(status) {
	case workStatusWaiting:
		return "WAITING"
	case workStatusCompleted, workStatusFailed:
		return "DONE"
	default:
		return "ACTIVE"
	}
}

func workGroupRank(status string) int {
	switch workGroup(status) {
	case "WAITING":
		return 1
	case "DONE":
		return 2
	default:
		return 0
	}
}

func workGlyph(status string) string {
	switch normalizeWorkStatus(status) {
	case workStatusWaiting:
		return "?"
	case workStatusCompleted:
		return "✓"
	case workStatusFailed:
		return "×"
	default:
		return "●"
	}
}

func workEventSessionID(record workItemRecord) string {
	if record.SessionID != "" {
		return record.SessionID
	}
	return "work:" + record.ID
}

func workDetailRow(builder *core.Builder, styles uiStyles, label, value string) {
	if value == "" {
		return
	}
	builder.WriteString(styles.status.Render(label + "  "))
	builder.WriteString(styles.answer.Render(value))
	builder.WriteString("\n")
}
