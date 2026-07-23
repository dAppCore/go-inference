// SPDX-License-Identifier: EUPL-1.2

package tui

import (
	"context"
	"sort"
	"time"

	tea "dappco.re/go/html/tui"
	"dappco.re/go/html/tui/list"
	"dappco.re/go/html/tui/style"

	core "dappco.re/go"
)

const (
	workStatusActive    = "active"
	workStatusWaiting   = "waiting"
	workStatusCompleted = "completed"
	workStatusFailed    = "failed"
)

const maxRenderedAgentEvents = 200

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
	repository  workspaceRepository
	provider    agentProvider
	ids         func() string
	now         func() time.Time
	items       []workItemRecord
	events      map[string][]eventRecord
	agentWork   map[string]agentWorkSnapshot
	agentEvents map[string][]agentEventSnapshot
	queueStatus string
	queueReason string
	list        list.Model
	actions     []agentCapability
	action      int
}

type agentEventOrder struct {
	at       time.Time
	class    uint8
	runID    string
	sequence int64
	id       string
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
		repository:  repository,
		provider:    provider,
		ids:         ids,
		now:         now,
		events:      make(map[string][]eventRecord),
		agentWork:   make(map[string]agentWorkSnapshot),
		agentEvents: make(map[string][]agentEventSnapshot),
		list:        model,
		actions:     provider.Capabilities(),
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
	return panel.ApplyAgentSnapshot(snapshot)
}

func (panel *workPanel) ApplyAgentSnapshot(snapshot agentSnapshot) core.Result {
	if panel == nil || panel.repository == nil {
		return core.Fail(core.E("tui.workPanel.ApplyAgentSnapshot", "work panel is unavailable", nil))
	}
	if result := panel.mergeAgentWork(snapshot.Work); !result.OK {
		return result
	}
	panel.mergeAgentEvents(snapshot.Events)
	panel.queueStatus, panel.queueReason = core.Trim(snapshot.QueueStatus), core.Trim(snapshot.QueueReason)
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
	items := make([]workItemRecord, len(panel.items))
	for index, item := range panel.items {
		items[index] = panel.effective(item)
	}
	return items
}

func (panel *workPanel) effective(record workItemRecord) workItemRecord {
	if panel == nil {
		return record
	}
	state := panel.agentWork[record.ID]
	if state.NativeRunID == "" {
		return record
	}
	if status := core.Trim(state.Status); status != "" {
		record.Status = status
	}
	record.Agent, record.Branch, record.Runtime, record.Question, record.PRURL = state.Agent, state.Branch, state.Runtime, state.Question, state.PRURL
	return record
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

func (panel *workPanel) AgentState(record workItemRecord) agentWorkSnapshot {
	if panel == nil {
		return agentWorkSnapshot{}
	}
	return panel.agentWork[record.ID]
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
		view = style.Row(style.Top,
			fitPane(listView, listWidth, height, styles.panel),
			separator,
			fitPane(detailView, detailWidth, height, styles.panel),
		)
	} else {
		view = style.Column(style.Left,
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
	if state := panel.AgentState(record); state.NativeRunID != "" {
		workDetailRow(builder, styles, "native run", state.NativeRunID)
	}
	if panel.queueStatus != "" {
		workDetailRow(builder, styles, "queue", core.Trim(core.Concat(panel.queueStatus, " ", panel.queueReason)))
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
			if event.Detail != "" {
				builder.WriteString(" ")
				builder.WriteString(styles.thought.Render(event.Detail))
			}
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
	byExternal := make(map[string]workItemRecord, len(panel.items)*2)
	for _, record := range panel.items {
		byExternal[record.ExternalID], byExternal[record.ID] = record, record
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
			continue
		}
		previous := panel.agentWork[record.ID]
		if previous.NativeRunID == snapshot.NativeRunID {
			if snapshot.AnswerID == "" {
				snapshot.AnswerID = previous.AnswerID
			}
			if snapshot.ResumeRunID == "" {
				snapshot.ResumeRunID = previous.ResumeRunID
			}
		}
		panel.agentWork[record.ID] = snapshot
	}
	return core.Ok(nil)
}

func (panel *workPanel) mergeAgentEvents(events []agentEventSnapshot) {
	if panel == nil {
		return
	}
	byExternal := make(map[string]string, len(panel.items)*2)
	for _, record := range panel.items {
		byExternal[record.ExternalID], byExternal[record.ID] = record.ID, record.ID
	}
	panel.agentEvents = make(map[string][]agentEventSnapshot, len(byExternal))
	ordered := append([]agentEventSnapshot(nil), events...)
	for index := range ordered {
		if ordered[index].CreatedAt.IsZero() {
			ordered[index].CreatedAt = panel.now().UTC()
		}
	}
	sortAgentEvents(ordered)
	for _, snapshot := range ordered {
		if localID := byExternal[snapshot.WorkID]; localID != "" {
			panel.agentEvents[localID] = append(panel.agentEvents[localID], snapshot)
		}
	}
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
		filtered := make([]eventRecord, 0, len(events)+maxRenderedAgentEvents)
		ordering := make(map[string]agentEventOrder, maxRenderedAgentEvents)
		for _, event := range events {
			if event.WorkItemID == item.ID {
				filtered = append(filtered, event)
			}
		}
		liveEvents := append([]agentEventSnapshot(nil), panel.agentEvents[item.ID]...)
		sortAgentEvents(liveEvents)
		logs := make([]agentEventSnapshot, 0, min(len(liveEvents), maxRenderedAgentEvents))
		durable := make([]agentEventSnapshot, 0, len(liveEvents))
		for _, event := range liveEvents {
			if event.Sequence > 0 {
				logs = append(logs, event)
			} else {
				durable = append(durable, event)
			}
		}
		if len(logs) > maxRenderedAgentEvents {
			logs = logs[len(logs)-maxRenderedAgentEvents:]
		}
		liveEvents = append(durable, logs...)
		sortAgentEvents(liveEvents)
		anchors := agentLogAnchors(liveEvents)
		for _, live := range liveEvents {
			createdAt := live.CreatedAt.UTC()
			if createdAt.IsZero() {
				createdAt = panel.now().UTC()
			}
			kind := core.Trim(live.Kind)
			if kind == "" {
				kind = "agent.event"
			}
			title := core.Trim(live.Title)
			if live.Stream != "" {
				title = core.Upper(core.Trim(live.Stream))
			}
			if title == "" {
				title = agentFeatureTitle(agentFeature(kind))
			}
			id := core.Trim(live.ExternalID)
			if id == "" {
				id = core.Concat(live.RunID, ":", core.Sprintf("%020d", live.Sequence), ":", kind)
			}
			if live.Sequence > 0 {
				id = core.Concat("agent-log:", core.Sprintf("%020d", live.Sequence), ":", id)
			} else {
				id = "agent-event:" + id
			}
			ordering[id] = agentEventOrdering(live, anchors)
			filtered = append(filtered, eventRecord{ID: id, SessionID: workEventSessionID(item), WorkItemID: item.ID, Kind: kind, Status: "live", Title: title, Detail: live.Detail, PayloadJSON: core.JSONMarshalString(live), CreatedAt: createdAt})
		}
		sort.SliceStable(filtered, func(left, right int) bool {
			leftOrder, exists := ordering[filtered[left].ID]
			if !exists {
				leftOrder = agentEventOrder{at: filtered[left].CreatedAt.UTC(), class: 1, id: filtered[left].ID}
			}
			rightOrder, exists := ordering[filtered[right].ID]
			if !exists {
				rightOrder = agentEventOrder{at: filtered[right].CreatedAt.UTC(), class: 1, id: filtered[right].ID}
			}
			return lessAgentEventOrder(leftOrder, rightOrder)
		})
		panel.events[item.ID] = filtered
	}
	return core.Ok(nil)
}

func sortAgentEvents(events []agentEventSnapshot) {
	anchors := agentLogAnchors(events)
	sort.SliceStable(events, func(left, right int) bool {
		return lessAgentEventOrder(agentEventOrdering(events[left], anchors), agentEventOrdering(events[right], anchors))
	})
}

func agentLogAnchors(events []agentEventSnapshot) map[string]time.Time {
	anchors := make(map[string]time.Time)
	for _, event := range events {
		if event.Sequence <= 0 {
			continue
		}
		at := event.CreatedAt.UTC()
		previous, exists := anchors[event.RunID]
		if !exists || at.Before(previous) {
			anchors[event.RunID] = at
		}
	}
	return anchors
}

func agentEventOrdering(event agentEventSnapshot, anchors map[string]time.Time) agentEventOrder {
	order := agentEventOrder{at: event.CreatedAt.UTC(), class: 1, id: event.ExternalID}
	if event.Sequence > 0 {
		order.at = anchors[event.RunID]
		order.class = 0
		order.runID = event.RunID
		order.sequence = event.Sequence
	}
	return order
}

func lessAgentEventOrder(left, right agentEventOrder) bool {
	if !left.at.Equal(right.at) {
		return left.at.Before(right.at)
	}
	if left.class != right.class {
		return left.class < right.class
	}
	if left.runID != right.runID {
		return left.runID < right.runID
	}
	if left.sequence != right.sequence {
		return left.sequence < right.sequence
	}
	return left.id < right.id
}

func (panel *workPanel) syncList() {
	selectedID := ""
	if selected, ok := panel.Selected(); ok {
		selectedID = selected.ID
	}
	items := make([]list.Item, 0, len(panel.items))
	for _, record := range panel.items {
		items = append(items, workListItem{record: panel.effective(record)})
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
	case "accepted":
		return workStatusCompleted
	case "failed", "error", "cancelled", "canceled", "interrupted", "rejected":
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
