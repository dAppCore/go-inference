// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"sort"
	"time"

	"github.com/charmbracelet/bubbles/list"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	core "dappco.re/go"
	"dappco.re/go/inference/dataset"
)

// dataAction names one review action the Data panel exposes — the design's
// "approve / reject / quarantine-clear / edit-as-derived / tag" list.
type dataAction uint8

const (
	dataActionApprove dataAction = iota
	dataActionReject
	dataActionQuarantineClear
	dataActionEditAsDerived
	dataActionTag
)

// needsNote reports whether action requires a human-entered note/label
// before it can apply — quarantine-clear (a mandatory justification) and
// tag (the label itself). Approve/reject/edit-as-derived do not.
func (action dataAction) needsNote() bool {
	switch action {
	case dataActionQuarantineClear, dataActionTag:
		return true
	default:
		return false
	}
}

func (action dataAction) title() string {
	switch action {
	case dataActionApprove:
		return "Approve"
	case dataActionReject:
		return "Reject"
	case dataActionQuarantineClear:
		return "Clear quarantine"
	case dataActionEditAsDerived:
		return "Edit as derived"
	case dataActionTag:
		return "Tag"
	default:
		return "Unknown action"
	}
}

// dataSortMode is the list's sort key — the design's "sort by score/date".
type dataSortMode uint8

const (
	dataSortDate dataSortMode = iota
	dataSortScore
)

// dataFilterState narrows the Data panel's cross-dataset list, per the
// design's "filters by dataset / status / score expression / source".
// Every field's zero value means "any" — the same "narrowing dimension"
// convention dataset.ItemFilter itself uses.
type dataFilterState struct {
	DatasetSlug string
	Status      dataset.ReviewStatus
	Kind        dataset.ItemKind
	Source      dataset.ItemSource
	Score       *dataset.ScoreExpression
}

// Equal reports whether f and other narrow to the same filter — used by
// the filter round-trip tests; Score is a pointer, so it is compared by
// value, not identity.
func (f dataFilterState) Equal(other dataFilterState) bool {
	if f.DatasetSlug != other.DatasetSlug || f.Status != other.Status || f.Kind != other.Kind || f.Source != other.Source {
		return false
	}
	if (f.Score == nil) != (other.Score == nil) {
		return false
	}
	if f.Score != nil && *f.Score != *other.Score {
		return false
	}
	return true
}

// String renders filter back in parseDataFilterExpr's grammar — the exact
// round trip the filter overlay pre-fills from and the list header shows.
func (f dataFilterState) String() string {
	clauses := make([]string, 0, 5)
	if f.DatasetSlug != "" {
		clauses = append(clauses, "dataset="+f.DatasetSlug)
	}
	if f.Status != "" {
		clauses = append(clauses, "status="+string(f.Status))
	}
	if f.Kind != "" {
		clauses = append(clauses, "kind="+string(f.Kind))
	}
	if f.Source != "" {
		clauses = append(clauses, "source="+string(f.Source))
	}
	if f.Score != nil {
		clauses = append(clauses, f.Score.String())
	}
	return core.Join(",", clauses...)
}

// parseDataFilterExpr parses the Data panel's filter grammar — the same
// tiny explicit grammar cli/data.go's --filter flag uses (comma-separated
// "field=value" clauses, or a bare dataset.ScoreExpression), plus
// "dataset=<slug>" to narrow the panel's own cross-dataset list. Duplicated
// rather than shared across the cli/tui <-> main package boundary (main is
// not importable, and the grammar is a CLI-shaped concern the root
// dataset package should not own) — kept intentionally tiny so the
// duplication stays cheap to eyeball against cli/data.go's parseItemFilter.
func parseDataFilterExpr(expr string) (dataFilterState, error) {
	filter := dataFilterState{}
	expr = core.Trim(expr)
	if expr == "" {
		return filter, nil
	}
	for _, clause := range core.Split(expr, ",") {
		clause = core.Trim(clause)
		if clause == "" {
			continue
		}
		if key, value, ok := splitDataFilterClause(clause); ok {
			switch key {
			case "dataset":
				filter.DatasetSlug = value
			case "status":
				filter.Status = dataset.ReviewStatus(value)
			case "kind":
				filter.Kind = dataset.ItemKind(value)
			case "source":
				filter.Source = dataset.ItemSource(value)
			default:
				return filter, core.NewError(core.Sprintf("tui: unknown filter field %q", key))
			}
			continue
		}
		exprResult := dataset.ParseScoreExpression(clause)
		if !exprResult.OK {
			return filter, core.E("tui.parseDataFilterExpr", core.Sprintf("unrecognised filter clause %q", clause), exprResult.Err())
		}
		parsed := exprResult.Value.(dataset.ScoreExpression)
		filter.Score = &parsed
	}
	return filter, nil
}

// splitDataFilterClause mirrors cli/data.go's splitFilterClause exactly:
// splits a "key=value" clause on the first bare '=', excluding >=/<=/!=/==
// so a bare score expression clause never matches.
func splitDataFilterClause(clause string) (key, value string, ok bool) {
	idx := core.Index(clause, "=")
	if idx < 0 {
		return "", "", false
	}
	if idx > 0 {
		switch clause[idx-1] {
		case '>', '<', '!':
			return "", "", false
		}
	}
	if idx+1 < len(clause) && clause[idx+1] == '=' {
		return "", "", false
	}
	return core.Trim(clause[:idx]), core.Trim(clause[idx+1:]), true
}

// dataItemRow is one list row — an Item plus the denormalised context the
// list/detail panes render without a second round trip per keystroke: the
// owning Dataset (slug shown in the list, filtered on), the latest Review
// (status/kind/top-score columns), and the top score of the panel's
// score kind (ScoreKindLEK — the heuristic tier's composite, always
// computed first per the design).
type dataItemRow struct {
	Item     dataset.Item
	Dataset  dataset.Dataset
	Review   dataset.Review
	TopScore dataset.Score
	HasScore bool
}

type dataListItem struct{ row dataItemRow }

func (item dataListItem) Title() string {
	score := "—"
	if item.row.HasScore {
		score = core.Sprintf("%.0f", item.row.TopScore.Value)
	}
	return core.Concat(dataStatusGlyph(item.row.Review.Status), " ", core.Upper(string(item.row.Review.Status)), " · ", string(item.row.Item.Kind), " · ", score)
}

func (item dataListItem) Description() string {
	detail := item.row.Dataset.Slug
	if item.row.Item.Source != "" {
		detail = core.Concat(detail, " · ", string(item.row.Item.Source))
	}
	return detail
}

func (item dataListItem) FilterValue() string {
	return core.Concat(
		item.row.Dataset.Slug, " ", string(item.row.Item.Kind), " ", string(item.row.Item.Source), " ",
		string(item.row.Review.Status), " ", item.row.Item.SourceRef,
	)
}

func dataStatusGlyph(status dataset.ReviewStatus) string {
	switch status {
	case dataset.StatusApproved:
		return "✓"
	case dataset.StatusRejected:
		return "×"
	case dataset.StatusQuarantined:
		return "!"
	default:
		return "●"
	}
}

// dataPanel is the fifth primary TUI panel — the human review surface for
// the dataset loop (docs/superpowers/specs/2026-07-19-lem-dataset-loop-
// design.md, "Review surface (TUI)"). Data ONLY ever flows through store
// (a [dataset.Store], concretely [OpenDatasetStore]'s duckDatasetStore in
// production) — this panel never opens DuckDB directly, per the design.
type dataPanel struct {
	store    dataset.Store
	markdown *markdownRenderer
	ids      func() string
	now      func() time.Time

	datasets []dataset.Dataset
	rows     []dataItemRow
	filter   dataFilterState
	sort     dataSortMode

	list list.Model
}

// newDataPanel builds a ready Data panel over store — never nil; pass a
// [dataset.MemoryStore] for lightweight tests or a duckDatasetStore for
// full-conformance/integration tests, exactly as go/dataset's own test
// suite does.
func newDataPanel(store dataset.Store, markdown *markdownRenderer, ids func() string, now func() time.Time) core.Result {
	if store == nil {
		return core.Fail(core.E("tui.newDataPanel", "dataset store is required", nil))
	}
	if ids == nil {
		ids = newRecordID
	}
	if now == nil {
		now = time.Now
	}
	model := list.New(nil, list.NewDefaultDelegate(), 48, 18)
	model.Title = "Data"
	model.SetShowStatusBar(false)
	model.SetFilteringEnabled(true)
	model.SetShowHelp(false)
	panel := &dataPanel{store: store, markdown: markdown, ids: ids, now: now, list: model}
	if result := panel.Refresh(); !result.OK {
		return result
	}
	return core.Ok(panel)
}

// Refresh reloads the dataset catalogue and the item list for the panel's
// current filter — items ACROSS every dataset matching filter.DatasetSlug
// (every dataset when empty), per the design's "items across datasets"
// list requirement. dataset.Store.Items requires one DatasetID per call
// (ItemFilter.DatasetID is mandatory), so a cross-dataset view means one
// Items call per matching dataset, merged and re-sorted here.
func (panel *dataPanel) Refresh() core.Result {
	if panel == nil || panel.store == nil {
		return core.Fail(core.E("tui.dataPanel.Refresh", "data panel is unavailable", nil))
	}
	datasetsResult := panel.store.Datasets(false)
	if !datasetsResult.OK {
		return datasetsResult
	}
	datasets, ok := datasetsResult.Value.([]dataset.Dataset)
	if !ok {
		return core.Fail(core.E("tui.dataPanel.Refresh", "invalid datasets result", nil))
	}
	panel.datasets = datasets

	rows := make([]dataItemRow, 0)
	for _, ds := range datasets {
		if panel.filter.DatasetSlug != "" && ds.Slug != panel.filter.DatasetSlug {
			continue
		}
		itemsResult := panel.store.Items(dataset.ItemFilter{
			DatasetID: ds.ID, Kind: panel.filter.Kind, Source: panel.filter.Source,
			Status: panel.filter.Status, Score: panel.filter.Score,
		})
		if !itemsResult.OK {
			return itemsResult
		}
		items, ok := itemsResult.Value.([]dataset.Item)
		if !ok {
			return core.Fail(core.E("tui.dataPanel.Refresh", "invalid items result", nil))
		}
		for _, item := range items {
			rows = append(rows, panel.buildRow(item, ds))
		}
	}
	sortDataRows(rows, panel.sort)
	panel.rows = rows
	panel.syncList()
	return core.Ok(panel.rows)
}

func (panel *dataPanel) buildRow(item dataset.Item, ds dataset.Dataset) dataItemRow {
	row := dataItemRow{Item: item, Dataset: ds, Review: dataset.Review{Status: dataset.StatusPending}}
	if reviewResult := panel.store.ReviewLatest(item.ID); reviewResult.OK {
		if review, ok := reviewResult.Value.(dataset.Review); ok && review.Status != "" {
			row.Review = review
		}
	}
	if scoresResult := panel.store.Scores(item.ID); scoresResult.OK {
		if scores, ok := scoresResult.Value.([]dataset.Score); ok {
			if top, found := latestScoreOfKind(scores, dataset.ScoreKindLEK); found {
				row.TopScore, row.HasScore = top, true
			}
		}
	}
	return row
}

func latestScoreOfKind(scores []dataset.Score, kind dataset.ScoreKind) (dataset.Score, bool) {
	found := false
	var latest dataset.Score
	for _, score := range scores {
		if score.Kind != kind {
			continue
		}
		if !found || !score.CreatedAt.Before(latest.CreatedAt) {
			latest, found = score, true
		}
	}
	return latest, found
}

func sortDataRows(rows []dataItemRow, mode dataSortMode) {
	sort.SliceStable(rows, func(i, j int) bool {
		if mode == dataSortScore && rows[i].HasScore != rows[j].HasScore {
			return rows[i].HasScore
		}
		if mode == dataSortScore && rows[i].HasScore && rows[i].TopScore.Value != rows[j].TopScore.Value {
			return rows[i].TopScore.Value > rows[j].TopScore.Value
		}
		if !rows[i].Item.CreatedAt.Equal(rows[j].Item.CreatedAt) {
			return rows[i].Item.CreatedAt.After(rows[j].Item.CreatedAt)
		}
		return rows[i].Item.ID < rows[j].Item.ID
	})
}

func (panel *dataPanel) syncList() {
	selectedID := ""
	if selected, ok := panel.Selected(); ok {
		selectedID = selected.Item.ID
	}
	items := make([]list.Item, 0, len(panel.rows))
	for _, row := range panel.rows {
		items = append(items, dataListItem{row: row})
	}
	panel.list.SetItems(items)
	if selectedID != "" {
		panel.selectItem(selectedID)
	}
}

func (panel *dataPanel) selectItem(id string) {
	for index, raw := range panel.list.VisibleItems() {
		item, ok := raw.(dataListItem)
		if ok && item.row.Item.ID == id {
			panel.list.Select(index)
			return
		}
	}
}

// Selected returns the list's current cursor row, if any.
func (panel *dataPanel) Selected() (dataItemRow, bool) {
	if panel == nil {
		return dataItemRow{}, false
	}
	item, ok := panel.list.SelectedItem().(dataListItem)
	if !ok {
		return dataItemRow{}, false
	}
	return item.row, true
}

// Filter returns the panel's current structural filter.
func (panel *dataPanel) Filter() dataFilterState {
	if panel == nil {
		return dataFilterState{}
	}
	return panel.filter
}

// SetFilter replaces the panel's structural filter and reloads the list.
func (panel *dataPanel) SetFilter(filter dataFilterState) core.Result {
	if panel == nil {
		return core.Fail(core.E("tui.dataPanel.SetFilter", "data panel is unavailable", nil))
	}
	panel.filter = filter
	return panel.Refresh()
}

// FilterExpr renders the current filter in parseDataFilterExpr's grammar.
func (panel *dataPanel) FilterExpr() string {
	if panel == nil {
		return ""
	}
	return panel.filter.String()
}

// SetFilterExpr parses expr and applies it as the panel's new filter — the
// inverse of FilterExpr, so FilterExpr()/SetFilterExpr() round-trip.
func (panel *dataPanel) SetFilterExpr(expr string) core.Result {
	if panel == nil {
		return core.Fail(core.E("tui.dataPanel.SetFilterExpr", "data panel is unavailable", nil))
	}
	filter, err := parseDataFilterExpr(expr)
	if err != nil {
		return core.Fail(core.E("tui.dataPanel.SetFilterExpr", err.Error(), err))
	}
	return panel.SetFilter(filter)
}

// ToggleSort flips between date and score sort and re-orders the list in
// place (no store round trip — every row already carries its top score).
func (panel *dataPanel) ToggleSort() core.Result {
	if panel == nil {
		return core.Fail(core.E("tui.dataPanel.ToggleSort", "data panel is unavailable", nil))
	}
	if panel.sort == dataSortDate {
		panel.sort = dataSortScore
	} else {
		panel.sort = dataSortDate
	}
	sortDataRows(panel.rows, panel.sort)
	panel.syncList()
	return core.Ok(panel.sort)
}

func (panel *dataPanel) SortMode() dataSortMode {
	if panel == nil {
		return dataSortDate
	}
	return panel.sort
}

// FilteredCount is how many items the current filter matches — the count
// a bulk action's confirmation overlay names.
func (panel *dataPanel) FilteredCount() int {
	if panel == nil {
		return 0
	}
	return len(panel.rows)
}

func (panel *dataPanel) Update(message tea.Msg) tea.Cmd {
	if panel == nil {
		return nil
	}
	var command tea.Cmd
	panel.list, command = panel.list.Update(message)
	return command
}

func (panel *dataPanel) View(width, height int, styles uiStyles) string {
	if panel == nil || width <= 0 || height <= 0 {
		return ""
	}
	panel.list.Styles.Title = styles.title
	listWidth, listHeight, detailWidth, detailHeight := width, height, width, height
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

func (panel *dataPanel) renderList(width, height int, styles uiStyles) string {
	builder := core.NewBuilder()
	builder.WriteString(styles.title.Render("DATA"))
	builder.WriteString("  ")
	sortLabel := "date"
	if panel.sort == dataSortScore {
		sortLabel = "score"
	}
	builder.WriteString(styles.status.Render(core.Sprintf("%d items · sort %s", len(panel.rows), sortLabel)))
	if filterExpr := panel.filter.String(); filterExpr != "" {
		builder.WriteString("  ")
		builder.WriteString(styles.thought.Render("filter " + filterExpr))
	}
	builder.WriteString("\n")
	if panel.list.SettingFilter() || panel.list.FilterState() == list.FilterApplied {
		builder.WriteString(panel.list.FilterInput.View())
		builder.WriteString("\n")
	}
	visible := panel.list.VisibleItems()
	if len(visible) == 0 {
		builder.WriteString("\n")
		builder.WriteString(styles.status.Render("○ No items match this filter"))
		builder.WriteString("\n")
		builder.WriteString(styles.thought.Render("Import or capture data with `lem data import` / `lem serve --capture`."))
		return fitPane(builder.String(), width, height, styles.panel)
	}
	for index, raw := range visible {
		item, ok := raw.(dataListItem)
		if !ok {
			continue
		}
		cursor := "  "
		rowStyle := styles.answer
		if index == panel.list.Index() {
			cursor = "› "
			rowStyle = styles.accent
		}
		score := "—"
		if item.row.HasScore {
			score = core.Sprintf("%.0f", item.row.TopScore.Value)
		}
		builder.WriteString(cursor)
		builder.WriteString(styles.status.Render(dataStatusGlyph(item.row.Review.Status) + " " + core.Upper(string(item.row.Review.Status))))
		builder.WriteString("  ")
		builder.WriteString(styles.thought.Render(string(item.row.Item.Kind) + " · " + score))
		builder.WriteString("  ")
		builder.WriteString(rowStyle.Render(item.row.Dataset.Slug))
		builder.WriteString("\n")
	}
	builder.WriteString(styles.thought.Render("/ filter · j/k select · f filters · s sort · a/r/c/e/t act · A/R/C/T bulk"))
	return fitPane(builder.String(), width, height, styles.panel)
}

func (panel *dataPanel) renderDetail(width, height int, styles uiStyles) string {
	row, ok := panel.Selected()
	if !ok {
		return styles.status.Render("Select an item for its content, scores, and lineage.")
	}
	builder := core.NewBuilder()
	builder.WriteString(styles.title.Render(row.Dataset.Slug))
	builder.WriteString("  ")
	builder.WriteString(styles.status.Render(dataStatusGlyph(row.Review.Status) + " " + core.Upper(string(row.Review.Status))))
	builder.WriteString("\n\n")
	dataDetailRow(builder, styles, "kind", string(row.Item.Kind))
	dataDetailRow(builder, styles, "source", string(row.Item.Source))
	dataDetailRow(builder, styles, "source ref", row.Item.SourceRef)
	dataDetailRow(builder, styles, "fingerprint", row.Item.ModelFingerprint)
	dataDetailRow(builder, styles, "created", row.Item.CreatedAt.Format(time.RFC3339))

	if row.Review.Reviewer == dataset.ReviewerAutoWelfare {
		builder.WriteString("\n")
		builder.WriteString(styles.attention.Render("! WELFARE FLAG"))
		builder.WriteString("\n")
		note := "the welfare screen quarantined this item at ingest"
		if row.Review.Note != "" {
			note = core.Concat(note, " — ", row.Review.Note)
		}
		builder.WriteString(styles.answer.Render(note))
		builder.WriteString("\n")
	}

	builder.WriteString("\n")
	builder.WriteString(styles.accent.Render("CONTENT"))
	builder.WriteString("\n")
	builder.WriteString(panel.renderContent(row.Item, width))
	builder.WriteString("\n")

	builder.WriteString("\n")
	builder.WriteString(styles.accent.Render("SCORES"))
	builder.WriteString("\n")
	builder.WriteString(panel.renderScores(row.Item.ID, width))

	builder.WriteString("\n")
	builder.WriteString(styles.accent.Render("REVIEW"))
	builder.WriteString("\n")
	builder.WriteString(panel.renderReviewHistory(row.Item.ID, styles))

	builder.WriteString("\n")
	builder.WriteString(styles.accent.Render("LINEAGE"))
	builder.WriteString("\n")
	builder.WriteString(panel.renderLineage(row.Item, styles))

	return fitPane(builder.String(), width, height, styles.panel)
}

// renderContent renders an item's prompt/response (or, for a KindTrace
// item, its raw opaque JSON) through the same Glamour markdown path Chat
// uses for turns (markdown.go) — the design's "existing Glamour path".
func (panel *dataPanel) renderContent(item dataset.Item, width int) string {
	prompt, response, ok := dataItemExchange(item)
	if !ok {
		return core.AsString(item.Content)
	}
	body := core.Concat("**Prompt**\n\n", prompt, "\n\n**Response**\n\n", response)
	return panel.markdown.Render(item.ID, body, max(1, width-2))
}

// dataItemExchange extracts the (prompt, response) pair renderContent and
// the edit-as-derived editor both use: KindPair carries it directly;
// KindMessages reduces via MessagesContent.LastExchange (the same
// trailing-assistant-turn reduction go/dataset's own heuristic scorer and
// export writers use); KindTrace is opaque (ok=false).
func dataItemExchange(item dataset.Item) (prompt, response string, ok bool) {
	switch item.Kind {
	case dataset.KindPair:
		var pc dataset.PairContent
		if r := core.JSONUnmarshal(item.Content, &pc); !r.OK {
			return "", "", false
		}
		return pc.Prompt, pc.Response, true
	case dataset.KindMessages:
		var mc dataset.MessagesContent
		if r := core.JSONUnmarshal(item.Content, &mc); !r.OK {
			return "", "", false
		}
		pc, found := mc.LastExchange()
		if !found {
			return "", "", false
		}
		return pc.Prompt, pc.Response, true
	default:
		return "", "", false
	}
}

// renderScores renders every Score row for itemID as Markdown — kind,
// value, scorer/judge identity, and timestamp, followed by the full
// payload pretty-printed in a fenced code block — the design's "full
// score breakdown from the Score payloads", generic across the heuristic
// kinds (lek/hostility/sycophancy) and any judge:<name> kind without this
// panel special-casing a payload shape.
func (panel *dataPanel) renderScores(itemID string, width int) string {
	scoresResult := panel.store.Scores(itemID)
	if !scoresResult.OK {
		return "○ scores unavailable: " + scoresResult.Error()
	}
	scores, ok := scoresResult.Value.([]dataset.Score)
	if !ok || len(scores) == 0 {
		return "○ not yet scored"
	}
	body := core.NewBuilder()
	for _, score := range scores {
		identity := score.ScorerName
		if score.ScorerVersion != "" {
			identity = core.Concat(identity, " v", score.ScorerVersion)
		}
		if identity == "" {
			identity = "judge:" + score.JudgeFingerprint
		}
		body.WriteString(core.Sprintf("**%s** = %.2f — %s @ %s\n\n", score.Kind, score.Value, identity, score.CreatedAt.Format(time.RFC3339)))
		body.WriteString("```json\n")
		body.WriteString(prettyJSON(score.Payload))
		body.WriteString("\n```\n\n")
	}
	return panel.markdown.Render("scores:"+itemID, body.String(), max(1, width-2))
}

// prettyJSON re-indents raw JSON bytes for display, degrading to the raw
// bytes verbatim if they do not parse (never hides a malformed payload).
func prettyJSON(raw []byte) string {
	if len(raw) == 0 {
		return "{}"
	}
	var value any
	if r := core.JSONUnmarshal(raw, &value); !r.OK {
		return core.AsString(raw)
	}
	if r := core.JSONMarshalIndent(value, "", "  "); r.OK {
		return core.AsString(r.Value.([]byte))
	}
	return core.AsString(raw)
}

// renderReviewHistory prefers the full append-only history (see
// [ReviewHistoryStore], a CLI-side optional capability duckDatasetStore
// implements) and falls back to ReviewLatest — the one call every
// dataset.Store implementation (including the root-module MemoryStore)
// supports — when the connected store does not expose it.
func (panel *dataPanel) renderReviewHistory(itemID string, styles uiStyles) string {
	if historyStore, ok := panel.store.(ReviewHistoryStore); ok {
		if historyResult := historyStore.ReviewHistory(itemID); historyResult.OK {
			if history, ok := historyResult.Value.([]dataset.Review); ok && len(history) > 0 {
				lines := make([]string, 0, len(history))
				for _, review := range history {
					lines = append(lines, dataReviewLine(review, styles))
				}
				return core.Join("\n", lines...) + "\n"
			}
		}
	}
	latestResult := panel.store.ReviewLatest(itemID)
	if !latestResult.OK {
		return styles.status.Render("○ review unavailable: " + latestResult.Error())
	}
	latest, ok := latestResult.Value.(dataset.Review)
	if !ok || latest.Status == "" || latest.Status == dataset.StatusPending {
		return styles.status.Render("○ pending review")
	}
	return dataReviewLine(latest, styles) + "\n"
}

func dataReviewLine(review dataset.Review, styles uiStyles) string {
	line := styles.status.Render(core.Sprintf("· %-12s %-14s ", review.Status, review.Reviewer)) +
		styles.thought.Render(review.CreatedAt.Format(time.RFC3339))
	if review.Note != "" {
		line = core.Concat(line, "\n  ", styles.answer.Render(review.Note))
	}
	return line
}

// renderLineage shows the item's parent (if edit-as-derived produced it)
// and any derived children — items in the SAME dataset whose
// ParentItemID names this one. dataset.Store has no "children of X"
// query, so children are found by scanning the dataset's full item list
// (including archived — the original is archived once superseded); an
// honest O(dataset size) scan, acceptable for a human review surface, not
// a hot path.
func (panel *dataPanel) renderLineage(item dataset.Item, styles uiStyles) string {
	lines := make([]string, 0, 2)
	if item.ParentItemID != "" {
		if parentResult := panel.store.Item(item.ParentItemID); parentResult.OK {
			if parent, ok := parentResult.Value.(dataset.Item); ok {
				note := ""
				if parent.Archived {
					note = "  (archived)"
				}
				lines = append(lines, styles.status.Render("parent  ")+styles.answer.Render(parent.ID)+styles.thought.Render(note))
			}
		}
	}
	if childrenResult := panel.store.Items(dataset.ItemFilter{DatasetID: item.DatasetID, IncludeArchived: true}); childrenResult.OK {
		if siblings, ok := childrenResult.Value.([]dataset.Item); ok {
			for _, candidate := range siblings {
				if candidate.ParentItemID == item.ID {
					lines = append(lines, styles.status.Render("derived ")+styles.answer.Render(candidate.ID))
				}
			}
		}
	}
	if len(lines) == 0 {
		return styles.status.Render("○ no lineage — an original item")
	}
	return core.Join("\n", lines...) + "\n"
}

func dataDetailRow(builder *core.Builder, styles uiStyles, label, value string) {
	if value == "" {
		return
	}
	builder.WriteString(styles.status.Render(label + "  "))
	builder.WriteString(styles.answer.Render(value))
	builder.WriteString("\n")
}

// currentReviewer identifies the human reviewer a manual Approve/Reject/
// QuarantineClear/Tag action records — the OS username (falling back to
// "local" when it cannot be resolved), distinguishing manual review rows
// from the auto:welfare / auto:threshold reviewer identities the design
// reserves for automated ones.
func currentReviewer() string {
	name := core.Trim(core.Username())
	if name == "" {
		return "local"
	}
	return name
}

// ---- actions ----

// Approve records an approved Review for itemID.
func (panel *dataPanel) Approve(itemID string) core.Result {
	if result := panel.reviewNoRefresh(itemID, dataset.StatusApproved, ""); !result.OK {
		return result
	}
	return panel.refreshAndSelect(itemID)
}

// Reject records a rejected Review for itemID.
func (panel *dataPanel) Reject(itemID string) core.Result {
	if result := panel.reviewNoRefresh(itemID, dataset.StatusRejected, ""); !result.OK {
		return result
	}
	return panel.refreshAndSelect(itemID)
}

// QuarantineClear approves a quarantined item with a mandatory
// justification note — the design's "quarantine-clear (requires a note)".
func (panel *dataPanel) QuarantineClear(itemID, note string) core.Result {
	note = core.Trim(note)
	if note == "" {
		return core.Fail(core.E("tui.dataPanel.QuarantineClear", "a note is required to clear a quarantine", nil))
	}
	if result := panel.reviewNoRefresh(itemID, dataset.StatusApproved, note); !result.OK {
		return result
	}
	return panel.refreshAndSelect(itemID)
}

// Tag records a freeform label on an item via a Review note — go/dataset
// carries no dedicated tag field (Task 1-7's shipped domain model has
// none, and widening it is out of this task's go/ lane; see
// datasetstore.go's "CLI-only capability extensions" for the same
// reasoning applied to item-archive/review-history), so a tag rides the
// same append-only Review channel notes already use, preserving the
// item's current review status rather than changing it — tagging is
// never itself a review decision.
func (panel *dataPanel) Tag(itemID, label string) core.Result {
	label = core.Trim(label)
	if label == "" {
		return core.Fail(core.E("tui.dataPanel.Tag", "a tag label is required", nil))
	}
	if panel == nil || panel.store == nil {
		return core.Fail(core.E("tui.dataPanel.Tag", "data panel is unavailable", nil))
	}
	trimmedID := core.Trim(itemID)
	status := dataset.StatusPending
	if latestResult := panel.store.ReviewLatest(trimmedID); latestResult.OK {
		if latest, ok := latestResult.Value.(dataset.Review); ok && latest.Status != "" {
			status = latest.Status
		}
	}
	if result := panel.reviewNoRefresh(trimmedID, status, "tag: "+label); !result.OK {
		return result
	}
	return panel.refreshAndSelect(trimmedID)
}

func (panel *dataPanel) reviewNoRefresh(itemID string, status dataset.ReviewStatus, note string) core.Result {
	if panel == nil || panel.store == nil {
		return core.Fail(core.E("tui.dataPanel.review", "data panel is unavailable", nil))
	}
	itemID = core.Trim(itemID)
	if itemID == "" {
		return core.Fail(core.E("tui.dataPanel.review", "an item id is required", nil))
	}
	review := dataset.Review{ItemID: itemID, Status: status, Reviewer: currentReviewer(), Note: note, CreatedAt: panel.now().UTC()}
	return panel.store.ReviewAppend(review)
}

func (panel *dataPanel) refreshAndSelect(itemID string) core.Result {
	if result := panel.Refresh(); !result.OK {
		return result
	}
	panel.selectItem(core.Trim(itemID))
	return core.Ok(itemID)
}

// dataEditedContent rebuilds an Item's Content with prompt/response
// substituted, per Kind: KindPair replaces both fields directly; a
// KindMessages item keeps every turn but the last assistant turn's
// content (the same reduction dataItemExchange used to populate the
// editor), so the derived item preserves multi-turn structure rather than
// collapsing it to a bare pair. KindTrace is opaque and not editable this
// way.
func dataEditedContent(item dataset.Item, prompt, response string) ([]byte, error) {
	switch item.Kind {
	case dataset.KindPair:
		marshalled := core.JSONMarshal(dataset.PairContent{Prompt: prompt, Response: response})
		if !marshalled.OK {
			return nil, marshalled.Err()
		}
		return marshalled.Value.([]byte), nil
	case dataset.KindMessages:
		var mc dataset.MessagesContent
		if r := core.JSONUnmarshal(item.Content, &mc); !r.OK {
			return nil, r.Err()
		}
		assistantIdx := -1
		for i := len(mc.Messages) - 1; i >= 0; i-- {
			if mc.Messages[i].Role == "assistant" {
				assistantIdx = i
				break
			}
		}
		if assistantIdx < 0 {
			return nil, core.NewError("dataset: messages content has no assistant turn to edit")
		}
		edited := append([]dataset.MessageTurn(nil), mc.Messages...)
		edited[assistantIdx].Content = response
		marshalled := core.JSONMarshal(dataset.MessagesContent{Messages: edited})
		if !marshalled.OK {
			return nil, marshalled.Err()
		}
		return marshalled.Value.([]byte), nil
	default:
		return nil, core.NewError("dataset: this item kind cannot be edited as a derived item")
	}
}

// EditAsDerived implements the design's edit-and-approve lineage flow
// exactly (go/dataset's own Item doc comment): create a derived Item
// (ParentItemID set, fresh content/content-hash), archive the original,
// approve the child. Requires the connected store to implement
// [ItemArchiver] — go/dataset's Store interface has no item-level archive
// (only DatasetArchive, one level up); a store that lacks it (e.g. the
// root-module dataset.MemoryStore in a lightweight test) fails loudly
// rather than silently skipping the archive step.
func (panel *dataPanel) EditAsDerived(original dataset.Item, prompt, response string) core.Result {
	if panel == nil || panel.store == nil {
		return core.Fail(core.E("tui.dataPanel.EditAsDerived", "data panel is unavailable", nil))
	}
	archiver, ok := panel.store.(ItemArchiver)
	if !ok {
		return core.Fail(core.E("tui.dataPanel.EditAsDerived", "the connected dataset store cannot archive the superseded original", nil))
	}
	content, contentErr := dataEditedContent(original, prompt, response)
	if contentErr != nil {
		return core.Fail(core.E("tui.dataPanel.EditAsDerived", contentErr.Error(), contentErr))
	}
	hashResult := dataset.ContentHash(original.Kind, content)
	if !hashResult.OK {
		return hashResult
	}
	now := panel.now().UTC()
	derived := dataset.Item{
		ID: panel.ids(), DatasetID: original.DatasetID, Kind: original.Kind, Content: content,
		Source: original.Source, SourceRef: original.SourceRef, ModelFingerprint: original.ModelFingerprint,
		ContentHash: hashResult.Value.(string), ParentItemID: original.ID, CreatedAt: now,
	}
	if result := panel.store.ItemAppend(derived); !result.OK {
		return result
	}
	if result := archiver.ItemArchive(original.ID); !result.OK {
		return result
	}
	review := dataset.Review{ItemID: derived.ID, Status: dataset.StatusApproved, Reviewer: currentReviewer(), Note: "edited from " + original.ID, CreatedAt: now}
	if result := panel.store.ReviewAppend(review); !result.OK {
		return result
	}
	// Returns the full derived Item (not just its id, unlike
	// refreshAndSelect's other callers) — the one action here whose
	// result a caller plausibly wants to inspect (e.g. to render the new
	// item immediately without a second Item() round trip).
	if result := panel.Refresh(); !result.OK {
		return result
	}
	panel.selectItem(derived.ID)
	return core.Ok(derived)
}

// BulkApply applies action to every item in the panel's CURRENT filtered
// view (panel.rows — every non-archived item matching filter, across every
// matching dataset), per the design's "bulk-apply-to-current-filter". The
// caller (the confirmation overlay) is the only gate: BulkApply itself
// performs no confirmation — "no confirm, no writes" is enforced by never
// calling this until the overlay's two-phase arm/confirm has completed.
// edit-as-derived has no bulk form (each item's edited text is unique;
// Capabilities() never offers it as a bulk action).
func (panel *dataPanel) BulkApply(action dataAction, note string) core.Result {
	if panel == nil || panel.store == nil {
		return core.Fail(core.E("tui.dataPanel.BulkApply", "data panel is unavailable", nil))
	}
	if action == dataActionEditAsDerived {
		return core.Fail(core.E("tui.dataPanel.BulkApply", "edit-as-derived has no bulk form", nil))
	}
	note = core.Trim(note)
	if action.needsNote() && note == "" {
		return core.Fail(core.E("tui.dataPanel.BulkApply", "a note is required for this bulk action", nil))
	}
	targets := append([]dataItemRow(nil), panel.rows...)
	applied := 0
	for _, row := range targets {
		var result core.Result
		switch action {
		case dataActionApprove:
			result = panel.reviewNoRefresh(row.Item.ID, dataset.StatusApproved, note)
		case dataActionReject:
			result = panel.reviewNoRefresh(row.Item.ID, dataset.StatusRejected, note)
		case dataActionQuarantineClear:
			result = panel.reviewNoRefresh(row.Item.ID, dataset.StatusApproved, note)
		case dataActionTag:
			status := row.Review.Status
			if status == "" {
				status = dataset.StatusPending
			}
			result = panel.reviewNoRefresh(row.Item.ID, status, "tag: "+note)
		default:
			result = core.Fail(core.E("tui.dataPanel.BulkApply", "unknown bulk action", nil))
		}
		if !result.OK {
			_ = panel.Refresh() // best-effort resync before surfacing the partial-batch error
			return core.Fail(core.E("tui.dataPanel.BulkApply", core.Sprintf("item %s: %s", row.Item.ID, result.Error()), result.Err()))
		}
		applied++
	}
	if result := panel.Refresh(); !result.OK {
		return result
	}
	return core.Ok(applied)
}

// dataCapability mirrors the agentcap pattern (agentcap.go) for the Data
// panel's own action set: always present in the palette, Available/Reason
// render honestly rather than hiding an action outright.
type dataCapability struct {
	Action    dataAction
	Bulk      bool
	Title     string
	Available bool
	Reason    string
}

// Capabilities reports every action the palette should mirror — single-
// item actions against the current selection, plus their bulk-apply-to-
// filter counterparts (excluding edit-as-derived, which has no bulk
// form), always present but rendered unavailable with an honest reason
// when a precondition is not met.
func (panel *dataPanel) Capabilities() []dataCapability {
	if panel == nil || panel.store == nil {
		reason := "dataset store is not connected"
		return []dataCapability{
			{Action: dataActionApprove, Title: "Approve", Reason: reason},
			{Action: dataActionReject, Title: "Reject", Reason: reason},
			{Action: dataActionQuarantineClear, Title: "Clear quarantine", Reason: reason},
			{Action: dataActionEditAsDerived, Title: "Edit as derived", Reason: reason},
			{Action: dataActionTag, Title: "Tag", Reason: reason},
			{Action: dataActionApprove, Bulk: true, Title: "Bulk approve", Reason: reason},
			{Action: dataActionReject, Bulk: true, Title: "Bulk reject", Reason: reason},
			{Action: dataActionQuarantineClear, Bulk: true, Title: "Bulk clear quarantine", Reason: reason},
			{Action: dataActionTag, Bulk: true, Title: "Bulk tag", Reason: reason},
		}
	}
	selected, hasSelection := panel.Selected()
	_, archiver := panel.store.(ItemArchiver)
	count := panel.FilteredCount()

	single := func(action dataAction, title string) dataCapability {
		capability := dataCapability{Action: action, Title: title, Available: hasSelection}
		if !hasSelection {
			capability.Reason = "a selected item is required"
			return capability
		}
		switch {
		case action == dataActionQuarantineClear && selected.Review.Status != dataset.StatusQuarantined:
			capability.Available, capability.Reason = false, "the selected item is not quarantined"
		case action == dataActionEditAsDerived && !archiver:
			capability.Available, capability.Reason = false, "the connected dataset store cannot archive the superseded original"
		}
		return capability
	}
	bulk := func(action dataAction, title string) dataCapability {
		capability := dataCapability{Action: action, Bulk: true, Title: title, Available: count > 0}
		if count == 0 {
			capability.Reason = "no items match the current filter"
		}
		return capability
	}

	return []dataCapability{
		single(dataActionApprove, "Approve"),
		single(dataActionReject, "Reject"),
		single(dataActionQuarantineClear, "Clear quarantine"),
		single(dataActionEditAsDerived, "Edit as derived"),
		single(dataActionTag, "Tag"),
		bulk(dataActionApprove, "Bulk approve"),
		bulk(dataActionReject, "Bulk reject"),
		bulk(dataActionQuarantineClear, "Bulk clear quarantine"),
		bulk(dataActionTag, "Bulk tag"),
	}
}
