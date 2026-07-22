// SPDX-Licence-Identifier: EUPL-1.2

package tui

import (
	"context"
	"sync"
	"time"

	"github.com/charmbracelet/bubbles/key"
	"github.com/charmbracelet/bubbles/list"
	"github.com/charmbracelet/bubbles/spinner"
	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"

	core "dappco.re/go"
	"dappco.re/go/inference"
	"dappco.re/go/inference/dataset"
	"dappco.re/go/inference/decode/parser"
)

// turn is one rendered element in the transcript.
type turn struct {
	id      string
	role    string // "user" | "assistant" | "tool"
	model   string
	thought string
	text    string
	calls   []string // rendered tool-call receipts on an assistant turn
}

type app struct {
	boot               bootState
	workspaceLoader    func() core.Result
	resources          *workspaceResources
	lifecycle          *appLifecycle
	warnings           []string
	runtimeDetector    runtimeDetector
	knowledgeScan      knowledgeScanner
	knowledgeMounts    []knowledgeMount
	knowledgeLimit     int64
	recentSessionLimit int

	activePanel           panelID
	inspectorOpen         bool
	styles                uiStyles
	keys                  keyMap
	markdown              *markdownRenderer
	activeOverlay         overlayKind
	palette               *commandPalette
	switcher              *sessionSwitcher
	search                *historySearch
	help                  *helpOverlay
	sessions              *sessionManager
	repository            workspaceRepository
	preferences           preferenceStore
	inspector             inspectorState
	agent                 agentProvider
	work                  *workPanel
	workEditor            *workEditor
	data                  *dataPanel
	dataEditor            *dataItemEditor
	dataNote              *dataNoteOverlay
	dataBulk              *dataBulkOverlay
	dataFilter            *dataFilterOverlay
	dataInitialSlug       string
	launchReview          *launchReviewOverlay
	answerOverlay         *agentAnswerOverlay
	changeOverlay         *changeAcceptanceOverlay
	agentReview           agentReview
	agentRequest          agentRequest
	agentCommand          tea.Cmd
	agentStage            agentReviewStage
	agentOperationID      uint64
	agentOperationNext    uint64
	agentInFlight         bool
	agentRefreshArmed     bool
	agentSnapshotNext     uint64
	agentSnapshotCurrent  uint64
	agentSnapshotInFlight bool
	agentSnapshotPending  bool
	knowledge             *knowledgeLibrary
	attachments           []attachmentRecord

	picker       list.Model
	spin         spinner.Model
	input        textarea.Model
	view         viewport.Model
	width        int
	height       int
	ready        bool
	loading      string // model path mid-load ("" = idle)
	pendingModel string
	modelLoader  func(path string, contextLength int) tea.Cmd

	model     inference.TextModel
	modelName string
	lane      *modelLane
	jobs      *jobManager
	sessionID string

	cfg   settings
	modes modeState
	tools toolState
	svc   serviceState

	turns       []turn
	gen         *generation
	generating  bool
	sessionJobs map[string]*sessionGeneration
	follow      bool
	newOutput   bool
	refreshAt   time.Time
	toolHops    int // auto-continuations this turn chain (bounded)
	lastTokS    float64
	errText     string
}

type loadedMsg struct {
	model inference.TextModel
	name  string
}

type agentReviewStage uint8

const (
	agentReviewNone agentReviewStage = iota
	agentReviewProject
	agentReviewLaunch
)

type loadErrMsg struct{ err error }

type bootPhase uint8

const (
	bootReady bootPhase = iota
	bootLoading
	bootFailed
)

type bootState struct {
	phase bootPhase
	err   error
}

type workspaceReadyMsg struct{ resources *workspaceResources }
type workspaceFailedMsg struct{ err error }
type runtimeDetectedMsg struct{ result core.Result }
type knowledgeDiscoveredMsg struct{ result core.Result }
type agentSnapshotMsg struct {
	requestID uint64
	result    core.Result
}
type agentRefreshMsg struct{}

type sessionGeneration struct {
	generation *generation
	answer     turnRecord
	job        generationJobRecord
	started    bool
	cancelled  bool
	persistErr string
	dirty      bool
	flushAt    time.Time
}

type appLifecycle struct {
	shutdownMu       sync.Mutex
	shutdownComplete bool
	result           core.Result
	mu               sync.Mutex
	stopped          bool
	cancel           context.CancelFunc
	context          context.Context
	workers          sync.WaitGroup
	stopCh           chan struct{}
	nextID           uint64
	pending          map[uint64]tea.Msg
}

type lifecycleStoppedMsg struct{}
type lifecycleResultMsg struct{ id uint64 }

func newAppLifecycle(ctx context.Context, cancel context.CancelFunc) *appLifecycle {
	if ctx == nil {
		ctx = context.Background()
	}
	return &appLifecycle{
		result:  core.Ok(nil),
		cancel:  cancel,
		context: ctx,
		stopCh:  make(chan struct{}),
		pending: make(map[uint64]tea.Msg),
	}
}

// command joins a resource-producing Bubble Tea command to the app lifetime.
// A result arriving after quit is closed here instead of being sent to a dead
// update loop.
func (lifecycle *appLifecycle) command(command tea.Cmd) tea.Cmd {
	if command == nil {
		return nil
	}
	return func() tea.Msg {
		if !lifecycle.beginWorker() {
			return lifecycleStoppedMsg{}
		}
		defer lifecycle.workers.Done()
		message := command()
		return lifecycle.adopt(message)
	}
}

func (lifecycle *appLifecycle) adopt(message tea.Msg) tea.Msg {
	if lifecycle == nil {
		closeLifecycleMessage(message)
		return lifecycleStoppedMsg{}
	}
	lifecycle.mu.Lock()
	if lifecycle.stopped {
		lifecycle.mu.Unlock()
		closeLifecycleMessage(message)
		return lifecycleStoppedMsg{}
	}
	if !lifecycleOwnsMessage(message) {
		lifecycle.mu.Unlock()
		return message
	}
	lifecycle.nextID++
	id := lifecycle.nextID
	lifecycle.pending[id] = message
	lifecycle.mu.Unlock()
	return lifecycleResultMsg{id: id}
}

func (lifecycle *appLifecycle) claim(id uint64) (tea.Msg, bool) {
	if lifecycle == nil || id == 0 {
		return nil, false
	}
	lifecycle.mu.Lock()
	defer lifecycle.mu.Unlock()
	message, ok := lifecycle.pending[id]
	if ok {
		delete(lifecycle.pending, id)
	}
	return message, ok
}

func (lifecycle *appLifecycle) closePending() {
	if lifecycle == nil {
		return
	}
	lifecycle.mu.Lock()
	messages := make([]tea.Msg, 0, len(lifecycle.pending))
	for id, message := range lifecycle.pending {
		messages = append(messages, message)
		delete(lifecycle.pending, id)
	}
	lifecycle.mu.Unlock()
	for _, message := range messages {
		closeLifecycleMessage(message)
	}
}

func lifecycleOwnsMessage(message tea.Msg) bool {
	switch message.(type) {
	case loadedMsg, workspaceReadyMsg, agentActionMsg, agentSnapshotMsg:
		return true
	default:
		return false
	}
}

func (lifecycle *appLifecycle) beginWorker() bool {
	if lifecycle == nil {
		return false
	}
	lifecycle.mu.Lock()
	defer lifecycle.mu.Unlock()
	if lifecycle.stopped {
		return false
	}
	lifecycle.workers.Add(1)
	return true
}

func (lifecycle *appLifecycle) isStopped() bool {
	if lifecycle == nil {
		return true
	}
	lifecycle.mu.Lock()
	defer lifecycle.mu.Unlock()
	return lifecycle.stopped
}

func (lifecycle *appLifecycle) stop() {
	if lifecycle == nil {
		return
	}
	lifecycle.mu.Lock()
	if lifecycle.stopped {
		lifecycle.mu.Unlock()
		return
	}
	lifecycle.stopped = true
	cancel := lifecycle.cancel
	if lifecycle.stopCh != nil {
		close(lifecycle.stopCh)
	}
	lifecycle.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

func (lifecycle *appLifecycle) wait() {
	if lifecycle != nil {
		lifecycle.workers.Wait()
	}
}

func closeLifecycleMessage(message tea.Msg) {
	switch value := message.(type) {
	case loadedMsg:
		if value.model != nil {
			if result := value.model.Close(); !result.OK {
				core.Warn("tui.lifecycle.close_late_model", "error", result.Value)
			}
		}
	case workspaceReadyMsg:
		if value.resources != nil {
			if result := value.resources.Close(); !result.OK {
				core.Warn("tui.lifecycle.close_late_workspace", "error", result.Value)
			}
		}
	case agentActionMsg:
		// Agent action messages hold no resource; they are intentionally dropped.
	}
}

// loadModel loads the checkpoint through the registered engine as a tea.Cmd.
func loadModel(path string, ctxLen int) tea.Cmd {
	return func() tea.Msg {
		var opts []inference.LoadOption
		if ctxLen > 0 {
			opts = append(opts, inference.WithContextLen(ctxLen))
		}
		r := inference.LoadModel(path, opts...)
		if !r.OK {
			if err, ok := r.Value.(error); ok {
				return loadErrMsg{err: err}
			}
			return loadErrMsg{err: core.NewError("load failed")}
		}
		return loadedMsg{model: r.Value.(inference.TextModel), name: displayName(path)}
	}
}

func newApp(modelPath string, ctxLen, maxTokens int) app {
	ctx, cancel := context.WithCancel(context.Background())
	lifecycle := newAppLifecycle(ctx, cancel)
	styles := newUIStyles(midnightTheme())
	keys := newKeyMap()
	agent := newUnavailableAgentProvider(defaultAgentUnavailableReason)
	sp := spinner.New()
	sp.Spinner = spinner.MiniDot

	in := textarea.New()
	in.Placeholder = "ask… (enter sends · esc cancels a reply · tab switches tabs · ctrl+c quits)"
	in.SetHeight(3)
	in.ShowLineNumbers = false
	in.Focus()

	cfg := newSettings()
	for i, v := range ctxSteps {
		if v == ctxLen {
			cfg.ctxIdx = i
		}
	}
	for i, v := range maxTokSteps {
		if v == maxTokens {
			cfg.maxTokIdx = i
		}
	}

	a := app{
		boot:               bootState{phase: bootReady},
		lifecycle:          lifecycle,
		runtimeDetector:    newContainerRuntimeDetector(),
		knowledgeScan:      newKnowledgeScanner(),
		knowledgeLimit:     knowledgeSystemMessageMaxBytes,
		recentSessionLimit: defaultPreferenceValues().RecentSessionLimit,
		modelLoader:        loadModel,
		activePanel:        panelChat,
		styles:             styles,
		keys:               keys,
		markdown:           newMarkdownRenderer(styles.theme.name),
		palette:            newCommandPalette(styles),
		help:               newHelpOverlay(keys, styles),
		inspector:          newInspector(),
		agent:              agent,
		picker:             newPicker(),
		spin:               sp,
		input:              in,
		cfg:                cfg,
		modes:              modeState{},
		tools:              newTools(),
		svc:                newService(),
		jobs:               newJobManager(ctx),
		sessionJobs:        make(map[string]*sessionGeneration),
		sessionID:          newRecordID(),
		follow:             true,
	}
	if modelPath != "" {
		a.activePanel = panelChat
		a.loading = modelPath
	}
	return a
}

func newWorkspaceApp(modelPath string, ctxLen, maxTokens int, loader func() core.Result) app {
	a := newApp(modelPath, ctxLen, maxTokens)
	a.boot = bootState{phase: bootLoading}
	a.workspaceLoader = loader
	a.width = 100
	a.height = 30
	return a
}

func workspaceBootstrap(loader func() core.Result) tea.Cmd {
	return func() tea.Msg {
		if loader == nil {
			return workspaceFailedMsg{err: core.E("tui.workspaceBootstrap", "workspace loader is unavailable", nil)}
		}
		result := loader()
		if !result.OK {
			return workspaceFailedMsg{err: workspaceOpenError(result, "open workspace")}
		}
		resources, ok := result.Value.(*workspaceResources)
		if !ok || resources == nil {
			return workspaceFailedMsg{err: core.E("tui.workspaceBootstrap", "invalid workspace result", nil)}
		}
		return workspaceReadyMsg{resources: resources}
	}
}

func (a *app) attachWork(repository workspaceRepository, provider agentProvider) core.Result {
	if a == nil {
		return core.Fail(core.E("tui.app.attachWork", "application is unavailable", nil))
	}
	if provider == nil {
		provider = newUnavailableAgentProvider(defaultAgentUnavailableReason)
	}
	opened := newWorkPanel(repository, provider, nil, nil)
	if !opened.OK {
		return opened
	}
	a.repository = repository
	a.agent = provider
	a.work = opened.Value.(*workPanel)
	a.refreshAgentPalette()
	return core.Ok(a.work)
}

// attachData wires the Data panel to store — the connected workspace's
// resources.DatasetStore (opened best-effort in openWorkspaceWithContext,
// bootstrap.go), or nil when the dataset store could not open (a damaged
// or missing datasets.duckdb never blocks the rest of the workspace, per
// the design's blast-radius rationale). A nil store leaves a.data nil —
// panelView and the palette both render that honestly rather than
// panicking on a nil panel. When dataInitialSlug was set (RunDataReview's
// `lem data review <slug>` seam, tui.go), it is consumed exactly once
// here to pre-filter the panel to that one dataset.
func (a *app) attachData(store dataset.Store) core.Result {
	if a == nil {
		return core.Fail(core.E("tui.app.attachData", "application is unavailable", nil))
	}
	if store == nil {
		a.data = nil
		return core.Ok(nil)
	}
	opened := newDataPanel(store, a.markdown, nil, nil)
	if !opened.OK {
		return opened
	}
	a.data = opened.Value.(*dataPanel)
	if a.dataInitialSlug != "" {
		if result := a.data.SetFilter(dataFilterState{DatasetSlug: a.dataInitialSlug}); !result.OK {
			a.warnings = append(a.warnings, core.Concat("data: initial dataset filter: ", result.Error()))
		}
		a.dataInitialSlug = ""
	}
	a.refreshDataPalette()
	return core.Ok(a.data)
}

func (a *app) refreshAgentPalette() {
	if a == nil || a.palette == nil || a.agent == nil {
		return
	}
	var selected *workItemRecord
	if a.work != nil {
		if record, ok := a.work.Selected(); ok {
			selected = &record
		}
	}
	state := agentWorkSnapshot{}
	if a.work != nil {
		state.QueueStatus, state.QueueReason = a.work.queueStatus, a.work.queueReason
	}
	if selected != nil {
		state = a.work.AgentState(*selected)
		state.QueueStatus, state.QueueReason = a.work.queueStatus, a.work.queueReason
	}
	a.palette.SetAgentContext(a.agent.Capabilities(), selected, state)
	a.palette.SetWorkSelection(selected != nil)
}

func (a *app) openWorkEditor(record workItemRecord) core.Result {
	if a == nil || a.work == nil {
		return core.Fail(core.E("tui.app.openWorkEditor", "work panel is unavailable", nil))
	}
	a.workEditor = newWorkEditor(record)
	a.activeOverlay = overlayWorkEditor
	return core.Ok(a.workEditor)
}

func (a *app) saveWorkEditor() core.Result {
	if a == nil || a.work == nil || a.workEditor == nil {
		return core.Fail(core.E("tui.app.saveWorkEditor", "work editor is unavailable", nil))
	}
	title, task, repository := a.workEditor.values()
	a.workEditor.validation = ""
	var result core.Result
	if a.workEditor.editingID == "" {
		result = a.work.CreateWork(title, task, repository)
	} else {
		result = a.work.EditWork(a.workEditor.editingID, title, task, repository)
	}
	if !result.OK {
		a.workEditor.validation = result.Error()
	}
	return result
}

// refreshDataPalette rebuilds the "data."-prefixed palette entries from
// the Data panel's current Capabilities() — called after every action
// that could change availability (a selection change, a status change),
// mirroring refreshAgentPalette's precedent. Nil-safe: a.data.Capabilities
// on a nil *dataPanel returns the honest "not connected" catalogue.
func (a *app) refreshDataPalette() {
	if a == nil || a.palette == nil {
		return
	}
	a.palette.SetDataContext(a.data.Capabilities())
}

// dataSelectedItemID reads the Data panel's current selection, or "" when
// nothing is selected — the single-item note overlays' itemID.
func dataSelectedItemID(panel *dataPanel) string {
	selected, ok := panel.Selected()
	if !ok {
		return ""
	}
	return selected.Item.ID
}

// applyDataAction runs a no-note single-item action (Approve/Reject)
// directly against the current selection — these need no overlay at all,
// matching workPanel's Complete/Reopen/Archive precedent (friction should
// scale with blast radius, and a single-item approve/reject is routine).
func (a *app) applyDataAction(action dataAction) core.Result {
	if a == nil || a.data == nil {
		return core.Fail(core.E("tui.app.applyDataAction", "data panel is unavailable", nil))
	}
	selected, ok := a.data.Selected()
	if !ok {
		return core.Fail(core.E("tui.app.applyDataAction", "a selected item is required", nil))
	}
	var result core.Result
	switch action {
	case dataActionApprove:
		result = a.data.Approve(selected.Item.ID)
	case dataActionReject:
		result = a.data.Reject(selected.Item.ID)
	default:
		return core.Fail(core.E("tui.app.applyDataAction", "this action requires its own overlay", nil))
	}
	if result.OK {
		a.refreshDataPalette()
	}
	return result
}

// openDataEditor opens the edit-as-derived editor seam over the current
// selection, pre-checking the ItemArchiver capability EditAsDerived will
// need, so an unsupported store fails loudly here rather than after the
// human has typed an edit.
func (a *app) openDataEditor() core.Result {
	if a == nil || a.data == nil {
		return core.Fail(core.E("tui.app.openDataEditor", "data panel is unavailable", nil))
	}
	selected, ok := a.data.Selected()
	if !ok {
		return core.Fail(core.E("tui.app.openDataEditor", "a selected item is required", nil))
	}
	if _, archiver := a.data.store.(ItemArchiver); !archiver {
		return core.Fail(core.E("tui.app.openDataEditor", "the connected dataset store cannot archive the superseded original", nil))
	}
	a.dataEditor = newDataItemEditor(selected.Item)
	a.activeOverlay = overlayDataEditor
	return core.Ok(a.dataEditor)
}

func (a *app) saveDataEditor() core.Result {
	if a == nil || a.data == nil || a.dataEditor == nil {
		return core.Fail(core.E("tui.app.saveDataEditor", "data editor is unavailable", nil))
	}
	prompt, response := a.dataEditor.values()
	result := a.data.EditAsDerived(a.dataEditor.original, prompt, response)
	if result.OK {
		a.refreshDataPalette()
	}
	return result
}

// dataNotePrompt builds the title/prompt/placeholder for a note overlay,
// worded for either a single item or the shared bulk note.
func dataNotePrompt(action dataAction, bulk bool) (title, prompt, placeholder string) {
	scope := "this item"
	if bulk {
		scope = "every item matching the current filter"
	}
	if action == dataActionTag {
		return "Tag", core.Concat("Tag label for ", scope), "label"
	}
	return "Clear quarantine", core.Concat("Why is the quarantine on ", scope, " being cleared?"), "note"
}

// openDataNote opens the note overlay for action — itemID set is a
// single-item note (Tag/QuarantineClear applied directly on submit);
// itemID empty is a bulk action's shared note, which hands off to the
// count-confirm overlay on submit rather than writing anything yet (see
// submitDataNote).
func (a *app) openDataNote(action dataAction, itemID string) core.Result {
	if a == nil || a.data == nil {
		return core.Fail(core.E("tui.app.openDataNote", "data panel is unavailable", nil))
	}
	if itemID == "" && a.data.FilteredCount() == 0 {
		return core.Fail(core.E("tui.app.openDataNote", "no items match the current filter", nil))
	}
	if itemID != "" {
		selected, ok := a.data.Selected()
		if !ok || selected.Item.ID != itemID {
			return core.Fail(core.E("tui.app.openDataNote", "a selected item is required", nil))
		}
	}
	title, prompt, placeholder := dataNotePrompt(action, itemID == "")
	a.dataNote = newDataNoteOverlay(action, itemID, title, prompt, placeholder)
	a.activeOverlay = overlayDataNote
	return core.Ok(a.dataNote)
}

// submitDataNote applies a completed single-item note (Tag/QuarantineClear
// with itemID set) directly, or — for a bulk note (itemID empty) — hands
// off into the count-confirm overlay rather than writing anything yet, so
// "no confirm, no writes" holds for bulk actions that also collect a note.
func (a *app) submitDataNote() core.Result {
	if a == nil || a.data == nil || a.dataNote == nil {
		return core.Fail(core.E("tui.app.submitDataNote", "data note overlay is unavailable", nil))
	}
	note := a.dataNote.Value()
	if note == "" {
		return core.Fail(core.E("tui.app.submitDataNote", "a value is required", nil))
	}
	if a.dataNote.Bulk() {
		a.dataBulk = newDataBulkOverlay(a.dataNote.action, a.data.FilteredCount(), note)
		a.dataNote = nil
		a.activeOverlay = overlayDataBulk
		return core.Ok(a.dataBulk)
	}
	action, itemID := a.dataNote.action, a.dataNote.itemID
	var result core.Result
	switch action {
	case dataActionTag:
		result = a.data.Tag(itemID, note)
	case dataActionQuarantineClear:
		result = a.data.QuarantineClear(itemID, note)
	default:
		result = core.Fail(core.E("tui.app.submitDataNote", "unknown note action", nil))
	}
	if !result.OK {
		return result
	}
	a.dataNote, a.activeOverlay = nil, overlayNone
	a.refreshDataPalette()
	return result
}

// openDataBulk opens the bulk-apply-to-current-filter flow for action —
// straight to the count-confirm overlay when action needs no note
// (Approve/Reject), or via openDataNote first (itemID empty) to collect
// the shared note/label when it does (QuarantineClear/Tag).
func (a *app) openDataBulk(action dataAction) core.Result {
	if a == nil || a.data == nil {
		return core.Fail(core.E("tui.app.openDataBulk", "data panel is unavailable", nil))
	}
	if action.needsNote() {
		return a.openDataNote(action, "")
	}
	count := a.data.FilteredCount()
	if count == 0 {
		return core.Fail(core.E("tui.app.openDataBulk", "no items match the current filter", nil))
	}
	a.dataBulk = newDataBulkOverlay(action, count, "")
	a.activeOverlay = overlayDataBulk
	return core.Ok(a.dataBulk)
}

// confirmDataBulk applies a bulk action once its two-phase overlay has
// been explicitly confirmed (dataBulkOverlay.Confirm) — the only call
// site that reaches dataPanel.BulkApply, so an overlay dismissed by
// Escape (or one that never receives the second Enter) never writes
// anything.
func (a *app) confirmDataBulk() core.Result {
	if a == nil || a.data == nil || a.dataBulk == nil {
		return core.Fail(core.E("tui.app.confirmDataBulk", "bulk action overlay is unavailable", nil))
	}
	result := a.data.BulkApply(a.dataBulk.action, a.dataBulk.note)
	a.dataBulk, a.activeOverlay = nil, overlayNone
	a.refreshDataPalette()
	return result
}

// openDataFilter opens the structural filter overlay, pre-filled from the
// panel's current filter.
func (a *app) openDataFilter() core.Result {
	if a == nil || a.data == nil {
		return core.Fail(core.E("tui.app.openDataFilter", "data panel is unavailable", nil))
	}
	a.dataFilter = newDataFilterOverlay(a.data.FilterExpr())
	a.activeOverlay = overlayDataFilter
	return core.Ok(a.dataFilter)
}

func (a *app) applyDataFilter() core.Result {
	if a == nil || a.data == nil || a.dataFilter == nil {
		return core.Fail(core.E("tui.app.applyDataFilter", "data filter overlay is unavailable", nil))
	}
	if result := a.data.SetFilterExpr(a.dataFilter.Value()); !result.OK {
		return result
	}
	a.dataFilter, a.activeOverlay = nil, overlayNone
	a.refreshDataPalette()
	return core.Ok(nil)
}

// runDataCommand is the palette invocation path for a "data."-prefixed
// command (dataWorkspaceCommandsForContext, palette.go) — dispatches to
// the exact same app methods the Data panel's own hotkeys call
// (applyDataAction/openDataEditor/openDataNote/openDataBulk), so the
// palette mirrors every available action rather than duplicating logic,
// and forces the Data panel into view first so the result (or the
// overlay it opens) is immediately visible.
func (a *app) runDataCommand(capability dataCapability) core.Result {
	if a == nil || a.data == nil {
		return core.Fail(core.E("tui.command.data", "data panel is unavailable", nil))
	}
	a.activePanel = panelData
	if capability.Bulk {
		return a.openDataBulk(capability.Action)
	}
	switch capability.Action {
	case dataActionApprove, dataActionReject:
		return a.applyDataAction(capability.Action)
	case dataActionQuarantineClear, dataActionTag:
		return a.openDataNote(capability.Action, dataSelectedItemID(a.data))
	case dataActionEditAsDerived:
		return a.openDataEditor()
	default:
		return core.Fail(core.E("tui.command.data", "unknown data action", nil))
	}
}

func (a *app) queueAgentAction(feature agentFeature) core.Result {
	if a == nil || a.work == nil || a.agent == nil {
		return core.Fail(core.E("tui.app.queueAgentAction", "agent work is unavailable", nil))
	}
	if a.agentStage != agentReviewNone || a.agentInFlight {
		return core.Fail(core.E("tui.app.queueAgentAction", "an agent operation is already in progress", nil))
	}
	capabilities := a.agent.Capabilities()
	available := false
	capability := agentCapability{Feature: feature}
	for _, candidate := range capabilities {
		if candidate.Feature == feature {
			capability = candidate
			available = candidate.Available
			if !available {
				return core.Fail(core.E("tui.app.queueAgentAction", core.Concat(agentFeatureTitle(feature), " is unavailable: ", candidate.Reason), nil))
			}
			break
		}
	}
	if !available {
		return core.Fail(core.E("tui.app.queueAgentAction", "agent action is unavailable", nil))
	}
	selected, hasSelected := a.work.Selected()
	if agentFeatureNeedsWork(feature) && !hasSelected {
		return core.Fail(core.E("tui.app.queueAgentAction", "select a Work item first", nil))
	}
	state := agentWorkSnapshot{QueueStatus: a.work.queueStatus, QueueReason: a.work.queueReason}
	if hasSelected {
		state = a.work.AgentState(selected)
		state.QueueStatus, state.QueueReason = a.work.queueStatus, a.work.queueReason
	}
	if hasSelected || feature == agentFeatureQueueStart || feature == agentFeatureQueueStop {
		var selection *workItemRecord
		if hasSelected {
			selection = &selected
		}
		if actionAvailable, reason := agentCommandAvailability(capability, selection, state); !actionAvailable {
			return core.Fail(core.E("tui.app.queueAgentAction", core.Concat(agentFeatureTitle(feature), " is unavailable: ", reason), nil))
		}
	}
	request := agentRequest{Feature: feature}
	if hasSelected {
		request.WorkID = selected.ID
		state := a.work.AgentState(selected)
		request.RunID, request.QuestionID = state.NativeRunID, state.QuestionID
		if feature == agentFeatureRecoveryAbandon {
			request.Recovery = state.Recovery
			request.RunID = state.Recovery.Receipt.RunID
		}
		if feature == agentFeatureResume {
			request.Input = state.AnswerID
			request.Provider, request.Model = state.Agent, state.Runtime
		}
		if feature == agentFeatureAccept {
			request.Review = state.Review
		}
		request.Work = agentWorkRequest{ID: selected.ID, ExternalID: selected.ExternalID, Title: selected.Title, Task: selected.Task, Repository: selected.Repo}
	}
	a.beginAgentOperation(request, agentReviewNone)
	if feature == agentFeatureAnswer {
		a.answerOverlay = newAgentAnswerOverlay(request.RunID, request.QuestionID, selected.Question)
		a.activeOverlay = overlayAgentAnswer
		return core.Ok(nil)
	}
	if feature == agentFeatureAccept {
		a.changeOverlay = newChangeAcceptanceOverlay(request.Review)
		a.activeOverlay = overlayChangeReview
		return core.Ok(nil)
	}
	if feature == agentFeatureDispatch || feature == agentFeatureRetry || feature == agentFeatureResume || feature == agentFeatureChangesReview || feature == agentFeatureRecoveryAbandon {
		if feature == agentFeatureDispatch {
			a.agentStage = agentReviewProject
			a.launchReview = newAgentSelectionOverlay(request.Provider, request.Model)
			a.activeOverlay = overlayAgentSelection
			return core.Ok(nil)
		}
		a.agentInFlight = true
		a.agentCommand = a.lifecycle.command(a.agentReviewCommand(a.agentOperationID, request, agentReviewNone))
		return core.Ok(nil)
	}
	a.agentInFlight = true
	a.agentCommand = a.lifecycle.command(a.agentRunCommand(a.agentOperationID, request, agentReviewNone))
	return core.Ok(nil)
}

func (a *app) beginAgentOperation(request agentRequest, stage agentReviewStage) {
	if a == nil {
		return
	}
	a.agentOperationNext++
	a.agentOperationID = a.agentOperationNext
	a.agentRequest = request
	a.agentStage = stage
	a.agentInFlight = false
}

func (a *app) abortAgentOperation() {
	if a == nil || a.agentInFlight {
		return
	}
	a.resetAgentOperation()
}

// resetAgentOperation ends a completed or abandoned transaction without
// rewinding agentOperationNext; stale command results can never match a later
// operation ID.
func (a *app) resetAgentOperation() {
	if a == nil {
		return
	}
	a.agentOperationID = 0
	a.agentRequest = agentRequest{}
	a.agentReview = agentReview{}
	a.agentStage = agentReviewNone
	a.agentCommand = nil
	a.agentInFlight = false
}

func (a *app) agentReviewCommand(operationID uint64, request agentRequest, stage agentReviewStage) tea.Cmd {
	return func() tea.Msg {
		workID := request.WorkID
		if (request.Feature == agentFeatureChangesReview || request.Feature == agentFeatureRetry || request.Feature == agentFeatureResume) && request.RunID != "" {
			workID = request.RunID
		}
		result := a.agent.Review(a.lifecycle.context, agentReviewRequest{
			Feature: request.Feature, WorkID: workID, Provider: request.Provider,
			Model: request.Model, Input: request.Input, Work: request.Work, Recovery: request.Recovery,
		})
		return agentActionMsg{operationID: operationID, feature: request.Feature, stage: stage, request: request, result: result}
	}
}

func (a *app) agentRunCommand(operationID uint64, request agentRequest, stage agentReviewStage) tea.Cmd {
	return func() tea.Msg {
		return agentActionMsg{operationID: operationID, feature: request.Feature, stage: stage, request: request, result: a.agent.Run(a.lifecycle.context, request)}
	}
}

func (a *app) queueAgentRun(confirmed, enableGit bool) {
	request := a.agentRequest
	request.Review = a.agentReview
	request.Confirmed = confirmed
	request.EnableGit = enableGit
	a.agentRequest = request
	a.agentStage = agentReviewLaunch
	a.agentInFlight = true
	a.agentCommand = a.lifecycle.command(a.agentRunCommand(a.agentOperationID, request, agentReviewLaunch))
}

func (a *app) takeAgentCommand() tea.Cmd {
	if a == nil {
		return nil
	}
	command := a.agentCommand
	a.agentCommand = nil
	return command
}

func (a *app) requestAgentSnapshot() tea.Cmd {
	if a == nil || a.lifecycle == nil {
		return nil
	}
	if a.agentSnapshotInFlight {
		a.agentSnapshotPending = true
		return nil
	}
	a.agentSnapshotNext++
	a.agentSnapshotCurrent = a.agentSnapshotNext
	a.agentSnapshotInFlight = true
	return a.lifecycle.command(a.agentSnapshotCommand(a.agentSnapshotCurrent))
}

func (a *app) agentSnapshotCommand(requestID uint64) tea.Cmd {
	return func() tea.Msg {
		if a == nil || a.agent == nil {
			return agentSnapshotMsg{requestID: requestID, result: core.Fail(core.E("tui.app.agentSnapshot", "agent provider is unavailable", nil))}
		}
		return agentSnapshotMsg{requestID: requestID, result: a.agent.Snapshot(a.lifecycle.context)}
	}
}

func (a *app) hasLiveAgentWork() bool {
	if a == nil || a.work == nil {
		return false
	}
	for _, record := range a.work.Items() {
		state := a.work.AgentState(record)
		if state.NativeRunID == "" {
			continue
		}
		switch core.Lower(core.Trim(record.Status)) {
		case "queued", "preparing", "running", "cancelling":
			return true
		}
	}
	return false
}

func (a *app) armAgentRefresh() tea.Cmd {
	if a == nil || a.lifecycle == nil || a.lifecycle.isStopped() || a.agentRefreshArmed || !a.hasLiveAgentWork() {
		return nil
	}
	a.agentRefreshArmed = true
	return a.lifecycle.command(func() tea.Msg {
		timer := time.NewTimer(time.Second)
		defer timer.Stop()
		select {
		case <-timer.C:
			return agentRefreshMsg{}
		case <-a.lifecycle.context.Done():
			return lifecycleStoppedMsg{}
		}
	})
}

func (a *app) confirmAgentSelection() core.Result {
	if a == nil || a.launchReview == nil {
		return core.Fail(core.E("tui.app.confirmAgentSelection", "agent selection is unavailable", nil))
	}
	provider, model := a.launchReview.selection()
	if provider == "" || model == "" {
		return core.Fail(core.E("tui.app.confirmAgentSelection", "provider and model are required before project review", nil))
	}
	a.agentRequest.Provider, a.agentRequest.Model = provider, model
	a.agentInFlight = true
	a.agentCommand = a.lifecycle.command(a.agentReviewCommand(a.agentOperationID, a.agentRequest, agentReviewProject))
	return core.Ok(nil)
}

func (a app) applyAgentAction(message agentActionMsg) (tea.Model, tea.Cmd) {
	if message.operationID == 0 || message.operationID != a.agentOperationID || !a.agentInFlight {
		return a, nil
	}
	a.agentInFlight = false
	if !message.result.OK {
		a.errText = message.result.Error()
		a.activeOverlay, a.launchReview = overlayNone, nil
		a.resetAgentOperation()
		return a, nil
	}
	if review, ok := message.result.Value.(agentReview); ok {
		if message.feature == agentFeatureChangesReview {
			a.agentRequest, a.agentReview = message.request, review
			a.changeOverlay = newChangeAcceptanceOverlay(review)
			a.activeOverlay = overlayChangeReview
			command := a.requestAgentSnapshot()
			return a, command
		}
		a.agentRequest, a.agentReview, a.agentStage = message.request, review, message.stage
		switch message.stage {
		case agentReviewProject:
			a.activeOverlay = overlayProjectReview
		case agentReviewLaunch:
			a.launchReview = newLaunchReviewOverlay(review, a.agentRequest.Provider, a.agentRequest.Model)
			a.activeOverlay = overlayLaunchReview
		default:
			a.launchReview = newLaunchReviewOverlay(review, a.agentRequest.Provider, a.agentRequest.Model)
			a.activeOverlay = overlayLaunchReview
		}
		return a, nil
	}
	if receipt, ok := message.result.Value.(agentActionReceipt); ok && receipt.Feature == agentFeatureAnswer && a.work != nil {
		state := a.work.agentWork[message.request.WorkID]
		if core.Trim(message.request.RunID) != "" && state.NativeRunID == message.request.RunID {
			state.AnswerID, state.ResumeRunID = receipt.Detail, receipt.RunID
			a.work.agentWork[message.request.WorkID] = state
		}
	}
	if receipt, ok := message.result.Value.(agentActionReceipt); ok && receipt.Feature == agentFeatureDispatch {
		workID := core.Trim(message.request.WorkID)
		if receiptWorkID := core.Trim(receipt.WorkID); receiptWorkID != "" && receiptWorkID != workID {
			a.errText = core.Sprintf("dispatch receipt WorkID %q does not match requested local Work %q", receiptWorkID, workID)
		}
		if a.work != nil && workID != "" && core.Trim(receipt.Status) != "" {
			if result := a.work.updateWork("AgentReceipt", workID, func(record *workItemRecord) { record.Status = core.Lower(core.Trim(receipt.Status)) }); !result.OK {
				if a.errText == "" {
					a.errText = result.Error()
				} else {
					a.errText = core.Concat(a.errText, "; ", result.Error())
				}
			}
		}
		if a.work != nil && workID != "" && receipt.RunID != "" {
			state := a.work.agentWork[workID]
			state.NativeRunID, state.Status = receipt.RunID, receipt.Status
			a.work.agentWork[workID] = state
		}
	}
	a.activeOverlay, a.launchReview = overlayNone, nil
	a.resetAgentOperation()
	a.refreshAgentPalette()
	return a, a.requestAgentSnapshot()
}

func (a *app) attachKnowledge(repository workspaceRepository, maxBytes int64) core.Result {
	if a == nil {
		return core.Fail(core.E("tui.app.attachKnowledge", "application is unavailable", nil))
	}
	opened := newKnowledgeLibrary(repository, maxBytes, nil, nil)
	if !opened.OK {
		return opened
	}
	a.knowledge = opened.Value.(*knowledgeLibrary)
	return a.reloadKnowledgeAttachments()
}

func (a *app) applyKnowledgeDiscovery(result core.Result) core.Result {
	if a == nil {
		return core.Fail(core.E("tui.app.applyKnowledgeDiscovery", "application is unavailable", nil))
	}
	a.inspector.ApplyKnowledge(result)
	if !result.OK || a.knowledge == nil || core.Trim(a.sessionID) == "" {
		return result
	}
	discovery, ok := result.Value.(knowledgeDiscovery)
	if !ok {
		return core.Fail(core.E("tui.app.applyKnowledgeDiscovery", "invalid knowledge discovery result", nil))
	}
	if refreshed := a.knowledge.RefreshStaleness(a.sessionID, discovery.Documents); !refreshed.OK {
		return refreshed
	}
	return a.reloadKnowledgeAttachments()
}

func (a *app) attachKnowledgeDocument(index int) core.Result {
	if a == nil || a.knowledge == nil {
		return core.Fail(core.E("tui.app.attachKnowledgeDocument", "knowledge library is unavailable", nil))
	}
	if index < 0 || index >= len(a.inspector.knowledge.documents) {
		return core.Fail(core.E("tui.app.attachKnowledgeDocument", "knowledge document selection is out of range", nil))
	}
	result := a.knowledge.Attach(a.sessionID, a.inspector.knowledge.documents[index])
	if !result.OK {
		return result
	}
	return a.reloadKnowledgeAttachments()
}

func (a *app) detachKnowledgeAttachment(index int) core.Result {
	if a == nil || a.knowledge == nil {
		return core.Fail(core.E("tui.app.detachKnowledgeAttachment", "knowledge library is unavailable", nil))
	}
	if index < 0 || index >= len(a.attachments) {
		return core.Fail(core.E("tui.app.detachKnowledgeAttachment", "knowledge attachment selection is out of range", nil))
	}
	if result := a.knowledge.Detach(a.sessionID, a.attachments[index].ID); !result.OK {
		return result
	}
	return a.reloadKnowledgeAttachments()
}

func (a *app) reloadKnowledgeAttachments() core.Result {
	if a == nil || a.knowledge == nil || core.Trim(a.sessionID) == "" {
		return core.Ok(nil)
	}
	result := a.knowledge.Attachments(a.sessionID)
	if !result.OK {
		return result
	}
	attachments, ok := result.Value.([]attachmentRecord)
	if !ok {
		return core.Fail(core.E("tui.app.reloadKnowledgeAttachments", "invalid attachment result", nil))
	}
	a.attachments = append([]attachmentRecord(nil), attachments...)
	return core.Ok(a.attachments)
}

func (a *app) attachPreferences(preferences preferenceStore) {
	if a == nil {
		return
	}
	a.preferences = preferences
	if preferences == nil {
		return
	}
	values := preferences.Values()
	a.cfg = a.cfg.withPreferenceValues(values)
	selectedTheme := themeForName(values.Theme)
	a.inspector.theme = selectedTheme.name
	a.rebuildTheme(selectedTheme)
}

func (a *app) rebuildTheme(selected theme) {
	if a == nil {
		return
	}
	a.styles = newUIStyles(selected)
	a.markdown = newMarkdownRenderer(selected.name)
	if a.palette != nil {
		a.palette.list.Styles.Title = a.styles.title
	}
	if a.switcher != nil {
		a.switcher.list.Styles.Title = a.styles.title
	}
	if a.search != nil {
		a.search.list.Styles.Title = a.styles.title
	}
	a.help = newHelpOverlay(a.keys, a.styles)
	if a.ready {
		a.refreshTranscript()
	}
}

func (a app) Init() tea.Cmd {
	cmds := []tea.Cmd{a.spin.Tick}
	if a.boot.phase == bootLoading {
		cmds = append(cmds, a.lifecycle.command(workspaceBootstrap(a.workspaceLoader)))
		return tea.Batch(cmds...)
	}
	cmds = append(cmds, discoverModels)
	if a.loading != "" {
		cmds = append(cmds, a.lifecycle.command(a.modelLoader(a.loading, a.cfg.contextLen())))
	}
	return tea.Batch(cmds...)
}

func (a app) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case workspaceReadyMsg:
		if a.lifecycle.isStopped() {
			closeLifecycleMessage(msg)
			return a, nil
		}
		if result := a.connectWorkspace(msg.resources); !result.OK {
			if msg.resources != nil {
				_ = msg.resources.Close()
			}
			a.boot = bootState{phase: bootFailed, err: resultError(result)}
			if a.boot.err == nil {
				a.boot.err = core.E("tui.app.connectWorkspace", result.Error(), nil)
			}
			return a, nil
		}
		a.boot = bootState{phase: bootReady}
		return a, a.workspaceReadyCommands()

	case workspaceFailedMsg:
		a.boot = bootState{phase: bootFailed, err: msg.err}
		return a, nil

	case runtimeDetectedMsg:
		a.inspector.ApplyRuntime(msg.result)
		return a, nil

	case knowledgeDiscoveredMsg:
		if result := a.applyKnowledgeDiscovery(msg.result); !result.OK {
			a.errText = result.Error()
		}
		return a, nil

	case agentActionMsg:
		if a.lifecycle.isStopped() {
			return a, nil
		}
		return a.applyAgentAction(msg)

	case agentRefreshMsg:
		a.agentRefreshArmed = false
		return a, a.requestAgentSnapshot()

	case agentSnapshotMsg:
		if msg.requestID == 0 || msg.requestID != a.agentSnapshotCurrent {
			return a, nil
		}
		a.agentSnapshotInFlight = false
		pending := a.agentSnapshotPending
		a.agentSnapshotPending = false
		if !msg.result.OK {
			a.errText = msg.result.Error()
			if pending {
				return a, a.requestAgentSnapshot()
			}
			return a, a.armAgentRefresh()
		}
		snapshot, ok := msg.result.Value.(agentSnapshot)
		if !ok {
			a.errText = "invalid agent snapshot"
			if pending {
				return a, a.requestAgentSnapshot()
			}
			return a, a.armAgentRefresh()
		}
		if a.work != nil {
			if result := a.work.ApplyAgentSnapshot(snapshot); !result.OK {
				a.errText = result.Error()
			}
		}
		a.refreshAgentPalette()
		if pending {
			return a, a.requestAgentSnapshot()
		}
		return a, a.armAgentRefresh()

	case tea.WindowSizeMsg:
		offset := a.view.YOffset
		a.width, a.height = msg.Width, msg.Height
		metrics := measureFrame(msg.Width, msg.Height, a.inspectorOpen)
		a.picker.SetSize(max(1, metrics.mainWidth), max(1, metrics.mainHeight))
		a.input.SetWidth(max(1, metrics.mainWidth-4))
		a.view = viewport.New(max(1, metrics.mainWidth), a.transcriptHeight())
		a.view.SetContent(a.renderTranscript())
		if a.follow {
			a.view.GotoBottom()
		} else {
			a.view.SetYOffset(offset)
		}
		a.ready = true
		return a, nil

	case discoveredMsg:
		return a, a.picker.SetItems(msg.items)

	case loadedMsg:
		if a.lifecycle.isStopped() {
			closeLifecycleMessage(msg)
			return a, nil
		}
		laneResult := newModelLane(msg.model, msg.name)
		if !laneResult.OK {
			if msg.model != nil {
				_ = msg.model.Close()
			}
			a.errText = laneResult.Error()
			a.loading = ""
			a.activePanel = panelModels
			return a, nil
		}
		if a.lane != nil {
			_ = a.lane.Close()
		}
		a.lane = laneResult.Value.(*modelLane)
		a.model, a.modelName = a.lane.Model(), msg.name
		a.loading = ""
		a.errText = ""
		a.activePanel = panelChat
		a.refreshTranscript()
		return a, nil

	case loadErrMsg:
		a.errText = msg.err.Error()
		a.loading = ""
		a.activePanel = panelModels
		return a, nil

	case lifecycleStoppedMsg:
		return a, nil

	case lifecycleResultMsg:
		message, ok := a.lifecycle.claim(msg.id)
		if !ok {
			return a, nil
		}
		return a.Update(message)

	case spinner.TickMsg:
		var cmd tea.Cmd
		a.spin, cmd = a.spin.Update(msg)
		return a, cmd

	case streamMsg:
		return a.onStream(msg)

	case streamRefreshMsg:
		if managed := a.sessionJobs[msg.SessionID]; managed != nil && managed.generation != nil && managed.generation.JobID == msg.JobID {
			a.flushManagedGeneration(managed)
			return a, waitEvent(managed.generation)
		}
		if a.gen == nil || msg.SessionID != a.gen.SessionID || msg.JobID != a.gen.JobID {
			return a, nil
		}
		a.refreshAt = time.Time{}
		a.refreshTranscriptOutput()
		return a, waitEvent(a.gen)

	case serviceMsg:
		a.svc.finish(msg.ev.err)
		if a.pendingModel != "" {
			path := a.pendingModel
			a.pendingModel = ""
			return a, a.beginModelLoad(path)
		}
		return a, nil

	case serviceTickMsg:
		if a.svc.running {
			return a, serviceTick() // re-arm: keeps the requests counter live
		}
		return a, nil

	case tea.KeyMsg:
		return a.onKey(msg)

	case tea.MouseMsg:
		return a.onMouse(msg)
	}
	return a.route(msg)
}

// onMouse gives left presses to the panel bar first: the bar's .ctml render
// exposes a box map, so a click resolves through teabox to the tab that
// painted at that cell (see tabs.go). The keyboard gates apply unchanged —
// during boot or under an overlay the bar takes no clicks — and every
// unclaimed mouse message keeps its existing route to the focused panel
// (wheel scrolling in the chat transcript).
func (a app) onMouse(msg tea.MouseMsg) (tea.Model, tea.Cmd) {
	if a.boot.phase != bootReady || a.activeOverlay != overlayNone {
		return a.route(msg)
	}
	if msg.Action != tea.MouseActionPress || msg.Button != tea.MouseButtonLeft {
		return a.route(msg)
	}
	metrics := measureFrame(a.width, a.height, a.inspectorOpen)
	_, boxes := renderPanelBarBoxes(a.activePanel, metrics.innerWidth, metrics.kind, a.styles)
	panel, ok := panelBarHit(boxes, msg.X-frameInsetCols, msg.Y-frameInsetRows)
	if !ok {
		return a.route(msg)
	}
	return a, a.selectPanel(panel)
}

// selectPanel activates target and returns the Models discovery command a
// first visit needs — the single switching path shared by the tab keys and
// panel-bar clicks, so both land with identical side effects.
func (a *app) selectPanel(target panelID) tea.Cmd {
	a.activePanel = target
	if target == panelModels && len(a.picker.Items()) == 0 {
		return discoverModels
	}
	return nil
}

func (a *app) connectWorkspace(resources *workspaceResources) core.Result {
	if a == nil || resources == nil || resources.Repository == nil || resources.State == nil || resources.Preferences == nil {
		return core.Fail(core.E("tui.app.connectWorkspace", "workspace resources are incomplete", nil))
	}
	// activateManagedSession below unconditionally focuses Chat (a fresh
	// or resumed session is always its default landing panel) — preserve
	// an explicitly requested starting panel (e.g. RunDataReview's Data
	// panel focus, tui.go) across that reset. Every other caller already
	// starts on panelChat (newApp/newWorkspaceApp's own default), so this
	// is a no-op for them.
	requestedPanel := a.activePanel
	managerResult := newSessionManager(resources.Repository, resources.State, nil, nil)
	if !managerResult.OK {
		return managerResult
	}
	manager := managerResult.Value.(*sessionManager)
	if manager.Active() == nil {
		created := manager.Create()
		if !created.OK {
			return created
		}
	}
	a.resources = resources
	a.repository = resources.Repository
	a.sessions = manager
	a.warnings = append([]string(nil), resources.Warnings...)
	a.attachPreferences(resources.Preferences)
	if result := a.attachWork(resources.Repository, resources.Agent); !result.OK {
		return result
	}
	// A dataset store problem never blocks the rest of the workspace (the
	// design's blast-radius rationale, bootstrap.go) — attachData already
	// degrades a nil resources.DatasetStore to a nil a.data; any other
	// failure here (e.g. the panel's own initial Refresh) is likewise a
	// warning, not a connectWorkspace failure.
	if result := a.attachData(resources.DatasetStore); !result.OK {
		a.warnings = append(a.warnings, core.Concat("data: ", result.Error()))
	}
	values := resources.Preferences.Values()
	a.knowledgeLimit = values.KnowledgeMaxBytes
	a.recentSessionLimit = values.RecentSessionLimit
	if result := a.attachKnowledge(resources.Repository, values.KnowledgeMaxBytes); !result.OK {
		return result
	}
	a.knowledgeMounts = []knowledgeMount{{Name: "local", Root: resources.Paths.Packs, Medium: resources.Files}}
	a.palette.SetExporter(
		newWorkspaceSessionExporter(resources.Repository, values.ShowThinking, nil, nil),
		resources.Files,
		resources.Paths.Exports,
	)
	a.activateManagedSession(manager.Active())
	if requestedPanel != panelChat {
		a.activePanel = requestedPanel
	}
	return core.Ok(nil)
}

func (a *app) workspaceReadyCommands() tea.Cmd {
	commands := []tea.Cmd{discoverModels}
	commands = append(commands, a.requestAgentSnapshot())
	if a.runtimeDetector != nil {
		detector := a.runtimeDetector
		commands = append(commands, func() tea.Msg { return runtimeDetectedMsg{result: detector.Detect()} })
	}
	if a.knowledgeScan != nil {
		scanner := a.knowledgeScan
		mounts := append([]knowledgeMount(nil), a.knowledgeMounts...)
		limit := a.knowledgeLimit
		commands = append(commands, func() tea.Msg {
			return knowledgeDiscoveredMsg{result: scanner.Discover(mounts, limit)}
		})
	}
	if a.loading != "" {
		commands = append(commands, a.lifecycle.command(a.modelLoader(a.loading, a.cfg.contextLen())))
	}
	return tea.Batch(commands...)
}

// onStream folds one generation event into the live assistant turn; on done it
// runs the tool loop when armed.
func (a app) onStream(ev streamMsg) (tea.Model, tea.Cmd) {
	if managed := a.sessionJobs[ev.SessionID]; managed != nil && managed.generation != nil && managed.generation.JobID == ev.JobID {
		return a.onManagedStream(ev, managed)
	}
	return a.onEphemeralStream(ev)
}

func (a app) onEphemeralStream(ev streamMsg) (tea.Model, tea.Cmd) {
	if a.gen == nil || ev.SessionID != a.gen.SessionID || ev.JobID != a.gen.JobID {
		return a, nil
	}
	if len(a.turns) > 0 {
		last := &a.turns[len(a.turns)-1]
		if last.role == "assistant" {
			last.thought += ev.thought
			last.text += ev.visible
		}
	}
	if ev.metrics != nil {
		a.lastTokS = ev.metrics.DecodeTokensPerSec
	}
	if ev.err != nil {
		a.errText = ev.err.Error()
	}
	if !ev.done {
		if a.refreshAt.IsZero() {
			a.refreshAt = time.Now().Add(streamRefreshInterval)
		}
		return a, waitEventOrRefresh(a.gen, a.refreshAt)
	}
	a.refreshAt = time.Time{}
	a.generating = false
	a.gen = nil
	if cmd := a.runToolLoop(); cmd != nil {
		a.refreshTranscriptOutput()
		return a, cmd
	}
	a.toolHops = 0
	a.refreshTranscriptOutput()
	return a, nil
}

func (a app) onManagedStream(ev streamMsg, managed *sessionGeneration) (tea.Model, tea.Cmd) {
	if managed == nil || a.sessions == nil || a.repository == nil {
		return a, nil
	}
	now := time.Now().UTC()
	if !managed.started {
		managed.started = true
		managed.job.Status = "generating"
		managed.job.StartedAt = now
		a.recordManagedPersistence(managed, a.repository.SaveJob(managed.job))
		a.recordManagedPersistence(managed, a.sessions.MarkGenerating(ev.SessionID, ev.JobID))
	}
	managed.answer.Visible += ev.visible
	managed.answer.Thought += ev.thought
	managed.answer.UpdatedAt = now
	if ev.visible != "" || ev.thought != "" {
		managed.dirty = true
		if managed.flushAt.IsZero() {
			managed.flushAt = now.Add(streamRefreshInterval)
		}
	}
	if ev.metrics != nil {
		managed.job.MetricsJSON = core.JSONMarshalString(ev.metrics)
		if a.sessionID == ev.SessionID {
			a.lastTokS = ev.metrics.DecodeTokensPerSec
		}
	}
	if ev.err != nil {
		managed.job.Error = ev.err.Error()
	}
	if ev.done || ev.err != nil || (!managed.flushAt.IsZero() && !now.Before(managed.flushAt)) {
		a.flushManagedGeneration(managed)
	}
	if !ev.done {
		if managed.dirty {
			return a, waitEventOrRefresh(managed.generation, managed.flushAt)
		}
		return a, waitEvent(managed.generation)
	}
	managed.job.FinishedAt = now
	switch {
	case managed.persistErr != "":
		managed.job.Status = "failed"
		managed.job.Error = managed.persistErr
	case managed.cancelled:
		managed.job.Status = "cancelled"
	case ev.err != nil:
		managed.job.Status = "failed"
	default:
		managed.job.Status = "completed"
	}
	if result := a.repository.SaveJob(managed.job); !result.OK {
		a.recordManagedPersistence(managed, result)
		managed.job.Status = "failed"
		managed.job.Error = managed.persistErr
	}
	a.jobs.finish(ev.SessionID, ev.JobID)
	delete(a.sessionJobs, ev.SessionID)
	var continuation tea.Cmd
	if managed.job.Status == "failed" {
		if result := a.sessions.FailGeneration(ev.SessionID, ev.JobID); !result.OK {
			a.errText = result.Error()
		}
		if result := a.persistGenerationEvent(ev.SessionID, ev.JobID, "generation.failed", "failed", managed.job.Error); !result.OK {
			a.errText = result.Error()
		}
	} else if managed.job.Status == "completed" {
		continued := a.continueManagedToolLoop(ev.SessionID, managed)
		if !continued.OK {
			a.errText = continued.Error()
		} else {
			continuation, _ = continued.Value.(tea.Cmd)
		}
		if continuation == nil {
			if result := a.sessions.Complete(ev.SessionID); !result.OK {
				a.errText = result.Error()
			}
		}
	} else {
		if result := a.sessions.cancelGeneration(ev.SessionID, ev.JobID); !result.OK {
			a.errText = result.Error()
		}
		if result := a.persistGenerationEvent(ev.SessionID, ev.JobID, "generation.cancelled", "cancelled", managed.job.Error); !result.OK {
			a.errText = result.Error()
		}
	}
	active := a.sessions.Active()
	if active != nil && active.Record.ID == ev.SessionID {
		a.syncManagedSession(active, false)
		if continuation == nil {
			a.gen = nil
			a.generating = false
			a.toolHops = 0
		}
		a.refreshTranscriptOutput()
	}
	return a, continuation
}

func (a *app) flushManagedGeneration(managed *sessionGeneration) {
	if a == nil || managed == nil || a.sessions == nil {
		return
	}
	if managed.dirty {
		a.recordManagedPersistence(managed, a.sessions.AddTurn(managed.answer))
		managed.dirty = false
	}
	managed.flushAt = time.Time{}
	active := a.sessions.Active()
	if active != nil && active.Record.ID == managed.answer.SessionID {
		a.syncManagedSession(active, false)
		a.refreshTranscriptOutput()
	}
}

func (a *app) recordManagedPersistence(managed *sessionGeneration, result core.Result) {
	if result.OK || managed == nil {
		return
	}
	if managed.persistErr == "" {
		managed.persistErr = result.Error()
	}
	managed.job.Error = managed.persistErr
	a.errText = result.Error()
}

func (a *app) persistGenerationEvent(sessionID, jobID, kind, status, detail string) core.Result {
	if a == nil || a.repository == nil {
		return core.Ok(nil)
	}
	return a.repository.SaveEvent(eventRecord{
		ID: newRecordID(), SessionID: sessionID, JobID: jobID, Kind: kind, Status: status,
		Title: core.Replace(kind, ".", " "), Detail: detail, PayloadJSON: "{}", CreatedAt: time.Now().UTC(),
	})
}

func (a *app) continueManagedToolLoop(sessionID string, managed *sessionGeneration) core.Result {
	if a == nil || managed == nil || !a.tools.enabled {
		return core.Ok(tea.Cmd(nil))
	}
	session := a.sessions.sessions[sessionID]
	if session == nil || session.ToolHops >= 2 {
		return core.Ok(tea.Cmd(nil))
	}
	calls, visible := parser.ParseGemmaToolCalls(managed.answer.Visible)
	if len(calls) == 0 {
		if !core.Contains(managed.answer.Visible, parser.ToolCallOpenMarker) {
			return core.Ok(tea.Cmd(nil))
		}
		managed.answer.Visible = core.Trim(visible)
		managed.answer.UpdatedAt = time.Now().UTC()
		if result := a.sessions.AddTurn(managed.answer); !result.OK {
			return result
		}
		failure := "error: malformed tool call"
		toolTurn := turn{id: newRecordID(), role: "tool", text: parser.RenderGemmaToolResponse(failure)}
		if result := a.persistSessionToolInteraction(sessionID, toolTurn, nil, failure, "tool.parse", "failed"); !result.OK {
			return result
		}
		return core.Ok(tea.Cmd(nil))
	}

	managed.answer.Visible = core.Trim(visible)
	managed.answer.ToolCallJSON = core.JSONMarshalString(calls)
	managed.answer.UpdatedAt = time.Now().UTC()
	if result := a.sessions.AddTurn(managed.answer); !result.OK {
		return result
	}
	promptTurnID := managed.answer.ID
	for _, call := range calls {
		output := a.tools.execute(call)
		status := "completed"
		if core.HasPrefix(output, "error:") {
			status = "failed"
		}
		toolTurn := turn{id: newRecordID(), role: "tool", text: parser.RenderGemmaToolResponse(output)}
		if result := a.persistSessionToolInteraction(sessionID, toolTurn, &call, output, "tool.call", status); !result.OK {
			return result
		}
		promptTurnID = toolTurn.id
	}
	session.ToolHops++
	return a.startManagedContinuation(session, promptTurnID)
}

func (a *app) startManagedContinuation(session *chatSession, promptTurnID string) core.Result {
	if a == nil || session == nil {
		return core.Fail(core.E("tui.app.startManagedContinuation", "session is unavailable", nil))
	}
	now := time.Now().UTC()
	answer := turnRecord{
		ID: newRecordID(), SessionID: session.Record.ID, Sequence: int64(len(session.Turns) + 1), Role: "assistant",
		ToolCallJSON: "{}", ToolResultJSON: "{}", Model: a.modelName, CreatedAt: now, UpdatedAt: now,
	}
	if result := a.sessions.AddTurn(answer); !result.OK {
		return result
	}
	job := generationJobRecord{
		ID: newRecordID(), SessionID: session.Record.ID, PromptTurnID: promptTurnID, AnswerTurnID: answer.ID,
		Status: "queued", Model: a.modelName, MetricsJSON: "{}", CreatedAt: now,
		StartedAt: unsetRecordTime(), FinishedAt: unsetRecordTime(),
	}
	if result := a.repository.SaveJob(job); !result.OK {
		return result
	}
	if result := a.sessions.BeginGeneration(session.Record.ID, job.ID); !result.OK {
		return result
	}
	started := a.jobs.Start(session.Record.ID, job.ID, a.chatModel(), a.historyForSession(session), a.generateOpts())
	if !started.OK {
		job.Status = "failed"
		job.Error = started.Error()
		job.FinishedAt = time.Now().UTC()
		_ = a.repository.SaveJob(job)
		_ = a.sessions.FailGeneration(session.Record.ID, job.ID)
		return started
	}
	generation := started.Value.(*generation)
	a.sessionJobs[session.Record.ID] = &sessionGeneration{generation: generation, answer: answer, job: job}
	if active := a.sessions.Active(); active != nil && active.Record.ID == session.Record.ID {
		a.syncManagedSession(active, false)
	}
	return core.Ok(waitEvent(generation))
}

// runToolLoop parses the finished assistant turn for tool calls; when the
// Tools tab armed them it executes each locally, appends the wrapped tool
// results, and auto-continues the conversation (bounded hops).
func (a *app) runToolLoop() tea.Cmd {
	if !a.tools.enabled || a.toolHops >= 2 || len(a.turns) == 0 {
		return nil
	}
	last := &a.turns[len(a.turns)-1]
	if last.role != "assistant" {
		return nil
	}
	calls, visible := parser.ParseGemmaToolCalls(last.text)
	if len(calls) == 0 {
		if !core.Contains(last.text, parser.ToolCallOpenMarker) {
			return nil
		}
		last.text = core.Trim(visible)
		failure := "error: malformed tool call"
		last.calls = append(last.calls, "malformed tool call → failed")
		toolTurn := turn{id: newRecordID(), role: "tool", text: parser.RenderGemmaToolResponse(failure)}
		a.turns = append(a.turns, toolTurn)
		if result := a.persistToolInteraction(toolTurn, nil, failure, "tool.parse", "failed"); !result.OK {
			a.errText = result.Error()
		}
		return nil
	}
	last.text = core.Trim(visible)
	for _, call := range calls {
		result := a.tools.execute(call)
		last.calls = append(last.calls, call.Name+" → "+result)
		toolTurn := turn{id: newRecordID(), role: "tool", text: parser.RenderGemmaToolResponse(result)}
		a.turns = append(a.turns, toolTurn)
		status := "completed"
		if core.HasPrefix(result, "error:") {
			status = "failed"
		}
		if persisted := a.persistToolInteraction(toolTurn, &call, result, "tool.call", status); !persisted.OK {
			a.errText = persisted.Error()
			return nil
		}
	}
	a.turns = append(a.turns, turn{id: newRecordID(), role: "assistant", model: a.modelName})
	a.toolHops++
	return a.beginGeneration()
}

func (a *app) persistToolInteraction(
	toolTurn turn,
	call *inference.ToolCall,
	result string,
	kind string,
	status string,
) core.Result {
	if a == nil {
		return core.Ok(nil)
	}
	return a.persistSessionToolInteraction(a.sessionID, toolTurn, call, result, kind, status)
}

func (a *app) persistSessionToolInteraction(
	sessionID string,
	toolTurn turn,
	call *inference.ToolCall,
	result string,
	kind string,
	status string,
) core.Result {
	if a == nil || a.repository == nil || core.Trim(sessionID) == "" {
		return core.Ok(nil)
	}
	now := time.Now().UTC()
	toolName := "parse"
	callJSON := "{}"
	if call != nil {
		toolName = call.Name
		callJSON = core.JSONMarshalString(call)
	}
	sequence := int64(len(a.turns))
	if a.sessions != nil {
		if session := a.sessions.sessions[sessionID]; session != nil {
			sequence = int64(len(session.Turns) + 1)
		}
	}
	record := turnRecord{
		ID:             toolTurn.id,
		SessionID:      sessionID,
		Sequence:       sequence,
		Role:           toolTurn.role,
		Visible:        toolTurn.text,
		ToolName:       toolName,
		ToolCallJSON:   callJSON,
		ToolResultJSON: core.JSONMarshalString(map[string]any{"result": result, "status": status}),
		Model:          a.modelName,
		CreatedAt:      now,
		UpdatedAt:      now,
	}
	var saved core.Result
	if a.sessions != nil {
		saved = a.sessions.AddTurn(record)
	} else {
		saved = a.repository.SaveTurn(record)
	}
	if !saved.OK {
		return saved
	}
	event := eventRecord{
		ID:          newRecordID(),
		SessionID:   sessionID,
		Kind:        kind,
		Status:      status,
		Title:       core.Concat("Tool ", toolName, " ", status),
		Detail:      result,
		PayloadJSON: callJSON,
		CreatedAt:   now,
	}
	return a.repository.SaveEvent(event)
}

func (a app) onKey(msg tea.KeyMsg) (tea.Model, tea.Cmd) {
	if a.boot.phase != bootReady {
		return a.onBootKey(msg)
	}
	if a.activeOverlay != overlayNone && msg.String() != "ctrl+c" {
		return a.onOverlayKey(msg)
	}
	if key.Matches(msg, a.keys.CommandPalette) {
		a.refreshAgentPalette()
		// Mirrors refreshAgentPalette's own precedent immediately above:
		// a selection/status change on the Data panel with no action yet
		// (e.g. j/k navigation alone) never itself calls
		// refreshDataPalette, so the palette must refresh availability
		// itself right before it opens, not rely on stale state from the
		// last write.
		a.refreshDataPalette()
		a.palette.Open()
		a.activeOverlay = overlayCommands
		return a, nil
	}
	if key.Matches(msg, a.keys.SwitchSession) {
		if result := a.openSessionSwitcher(); !result.OK {
			a.errText = result.Error()
		}
		return a, nil
	}
	if key.Matches(msg, a.keys.Search) {
		if result := a.openHistorySearch(); !result.OK {
			a.errText = result.Error()
		}
		return a, nil
	}
	if key.Matches(msg, a.keys.Help) {
		a.activeOverlay = overlayHelp
		return a, nil
	}
	if key.Matches(msg, a.keys.NewSession) {
		if result := a.createSession(); !result.OK {
			a.errText = result.Error()
		}
		return a, nil
	}
	if key.Matches(msg, a.keys.PreviousSession) {
		if a.sessions == nil {
			a.errText = "session workspace is not connected"
		} else if result := a.sessions.Previous(); !result.OK {
			a.errText = result.Error()
		} else {
			a.activateManagedSession(a.sessions.Active())
		}
		return a, nil
	}
	if key.Matches(msg, a.keys.NextSession) {
		if a.sessions == nil {
			a.errText = "session workspace is not connected"
		} else if result := a.sessions.Next(); !result.OK {
			a.errText = result.Error()
		} else {
			a.activateManagedSession(a.sessions.Active())
		}
		return a, nil
	}
	if key.Matches(msg, a.keys.Save) {
		if result := a.inspector.Save(&a); !result.OK {
			a.errText = result.Error()
		}
		return a, nil
	}
	if key.Matches(msg, a.keys.ToggleInspector) {
		a.toggleInspector()
		return a, nil
	}
	if a.inspectorOpen {
		if a.activePanel == panelWork {
			switch msg.String() {
			case "up", "k":
				if a.work != nil {
					a.work.MoveAction(-1)
				}
				return a, nil
			case "down", "j":
				if a.work != nil {
					a.work.MoveAction(1)
				}
				return a, nil
			case "enter":
				if a.work == nil {
					a.errText = defaultAgentUnavailableReason
				} else if result := a.queueAgentAction(a.work.SelectedAction().Feature); !result.OK {
					a.errText = result.Error()
				}
				return a, a.takeAgentCommand()
			case "left", "h", "right", "l":
				// Work inspector actions do not edit Chat generation settings.
				return a, nil
			}
		}
		switch msg.String() {
		case "up", "k":
			a.inspector.Move(-1)
			return a, nil
		case "down", "j":
			a.inspector.Move(1)
			return a, nil
		case "left", "h":
			if result := a.inspector.Adjust(&a, -1); !result.OK {
				a.errText = result.Error()
			}
			return a, nil
		case "right", "l":
			if result := a.inspector.Adjust(&a, 1); !result.OK {
				a.errText = result.Error()
			}
			return a, nil
		case "enter":
			if result := a.inspector.Adjust(&a, 0); !result.OK {
				a.errText = result.Error()
			}
			return a, nil
		}
	}
	switch msg.String() {
	case "ctrl+c":
		_ = a.shutdown()
		return a, tea.Quit
	case "tab", "shift+tab":
		if msg.String() == "tab" {
			return a, a.selectPanel(a.activePanel.next())
		}
		return a, a.selectPanel(a.activePanel.prev())
	case "esc":
		if a.generating && a.gen != nil {
			if managed := a.sessionJobs[a.gen.SessionID]; managed != nil {
				managed.cancelled = true
			}
			_ = a.jobs.Cancel(a.gen.SessionID) // stream drains to tagged done
			return a, nil
		}
	case "ctrl+t":
		// quick thinking toggle: flips between explicit on and off
		if a.cfg.thinkIdx == 2 {
			a.cfg.thinkIdx = 1
		} else {
			a.cfg.thinkIdx = 2
		}
		return a, nil
	case "end":
		if a.activePanel == panelChat {
			a.view.GotoBottom()
			a.follow = true
			a.newOutput = false
			return a, nil
		}
	case "home":
		if a.activePanel == panelChat {
			a.view.GotoTop()
			a.follow = false
			return a, nil
		}
	}

	switch a.activePanel {
	case panelModels:
		if msg.String() == "enter" && a.loading == "" {
			if item, ok := a.picker.SelectedItem().(modelItem); ok {
				return a, a.requestModelLoad(item.path)
			}
			return a, nil
		}
	case panelService:
		switch msg.String() {
		case "enter":
			if a.svc.running {
				a.svc.stop()
				return a, nil
			}
			return a, a.svc.start(a.model)
		case "left", "h":
			if !a.svc.running {
				a.svc.addrIdx = (a.svc.addrIdx + len(serviceAddrs) - 1) % len(serviceAddrs)
			}
			return a, nil
		case "right", "l":
			if !a.svc.running {
				a.svc.addrIdx = (a.svc.addrIdx + 1) % len(serviceAddrs)
			}
			return a, nil
		}
	case panelData:
		// The keyboard idiom: j/k select (bubbles list default, reached via
		// route() below for anything not matched here); lowercase acts on
		// the selected item; uppercase runs the same action in bulk across
		// the current filter. Every key below has a mirrored palette entry
		// (dataWorkspaceCommandsForContext, palette.go) — this switch is a
		// keyboard shortcut for the same dataPanel/app methods the palette
		// invokes, never a second code path.
		if a.data != nil {
			switch msg.String() {
			case "a":
				if result := a.applyDataAction(dataActionApprove); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
			case "r":
				if result := a.applyDataAction(dataActionReject); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
			case "c":
				if result := a.openDataNote(dataActionQuarantineClear, dataSelectedItemID(a.data)); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
			case "e":
				if result := a.openDataEditor(); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
			case "t":
				if result := a.openDataNote(dataActionTag, dataSelectedItemID(a.data)); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
			case "A":
				if result := a.openDataBulk(dataActionApprove); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
			case "R":
				if result := a.openDataBulk(dataActionReject); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
			case "C":
				if result := a.openDataBulk(dataActionQuarantineClear); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
			case "T":
				if result := a.openDataBulk(dataActionTag); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
			case "s":
				if result := a.data.ToggleSort(); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
			case "f":
				if result := a.openDataFilter(); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
			}
		}
	case panelChat:
		if msg.Type == tea.KeyEnter && msg.Alt && !a.generating {
			// Textarea treats Enter as a newline; the app reserves plain Enter
			// for send, so explicitly forward the modified form to the editor.
			return a.route(tea.KeyMsg{Type: tea.KeyEnter})
		}
		if msg.String() == "enter" && !a.generating && a.model != nil {
			prompt := core.Trim(a.input.Value())
			if prompt == "" {
				return a, nil
			}
			result := a.sendPrompt(prompt)
			if !result.OK {
				a.errText = result.Error()
				return a, nil
			}
			command, _ := result.Value.(tea.Cmd)
			return a, command
		}
	}
	return a.route(msg)
}

func (a *app) toggleInspector() {
	if a == nil {
		return
	}
	a.inspectorOpen = !a.inspectorOpen
	if !a.ready {
		return
	}
	metrics := measureFrame(a.width, a.height, a.inspectorOpen)
	a.picker.SetSize(max(1, metrics.mainWidth), max(1, metrics.mainHeight))
	a.input.SetWidth(max(1, metrics.mainWidth-4))
	a.view.Width = max(1, metrics.mainWidth)
	a.view.Height = a.transcriptHeight()
	a.refreshTranscript()
}

func (a *app) requestModelLoad(path string) tea.Cmd {
	if a == nil {
		return nil
	}
	path = core.Trim(path)
	if path == "" {
		a.errText = "model path is empty"
		return nil
	}
	if a.jobs != nil && a.jobs.ActiveCount() > 0 {
		a.errText = "model change refused while session jobs are active"
		return nil
	}
	if a.svc.running {
		a.pendingModel = path
		a.svc.note = "draining service before model change"
		a.svc.stop()
		return nil
	}
	return a.beginModelLoad(path)
}

func (a *app) beginModelLoad(path string) tea.Cmd {
	if a == nil {
		return nil
	}
	if a.lane != nil {
		if result := a.lane.Close(); !result.OK {
			a.errText = result.Error()
			return nil
		}
	}
	a.lane = nil
	a.model = nil
	a.modelName = ""
	a.loading = path
	a.activePanel = panelModels
	loader := a.modelLoader
	if loader == nil {
		loader = loadModel
	}
	return a.lifecycle.command(loader(path, a.cfg.contextLen()))
}

func (a *app) sendPrompt(prompt string) core.Result {
	if a == nil || a.model == nil {
		return core.Fail(core.E("tui.app.sendPrompt", "a loaded model is required", nil))
	}
	prompt = core.Trim(prompt)
	if prompt == "" {
		return core.Fail(core.E("tui.app.sendPrompt", "prompt is required", nil))
	}
	if a.sessions == nil || a.repository == nil {
		a.input.Reset()
		a.turns = append(a.turns,
			turn{id: newRecordID(), role: "user", text: prompt},
			turn{id: newRecordID(), role: "assistant", model: a.modelName},
		)
		a.follow = true
		a.newOutput = false
		a.toolHops = 0
		a.errText = ""
		command := a.beginGeneration()
		a.refreshTranscript()
		return core.Ok(command)
	}
	session := a.sessions.Active()
	if session == nil || session.Record.ID != a.sessionID {
		return core.Fail(core.E("tui.app.sendPrompt", "active session is unavailable", nil))
	}
	if a.sessionJobs[session.Record.ID] != nil || session.ActiveJobID != "" {
		return core.Fail(core.E("tui.app.sendPrompt", "session already has a running generation", nil))
	}
	now := time.Now().UTC()
	sequence := int64(len(session.Turns) + 1)
	userRecord := turnRecord{
		ID: newRecordID(), SessionID: session.Record.ID, Sequence: sequence, Role: "user", Visible: prompt,
		ToolCallJSON: "{}", ToolResultJSON: "{}", CreatedAt: now, UpdatedAt: now,
	}
	answerRecord := turnRecord{
		ID: newRecordID(), SessionID: session.Record.ID, Sequence: sequence + 1, Role: "assistant",
		ToolCallJSON: "{}", ToolResultJSON: "{}", Model: a.modelName, CreatedAt: now, UpdatedAt: now,
	}
	if result := a.sessions.AddTurn(userRecord); !result.OK {
		return result
	}
	if result := a.sessions.AddTurn(answerRecord); !result.OK {
		return result
	}
	a.syncManagedSession(session, false)
	job := generationJobRecord{
		ID: newRecordID(), SessionID: session.Record.ID, PromptTurnID: userRecord.ID, AnswerTurnID: answerRecord.ID,
		Status: "queued", Model: a.modelName, MetricsJSON: "{}", CreatedAt: now,
		StartedAt: unsetRecordTime(), FinishedAt: unsetRecordTime(),
	}
	if result := a.repository.SaveJob(job); !result.OK {
		return result
	}
	if result := a.sessions.BeginGeneration(session.Record.ID, job.ID); !result.OK {
		return result
	}
	started := a.jobs.Start(session.Record.ID, job.ID, a.chatModel(), a.history(), a.generateOpts())
	if !started.OK {
		job.Status = "failed"
		job.Error = started.Error()
		job.FinishedAt = time.Now().UTC()
		_ = a.repository.SaveJob(job)
		_ = a.sessions.FailGeneration(session.Record.ID, job.ID)
		return started
	}
	generation := started.Value.(*generation)
	a.sessionJobs[session.Record.ID] = &sessionGeneration{
		generation: generation,
		answer:     answerRecord,
		job:        job,
	}
	session.ActiveJobID = job.ID
	a.gen = generation
	a.generating = true
	a.input.Reset()
	_ = a.sessions.SetDraft(session.Record.ID, "")
	a.follow = true
	a.newOutput = false
	a.toolHops = 0
	a.errText = ""
	a.refreshTranscript()
	return core.Ok(waitEvent(generation))
}

func (a app) onBootKey(message tea.KeyMsg) (tea.Model, tea.Cmd) {
	switch message.String() {
	case "ctrl+c", "q":
		_ = a.shutdown()
		return a, tea.Quit
	case "r":
		if a.boot.phase == bootFailed {
			a.boot = bootState{phase: bootLoading}
			return a, a.lifecycle.command(workspaceBootstrap(a.workspaceLoader))
		}
	}
	return a, nil
}

func (a *app) shutdown() core.Result {
	if a == nil {
		return core.Ok(nil)
	}
	if a.lifecycle == nil {
		a.lifecycle = newAppLifecycle(context.Background(), nil)
	}
	a.lifecycle.shutdownMu.Lock()
	defer a.lifecycle.shutdownMu.Unlock()
	if a.lifecycle.shutdownComplete {
		return a.lifecycle.result
	}
	failures := make([]string, 0)
	record := func(candidate core.Result) {
		if !candidate.OK {
			failures = append(failures, candidate.Error())
		}
	}
	a.lifecycle.stop()
	if a.jobs != nil {
		record(a.jobs.CancelAll())
	}
	record(a.drainManagedGenerations())
	agentClosed := true
	if a.resources != nil {
		closeResult := a.resources.closeAgent()
		record(closeResult)
		agentClosed = closeResult.OK
	} else if a.agent != nil {
		closeResult := a.agent.Close()
		record(closeResult)
		agentClosed = closeResult.OK
		if closeResult.OK {
			a.agent = nil
		}
	}
	a.svc.teardown("stopped (quit)")
	if a.lane != nil {
		record(a.lane.Close())
		a.lane = nil
		a.model = nil
	}
	a.lifecycle.wait()
	a.lifecycle.closePending()
	if a.resources != nil && agentClosed {
		record(a.resources.Close())
	}
	if len(failures) > 0 {
		a.lifecycle.result = core.Fail(core.E("tui.app.shutdown", core.Join("; ", failures...), nil))
		return a.lifecycle.result
	}
	a.lifecycle.result = core.Ok(nil)
	a.lifecycle.shutdownComplete = true
	return a.lifecycle.result
}

// drainManagedGenerations folds every buffered delta and a terminal
// cancellation into durable state before the workspace database closes.
func (a *app) drainManagedGenerations() core.Result {
	if a == nil || len(a.sessionJobs) == 0 {
		return core.Ok(nil)
	}
	managedJobs := make([]*sessionGeneration, 0, len(a.sessionJobs))
	firstFailure := ""
	for _, managed := range a.sessionJobs {
		if managed != nil && managed.generation != nil {
			managed.cancelled = true
			managedJobs = append(managedJobs, managed)
		}
	}
	for _, managed := range managedJobs {
		generation := managed.generation
		terminal := false
		for event := range generation.events {
			model, _ := a.onManagedStream(streamMsg(event), managed)
			*a = model.(app)
			if event.done {
				terminal = true
			}
		}
		if !terminal {
			model, _ := a.onManagedStream(streamMsg{
				SessionID: generation.SessionID,
				JobID:     generation.JobID,
				done:      true,
			}, managed)
			*a = model.(app)
		}
		if managed.persistErr != "" && firstFailure == "" {
			firstFailure = managed.persistErr
		}
	}
	if firstFailure != "" {
		return core.Fail(core.E("tui.app.drainManagedGenerations", "persist terminal generation state", core.NewError(firstFailure)))
	}
	return core.Ok(nil)
}

func (a app) onOverlayKey(message tea.KeyMsg) (tea.Model, tea.Cmd) {
	if message.String() == "esc" {
		overlay := a.activeOverlay
		if a.launchReview != nil {
			a.launchReview.Update(message)
		}
		a.activeOverlay = overlayNone
		a.workEditor, a.launchReview, a.answerOverlay, a.changeOverlay = nil, nil, nil, nil
		a.dataEditor, a.dataNote, a.dataBulk, a.dataFilter = nil, nil, nil, nil
		switch overlay {
		case overlayAgentSelection, overlayProjectReview, overlayGitEnableReview, overlayLaunchReview, overlayAgentAnswer, overlayChangeReview:
			a.abortAgentOperation()
		}
		return a, nil
	}
	if a.activeOverlay == overlayWorkEditor && message.String() == "enter" && a.workEditor != nil && a.workEditor.focus == 1 {
		return a, a.workEditor.Update(message)
	}
	if a.activeOverlay == overlayWorkEditor && message.String() == "ctrl+s" {
		if result := a.saveWorkEditor(); !result.OK {
			a.errText = result.Error()
			return a, nil
		}
		a.activeOverlay, a.workEditor = overlayNone, nil
		return a, nil
	}
	if a.activeOverlay == overlayDataEditor && message.String() == "enter" {
		// Both fields are multi-line textareas — Enter always inserts a
		// newline in the focused one; only ctrl+s saves (unlike
		// overlayWorkEditor, which mixes single-line and multi-line
		// fields and so only forwards Enter for its one textarea field).
		return a, a.dataEditor.Update(message)
	}
	if a.activeOverlay == overlayDataEditor && message.String() == "ctrl+s" {
		if result := a.saveDataEditor(); !result.OK {
			a.errText = result.Error()
			return a, nil
		}
		a.activeOverlay, a.dataEditor = overlayNone, nil
		return a, nil
	}
	if a.activeOverlay == overlayGitEnableReview && message.String() != "enter" {
		if a.launchReview != nil {
			a.launchReview.Update(message)
		}
		return a, nil
	}
	if a.activeOverlay == overlayAgentSelection && message.String() != "enter" {
		if a.launchReview != nil {
			a.launchReview.Update(message)
		}
		return a, nil
	}
	if message.String() == "enter" {
		switch a.activeOverlay {
		case overlayCommands:
			id := a.palette.SelectedID()
			a.activeOverlay = overlayNone
			if result := a.palette.Invoke(id, &a); !result.OK {
				a.errText = result.Error()
			}
			return a, a.takeAgentCommand()
		case overlayWorkEditor:
			if result := a.saveWorkEditor(); !result.OK {
				a.errText = result.Error()
				return a, nil
			}
			a.activeOverlay, a.workEditor = overlayNone, nil
		case overlayAgentAnswer:
			if a.answerOverlay == nil || !a.answerOverlay.Update(message) {
				a.errText = "an answer is required"
				return a, nil
			}
			a.agentRequest.Input = a.answerOverlay.answer()
			a.agentInFlight = true
			a.agentCommand = a.lifecycle.command(a.agentRunCommand(a.agentOperationID, a.agentRequest, agentReviewNone))
			a.activeOverlay, a.answerOverlay = overlayNone, nil
			return a, a.takeAgentCommand()
		case overlayChangeReview:
			if a.changeOverlay == nil {
				a.errText = "change review is unavailable"
				return a, nil
			}
			if !a.changeOverlay.review.AcceptanceAllowed {
				a.errText = "conflicts or failed validation prevent acceptance; reject this reviewed result or retry the run"
				return a, nil
			}
			if a.changeOverlay.review.NeedsAcknowledgement && !a.changeOverlay.acknowledged {
				a.errText = "acknowledge the missing validation before acceptance"
				return a, nil
			}
			if !a.changeOverlay.final {
				a.changeOverlay.final = true
				return a, nil
			}
			request := a.agentRequest
			request.Feature, request.Review, request.Confirmed = agentFeatureAccept, a.changeOverlay.review, true
			a.agentRequest, a.agentInFlight = request, true
			a.agentCommand = a.lifecycle.command(a.agentRunCommand(a.agentOperationID, request, agentReviewLaunch))
			a.activeOverlay, a.changeOverlay = overlayNone, nil
			return a, a.takeAgentCommand()
		case overlayAgentSelection:
			if result := a.confirmAgentSelection(); !result.OK {
				a.errText = result.Error()
				return a, nil
			}
			a.activeOverlay, a.launchReview = overlayNone, nil
			return a, a.takeAgentCommand()
		case overlayProjectReview:
			if a.agentReview.GitConfirmRequired {
				a.launchReview = newLaunchReviewOverlay(agentReview{
					Feature: agentFeatureDispatch, Title: "Enable Git for this directory", Body: a.agentReview.Body,
					Warning: "This creates Git metadata and an initial local commit only after confirmation.", ConfirmRequired: true,
				}, a.agentRequest.Provider, a.agentRequest.Model)
				a.activeOverlay = overlayGitEnableReview
				return a, nil
			}
			a.queueAgentRun(true, false)
			a.activeOverlay = overlayNone
			return a, a.takeAgentCommand()
		case overlayGitEnableReview:
			if a.launchReview != nil {
				a.launchReview.Update(message)
			}
			a.queueAgentRun(true, true)
			a.activeOverlay, a.launchReview = overlayNone, nil
			return a, a.takeAgentCommand()
		case overlayLaunchReview:
			if a.launchReview != nil {
				a.launchReview.Update(message)
			}
			a.queueAgentRun(true, false)
			a.activeOverlay, a.launchReview = overlayNone, nil
			return a, a.takeAgentCommand()
		case overlaySessions:
			if a.switcher == nil {
				a.errText = "session switcher is unavailable"
			} else if result := a.switcher.ActivateSelected(); !result.OK {
				a.errText = result.Error()
			} else {
				a.activateManagedSession(a.sessions.Active())
				a.activeOverlay = overlayNone
			}
		case overlaySearch:
			if a.search == nil {
				a.errText = "history search is unavailable"
			} else if result := a.search.ActivateSelected(); !result.OK {
				a.errText = result.Error()
			} else {
				a.activateManagedSession(a.sessions.Active())
				offset := a.transcriptTurnOffset(a.search.MatchTurnID())
				a.view.SetYOffset(offset)
				a.follow = false
				if result := a.sessions.SetViewport(a.sessionID, a.view.YOffset, false); !result.OK {
					a.errText = result.Error()
				}
				a.activeOverlay = overlayNone
			}
		case overlayDataNote:
			if a.dataNote == nil || !a.dataNote.Update(message) {
				a.errText = "a value is required"
				return a, nil
			}
			if result := a.submitDataNote(); !result.OK {
				a.errText = result.Error()
			}
		case overlayDataFilter:
			if a.dataFilter == nil || !a.dataFilter.Update(message) {
				return a, nil
			}
			if result := a.applyDataFilter(); !result.OK {
				a.errText = result.Error()
			}
		case overlayDataBulk:
			if a.dataBulk == nil || !a.dataBulk.Confirm(message.String()) {
				// The first Enter only arms the overlay (see
				// dataBulkOverlay.Confirm) — nothing to apply yet, and the
				// re-rendered View() shows the "confirm again" prompt.
				return a, nil
			}
			if result := a.confirmDataBulk(); !result.OK {
				a.errText = result.Error()
			}
		}
		return a, nil
	}

	var command tea.Cmd
	switch a.activeOverlay {
	case overlayCommands:
		command = a.palette.Update(message)
	case overlaySessions:
		if a.switcher != nil {
			command = a.switcher.Update(message)
		}
	case overlaySearch:
		if a.search != nil {
			command = a.search.Update(message)
		}
	case overlayHelp:
		// Help is read-only; every key except Escape is intentionally consumed.
	case overlayWorkEditor:
		if a.workEditor != nil {
			command = a.workEditor.Update(message)
		}
	case overlayDataEditor:
		if a.dataEditor != nil {
			command = a.dataEditor.Update(message)
		}
	case overlayDataNote:
		if a.dataNote != nil {
			a.dataNote.Update(message)
		}
	case overlayDataFilter:
		if a.dataFilter != nil {
			a.dataFilter.Update(message)
		}
	case overlayDataBulk:
		// Deliberately consumes keys until an explicit confirm or cancel —
		// Confirm only ever fires from the "enter" branch above.
	case overlayAgentAnswer:
		if a.answerOverlay != nil {
			a.answerOverlay.Update(message)
		}
	case overlayChangeReview:
		if a.changeOverlay != nil {
			if message.String() == "a" {
				a.changeOverlay.acknowledged = true
			} else {
				var ignored tea.Cmd
				a.changeOverlay.viewport, ignored = a.changeOverlay.viewport.Update(message)
				_ = ignored
			}
		}
	case overlayProjectReview, overlayGitEnableReview, overlayLaunchReview, overlayAgentSelection:
		// Reviews deliberately consume keys until an explicit confirm or cancel.
	}
	return a, command
}

func (a *app) createSession() core.Result {
	if a.sessions != nil {
		result := a.sessions.Create()
		if !result.OK {
			return result
		}
		a.activateManagedSession(result.Value.(*chatSession))
		return core.Ok(result.Value)
	}
	if a.generating {
		return core.Fail(core.E("tui.app.createSession", "connect the persistent workspace before creating a session during generation", nil))
	}
	a.sessionID = newRecordID()
	a.turns = nil
	a.attachments = nil
	a.input.Reset()
	a.follow = true
	a.newOutput = false
	a.activePanel = panelChat
	a.refreshTranscript()
	return core.Ok(a.sessionID)
}

func (a *app) openSessionSwitcher() core.Result {
	if a.sessions == nil {
		return core.Fail(core.E("tui.app.openSessionSwitcher", "session workspace is not connected", nil))
	}
	metrics := measureFrame(a.width, a.height, a.inspectorOpen)
	result := newSessionSwitcher(a.sessions, a.styles, metrics.mainWidth, metrics.mainHeight)
	if !result.OK {
		return result
	}
	a.switcher = result.Value.(*sessionSwitcher)
	a.activeOverlay = overlaySessions
	return core.Ok(a.switcher)
}

func (a *app) openHistorySearch() core.Result {
	if a.sessions == nil || a.repository == nil {
		return core.Fail(core.E("tui.app.openHistorySearch", "history workspace is not connected", nil))
	}
	metrics := measureFrame(a.width, a.height, a.inspectorOpen)
	result := newHistorySearch(a.repository, a.sessions, a.styles, metrics.mainWidth, metrics.mainHeight)
	if !result.OK {
		return result
	}
	a.search = result.Value.(*historySearch)
	a.activeOverlay = overlaySearch
	return core.Ok(a.search)
}

func (a *app) activateManagedSession(session *chatSession) {
	if a == nil || session == nil {
		return
	}
	a.syncManagedSession(session, true)
	a.attachments = nil
	if result := a.reloadKnowledgeAttachments(); !result.OK {
		a.errText = result.Error()
	}
	a.refreshTranscript()
	if !a.follow {
		a.view.SetYOffset(session.ViewportOffset)
	}
}

func (a *app) syncManagedSession(session *chatSession, activatePanel bool) {
	if a == nil || session == nil {
		return
	}
	a.sessionID = session.Record.ID
	a.turns = make([]turn, 0, len(session.Turns))
	for _, record := range session.Turns {
		a.turns = append(a.turns, turn{
			id:      record.ID,
			role:    record.Role,
			model:   record.Model,
			thought: record.Thought,
			text:    record.Visible,
			calls:   persistedTurnCalls(record),
		})
	}
	a.input.SetValue(session.Draft)
	a.follow = session.Follow
	a.newOutput = session.Attention
	a.toolHops = session.ToolHops
	if managed := a.sessionJobs[session.Record.ID]; managed != nil {
		a.gen = managed.generation
		a.generating = true
	} else {
		a.gen = nil
		a.generating = false
	}
	if activatePanel {
		a.activePanel = panelChat
	}
}

func persistedTurnCalls(record turnRecord) []string {
	if record.Role == "tool" && record.ToolName != "" {
		return []string{record.ToolName}
	}
	if record.Role != "assistant" || record.ToolCallJSON == "" || record.ToolCallJSON == "{}" {
		return nil
	}
	var calls []inference.ToolCall
	if result := core.JSONUnmarshalString(record.ToolCallJSON, &calls); !result.OK {
		return nil
	}
	receipts := make([]string, 0, len(calls))
	for _, call := range calls {
		receipts = append(receipts, core.Concat(call.Name, " → requested"))
	}
	return receipts
}

// route hands the message to the focused component for the active panel.
func (a app) route(msg tea.Msg) (tea.Model, tea.Cmd) {
	var cmd tea.Cmd
	switch a.activePanel {
	case panelModels:
		a.picker, cmd = a.picker.Update(msg)
	case panelWork:
		if a.work != nil {
			cmd = a.work.Update(msg)
		}
	case panelData:
		if a.data != nil {
			cmd = a.data.Update(msg)
		}
	case panelChat:
		_, mouse := msg.(tea.MouseMsg)
		if a.generating || mouse || isTranscriptPageKey(msg) {
			cmd = a.updateTranscriptViewport(msg)
		} else {
			before := a.input.Value()
			a.input, cmd = a.input.Update(msg)
			if a.sessions != nil && a.input.Value() != before {
				if result := a.sessions.SetDraft(a.sessionID, a.input.Value()); !result.OK {
					a.errText = result.Error()
				}
			}
		}
	}
	return a, cmd
}

func (a *app) updateTranscriptViewport(msg tea.Msg) tea.Cmd {
	before := a.view.YOffset
	var cmd tea.Cmd
	a.view, cmd = a.view.Update(msg)
	if a.view.YOffset < before || scrollsTranscriptUp(msg) {
		a.follow = false
	}
	if a.view.AtBottom() && scrollsTranscriptDown(msg) {
		a.follow = true
		a.newOutput = false
	}
	if a.sessions != nil {
		if result := a.sessions.SetViewport(a.sessionID, a.view.YOffset, a.follow); !result.OK {
			a.errText = result.Error()
		}
	}
	return cmd
}

func isTranscriptPageKey(msg tea.Msg) bool {
	keyMsg, ok := msg.(tea.KeyMsg)
	if !ok {
		return false
	}
	switch keyMsg.String() {
	case "pgup", "pgdown", "ctrl+u", "ctrl+d":
		return true
	default:
		return false
	}
}

func scrollsTranscriptUp(msg tea.Msg) bool {
	switch value := msg.(type) {
	case tea.KeyMsg:
		switch value.String() {
		case "up", "k", "pgup", "ctrl+u":
			return true
		}
	case tea.MouseMsg:
		return value.Action == tea.MouseActionPress && value.Button == tea.MouseButtonWheelUp && !value.Shift
	}
	return false
}

func scrollsTranscriptDown(msg tea.Msg) bool {
	switch value := msg.(type) {
	case tea.KeyMsg:
		switch value.String() {
		case "down", "j", "pgdown", "ctrl+d":
			return true
		}
	case tea.MouseMsg:
		return value.Action == tea.MouseActionPress && value.Button == tea.MouseButtonWheelDown && !value.Shift
	}
	return false
}

// chatModel is always the application's shared serial lane. Starting or
// stopping the HTTP service cannot change generation ownership or ordering.
func (a app) chatModel() inference.TextModel {
	return a.model
}

// beginGeneration registers the active session turn with the job manager and
// returns Bubble Tea's first wait command.
func (a *app) beginGeneration() tea.Cmd {
	result := a.jobs.Start(a.sessionID, newRecordID(), a.chatModel(), a.history(), a.generateOpts())
	if !result.OK {
		a.errText = result.Error()
		a.generating = false
		a.gen = nil
		return nil
	}
	a.gen = result.Value.(*generation)
	a.generating = true
	return waitEvent(a.gen)
}

// generateOpts folds the Modes preset with the Settings overrides — an
// explicit Settings thinking choice wins over the preset's.
func (a app) generateOpts() []inference.GenerateOption {
	opts := append([]inference.GenerateOption{}, a.modes.current().opts()...)
	opts = append(opts, inference.WithMaxTokens(a.cfg.maxTokens()))
	if th := a.cfg.thinking(); th != nil {
		opts = append(opts, inference.WithEnableThinking(th))
	}
	return opts
}

// history rebuilds the inference messages from the transcript — the message
// slice IS the memory, exactly like a stateless API client. Tool declarations
// ride the system turn when the Tools tab armed them (the serve convention).
func (a app) history() []inference.Message {
	if a.sessions != nil {
		if session := a.sessions.sessions[a.sessionID]; session != nil {
			return a.historyForSession(session)
		}
	}
	msgs := make([]inference.Message, 0, len(a.turns)+1)
	msgs = append(msgs, a.systemHistory(a.attachments)...)
	for _, t := range a.turns {
		if t.role == "assistant" && t.text == "" && t.thought == "" {
			continue // the live, still-empty turn
		}
		msgs = append(msgs, inference.Message{Role: t.role, Content: t.text})
	}
	return msgs
}

func (a app) historyForSession(session *chatSession) []inference.Message {
	if session == nil {
		return nil
	}
	attachments := a.attachments
	if session.Record.ID != a.sessionID && a.knowledge != nil {
		if result := a.knowledge.Attachments(session.Record.ID); result.OK {
			attachments, _ = result.Value.([]attachmentRecord)
		}
	}
	msgs := make([]inference.Message, 0, len(session.Turns)+1)
	msgs = append(msgs, a.systemHistory(attachments)...)
	for _, record := range session.Turns {
		content := record.Visible
		if record.Role == "assistant" && record.ToolCallJSON != "" && record.ToolCallJSON != "{}" {
			var calls []inference.ToolCall
			if result := core.JSONUnmarshalString(record.ToolCallJSON, &calls); result.OK {
				for _, call := range calls {
					content += parser.RenderGemmaToolCall(call.Name, call.ArgumentsJSON)
				}
			}
		}
		if record.Role == "assistant" && content == "" && record.Thought == "" {
			continue
		}
		msgs = append(msgs, inference.Message{Role: record.Role, Content: content})
	}
	return msgs
}

func (a app) systemHistory(attachments []attachmentRecord) []inference.Message {
	systemParts := make([]string, 0, 2)
	knowledgeLimit := knowledgeSystemMessageMaxBytes
	if a.knowledge != nil {
		knowledgeLimit = int(a.knowledge.maxBytes)
	}
	if knowledge := knowledgeSystemMessageBounded(attachments, knowledgeLimit); knowledge != "" {
		systemParts = append(systemParts, knowledge)
	}
	if decl := a.tools.declarations(); decl != "" {
		systemParts = append(systemParts, decl)
	}
	if len(systemParts) > 0 {
		return []inference.Message{{Role: "system", Content: core.Join("\n\n", systemParts...)}}
	}
	return nil
}

func (a *app) refreshTranscript() {
	a.updateTranscript(false)
}

func (a *app) refreshTranscriptOutput() {
	a.updateTranscript(true)
}

func (a *app) updateTranscript(hasOutput bool) {
	if !a.ready {
		return
	}
	offset := a.view.YOffset
	a.view.Height = a.transcriptHeight()
	a.view.SetContent(a.renderTranscript())
	if a.follow {
		a.view.GotoBottom()
		a.newOutput = false
		return
	}
	a.view.SetYOffset(offset)
	if hasOutput {
		a.newOutput = true
	}
}

func (a app) contentHeight() int {
	return max(1, measureFrame(a.width, a.height, a.inspectorOpen).mainHeight)
}

func (a app) transcriptHeight() int {
	h := a.contentHeight() - a.input.Height() - 2 // composer border
	if h < 3 {
		h = 3
	}
	return h
}

func (a app) renderTranscript() string {
	var b core.Builder
	notice := a.sessionTranscriptNotice()
	if notice != "" {
		b.WriteString(a.styles.attention.Render(notice))
	}
	for i, t := range a.turns {
		if i > 0 || notice != "" {
			b.WriteString("\n\n")
		}
		switch t.role {
		case "user":
			b.WriteString(a.styles.user.Render("you ") + a.styles.answer.Render(t.text))
		case "tool":
			label := "tool result"
			if len(t.calls) > 0 {
				label = core.Concat("tool · ", t.calls[0])
			}
			b.WriteString(a.styles.thought.Render(label))
			if result := visibleToolResult(t.text); result != "" {
				b.WriteString("\n" + a.styles.answer.Render(result))
			}
		default:
			if t.thought != "" {
				b.WriteString(a.styles.thought.Render("· thinking · "+core.Trim(t.thought)) + "\n")
			}
			label := t.model
			if label == "" {
				label = a.modelName
			}
			if label == "" {
				label = "assistant"
			}
			b.WriteString(a.styles.assistant.Render(label))
			streaming := a.generating && i == len(a.turns)-1
			if t.text != "" {
				b.WriteString("\n")
				if streaming {
					b.WriteString(a.styles.answer.Render(t.text))
				} else {
					turnID := t.id
					if turnID == "" {
						turnID = core.Sprintf("transcript-%d", i)
					}
					b.WriteString(a.markdown.Render(turnID, t.text, max(1, a.view.Width-2)))
				}
			}
			for _, c := range t.calls {
				b.WriteString("\n" + a.styles.thought.Render("→ "+c))
			}
			if t.text == "" && t.thought == "" && a.generating && i == len(a.turns)-1 {
				b.WriteString(a.styles.thought.Render(a.spin.View() + " …"))
			}
		}
	}
	return lipgloss.NewStyle().Width(a.view.Width).Render(b.String())
}

func (a app) transcriptTurnOffset(turnID string) int {
	turnID = core.Trim(turnID)
	if turnID == "" {
		return 0
	}
	index := -1
	for candidate, item := range a.turns {
		if item.id == turnID {
			index = candidate
			break
		}
	}
	if index < 0 {
		return 0
	}
	if index == 0 && a.sessionTranscriptNotice() == "" {
		return 0
	}
	prefix := a
	prefix.turns = a.turns[:index]
	return core.Count(prefix.renderTranscript(), "\n") + 2
}

func (a app) sessionTranscriptNotice() string {
	if a.sessions == nil || a.sessions.Active() == nil {
		return ""
	}
	switch a.sessions.Active().Record.Status {
	case "interrupted":
		return "! Generation interrupted · partial output preserved"
	case "cancelled":
		return "! Generation cancelled · partial output preserved"
	case "failed":
		return "! Generation failed · inspect the error before retrying"
	default:
		return ""
	}
}

func visibleToolResult(value string) string {
	value = core.TrimPrefix(value, parser.ToolResponseOpenMarker)
	value = core.TrimSuffix(value, parser.ToolResponseCloseMarker)
	return core.Trim(value)
}

func (a app) statusLine() string {
	parts := []string{}
	if len(a.warnings) > 0 {
		parts = append(parts, a.styles.attention.Render("warning: "+a.warnings[0]))
	}
	if a.errText != "" {
		parts = append(parts, a.styles.err.Render("error: "+a.errText))
	}
	if a.newOutput {
		parts = append(parts, a.styles.attention.Render("↓ new output"))
	}
	if a.modelName != "" {
		parts = append(parts, a.modelName)
	} else {
		parts = append(parts, "○ no model")
	}
	parts = append(parts, "mode "+a.modes.current().name, "thinking "+thinkNames[a.cfg.thinkIdx])
	if a.tools.enabled {
		parts = append(parts, "tools on")
	}
	if a.svc.running {
		parts = append(parts, core.Sprintf("api %s · %d req", a.svc.addr(), a.svc.requests.Load()))
	}
	if a.lastTokS > 0 {
		parts = append(parts, core.Sprintf("%.1f tok/s", a.lastTokS))
	}
	if a.loading != "" {
		parts = append(parts, a.spin.View()+" loading "+displayName(a.loading))
	}
	if a.generating {
		parts = append(parts, a.spin.View()+" generating (esc cancels)")
	}
	return a.styles.status.Render(core.Join("  ·  ", parts...))
}

func (a app) View() string {
	content := a.panelView()
	return renderFrame(frameSpec{
		Width:         a.width,
		Height:        a.height,
		Active:        a.activePanel,
		SessionStrip:  a.sessionStrip(),
		Main:          content,
		Inspector:     a.inspectorView(),
		Footer:        a.footerLine(),
		InspectorOpen: a.inspectorOpen,
	}, a.styles)
}

func (a app) panelView() string {
	if a.boot.phase != bootReady {
		return a.bootView()
	}
	if a.activeOverlay != overlayNone {
		return a.overlayView()
	}
	switch a.activePanel {
	case panelModels:
		return renderPicker(a.picker, measureFrame(a.width, a.height, a.inspectorOpen).mainWidth, a.styles)
	case panelService:
		return a.svc.view(a.modelName, a.width, a.styles)
	case panelWork:
		if a.work == nil {
			return lipgloss.JoinVertical(lipgloss.Left,
				a.styles.title.Render("Work"),
				"",
				a.styles.status.Render("○ No work yet"),
				a.styles.thought.Render("Local work becomes durable when the workspace store connects."),
				"",
				a.styles.attention.Render("Agent actions unavailable"),
				a.styles.thought.Render(defaultAgentUnavailableReason),
			)
		}
		metrics := measureFrame(a.width, a.height, a.inspectorOpen)
		return a.work.View(metrics.mainWidth, metrics.mainHeight, a.styles)
	case panelData:
		if a.data == nil {
			return lipgloss.JoinVertical(lipgloss.Left,
				a.styles.title.Render("Data"),
				"",
				a.styles.status.Render("○ No dataset store connected"),
				a.styles.thought.Render("datasets.duckdb becomes available when the workspace store connects."),
				"",
				a.styles.attention.Render("Review actions unavailable"),
				a.styles.thought.Render("open ~/.lem/datasets.duckdb via `lem data create`/`lem data import` first"),
			)
		}
		metrics := measureFrame(a.width, a.height, a.inspectorOpen)
		return a.data.View(metrics.mainWidth, metrics.mainHeight, a.styles)
	default: // chat
		if a.model == nil && a.loading == "" {
			return "\n  " + a.styles.status.Render("○ no model loaded — open Models and choose one")
		}
		if a.model == nil {
			return "\n  " + a.spin.View() + a.styles.status.Render(" loading "+displayName(a.loading)+" …")
		}
		return lipgloss.JoinVertical(lipgloss.Left,
			a.view.View(),
			a.styles.inputBorder.Render(a.input.View()),
		)
	}
}

func (a app) bootView() string {
	if a.boot.phase == bootFailed {
		reason := "workspace storage is unavailable"
		if a.boot.err != nil {
			reason = a.boot.err.Error()
		}
		return lipgloss.JoinVertical(lipgloss.Left,
			a.styles.err.Render("Workspace could not open"),
			"",
			a.styles.answer.Render(reason),
			"",
			a.styles.accent.Render("R  Retry"),
			a.styles.status.Render("Q  Quit without changing storage"),
		)
	}
	return lipgloss.JoinVertical(lipgloss.Left,
		a.spin.View()+a.styles.title.Render(" Opening ~/.lem workspace"),
		"",
		a.styles.status.Render("Preparing durable sessions, preferences, work, and local knowledge…"),
	)
}

func (a app) overlayView() string {
	metrics := measureFrame(a.width, a.height, a.inspectorOpen)
	width := max(1, metrics.mainWidth)
	height := max(1, metrics.mainHeight)
	bodyWidth := max(1, min(68, width-8))
	bodyHeight := max(5, min(24, height-4))
	var body string
	switch a.activeOverlay {
	case overlayCommands:
		body = a.palette.View(bodyWidth, bodyHeight)
	case overlaySessions:
		if a.switcher == nil {
			body = overlayEmpty("Recent sessions", "session workspace is not connected")
		} else {
			body = a.switcher.View(bodyWidth, bodyHeight)
		}
	case overlaySearch:
		if a.search == nil {
			body = overlayEmpty("History search", "history workspace is not connected")
		} else {
			body = a.search.View(bodyWidth, bodyHeight)
		}
	case overlayHelp:
		body = a.help.View(bodyWidth)
	case overlayWorkEditor:
		if a.workEditor == nil {
			body = overlayEmpty("Work editor", "work editor is unavailable")
		} else {
			body = a.workEditor.View(bodyWidth, bodyHeight, a.styles)
		}
	case overlayDataEditor:
		if a.dataEditor == nil {
			body = overlayEmpty("Edit as derived", "data editor is unavailable")
		} else {
			body = a.dataEditor.View(bodyWidth, bodyHeight, a.styles)
		}
	case overlayDataNote:
		if a.dataNote == nil {
			body = overlayEmpty("Data note", "note overlay is unavailable")
		} else {
			body = a.dataNote.View(bodyWidth, bodyHeight, a.styles)
		}
	case overlayDataFilter:
		if a.dataFilter == nil {
			body = overlayEmpty("Filter", "filter overlay is unavailable")
		} else {
			body = a.dataFilter.View(bodyWidth, bodyHeight, a.styles)
		}
	case overlayDataBulk:
		if a.dataBulk == nil {
			body = overlayEmpty("Bulk action", "bulk action overlay is unavailable")
		} else {
			body = a.dataBulk.View(bodyWidth, bodyHeight, a.styles)
		}
	case overlayAgentAnswer:
		if a.answerOverlay == nil {
			body = overlayEmpty("Answer agent question", "question is unavailable")
		} else {
			body = a.answerOverlay.View(bodyWidth, bodyHeight, a.styles)
		}
	case overlayChangeReview:
		if a.changeOverlay == nil {
			body = overlayEmpty("Review agent changes", "review receipt is unavailable")
		} else {
			body = a.changeOverlay.View(bodyWidth, bodyHeight, a.styles)
		}
	case overlayProjectReview:
		body = newLaunchReviewOverlay(a.agentReview, a.agentRequest.Provider, a.agentRequest.Model).View(bodyWidth, bodyHeight, a.styles)
	case overlayGitEnableReview, overlayLaunchReview, overlayAgentSelection:
		if a.launchReview == nil {
			body = overlayEmpty("Launch review", "review is unavailable")
		} else {
			body = a.launchReview.View(bodyWidth, bodyHeight, a.styles)
		}
	}
	return renderOverlay(body, width, height, a.styles)
}

func (a app) sessionStrip() string {
	if a.sessions == nil || len(a.sessions.Recent()) == 0 {
		title := newSessionTitle
		for _, item := range a.turns {
			if item.role == "user" && core.Trim(item.text) != "" {
				title = compactStripTitle(item.text, 24)
				break
			}
		}
		marker := "●"
		if a.generating {
			marker = "◉"
		}
		return core.Concat(marker, " ", title)
	}

	recent := a.sessions.Recent()
	limit := a.recentSessionLimit
	if limit < 1 {
		limit = defaultPreferenceValues().RecentSessionLimit
	}
	shown := min(len(recent), limit)
	parts := make([]string, 0, shown+1)
	for _, session := range recent[:shown] {
		title := core.Trim(session.Record.Title)
		if title == "" {
			title = newSessionTitle
		}
		parts = append(parts, core.Concat(sessionStripMarker(session, a.sessions.activeID), " ", compactStripTitle(title, 24)))
	}
	hidden := len(recent) - shown
	maxWidth := a.width - 14
	if maxWidth < 20 {
		maxWidth = 20
	}
	for len(parts) > 1 {
		candidate := append([]string(nil), parts...)
		if hidden > 0 {
			candidate = append(candidate, core.Sprintf("+%d", hidden))
		}
		if lipgloss.Width(core.Join("  ·  ", candidate...)) <= maxWidth {
			break
		}
		parts = parts[:len(parts)-1]
		hidden++
	}
	if hidden > 0 {
		parts = append(parts, core.Sprintf("+%d", hidden))
	}
	return core.Join("  ·  ", parts...)
}

func sessionStripMarker(session *chatSession, activeID string) string {
	if session == nil {
		return "○"
	}
	if session.Record.Status == "queued" || session.Record.Status == "generating" || session.ActiveJobID != "" {
		return "◉"
	}
	terminal := session.Record.Status == "failed" || session.Record.Status == "interrupted" || session.Record.Status == "cancelled"
	if session.Record.ID == activeID && terminal {
		return "!"
	}
	if session.Record.ID == activeID {
		return "●"
	}
	if session.Attention {
		return "◆"
	}
	if terminal {
		return "!"
	}
	return "○"
}

func compactStripTitle(value string, maxCells int) string {
	value = core.Trim(value)
	if maxCells < 2 || lipgloss.Width(value) <= maxCells {
		return value
	}
	runes := []rune(value)
	for len(runes) > 1 && lipgloss.Width(string(runes)+"…") > maxCells {
		runes = runes[:len(runes)-1]
	}
	return string(runes) + "…"
}

func (a app) inspectorView() string {
	metrics := measureFrame(a.width, a.height, a.inspectorOpen)
	width := metrics.inspectorWidth
	height := metrics.inspectorHeight
	if width <= 0 {
		width = metrics.innerWidth
	}
	if height <= 0 {
		height = metrics.regionHeight
	}
	return a.inspector.View(a, width, height)
}

func (a app) footerLine() string {
	keys := "tab panels  ·  ctrl+k commands  ·  ctrl+o inspector  ·  f1 help"
	if chooseLayout(a.width) == layoutNarrow {
		keys = "tab panels  ·  ^K commands  ·  ^O info  ·  F1 help"
	}
	status := a.statusLine()
	if status == "" {
		return keys
	}
	return status + "  │  " + keys
}
