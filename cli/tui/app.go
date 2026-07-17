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
	"dappco.re/go/inference/decode/parser"
)

// turn is one rendered element in the transcript.
type turn struct {
	id      string
	role    string // "user" | "assistant" | "tool"
	thought string
	text    string
	calls   []string // rendered tool-call receipts on an assistant turn
}

type app struct {
	boot            bootState
	workspaceLoader func() core.Result
	resources       *workspaceResources
	lifecycle       *appLifecycle
	warnings        []string
	runtimeDetector runtimeDetector
	knowledgeScan   knowledgeScanner
	knowledgeMounts []knowledgeMount
	knowledgeLimit  int64

	activePanel   panelID
	inspectorOpen bool
	styles        uiStyles
	keys          keyMap
	markdown      *markdownRenderer
	activeOverlay overlayKind
	palette       *commandPalette
	switcher      *sessionSwitcher
	search        *historySearch
	help          *helpOverlay
	sessions      *sessionManager
	repository    workspaceRepository
	preferences   preferenceStore
	inspector     inspectorState
	agent         agentProvider
	work          *workPanel
	knowledge     *knowledgeLibrary
	attachments   []attachmentRecord

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
	cancel    context.CancelFunc
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

type sessionGeneration struct {
	generation *generation
	answer     turnRecord
	job        generationJobRecord
	started    bool
	cancelled  bool
}

type appLifecycle struct {
	once   sync.Once
	result core.Result
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
		boot:            bootState{phase: bootReady},
		lifecycle:       &appLifecycle{result: core.Ok(nil)},
		runtimeDetector: newContainerRuntimeDetector(),
		knowledgeScan:   newKnowledgeScanner(),
		knowledgeLimit:  knowledgeSystemMessageMaxBytes,
		modelLoader:     loadModel,
		activePanel:     panelChat,
		styles:          styles,
		keys:            keys,
		markdown:        newMarkdownRenderer(styles.theme.name),
		palette:         newCommandPalette(styles),
		help:            newHelpOverlay(keys, styles),
		inspector:       newInspector(),
		agent:           agent,
		picker:          newPicker(styles),
		spin:            sp,
		input:           in,
		cfg:             cfg,
		modes:           modeState{},
		tools:           newTools(),
		svc:             newService(),
		jobs:            newJobManager(ctx),
		sessionJobs:     make(map[string]*sessionGeneration),
		cancel:          cancel,
		sessionID:       newRecordID(),
		follow:          true,
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
	a.palette.SetAgentCapabilities(provider.Capabilities())
	return core.Ok(a.work)
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
	a.picker.Styles.Title = a.styles.title
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
		cmds = append(cmds, workspaceBootstrap(a.workspaceLoader))
		return tea.Batch(cmds...)
	}
	cmds = append(cmds, discoverModels)
	if a.loading != "" {
		cmds = append(cmds, a.modelLoader(a.loading, a.cfg.contextLen()))
	}
	return tea.Batch(cmds...)
}

func (a app) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case workspaceReadyMsg:
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

	case spinner.TickMsg:
		var cmd tea.Cmd
		a.spin, cmd = a.spin.Update(msg)
		return a, cmd

	case streamMsg:
		return a.onStream(msg)

	case streamRefreshMsg:
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
	}
	return a.route(msg)
}

func (a *app) connectWorkspace(resources *workspaceResources) core.Result {
	if a == nil || resources == nil || resources.Repository == nil || resources.State == nil || resources.Preferences == nil {
		return core.Fail(core.E("tui.app.connectWorkspace", "workspace resources are incomplete", nil))
	}
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
	if result := a.attachWork(resources.Repository, a.agent); !result.OK {
		return result
	}
	values := resources.Preferences.Values()
	a.knowledgeLimit = values.KnowledgeMaxBytes
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
	return core.Ok(nil)
}

func (a app) workspaceReadyCommands() tea.Cmd {
	commands := []tea.Cmd{discoverModels}
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
		commands = append(commands, a.modelLoader(a.loading, a.cfg.contextLen()))
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
		if result := a.repository.SaveJob(managed.job); !result.OK {
			a.errText = result.Error()
		}
		if result := a.sessions.MarkGenerating(ev.SessionID, ev.JobID); !result.OK {
			a.errText = result.Error()
		}
	}
	managed.answer.Visible += ev.visible
	managed.answer.Thought += ev.thought
	managed.answer.UpdatedAt = now
	if ev.metrics != nil {
		managed.job.MetricsJSON = core.JSONMarshalString(ev.metrics)
		if a.sessionID == ev.SessionID {
			a.lastTokS = ev.metrics.DecodeTokensPerSec
		}
	}
	if ev.err != nil {
		managed.job.Error = ev.err.Error()
	}
	if ev.visible != "" || ev.thought != "" || ev.done || ev.err != nil {
		if result := a.sessions.AddTurn(managed.answer); !result.OK {
			a.errText = result.Error()
		}
	}
	active := a.sessions.Active()
	if active != nil && active.Record.ID == ev.SessionID {
		a.syncManagedSession(active, false)
		a.refreshTranscriptOutput()
	}
	if !ev.done {
		return a, waitEvent(managed.generation)
	}
	managed.job.FinishedAt = now
	switch {
	case managed.cancelled:
		managed.job.Status = "cancelled"
	case ev.err != nil:
		managed.job.Status = "failed"
	default:
		managed.job.Status = "completed"
	}
	if result := a.repository.SaveJob(managed.job); !result.OK {
		a.errText = result.Error()
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
	} else if result := a.sessions.Complete(ev.SessionID); !result.OK {
		a.errText = result.Error()
	}
	active = a.sessions.Active()
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
	a.turns = append(a.turns, turn{id: newRecordID(), role: "assistant"})
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
		a.inspectorOpen = !a.inspectorOpen
		if a.ready {
			metrics := measureFrame(a.width, a.height, a.inspectorOpen)
			a.picker.SetSize(max(1, metrics.mainWidth), max(1, metrics.mainHeight))
			a.input.SetWidth(max(1, metrics.mainWidth-4))
			a.view.Width = max(1, metrics.mainWidth)
			a.view.Height = a.transcriptHeight()
			a.refreshTranscript()
		}
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
				} else if result := a.work.ActivateSelectedAction(context.Background(), ""); !result.OK {
					a.errText = result.Error()
				}
				return a, nil
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
			a.activePanel = a.activePanel.next()
		} else {
			a.activePanel = a.activePanel.prev()
		}
		if a.activePanel == panelModels && len(a.picker.Items()) == 0 {
			return a, discoverModels
		}
		return a, nil
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
	case panelChat:
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
	return loader(path, a.cfg.contextLen())
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
			turn{id: newRecordID(), role: "assistant"},
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
			return a, workspaceBootstrap(a.workspaceLoader)
		}
	}
	return a, nil
}

func (a *app) shutdown() core.Result {
	if a == nil {
		return core.Ok(nil)
	}
	if a.lifecycle == nil {
		a.lifecycle = &appLifecycle{result: core.Ok(nil)}
	}
	a.lifecycle.once.Do(func() {
		result := core.Ok(nil)
		record := func(candidate core.Result) {
			if !candidate.OK && result.OK {
				result = candidate
			}
		}
		if a.jobs != nil {
			record(a.jobs.CancelAll())
		}
		if a.cancel != nil {
			a.cancel()
		}
		if a.agent != nil {
			record(a.agent.Close())
		}
		a.svc.teardown("stopped (quit)")
		if a.lane != nil {
			record(a.lane.Close())
			a.lane = nil
			a.model = nil
		}
		if a.resources != nil {
			record(a.resources.Close())
		}
		a.lifecycle.result = result
	})
	return a.lifecycle.result
}

func (a app) onOverlayKey(message tea.KeyMsg) (tea.Model, tea.Cmd) {
	if message.String() == "esc" {
		a.activeOverlay = overlayNone
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
				a.activeOverlay = overlayNone
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
			thought: record.Thought,
			text:    record.Visible,
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
	for i, t := range a.turns {
		if i > 0 {
			b.WriteString("\n\n")
		}
		switch t.role {
		case "user":
			b.WriteString(a.styles.user.Render("you ") + a.styles.answer.Render(t.text))
		case "tool":
			b.WriteString(a.styles.thought.Render("tool result fed back"))
		default:
			if t.thought != "" {
				b.WriteString(a.styles.thought.Render("· thinking · "+core.Trim(t.thought)) + "\n")
			}
			label := a.modelName
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
		return a.picker.View()
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
	bodyHeight := max(5, min(14, height-4))
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
	}
	return renderOverlay(body, width, height, a.styles)
}

func (a app) sessionStrip() string {
	title := "New session"
	for _, turn := range a.turns {
		if turn.role == "user" && core.Trim(turn.text) != "" {
			title = core.Trim(turn.text)
			break
		}
	}
	marker := "●"
	state := "idle"
	if a.generating {
		marker, state = "◉", "generating"
	}
	return marker + " " + title + "  ·  " + state
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
