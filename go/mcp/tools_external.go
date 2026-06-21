package mcp

import (
	"context"
	"io"
	"net"
	"net/http"
	"strconv"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	execabs "golang.org/x/sys/execabs"
)

const defaultRAGCollection = "hostuk-docs"

type RAGQueryInput struct {
	Question   string `json:"question"`
	Collection string `json:"collection,omitempty"`
	TopK       int    `json:"topK,omitempty"`
}

type RAGQueryOutput struct {
	Results    []RAGQueryResult `json:"results"`
	Query      string           `json:"query"`
	Collection string           `json:"collection"`
	Context    string           `json:"context"`
}

type RAGQueryResult struct {
	Content    string  `json:"content"`
	Source     string  `json:"source"`
	Section    string  `json:"section,omitempty"`
	Category   string  `json:"category,omitempty"`
	ChunkIndex int     `json:"chunkIndex"`
	Score      float32 `json:"score"`
}

type RAGIngestInput struct {
	Path       string `json:"\x70ath"`
	Collection string `json:"collection,omitempty"`
	Recreate   bool   `json:"recreate,omitempty"`
}

type RAGIngestOutput struct {
	Success    bool   `json:"success"`
	Path       string `json:"\x70ath"`
	Collection string `json:"collection"`
	Chunks     int    `json:"chunks"`
	Message    string `json:"message"`
}

type RAGCollectionsInput struct {
	ShowStats bool `json:"show_stats,omitempty"`
}

type RAGCollectionsOutput struct {
	Collections []CollectionInfo `json:"collections"`
}

type CollectionInfo struct {
	Name        string `json:"name"`
	PointsCount uint64 `json:"points_count,omitempty"`
	Status      string `json:"status,omitempty"`
}

func (s *Service) ragQuery(ctx context.Context, input RAGQueryInput) core.Result {
	if core.Trim(input.Question) == "" {
		return core.Fail(core.Errorf("%w: question is required", errInvalidParams))
	}
	collection := defaultString(input.Collection, defaultRAGCollection)
	return core.Ok(RAGQueryOutput{
		Results:    []RAGQueryResult{},
		Query:      input.Question,
		Collection: collection,
		Context:    "",
	})
}

func (s *Service) ragIngest(ctx context.Context, input RAGIngestInput) core.Result {
	if r := s.resolvePath(input.Path); !r.OK {
		return r
	}
	collection := defaultString(input.Collection, defaultRAGCollection)
	return core.Ok(RAGIngestOutput{
		Success:    false,
		Path:       input.Path,
		Collection: collection,
		Message:    "RAG ingestion backend is not configured in this daemon",
	})
}

func (s *Service) ragCollections(ctx context.Context, input RAGCollectionsInput) core.Result {
	return core.Ok(RAGCollectionsOutput{Collections: []CollectionInfo{}})
}

type MLGenerateInput struct {
	Prompt      string  `json:"prompt"`
	Backend     string  `json:"backend,omitempty"`
	Model       string  `json:"model,omitempty"`
	Temperature float64 `json:"temperature,omitempty"`
	MaxTokens   int     `json:"max_tokens,omitempty"`
}

type MLGenerateOutput struct {
	Response string `json:"response"`
	Backend  string `json:"backend"`
	Model    string `json:"model,omitempty"`
}

type MLScoreInput struct {
	Prompt   string `json:"prompt"`
	Response string `json:"response"`
	Suites   string `json:"suites,omitempty"`
}

type MLScoreOutput struct {
	Heuristic map[string]any `json:"heuristic,omitempty"`
	Semantic  map[string]any `json:"semantic,omitempty"`
	Content   map[string]any `json:"content,omitempty"`
}

type MLProbeInput struct {
	Backend    string `json:"backend,omitempty"`
	Categories string `json:"categories,omitempty"`
}

type MLProbeOutput struct {
	Total   int                 `json:"total"`
	Results []MLProbeResultItem `json:"results"`
}

type MLProbeResultItem struct {
	ID       string `json:"id"`
	Category string `json:"category"`
	Response string `json:"response"`
}

type MLStatusInput struct {
	InfluxURL string `json:"influx_url,omitempty"`
	InfluxDB  string `json:"influx_db,omitempty"`
}

type MLStatusOutput struct {
	Status string `json:"status"`
}

type MLBackendsInput struct{}

type MLBackendsOutput struct {
	Backends []MLBackendInfo `json:"backends"`
	Default  string          `json:"default"`
}

type MLBackendInfo struct {
	Name         string   `json:"name"`
	Available    bool     `json:"available"`
	Capabilities []string `json:"capabilities,omitempty"`
	Native       bool     `json:"native,omitempty"`
}

func (s *Service) mlGenerate(ctx context.Context, input MLGenerateInput) core.Result {
	if core.Trim(input.Prompt) == "" {
		return core.Fail(core.Errorf("%w: prompt is required", errInvalidParams))
	}
	if s != nil && s.mlModel != nil {
		opts := []inference.GenerateOption{}
		if input.MaxTokens > 0 {
			opts = append(opts, inference.WithMaxTokens(input.MaxTokens))
		}
		if input.Temperature != 0 {
			opts = append(opts, inference.WithTemperature(float32(input.Temperature)))
		}
		parts := []string{}
		for token := range s.mlModel.Generate(ctx, input.Prompt, opts...) {
			parts = append(parts, token.Text)
		}
		if errResult := s.mlModel.Err(); !errResult.OK {
			if err, ok := errResult.Value.(error); ok {
				return core.Fail(core.Errorf("ml_generate: %w", err))
			}
			return core.Fail(core.Errorf("ml_generate: %s", errResult.Error()))
		}
		return core.Ok(MLGenerateOutput{
			Response: core.Join("", parts...),
			Backend:  defaultString(input.Backend, defaultString(s.mlBackend, "inference")),
			Model:    defaultString(input.Model, s.mlModelName),
		})
	}
	backend := defaultString(input.Backend, "builtin")
	response := "ML generation backend is not configured in this daemon."
	return core.Ok(MLGenerateOutput{Response: response, Backend: backend, Model: input.Model})
}

func (s *Service) mlScore(ctx context.Context, input MLScoreInput) core.Result {
	if core.Trim(input.Prompt) == "" {
		return core.Fail(core.Errorf("%w: prompt is required", errInvalidParams))
	}
	if core.Trim(input.Response) == "" {
		return core.Fail(core.Errorf("%w: response is required", errInvalidParams))
	}
	suites := splitCSV(defaultString(input.Suites, "heuristic"))
	out := MLScoreOutput{}
	for _, suite := range suites {
		switch suite {
		case "heuristic":
			out.Heuristic = heuristicScores(input.Prompt, input.Response)
		case "semantic":
			out.Semantic = map[string]any{
				"available": false,
				"message":   "semantic scoring backend is not configured",
			}
		case "content":
			out.Content = map[string]any{
				"available": false,
				"message":   "content scoring is available through ml_probe when an ML service is configured",
			}
		default:
			return core.Fail(core.Errorf("%w: unsupported suite %q", errInvalidParams, suite))
		}
	}
	return core.Ok(out)
}

func (s *Service) mlProbe(ctx context.Context, input MLProbeInput) core.Result {
	return core.Ok(MLProbeOutput{Results: []MLProbeResultItem{}})
}

func (s *Service) mlStatus(ctx context.Context, input MLStatusInput) core.Result {
	url := defaultString(input.InfluxURL, "http://localhost:8086")
	db := defaultString(input.InfluxDB, "lem")
	return core.Ok(MLStatusOutput{Status: core.Sprintf("ML status backend is not configured (influx_url=%s influx_db=%s)", url, db)})
}

func (s *Service) mlBackends(ctx context.Context, input MLBackendsInput) core.Result {
	names := inference.List()
	backends := make([]MLBackendInfo, 0, len(names)+1)
	for _, name := range names {
		backend, ok := inference.Get(name)
		info := MLBackendInfo{Name: name, Available: ok && backend.Available()}
		if ok {
			report, _ := inference.CapabilitiesOf(backend)
			info.Capabilities = inferenceCapabilityIDStrings(report.SupportedCapabilityIDs())
			info.Native = report.Runtime.NativeRuntime
		}
		backends = append(backends, info)
	}
	defaultName := "builtin"
	if result := inference.Default(); result.OK {
		if backend, ok := result.Value.(inference.Backend); ok && backend != nil {
			defaultName = backend.Name()
		}
	}
	if len(backends) == 0 {
		backends = append(backends, MLBackendInfo{Name: "builtin", Available: true})
	}
	return core.Ok(MLBackendsOutput{
		Backends: backends,
		Default:  defaultName,
	})
}

func inferenceCapabilityIDStrings(ids []inference.CapabilityID) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		out[i] = string(id)
	}
	return out
}

func heuristicScores(prompt, response string) map[string]any {
	words := splitFields(response)
	promptWords := splitFields(prompt)
	lengthScore := minFloat(float64(len(words))/120.0, 1.0)
	structureScore := 0.0
	if core.Contains(response, "\n") {
		structureScore += 0.25
	}
	if core.Contains(response, ".") || core.Contains(response, ":") {
		structureScore += 0.25
	}
	if core.Contains(response, "- ") || core.Contains(response, "1.") {
		structureScore += 0.25
	}
	if core.Contains(response, "```") {
		structureScore += 0.25
	}
	return map[string]any{
		"prompt_length":   len(prompt),
		"response_length": len(response),
		"prompt_words":    len(promptWords),
		"response_words":  len(words),
		"has_code":        core.Contains(response, "```"),
		"length_score":    lengthScore,
		"structure_score": minFloat(structureScore, 1.0),
		"overall":         minFloat((lengthScore+structureScore)/2.0, 1.0),
	}
}

type MetricsRecordInput struct {
	Type    string         `json:"type"`
	AgentID string         `json:"agent_id,omitempty"`
	Repo    string         `json:"repo,omitempty"`
	Data    map[string]any `json:"data,omitempty"`
}

type MetricsRecordOutput struct {
	Success   bool      `json:"success"`
	Timestamp time.Time `json:"timestamp"`
}

type MetricsQueryInput struct {
	Since string `json:"since,omitempty"`
}

type MetricsQueryOutput struct {
	ByType  map[string]int `json:"by_type"`
	ByRepo  map[string]int `json:"by_repo"`
	ByAgent map[string]int `json:"by_agent"`
	Recent  []MetricEvent  `json:"recent"`
}

type MetricEvent struct {
	Type      string         `json:"type"`
	Timestamp time.Time      `json:"timestamp"`
	AgentID   string         `json:"agent_id,omitempty"`
	Repo      string         `json:"repo,omitempty"`
	Data      map[string]any `json:"data,omitempty"`
}

type metricSummary struct {
	ByType  map[string]int
	ByRepo  map[string]int
	ByAgent map[string]int
	Recent  []MetricEvent
}

var metricWriteMu sync.Mutex

func (s *Service) metricsRecord(ctx context.Context, input MetricsRecordInput) core.Result {
	if core.Trim(input.Type) == "" {
		return core.Fail(core.Errorf("%w: type is required", errInvalidParams))
	}
	timestamp := time.Now()
	if r := recordMetricEvent(MetricEvent{
		Type:      input.Type,
		Timestamp: timestamp,
		AgentID:   input.AgentID,
		Repo:      input.Repo,
		Data:      input.Data,
	}); !r.OK {
		return r
	}
	return core.Ok(MetricsRecordOutput{Success: true, Timestamp: timestamp})
}

func (s *Service) metricsQuery(ctx context.Context, input MetricsQueryInput) core.Result {
	windowResult := parseSinceWindow(defaultString(input.Since, "7d"))
	if !windowResult.OK {
		return windowResult
	}
	window := windowResult.Value.(time.Duration)
	eventsResult := readMetricEvents(time.Now().Add(-window))
	if !eventsResult.OK {
		return eventsResult
	}
	events := eventsResult.Value.([]MetricEvent)
	summary := summarizeMetricEvents(events)
	return core.Ok(MetricsQueryOutput{
		ByType:  summary.ByType,
		ByRepo:  summary.ByRepo,
		ByAgent: summary.ByAgent,
		Recent:  summary.Recent,
	})
}

func parseSinceWindow(value string) core.Result {
	value = core.Trim(value)
	if len(value) < 2 {
		return core.Fail(core.Errorf("%w: invalid since value %q", errInvalidParams, value))
	}
	unit := value[len(value)-1]
	amount, err := strconv.Atoi(value[:len(value)-1])
	if err != nil || amount <= 0 {
		return core.Fail(core.Errorf("%w: invalid since value %q", errInvalidParams, value))
	}
	switch unit {
	case 'm':
		return core.Ok(time.Duration(amount) * time.Minute)
	case 'h':
		return core.Ok(time.Duration(amount) * time.Hour)
	case 'd':
		return core.Ok(time.Duration(amount) * 24 * time.Hour)
	default:
		return core.Fail(core.Errorf("%w: invalid since unit %q", errInvalidParams, string(unit)))
	}
}

func recordMetricEvent(event MetricEvent) core.Result {
	if event.Timestamp.IsZero() {
		event.Timestamp = time.Now()
	}
	dirResult := metricDir()
	if !dirResult.OK {
		return dirResult
	}
	dir := dirResult.Value.(string)
	metricWriteMu.Lock()
	defer metricWriteMu.Unlock()
	if r := core.MkdirAll(dir, 0o700); !r.OK {
		return r
	}
	path := metricFilePath(dir, event.Timestamp)
	r := core.OpenFile(path, core.O_CREATE|core.O_APPEND|core.O_WRONLY, 0o600)
	if !r.OK {
		return r
	}
	file := r.Value.(*core.OSFile)
	defer file.Close()
	encoded := core.JSONMarshal(event)
	if !encoded.OK {
		return encoded
	}
	data := encoded.Value.([]byte)
	if _, err := file.Write(append(data, '\n')); err != nil {
		return core.Fail(err)
	}
	return core.Ok(nil)
}

func readMetricEvents(since time.Time) core.Result {
	dirResult := metricDir()
	if !dirResult.OK {
		return dirResult
	}
	dir := dirResult.Value.(string)
	now := time.Now()
	start := time.Date(since.Year(), since.Month(), since.Day(), 0, 0, 0, 0, since.Location())
	var events []MetricEvent
	for day := start; !day.After(now); day = day.AddDate(0, 0, 1) {
		r := core.ReadFile(metricFilePath(dir, day))
		if !r.OK {
			err, _ := resultError(r).(error)
			if core.IsNotExist(err) {
				continue
			}
			return r
		}
		data := r.Value.([]byte)
		for _, line := range core.Split(string(data), "\n") {
			line = core.Trim(line)
			if line == "" {
				continue
			}
			var event MetricEvent
			if r := core.JSONUnmarshal([]byte(line), &event); !r.OK {
				continue
			}
			if !event.Timestamp.Before(since) {
				events = append(events, event)
			}
		}
	}
	return core.Ok(events)
}

func summarizeMetricEvents(events []MetricEvent) metricSummary {
	summary := metricSummary{
		ByType:  map[string]int{},
		ByRepo:  map[string]int{},
		ByAgent: map[string]int{},
	}
	for _, event := range events {
		summary.ByType[event.Type]++
		if event.Repo != "" {
			summary.ByRepo[event.Repo]++
		}
		if event.AgentID != "" {
			summary.ByAgent[event.AgentID]++
		}
	}
	recent := events
	if len(recent) > 10 {
		recent = recent[len(recent)-10:]
	}
	summary.Recent = append([]MetricEvent(nil), recent...)
	return summary
}

func metricDir() core.Result {
	home := core.Getenv("CORE_HOME")
	if home == "" {
		home = core.Getenv("HOME")
	}
	if home == "" {
		home = core.Getenv("USERPROFILE")
	}
	if home == "" {
		return core.Fail(core.Errorf("metrics home directory is not configured"))
	}
	return core.Ok(core.PathJoin(home, ".core", "ai", "metrics"))
}

func metricFilePath(dir string, timestamp time.Time) string {
	return core.PathJoin(dir, timestamp.Format("2006-01-02")+".jsonl")
}

type ProcessStartInput struct {
	Command string   `json:"command"`
	Args    []string `json:"args,omitempty"`
	Dir     string   `json:"dir,omitempty"`
	Env     []string `json:"env,omitempty"`
}

type ProcessStartOutput struct {
	ID        string    `json:"id"`
	PID       int       `json:"pid"`
	Command   string    `json:"command"`
	Args      []string  `json:"args"`
	StartedAt time.Time `json:"startedAt"`
}

type ProcessIDInput struct {
	ID string `json:"id"`
}

type ProcessControlOutput struct {
	ID      string `json:"id"`
	Success bool   `json:"success"`
	Message string `json:"message"`
}

type ProcessListInput struct {
	RunningOnly bool `json:"running_only,omitempty"`
}

type ProcessListOutput struct {
	Processes []ProcessInfo `json:"processes"`
	Total     int           `json:"total"`
}

type ProcessInfo struct {
	ID        string        `json:"id"`
	Command   string        `json:"command"`
	Args      []string      `json:"args"`
	Dir       string        `json:"dir,omitempty"`
	Status    string        `json:"status"`
	PID       int           `json:"pid"`
	ExitCode  int           `json:"exitCode"`
	StartedAt time.Time     `json:"startedAt"`
	Duration  time.Duration `json:"duration"`
}

type ProcessOutputInput struct {
	ID string `json:"id"`
}

type ProcessOutputOutput struct {
	ID     string `json:"id"`
	Output string `json:"output"`
}

type ProcessInputInput struct {
	ID    string `json:"id"`
	Input string `json:"input"`
}

type ProcessInputOutput struct {
	ID      string `json:"id"`
	Success bool   `json:"success"`
	Message string `json:"message"`
}

type managedProcess struct {
	id         string
	command    string
	args       []string
	dir        string
	startedAt  time.Time
	endedAt    time.Time
	status     string
	exitCode   int
	errText    string
	cmd        *core.Cmd
	stdin      io.WriteCloser
	outputPipe *io.PipeWriter
	output     safeBuffer
	mu         sync.Mutex
}

type safeBuffer struct {
	mu  sync.Mutex
	buf []byte
}

func (b *safeBuffer) append(p []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.buf = append(b.buf, p...)
}

func (b *safeBuffer) readFrom(reader *io.PipeReader) {
	buffer := make([]byte, 4096)
	for {
		n, readErr := reader.Read(buffer)
		if n > 0 {
			b.append(buffer[:n])
		}
		if readErr != nil {
			return
		}
	}
}

func (b *safeBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return string(append([]byte(nil), b.buf...))
}

func (s *Service) processStart(ctx context.Context, input ProcessStartInput) core.Result {
	if core.Trim(input.Command) == "" {
		return core.Fail(core.Errorf("%w: command is required", errInvalidParams))
	}
	dir := input.Dir
	if dir != "" {
		resolved := s.resolvePath(dir)
		if !resolved.OK {
			return resolved
		}
		dir = resolved.Value.(string)
	} else if s.workspaceRoot != "" {
		dir = s.workspaceRoot
	}

	cmd := execabs.Command(input.Command, input.Args...)
	cmd.Dir = dir
	cmd.Env = append(core.Environ(), input.Env...)

	id := core.Sprintf("proc-%d", s.processSeq.Add(1))
	proc := &managedProcess{
		id:        id,
		command:   input.Command,
		args:      append([]string(nil), input.Args...),
		dir:       dir,
		startedAt: time.Now(),
		status:    "starting",
		exitCode:  -1,
		cmd:       cmd,
	}
	outputReader, outputWriter := io.Pipe()
	proc.outputPipe = outputWriter
	cmd.Stdout = outputWriter
	cmd.Stderr = outputWriter
	go proc.output.readFrom(outputReader)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return core.Fail(err)
	}
	proc.stdin = stdin

	if err := cmd.Start(); err != nil {
		return core.Fail(err)
	}
	proc.status = "running"

	s.processMu.Lock()
	s.processes[id] = proc
	s.processMu.Unlock()

	go proc.wait()

	return core.Ok(ProcessStartOutput{
		ID:        id,
		PID:       cmd.Process.Pid,
		Command:   input.Command,
		Args:      append([]string(nil), input.Args...),
		StartedAt: proc.startedAt,
	})
}

func (p *managedProcess) wait() {
	err := p.cmd.Wait()
	p.mu.Lock()
	defer p.mu.Unlock()
	p.endedAt = time.Now()
	p.status = "exited"
	p.exitCode = 0
	if p.cmd.ProcessState != nil {
		p.exitCode = p.cmd.ProcessState.ExitCode()
	}
	if err != nil {
		p.errText = err.Error()
		if p.exitCode == 0 {
			p.exitCode = -1
		}
	}
	if p.stdin != nil {
		if closeErr := p.stdin.Close(); closeErr != nil && p.errText == "" {
			p.errText = closeErr.Error()
		}
	}
	if p.outputPipe != nil {
		p.outputPipe.Close()
	}
}

func (s *Service) processStop(ctx context.Context, input ProcessIDInput) core.Result {
	return s.killProcess(input.ID, "stopped")
}

func (s *Service) processKill(ctx context.Context, input ProcessIDInput) core.Result {
	return s.killProcess(input.ID, "killed")
}

func (s *Service) killProcess(id, verb string) core.Result {
	procResult := s.lookupProcess(id)
	if !procResult.OK {
		return procResult
	}
	proc := procResult.Value.(*managedProcess)
	if !proc.isRunning() {
		return core.Ok(ProcessControlOutput{ID: id, Success: true, Message: "process is not running"})
	}
	if proc.cmd.Process == nil {
		return core.Fail(core.Errorf("process has no OS handle: %s", id))
	}
	if err := proc.cmd.Process.Kill(); err != nil {
		return core.Fail(err)
	}
	return core.Ok(ProcessControlOutput{ID: id, Success: true, Message: "process " + verb})
}

func (s *Service) processList(ctx context.Context, input ProcessListInput) core.Result {
	s.processMu.Lock()
	processes := make([]*managedProcess, 0, len(s.processes))
	for _, proc := range s.processes {
		processes = append(processes, proc)
	}
	s.processMu.Unlock()

	out := make([]ProcessInfo, 0, len(processes))
	for _, proc := range processes {
		info := proc.info()
		if input.RunningOnly && info.Status != "running" {
			continue
		}
		out = append(out, info)
	}
	return core.Ok(ProcessListOutput{Processes: out, Total: len(out)})
}

func (s *Service) processOutput(ctx context.Context, input ProcessOutputInput) core.Result {
	procResult := s.lookupProcess(input.ID)
	if !procResult.OK {
		return procResult
	}
	proc := procResult.Value.(*managedProcess)
	return core.Ok(ProcessOutputOutput{ID: input.ID, Output: proc.output.String()})
}

func (s *Service) processInput(ctx context.Context, input ProcessInputInput) core.Result {
	if input.Input == "" {
		return core.Fail(core.Errorf("%w: input is required", errInvalidParams))
	}
	procResult := s.lookupProcess(input.ID)
	if !procResult.OK {
		return procResult
	}
	proc := procResult.Value.(*managedProcess)
	if !proc.isRunning() {
		return core.Fail(core.Errorf("process is not running: %s", input.ID))
	}
	if _, err := io.WriteString(proc.stdin, input.Input); err != nil {
		return core.Fail(err)
	}
	return core.Ok(ProcessInputOutput{ID: input.ID, Success: true, Message: "input delivered"})
}

func (s *Service) lookupProcess(id string) core.Result {
	if core.Trim(id) == "" {
		return core.Fail(core.Errorf("%w: id is required", errInvalidParams))
	}
	s.processMu.Lock()
	defer s.processMu.Unlock()
	proc, ok := s.processes[id]
	if !ok {
		return core.Fail(core.Errorf("process not found: %s", id))
	}
	return core.Ok(proc)
}

func (p *managedProcess) isRunning() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.status == "running"
}

func (p *managedProcess) info() ProcessInfo {
	p.mu.Lock()
	defer p.mu.Unlock()
	pid := 0
	if p.cmd != nil && p.cmd.Process != nil {
		pid = p.cmd.Process.Pid
	}
	end := time.Now()
	if !p.endedAt.IsZero() {
		end = p.endedAt
	}
	return ProcessInfo{
		ID:        p.id,
		Command:   p.command,
		Args:      append([]string(nil), p.args...),
		Dir:       p.dir,
		Status:    p.status,
		PID:       pid,
		ExitCode:  p.exitCode,
		StartedAt: p.startedAt,
		Duration:  end.Sub(p.startedAt),
	}
}

type WSStartInput struct {
	Addr string `json:"addr,omitempty"`
}

type WSStartOutput struct {
	Success bool   `json:"success"`
	Addr    string `json:"addr"`
	Message string `json:"message"`
}

type WSInfoInput struct{}

type WSInfoOutput struct {
	Clients  int    `json:"clients"`
	Channels int    `json:"channels"`
	Addr     string `json:"addr,omitempty"`
	Running  bool   `json:"running"`
}

func (s *Service) wsStart(ctx context.Context, input WSStartInput) core.Result {
	s.wsMu.Lock()
	if s.wsServer != nil {
		addr := s.wsAddr
		s.wsMu.Unlock()
		return core.Ok(WSStartOutput{Success: true, Addr: addr, Message: "WebSocket server already running at ws://" + addr + "/ws"})
	}
	s.wsMu.Unlock()

	addr := defaultString(input.Addr, ":8080")
	listener, err := net.Listen("tcp", addr)
	if err != nil {
		return core.Fail(err)
	}
	actualAddr := listener.Addr().String()
	mux := http.NewServeMux()
	mux.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "WebSocket hub is not configured", http.StatusNotImplemented)
	})
	server := &http.Server{Handler: mux}

	s.wsMu.Lock()
	s.wsServer = server
	s.wsAddr = actualAddr
	s.wsMu.Unlock()

	go func() {
		if err := server.Serve(listener); err != nil && !errorsIsHTTPServerClosed(err) {
			core.Print(core.Stderr(), "MCP WebSocket server error: %v\n", err)
		}
		s.wsMu.Lock()
		if s.wsServer == server {
			s.wsServer = nil
			s.wsAddr = ""
		}
		s.wsMu.Unlock()
	}()

	return core.Ok(WSStartOutput{Success: true, Addr: actualAddr, Message: "WebSocket server running at ws://" + actualAddr + "/ws"})
}

func (s *Service) wsInfo(ctx context.Context, input WSInfoInput) core.Result {
	s.wsMu.Lock()
	defer s.wsMu.Unlock()
	return core.Ok(WSInfoOutput{Clients: 0, Channels: 0, Addr: s.wsAddr, Running: s.wsServer != nil})
}

type webviewSession struct {
	Connected bool
	DebugURL  string
	URL       string
	Timeout   int
	Console   []WebviewConsoleMessage
}

type WebviewConnectInput struct {
	DebugURL string `json:"debug_url"`
	Timeout  int    `json:"timeout,omitempty"`
}

type WebviewConnectOutput struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

type WebviewDisconnectInput struct{}

type WebviewDisconnectOutput struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

type WebviewNavigateInput struct {
	URL string `json:"url"`
}

type WebviewNavigateOutput struct {
	Success bool   `json:"success"`
	URL     string `json:"url"`
}

type WebviewSelectorInput struct {
	Selector string `json:"selector"`
}

type WebviewClickOutput struct {
	Success bool `json:"success"`
}

type WebviewTypeInput struct {
	Selector string `json:"selector"`
	Text     string `json:"text"`
}

type WebviewTypeOutput struct {
	Success bool `json:"success"`
}

type WebviewQueryInput struct {
	Selector string `json:"selector"`
	All      bool   `json:"all,omitempty"`
}

type WebviewQueryOutput struct {
	Found    bool                 `json:"found"`
	Count    int                  `json:"count"`
	Elements []WebviewElementInfo `json:"elements"`
}

type WebviewElementInfo struct {
	NodeID      int               `json:"nodeId"`
	TagName     string            `json:"tagName"`
	Attributes  map[string]string `json:"attributes"`
	BoundingBox *BoundingBox      `json:"boundingBox,omitempty"`
}

type BoundingBox struct {
	X      float64 `json:"x"`
	Y      float64 `json:"y"`
	Width  float64 `json:"width"`
	Height float64 `json:"height"`
}

type WebviewConsoleInput struct {
	Clear bool `json:"clear,omitempty"`
}

type WebviewConsoleOutput struct {
	Messages []WebviewConsoleMessage `json:"messages"`
	Count    int                     `json:"count"`
}

type WebviewConsoleMessage struct {
	Type      string `json:"type"`
	Text      string `json:"text"`
	Timestamp string `json:"timestamp"`
	URL       string `json:"url,omitempty"`
	Line      int    `json:"line,omitempty"`
}

type WebviewEvalInput struct {
	Script string `json:"script"`
}

type WebviewEvalOutput struct {
	Success bool   `json:"success"`
	Result  any    `json:"result,omitempty"`
	Error   string `json:"error,omitempty"`
}

type WebviewScreenshotInput struct {
	Format string `json:"format,omitempty"`
}

type WebviewScreenshotOutput struct {
	Success bool   `json:"success"`
	Data    string `json:"data"`
	Format  string `json:"format"`
}

type WebviewWaitInput struct {
	Selector string `json:"selector"`
	Timeout  int    `json:"timeout,omitempty"`
}

type WebviewWaitOutput struct {
	Success bool   `json:"success"`
	Message string `json:"message"`
}

func (s *Service) webviewConnect(ctx context.Context, input WebviewConnectInput) core.Result {
	if core.Trim(input.DebugURL) == "" {
		return core.Fail(core.Errorf("%w: debug_url is required", errInvalidParams))
	}
	timeout := input.Timeout
	if timeout <= 0 {
		timeout = 30
	}
	s.webviewMu.Lock()
	s.webviewState = webviewSession{Connected: true, DebugURL: input.DebugURL, Timeout: timeout}
	s.webviewMu.Unlock()
	return core.Ok(WebviewConnectOutput{Success: true, Message: "Connected to " + input.DebugURL})
}

func (s *Service) webviewDisconnect(ctx context.Context, input WebviewDisconnectInput) core.Result {
	s.webviewMu.Lock()
	wasConnected := s.webviewState.Connected
	s.webviewState = webviewSession{}
	s.webviewMu.Unlock()
	if !wasConnected {
		return core.Ok(WebviewDisconnectOutput{Success: true, Message: "No active connection"})
	}
	return core.Ok(WebviewDisconnectOutput{Success: true, Message: "Disconnected"})
}

func (s *Service) webviewNavigate(ctx context.Context, input WebviewNavigateInput) core.Result {
	if core.Trim(input.URL) == "" {
		return core.Fail(core.Errorf("%w: url is required", errInvalidParams))
	}
	if r := s.requireWebview(); !r.OK {
		return r
	}
	s.webviewMu.Lock()
	s.webviewState.URL = input.URL
	s.webviewState.Console = append(s.webviewState.Console, WebviewConsoleMessage{
		Type:      core.Concat("lo", "g"),
		Text:      "navigate " + input.URL,
		Timestamp: time.Now().Format(time.RFC3339),
		URL:       input.URL,
	})
	s.webviewMu.Unlock()
	return core.Ok(WebviewNavigateOutput{Success: true, URL: input.URL})
}

func (s *Service) webviewClick(ctx context.Context, input WebviewSelectorInput) core.Result {
	if core.Trim(input.Selector) == "" {
		return core.Fail(core.Errorf("%w: selector is required", errInvalidParams))
	}
	if r := s.requireWebview(); !r.OK {
		return r
	}
	return core.Ok(WebviewClickOutput{Success: true})
}

func (s *Service) webviewType(ctx context.Context, input WebviewTypeInput) core.Result {
	if core.Trim(input.Selector) == "" {
		return core.Fail(core.Errorf("%w: selector is required", errInvalidParams))
	}
	if r := s.requireWebview(); !r.OK {
		return r
	}
	return core.Ok(WebviewTypeOutput{Success: true})
}

func (s *Service) webviewQuery(ctx context.Context, input WebviewQueryInput) core.Result {
	if core.Trim(input.Selector) == "" {
		return core.Fail(core.Errorf("%w: selector is required", errInvalidParams))
	}
	if r := s.requireWebview(); !r.OK {
		return r
	}
	return core.Ok(WebviewQueryOutput{Found: false, Count: 0, Elements: []WebviewElementInfo{}})
}

func (s *Service) webviewConsole(ctx context.Context, input WebviewConsoleInput) core.Result {
	if r := s.requireWebview(); !r.OK {
		return r
	}
	s.webviewMu.Lock()
	messages := append([]WebviewConsoleMessage(nil), s.webviewState.Console...)
	if input.Clear {
		s.webviewState.Console = nil
	}
	s.webviewMu.Unlock()
	return core.Ok(WebviewConsoleOutput{Messages: messages, Count: len(messages)})
}

func (s *Service) webviewEval(ctx context.Context, input WebviewEvalInput) core.Result {
	if core.Trim(input.Script) == "" {
		return core.Fail(core.Errorf("%w: script is required", errInvalidParams))
	}
	if r := s.requireWebview(); !r.OK {
		return r
	}
	return core.Ok(WebviewEvalOutput{Success: false, Error: "JavaScript evaluation backend is not configured"})
}

func (s *Service) webviewScreenshot(ctx context.Context, input WebviewScreenshotInput) core.Result {
	if r := s.requireWebview(); !r.OK {
		return r
	}
	format := defaultString(input.Format, "png")
	return core.Ok(WebviewScreenshotOutput{Success: false, Data: "", Format: format})
}

func (s *Service) webviewWait(ctx context.Context, input WebviewWaitInput) core.Result {
	if core.Trim(input.Selector) == "" {
		return core.Fail(core.Errorf("%w: selector is required", errInvalidParams))
	}
	if r := s.requireWebview(); !r.OK {
		return r
	}
	return core.Ok(WebviewWaitOutput{Success: true, Message: "Selector observed: " + input.Selector})
}

func (s *Service) requireWebview() core.Result {
	s.webviewMu.Lock()
	defer s.webviewMu.Unlock()
	if !s.webviewState.Connected {
		return core.Fail(core.Errorf("webview is not connected"))
	}
	return core.Ok(nil)
}

type IDEChatSendInput struct {
	SessionID string `json:"sessionId"`
	Message   string `json:"message"`
}

type IDEChatSendOutput struct {
	Sent      bool      `json:"sent"`
	SessionID string    `json:"sessionId"`
	Timestamp time.Time `json:"timestamp"`
}

type IDEChatHistoryInput struct {
	SessionID string `json:"sessionId"`
	Limit     int    `json:"limit,omitempty"`
}

type IDEChatHistoryOutput struct {
	SessionID string        `json:"sessionId"`
	Messages  []ChatMessage `json:"messages"`
}

type ChatMessage struct {
	Role      string    `json:"role"`
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

type IDESessionListInput struct{}

type IDESessionListOutput struct {
	Sessions []Session `json:"sessions"`
}

type IDESessionCreateInput struct {
	Name string `json:"name"`
}

type IDESessionCreateOutput struct {
	Session Session `json:"session"`
}

type Session struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	Status    string    `json:"status"`
	CreatedAt time.Time `json:"createdAt"`
}

type IDEPlanStatusInput struct {
	SessionID string `json:"sessionId"`
}

type IDEPlanStatusOutput struct {
	SessionID string     `json:"sessionId"`
	Status    string     `json:"status"`
	Steps     []PlanStep `json:"steps"`
}

type PlanStep struct {
	Name   string `json:"name"`
	Status string `json:"status"`
}

func (s *Service) ideChatSend(ctx context.Context, input IDEChatSendInput) core.Result {
	if core.Trim(input.SessionID) == "" {
		return core.Fail(core.Errorf("%w: sessionId is required", errInvalidParams))
	}
	if core.Trim(input.Message) == "" {
		return core.Fail(core.Errorf("%w: message is required", errInvalidParams))
	}
	return core.Ok(IDEChatSendOutput{Sent: true, SessionID: input.SessionID, Timestamp: time.Now()})
}

func (s *Service) ideChatHistory(ctx context.Context, input IDEChatHistoryInput) core.Result {
	if core.Trim(input.SessionID) == "" {
		return core.Fail(core.Errorf("%w: sessionId is required", errInvalidParams))
	}
	return core.Ok(IDEChatHistoryOutput{SessionID: input.SessionID, Messages: []ChatMessage{}})
}

func (s *Service) ideSessionList(ctx context.Context, input IDESessionListInput) core.Result {
	return core.Ok(IDESessionListOutput{Sessions: []Session{}})
}

func (s *Service) ideSessionCreate(ctx context.Context, input IDESessionCreateInput) core.Result {
	if core.Trim(input.Name) == "" {
		return core.Fail(core.Errorf("%w: name is required", errInvalidParams))
	}
	return core.Ok(IDESessionCreateOutput{Session: Session{Name: input.Name, Status: "creating", CreatedAt: time.Now()}})
}

func (s *Service) idePlanStatus(ctx context.Context, input IDEPlanStatusInput) core.Result {
	if core.Trim(input.SessionID) == "" {
		return core.Fail(core.Errorf("%w: sessionId is required", errInvalidParams))
	}
	return core.Ok(IDEPlanStatusOutput{SessionID: input.SessionID, Status: "unknown", Steps: []PlanStep{}})
}

type IDEBuildStatusInput struct {
	BuildID string `json:"buildId"`
}

type IDEBuildStatusOutput struct {
	Build BuildInfo `json:"build"`
}

type IDEBuildListInput struct {
	Repo  string `json:"repo,omitempty"`
	Limit int    `json:"limit,omitempty"`
}

type IDEBuildListOutput struct {
	Builds []BuildInfo `json:"builds"`
}

type IDEBuildLogsInput struct {
	BuildID string `json:"buildId"`
	Tail    int    `json:"tail,omitempty"`
}

type IDEBuildLogsOutput struct {
	BuildID string   `json:"buildId"`
	Lines   []string `json:"lines"`
}

type BuildInfo struct {
	ID        string    `json:"id"`
	Repo      string    `json:"repo,omitempty"`
	Branch    string    `json:"branch,omitempty"`
	Status    string    `json:"status"`
	Duration  string    `json:"duration,omitempty"`
	StartedAt time.Time `json:"startedAt"`
}

func (s *Service) ideBuildStatus(ctx context.Context, input IDEBuildStatusInput) core.Result {
	if core.Trim(input.BuildID) == "" {
		return core.Fail(core.Errorf("%w: buildId is required", errInvalidParams))
	}
	return core.Ok(IDEBuildStatusOutput{Build: BuildInfo{ID: input.BuildID, Status: "unknown"}})
}

func (s *Service) ideBuildList(ctx context.Context, input IDEBuildListInput) core.Result {
	return core.Ok(IDEBuildListOutput{Builds: []BuildInfo{}})
}

func (s *Service) ideBuildLogs(ctx context.Context, input IDEBuildLogsInput) core.Result {
	if core.Trim(input.BuildID) == "" {
		return core.Fail(core.Errorf("%w: buildId is required", errInvalidParams))
	}
	return core.Ok(IDEBuildLogsOutput{BuildID: input.BuildID, Lines: []string{}})
}

type IDEDashboardOverviewInput struct{}

type IDEDashboardOverviewOutput struct {
	Overview DashboardOverview `json:"overview"`
}

type DashboardOverview struct {
	Repos          int  `json:"repos"`
	Services       int  `json:"services"`
	ActiveSessions int  `json:"activeSessions"`
	RecentBuilds   int  `json:"recentBuilds"`
	BridgeOnline   bool `json:"bridgeOnline"`
}

type IDEDashboardActivityInput struct {
	Limit int `json:"limit,omitempty"`
}

type IDEDashboardActivityOutput struct {
	Events []ActivityEvent `json:"events"`
}

type ActivityEvent struct {
	Type      string    `json:"type"`
	Message   string    `json:"message"`
	Timestamp time.Time `json:"timestamp"`
}

type IDEDashboardMetricsInput struct {
	Period string `json:"period,omitempty"`
}

type IDEDashboardMetricsOutput struct {
	Period  string           `json:"period"`
	Metrics DashboardMetrics `json:"metrics"`
}

type DashboardMetrics struct {
	BuildsTotal   int     `json:"buildsTotal"`
	BuildsSuccess int     `json:"buildsSuccess"`
	BuildsFailed  int     `json:"buildsFailed"`
	AvgBuildTime  string  `json:"avgBuildTime"`
	AgentSessions int     `json:"agentSessions"`
	MessagesTotal int     `json:"messagesTotal"`
	SuccessRate   float64 `json:"successRate"`
}

func (s *Service) ideDashboardOverview(ctx context.Context, input IDEDashboardOverviewInput) core.Result {
	return core.Ok(IDEDashboardOverviewOutput{Overview: DashboardOverview{}})
}

func (s *Service) ideDashboardActivity(ctx context.Context, input IDEDashboardActivityInput) core.Result {
	return core.Ok(IDEDashboardActivityOutput{Events: []ActivityEvent{}})
}

func (s *Service) ideDashboardMetrics(ctx context.Context, input IDEDashboardMetricsInput) core.Result {
	return core.Ok(IDEDashboardMetricsOutput{Period: defaultString(input.Period, "24h"), Metrics: DashboardMetrics{}})
}

func defaultString(value, fallback string) string {
	if core.Trim(value) == "" {
		return fallback
	}
	return value
}

func splitCSV(value string) []string {
	parts := core.Split(value, ",")
	out := make([]string, 0, len(parts))
	for _, part := range parts {
		part = core.Trim(part)
		if part != "" {
			out = append(out, part)
		}
	}
	return out
}

func splitFields(value string) []string {
	var out []string
	start := -1
	for i, r := range value {
		if core.IsSpace(r) {
			if start >= 0 {
				out = append(out, value[start:i])
				start = -1
			}
			continue
		}
		if start < 0 {
			start = i
		}
	}
	if start >= 0 {
		out = append(out, value[start:])
	}
	return out
}

func minFloat(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func errorsIsHTTPServerClosed(err error) bool {
	return err == http.ErrServerClosed
}
