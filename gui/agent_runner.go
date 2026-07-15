package main

import (
	"context"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference/gui/internal/lem"
	"github.com/wailsapp/wails/v3/pkg/application"
)

// AgentRunner wraps the scoring agent for desktop use.
// Provides start/stop/status for the tray and dashboard.
type AgentRunner struct {
	apiURL    string
	influxURL string
	influxDB  string
	m3Host    string
	baseModel string
	workDir   string

	mu      sync.RWMutex
	running bool
	task    string
	cancel  context.CancelFunc
}

// NewAgentRunner creates an AgentRunner.
func NewAgentRunner(apiURL, influxURL, influxDB, m3Host, baseModel, workDir string) *AgentRunner {
	return &AgentRunner{
		apiURL:    apiURL,
		influxURL: influxURL,
		influxDB:  influxDB,
		m3Host:    m3Host,
		baseModel: baseModel,
		workDir:   workDir,
	}
}

// ServiceName returns the Wails service name.
func (a *AgentRunner) ServiceName() string {
	return "AgentRunner"
}

// ServiceStartup is called when the Wails app starts.
func (a *AgentRunner) ServiceStartup(ctx context.Context, options application.ServiceOptions) core.Result {
	core.Print(core.Stderr(), "AgentRunner started\n")
	return core.Ok(nil)
}

// IsRunning returns whether the agent is currently running.
func (a *AgentRunner) IsRunning() bool {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.running
}

// CurrentTask returns the current task description.
func (a *AgentRunner) CurrentTask() string {
	a.mu.RLock()
	defer a.mu.RUnlock()
	return a.task
}

// Start begins the scoring agent in a background goroutine.
func (a *AgentRunner) Start() core.Result {
	a.mu.Lock()
	if a.running {
		a.mu.Unlock()
		return core.Ok(nil)
	}

	ctx, cancel := context.WithCancel(context.Background())
	a.cancel = cancel
	a.running = true
	a.task = "Starting..."
	a.mu.Unlock()

	go func() {
		defer func() {
			a.mu.Lock()
			a.running = false
			a.task = ""
			a.cancel = nil
			a.mu.Unlock()
		}()

		core.Print(core.Stderr(), "Scoring agent started via desktop\n")

		// Use the same RunAgent function from pkg/lem.
		// Build args matching the CLI flags.
		args := []string{
			"--api-url", a.apiURL,
			"--influx", a.influxURL,
			"--influx-db", a.influxDB,
			"--m3-host", a.m3Host,
			"--base-model", a.baseModel,
			"--work-dir", a.workDir,
		}

		// Run in the background — RunAgent blocks until cancelled.
		// We use a goroutine-safe wrapper here.
		_ = ctx // Agent doesn't support context cancellation yet.
		_ = args
		lem.RunAgent(args)
	}()

	return core.Ok(nil)
}

// Stop stops the scoring agent.
func (a *AgentRunner) Stop() {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.cancel != nil {
		a.cancel()
	}
	a.running = false
	a.task = ""
	core.Print(core.Stderr(), "Scoring agent stopped via desktop\n")
}
