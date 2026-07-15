// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"sync"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/gui/internal/serve"
	"dappco.re/go/inference/serving"
	"github.com/wailsapp/wails/v3/pkg/application"
)

// serveStatusPoll is how often the tray refreshes the serve daemon's status.
const serveStatusPoll = 5 * time.Second

// ServeService is the Wails service backing the tray's `lem serve` controls. It
// wraps internal/serve — the status client, the managed-process controller and
// model discovery — and mirrors DockerService's shape (Start/Stop/GetSnapshot
// plus a background statusLoop) so the tray drives it identically.
type ServeService struct {
	client    *serve.Client
	manager   *serve.Manager
	modelsDir string

	mu     sync.RWMutex
	status serve.Status
}

// ServeSnapshot is the tray-facing serve state. Up comes from the live status
// probe; Managed reports whether this app spawned the running daemon (versus one
// started externally by lem.sh or launchd, which shows Up but not Managed).
type ServeSnapshot struct {
	Up        bool   `json:"up"`
	Managed   bool   `json:"managed"`
	ModelPath string `json:"modelPath"`
	ModelName string `json:"modelName"`
	Runtime   string `json:"runtime"`
}

// NewServeService builds a serve service for a daemon on addr (e.g. ":36911"),
// spawning lemBinary and discovering models under modelsDir. An empty modelsDir
// defaults to serve.DefaultModelsDir(); an empty lemBinary resolves "lem" on PATH.
func NewServeService(addr, lemBinary, modelsDir string) *ServeService {
	if modelsDir == "" {
		modelsDir = serve.DefaultModelsDir()
	}
	return &ServeService{
		client:    serve.NewClient(statusURL(addr), readAdminToken),
		manager:   serve.NewManager(lemBinary, addr),
		modelsDir: modelsDir,
	}
}

// ServiceName returns the Wails service name.
func (s *ServeService) ServiceName() string { return "ServeService" }

// ServiceStartup starts the background status poll loop.
func (s *ServeService) ServiceStartup(ctx context.Context, options application.ServiceOptions) core.Result {
	core.Print(core.Stderr(), "ServeService started\n")
	go s.statusLoop(ctx)
	return core.Ok(nil)
}

// ServiceShutdown terminates a daemon this app spawned so quitting the tray does
// not leave an orphaned serve process. A daemon we did not spawn is untouched
// (Manager.Stop is a no-op there).
func (s *ServeService) ServiceShutdown() core.Result {
	return s.manager.Stop()
}

// Status probes the daemon and records the result. A client error (auth reject,
// transient failure) is folded to "down" for the tray — the status endpoint is
// the source of truth for up/down.
func (s *ServeService) Status(ctx context.Context) serve.Status {
	st, err := s.client.Status(ctx)
	if err != nil {
		st = serve.Status{Up: false}
	}
	s.mu.Lock()
	s.status = st
	s.mu.Unlock()
	return st
}

// GetSnapshot returns the last polled serve state for the tray.
func (s *ServeService) GetSnapshot() ServeSnapshot {
	s.mu.RLock()
	st := s.status
	s.mu.RUnlock()
	return ServeSnapshot{
		Up:        st.Up,
		Managed:   s.manager.Managed(),
		ModelPath: st.ModelPath,
		ModelName: core.PathBase(st.ModelPath),
		Runtime:   st.Runtime,
	}
}

// ListModels returns the discoverable models the picker offers (walked fresh so
// a just-downloaded model appears without a restart).
func (s *ServeService) ListModels() []serve.Model {
	return serve.DiscoverModels(s.modelsDir)
}

// Start spawns a managed `lem serve` with optional chat model, embeddings model
// and scheduler mode. Empty scheduler means off; supported enabled values are
// serial, batch and interleave, matching the lem serve flag.
func (s *ServeService) Start(modelPath, embedModelPath, scheduler string) core.Result {
	return s.manager.Start(modelPath, embedModelPath, scheduler)
}

// Stop terminates the managed serve daemon.
func (s *ServeService) Stop() core.Result {
	return s.manager.Stop()
}

// IsUp reports whether the last poll saw the daemon running.
func (s *ServeService) IsUp() bool {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.status.Up
}

func (s *ServeService) statusLoop(ctx context.Context) {
	s.Status(ctx)

	ticker := time.NewTicker(serveStatusPoll)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			s.Status(ctx)
		}
	}
}

// statusURL turns a serve listen address into the localhost admin base URL the
// client polls. A bare ":port" binds all interfaces, reached on 127.0.0.1; a
// host:port form is used as given.
func statusURL(addr string) string {
	if core.HasPrefix(addr, ":") {
		return "http://127.0.0.1" + addr
	}
	return "http://" + addr
}

// readAdminToken reads the serve admin Bearer token from its canonical file
// (~/Lethean/lem/admin.token), or "" when absent. Read fresh per poll so a token
// serve mints on first boot is picked up without restarting the tray.
func readAdminToken() string {
	return readTokenFile(serving.AdminTokenPath())
}

// readTokenFile reads and trims a Bearer token from path, returning "" for any
// read failure (a missing token file simply means serve has not run yet).
func readTokenFile(path string) string {
	r := core.ReadFile(path)
	if !r.OK {
		return ""
	}
	data, ok := r.Value.([]byte)
	if !ok {
		return ""
	}
	return core.Trim(string(data))
}
