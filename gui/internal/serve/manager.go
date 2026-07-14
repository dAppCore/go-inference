// SPDX-Licence-Identifier: EUPL-1.2

package serve

import (
	"sync"
	"syscall"
	"time"

	core "dappco.re/go"
	execabs "golang.org/x/sys/execabs"
)

// stopTimeout is how long Stop waits for a graceful SIGTERM exit before
// escalating to SIGKILL. serve drains in-flight requests on a 10s deadline, so
// this allows that to complete before a hard kill.
const stopTimeout = 12 * time.Second

// Manager spawns and terminates a single managed `lem serve` process. It owns
// only the process it started itself; a serve daemon launched elsewhere (the
// lem.sh harness, a launchd job) is observed through Client, never signalled
// here. A reaper goroutine watches the child so Managed() reflects reality — if
// serve exits on its own, the manager forgets it and Start can respawn.
type Manager struct {
	binary string
	addr   string

	mu   sync.Mutex
	cmd  *execabs.Cmd
	done chan struct{} // closed by the reaper when the current cmd exits
}

// NewManager builds a manager that runs `binary serve --addr addr [...]`. An
// empty binary defaults to "lem" (resolved on PATH at spawn time); an empty addr
// defaults to Lethean's own port.
//
//	m := serve.NewManager("", ":36911")
//	m.Start("/models/gemma-4-e2b-it-4bit", "/models/bge-small", "batch")
//	defer m.Stop()
func NewManager(binary, addr string) *Manager {
	if binary == "" {
		binary = "lem"
	}
	if addr == "" {
		addr = ":36911"
	}
	return &Manager{binary: binary, addr: addr}
}

// Addr returns the listen address the manager spawns serve on.
func (m *Manager) Addr() string { return m.addr }

// Start spawns `lem serve --addr <addr>` with optional chat model, embeddings
// model and scheduler flags. Empty option values leave their corresponding CLI
// flags absent; in particular an empty scheduler keeps scheduling off. Start is
// an idempotent no-op returning OK when this manager already owns a live process.
func (m *Manager) Start(modelPath, embedModelPath, scheduler string) core.Result {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.cmd != nil {
		return core.Ok(nil) // already managing a live process
	}

	// The desktop's webview is a different origin from the serve, so the
	// browser side of the app can only call it with CORS armed. Any-origin is
	// the right scope for a loopback serve the app itself manages.
	args := []string{"serve", "--addr", m.addr, "--cors", "*"}
	if modelPath != "" {
		args = append(args, "--model", modelPath)
	}
	if embedModelPath != "" {
		args = append(args, "--embed-model", embedModelPath)
	}
	if scheduler != "" {
		args = append(args, "--scheduler", scheduler)
	}
	cmd := execabs.Command(m.binary, args...)
	if err := cmd.Start(); err != nil {
		return core.Fail(core.Errorf("start %s serve: %w", m.binary, err))
	}

	done := make(chan struct{})
	m.cmd = cmd
	m.done = done
	// Reaper: Wait releases the child's resources and lets us detect an exit
	// (graceful stop, external kill, or crash) so Managed() cannot go stale.
	go func() {
		_ = cmd.Wait()
		m.mu.Lock()
		if m.cmd == cmd {
			m.cmd = nil
		}
		m.mu.Unlock()
		close(done)
	}()
	return core.Ok(nil)
}

// Stop terminates the managed process — SIGTERM for a graceful drain, escalating
// to SIGKILL if it outlives stopTimeout — and blocks until it has exited. It is
// a no-op returning OK when nothing is managed. The reaper (from Start) performs
// the Wait; Stop waits on its completion signal, so the child is never Waited
// twice.
func (m *Manager) Stop() core.Result {
	m.mu.Lock()
	cmd := m.cmd
	done := m.done
	m.mu.Unlock()

	if cmd == nil || cmd.Process == nil {
		return core.Ok(nil)
	}

	_ = cmd.Process.Signal(syscall.SIGTERM)
	if done == nil {
		return core.Ok(nil)
	}
	select {
	case <-done:
	case <-time.After(stopTimeout):
		_ = cmd.Process.Kill()
		<-done
	}
	return core.Ok(nil)
}

// Managed reports whether this manager currently owns a live serve process.
func (m *Manager) Managed() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.cmd != nil
}
