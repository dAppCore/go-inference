// SPDX-License-Identifier: EUPL-1.2

// Package driver orchestrates the model driver's lifecycle for lthn-ai. It
// turns a (model, profile, runtime) request into a supervised driver process
// (lthn-mlx / lthn-cuda / lthn-amd) via go-process, gates "live" on the driver
// answering /v1/health, restarts it on a crash, and tracks what is served so
// status/stop have a model-semantic view over the generic /api/process surface.
// lthn-ai is the host half of the LEM Runtime split; this package is where it
// manages the driver half.
//
// The driver stays CLI-instantiated — driver kernels (MLX / ROCm / CUDA) init
// at the process boundary. This package decides only WHICH driver runs WHICH
// (model, profile); it never loads weights itself.
//
// Usage example:
//
//	svc := driver.NewService(procSvc)
//	r := svc.Serve(driver.ServeRequest{Runtime: "mlx"}) // model-less start
//	if r.OK {
//	    served := r.Value.(driver.Served)
//	    _ = served.Addr
//	}
package driver

import (
	// AX-6: io/fs.DirEntry is the structural element type core.ReadDir returns.
	"io/fs"
	// AX-6: net/http is the structural client boundary for the driver readiness probe.
	"net/http"
	"sync"
	"time"

	core "dappco.re/go"
	coreprocess "dappco.re/go/process"
	ratelimit "dappco.re/go/ratelimit"
)

// Driver runtimes — each a sibling binary of lthn-ai in the LEM Runtime split.
const (
	// RuntimeMLX is the Apple-silicon MLX driver runtime.
	RuntimeMLX = "mlx"
	// RuntimeCUDA is the NVIDIA CUDA driver runtime.
	RuntimeCUDA = "cuda"
	// RuntimeAMD is the AMD ROCm driver runtime.
	RuntimeAMD = "amd"
)

// driverGracePeriod is the SIGTERM→SIGKILL window when stopping a driver, so an
// in-flight generation gets a chance to drain before the hard kill.
const driverGracePeriod = 10 * time.Second

// Readiness + crash-restart policy.
var (
	// driverReadyTimeout bounds how long Serve waits for the driver to answer
	// /v1/health after spawn. The driver eager-binds its listener before loading
	// weights, so readiness here means "accepting requests" — the first inference
	// triggers the lazy model load — and is reached well inside this window.
	//
	// A var (not const) purely so hermetic tests can shrink it to exercise the
	// spawned-but-never-ready path without a real 30s wait; production code
	// never assigns it, so live behaviour is unchanged.
	driverReadyTimeout = 30 * time.Second
	// readyPollInterval is the gap between /v1/health probes during the wait.
	// Also a var for the same test-only reason as driverReadyTimeout.
	readyPollInterval = 200 * time.Millisecond
)

const (
	// maxRestarts is how many crash-restarts a runtime gets within restartWindow
	// before the host gives up and leaves it down (restart-storm guard).
	maxRestarts = 3
	// restartWindow is the sliding window over which maxRestarts is counted.
	restartWindow = 60 * time.Second
)

// runtimeBinary maps a driver runtime to the binary that serves it.
var runtimeBinary = map[string]string{
	RuntimeMLX:  "lthn-mlx",
	RuntimeCUDA: "lthn-cuda",
	RuntimeAMD:  "lthn-amd",
}

// runtimeDefaultAddr is the loopback address a runtime's driver binds when the
// serve request doesn't pin one. mlx uses Lethean's own 36911 — an Ollama
// install on 11434 never collides (cuda/amd keep their go-rocm defaults
// until that lane makes the same move).
var runtimeDefaultAddr = map[string]string{
	RuntimeMLX:  "127.0.0.1:36911",
	RuntimeCUDA: "127.0.0.1:11435",
	RuntimeAMD:  "127.0.0.1:11436",
}

// ServeRequest asks the host to make a model live on a driver runtime.
type ServeRequest struct {
	// Model is the weights path or name passed through to the driver's --model.
	// Empty starts the driver model-less (binds immediately, load later via the
	// driver's admin reload) — the crew/fleet boot path.
	Model string `json:"model"`
	// Profile is a driver tuning-profile JSON path passed to --profile. Empty
	// lets the driver auto-discover one for this machine + model.
	Profile string `json:"profile"`
	// Runtime selects the driver: mlx | cuda | amd. Empty defaults to mlx.
	Runtime string `json:"runtime"`
	// Addr is the driver's listen address. Empty uses the runtime default.
	Addr string `json:"addr"`
	// Context overrides the model context length (--context). Zero uses the
	// model's own default.
	Context int `json:"context"`
	// NoAutoProfile skips the driver's profile auto-discovery (--no-auto-profile).
	NoAutoProfile bool `json:"noAutoProfile"`
}

// Served is a snapshot of one driver the host is supervising.
type Served struct {
	Runtime   string `json:"runtime"`
	Model     string `json:"model"`
	Profile   string `json:"profile,omitempty"`
	Addr      string `json:"addr"`
	ProcessID string `json:"processId"`
	Running   bool   `json:"running"`
	// Ready is true once the driver answered /v1/health — accepting requests.
	Ready bool `json:"ready"`
}

// Catalogue is what the host can serve — model weights and the serve profiles
// bound to them. Per the LEM Runtime layout a model (weights, one) carries N+1
// profiles.
type Catalogue struct {
	Models   []string `json:"models"`
	Profiles []string `json:"profiles"`
}

// Service supervises driver processes for one lthn-ai host. It holds the
// go-process Service it spawns through and tracks the active driver per runtime,
// so a second serve on the same runtime is a clear conflict rather than a silent
// second process (hot-swap lands in a later pass).
type Service struct {
	proc       *coreprocess.Service
	limiter    *ratelimit.RateLimiter
	mu         sync.Mutex
	served     map[string]*Served     // runtime → active driver
	everReady  map[string]bool        // runtime → driver answered /v1/health at least once
	restartLog map[string][]time.Time // runtime → recent crash-restart timestamps
}

// NewService binds a driver orchestrator to the go-process Service that spawns
// and supervises its children, plus the rate limiter that gates the inference
// path (nil disables the gate). It subscribes to process lifecycle events so a
// crashed driver is restarted on its last-good (model, profile).
//
//	svc := driver.NewService(procSvc, limiter)
func NewService(proc *coreprocess.Service, limiter *ratelimit.RateLimiter) *Service {
	s := &Service{
		proc:       proc,
		limiter:    limiter,
		served:     make(map[string]*Served),
		everReady:  make(map[string]bool),
		restartLog: make(map[string][]time.Time),
	}
	// A driver that exits while still tracked is a crash → restart. A driver
	// stopped deliberately is dropped from the tracked set before the kill, so
	// its exit is ignored here.
	if c := proc.Core(); c != nil {
		c.RegisterAction(s.onProcessEvent)
	}
	return s
}

// Serve cold-starts a driver for the requested (model, profile) on the given
// runtime, waits for it to answer /v1/health, and returns the Served snapshot.
// Refuses if that runtime is already serving — stop it first (single driver per
// runtime until hot-swap lands).
//
//	r := svc.Serve(driver.ServeRequest{Runtime: "mlx", Model: "/path/to/weights"})
func (s *Service) Serve(req ServeRequest) core.Result {
	runtime := req.Runtime
	if runtime == "" {
		runtime = RuntimeMLX
	}
	bin, ok := runtimeBinary[runtime]
	if !ok {
		return core.Fail(core.E("driver.Serve", core.Sprintf("unknown runtime %q (want mlx|cuda|amd)", runtime), nil))
	}
	addr := req.Addr
	if addr == "" {
		addr = runtimeDefaultAddr[runtime]
	}

	// Hot-swap: an already-serving runtime takes a model change in place of a
	// "stop first" refusal. Same model → no-op (return the current Served); a
	// different model → drain the old driver, then cold-start the new below.
	if res := s.swapOrPass(runtime, req); res != nil {
		return *res
	}

	r := s.spawn(runtime, bin, addr, req)
	if !r.OK {
		return r
	}
	proc := r.Value.(*coreprocess.Process)

	// Gate "live" on the driver answering /v1/health — polled outside the lock so
	// a slow cold start doesn't block status/stop/other serves.
	ready, reason := waitDriverReady(addr, driverReadyTimeout)

	s.mu.Lock()
	if cur := s.served[runtime]; cur != nil && cur.ProcessID == proc.ID {
		cur.Ready = ready
		if ready {
			s.everReady[runtime] = true
		}
	}
	s.mu.Unlock()

	if !ready {
		return core.Fail(core.E("driver.Serve", core.Sprintf("driver %q started but not ready at %s: %s", runtime, addr, reason), nil))
	}
	// Remember this choice so the next boot restores the operator's last model.
	// Model-less serves persist nothing — there's nothing meaningful to restore.
	persistServe(persistedServe{Runtime: runtime, Model: req.Model, Profile: req.Profile})
	return core.Ok(Served{
		Runtime: runtime, Model: req.Model, Profile: req.Profile,
		Addr: addr, ProcessID: proc.ID, Running: true, Ready: true,
	})
}

// persistedServe is the last-served (model, profile) the host remembers across
// restarts so a boot auto-serve can restore the operator's last choice.
type persistedServe struct {
	Runtime string `json:"runtime"`
	Model   string `json:"model"`
	Profile string `json:"profile"`
}

// servePersistPath is where the last-served choice is recorded —
// ~/Lethean/data/lthn-ai-serve.json. Empty when the home dir can't resolve.
func servePersistPath() string {
	home := core.UserHomeDir()
	if !home.OK {
		return ""
	}
	return core.PathJoin(home.Value.(string), "Lethean", "data", "lthn-ai-serve.json")
}

// persistServe records the last successful serve. Best-effort: a write failure
// must never break serving, and a model-less serve is not recorded (nothing to
// restore).
func persistServe(p persistedServe) {
	if p.Model == "" {
		return
	}
	path := servePersistPath()
	if path == "" {
		return
	}
	_ = core.MkdirAll(core.PathDir(path), 0o755)
	_ = core.WriteFile(path, []byte(core.JSONMarshalString(p)), 0o644)
}

// LastServed returns the last successfully-served (model, profile), or ok=false
// when nothing is persisted — the boot auto-serve uses it to restore the
// operator's last model when no explicit model env is set.
//
//	if req, ok := svc.LastServed(); ok { _ = svc.Serve(req) }
func (s *Service) LastServed() (ServeRequest, bool) {
	path := servePersistPath()
	if path == "" {
		return ServeRequest{}, false
	}
	r := core.ReadFile(path)
	if !r.OK {
		return ServeRequest{}, false
	}
	data, ok := r.Value.([]byte)
	if !ok {
		return ServeRequest{}, false
	}
	var p persistedServe
	if jr := core.JSONUnmarshalString(string(data), &p); !jr.OK || p.Model == "" {
		return ServeRequest{}, false
	}
	return ServeRequest{Runtime: p.Runtime, Model: p.Model, Profile: p.Profile}, true
}

// spawn claims the runtime slot, resolves the driver binary, and starts it under
// the lock — returning the live *coreprocess.Process. The readiness wait happens
// in Serve, outside the lock.
func (s *Service) spawn(runtime, bin, addr string, req ServeRequest) core.Result {
	s.mu.Lock()
	defer s.mu.Unlock()

	if cur := s.served[runtime]; cur != nil && s.running(cur.ProcessID) {
		return core.Fail(core.E("driver.Serve", core.Sprintf("runtime %q already serving %q — stop it first", runtime, cur.Model), nil))
	}

	prog := &coreprocess.Program{Name: resolveDriverBinary(bin)}
	if r := prog.Find(); !r.OK {
		cause, _ := r.Value.(error)
		return core.Fail(core.E("driver.Serve", core.Sprintf("driver %q not found (CORE_AI_DRIVER_DIR, exe dir, ~/Lethean/bin, PATH)", bin), cause))
	}

	r := s.proc.StartWithOptions(core.Background(), coreprocess.RunOptions{
		Command:     prog.Path,
		Args:        serveArgs(req, addr),
		Detach:      true,
		KillGroup:   true,
		GracePeriod: driverGracePeriod,
	})
	if !r.OK {
		return r
	}
	proc, ok := r.Value.(*coreprocess.Process)
	if !ok {
		return core.Fail(core.E("driver.Serve", "process service returned unexpected type", nil))
	}

	s.served[runtime] = &Served{
		Runtime:   runtime,
		Model:     req.Model,
		Profile:   req.Profile,
		Addr:      addr,
		ProcessID: proc.ID,
		Running:   true,
	}
	return core.Ok(proc)
}

// swapOrPass handles a Serve against an already-serving runtime. It returns a
// non-nil Result only for the same-model no-op (the caller returns it as-is);
// nil means "proceed to cold-start" — either nothing was serving, or a
// different model was draining and has now exited so the address is free.
//
// The old driver is dropped from the tracked set BEFORE the kill, so its exit
// reads as deliberate (handleExit won't restart it); Wait then blocks until it
// exits so the listen address frees before the replacement binds.
func (s *Service) swapOrPass(runtime string, req ServeRequest) *core.Result {
	s.mu.Lock()
	cur := s.served[runtime]
	if cur == nil || !s.running(cur.ProcessID) {
		s.mu.Unlock()
		return nil
	}
	if cur.Model == req.Model {
		snap := *cur
		s.mu.Unlock()
		r := core.Ok(snap)
		return &r
	}
	pid := cur.ProcessID
	delete(s.served, runtime)
	delete(s.everReady, runtime)
	delete(s.restartLog, runtime)
	s.mu.Unlock()

	if r := s.proc.Kill(pid); !r.OK {
		core.Print(core.Stderr(), "driver.swapOrPass: kill old %s: %s\n", pid, r.Error())
	}
	_ = s.proc.Wait(pid) // block until the old listener releases the address
	return nil
}

// Stop terminates the driver serving the given runtime (default mlx) and drops
// it from the served set BEFORE the kill, so the resulting process exit is read
// as deliberate (no restart). GracePeriod gives in-flight work the SIGTERM drain
// window before the hard kill.
//
//	r := svc.Stop("mlx")
func (s *Service) Stop(runtime string) core.Result {
	if runtime == "" {
		runtime = RuntimeMLX
	}
	s.mu.Lock()
	sv := s.served[runtime]
	if sv == nil {
		s.mu.Unlock()
		return core.Fail(core.E("driver.Stop", core.Sprintf("no driver serving runtime %q", runtime), nil))
	}
	processID := sv.ProcessID
	delete(s.served, runtime)
	delete(s.restartLog, runtime)
	delete(s.everReady, runtime)
	s.mu.Unlock()

	if r := s.proc.Kill(processID); !r.OK {
		return r
	}
	return core.Ok(runtime)
}

// Status returns a snapshot of every driver the host is supervising, each
// Running flag refreshed against the live process state.
//
//	for _, sv := range svc.Status() { _ = sv.Addr }
func (s *Service) Status() []Served {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make([]Served, 0, len(s.served))
	for _, sv := range s.served {
		snap := *sv
		snap.Running = s.running(sv.ProcessID)
		if !snap.Running {
			snap.Ready = false
		}
		out = append(out, snap)
	}
	return out
}

// Models lists what the host can serve: the model weights under
// ~/Lethean/data/models and the serve profiles under ~/Lethean/conf/models.
//
//	r := svc.Models()
//	if r.OK { cat := r.Value.(driver.Catalogue); _ = cat.Models }
func (s *Service) Models() core.Result {
	home := core.UserHomeDir()
	if !home.OK {
		return home
	}
	root := home.Value.(string)
	return core.Ok(Catalogue{
		Models:   listNames(core.PathJoin(root, "Lethean", "data", "models")),
		Profiles: listNames(core.PathJoin(root, "Lethean", "conf", "models")),
	})
}

// onProcessEvent receives the conclave's process lifecycle broadcasts. A tracked
// driver exiting is a crash (deliberate stops are untracked first) → restart.
func (s *Service) onProcessEvent(_ *core.Core, msg core.Message) core.Result {
	if exited, ok := msg.(coreprocess.ActionProcessExited); ok {
		s.handleExit(exited.ID)
	}
	return core.Ok(nil)
}

// handleExit restarts a crashed driver on its last-good (model, profile), within
// the restart-storm guard. Only drivers that became ready at least once are
// restarted — one that never came up (e.g. a bad model path) is left down so the
// operator sees the Serve error instead of a restart loop.
func (s *Service) handleExit(processID string) {
	s.mu.Lock()
	runtime, sv := s.trackedByPID(processID)
	if sv == nil {
		s.mu.Unlock()
		return // foreign process, or stopped deliberately (already dropped)
	}
	sv.Running = false
	sv.Ready = false
	last := ServeRequest{Model: sv.Model, Profile: sv.Profile, Runtime: runtime, Addr: sv.Addr}
	wasReady := s.everReady[runtime]
	restart := wasReady && s.allowRestart(runtime)
	s.mu.Unlock()

	switch {
	case restart:
		core.Print(core.Stderr(), "driver %q exited — restarting on %q\n", runtime, last.Model)
		go func() { _ = s.Serve(last) }()
	case wasReady:
		core.Print(core.Stderr(), "driver %q exited — restart cap (%d/%s) reached, leaving down\n", runtime, maxRestarts, restartWindow)
	}
}

// trackedByPID returns the runtime + Served owning a process id, or "", nil.
// Caller holds s.mu.
func (s *Service) trackedByPID(processID string) (string, *Served) {
	for rt, sv := range s.served {
		if sv.ProcessID == processID {
			return rt, sv
		}
	}
	return "", nil
}

// allowRestart prunes the runtime's restart log to restartWindow and reports
// whether another restart is within the maxRestarts budget, recording it when
// allowed. Caller holds s.mu.
func (s *Service) allowRestart(runtime string) bool {
	cutoff := time.Now().Add(-restartWindow)
	recent := s.restartLog[runtime][:0]
	for _, t := range s.restartLog[runtime] {
		if t.After(cutoff) {
			recent = append(recent, t)
		}
	}
	if len(recent) >= maxRestarts {
		s.restartLog[runtime] = recent
		return false
	}
	s.restartLog[runtime] = append(recent, time.Now())
	return true
}

// running reports whether the tracked process is still alive.
func (s *Service) running(processID string) bool {
	r := s.proc.Get(processID)
	if !r.OK {
		return false
	}
	proc, ok := r.Value.(*coreprocess.Process)
	if !ok {
		return false
	}
	return proc.IsRunning()
}

// serveArgs builds the driver argv for the serve subcommand:
// `serve --addr <addr> [--model <path>] [--context N] [--profile P]
// [--no-auto-profile]`. An empty Model starts the driver model-less, a
// first-class driver mode.
func serveArgs(req ServeRequest, addr string) []string {
	args := []string{"serve", "--addr", addr}
	if req.Model != "" {
		args = append(args, "--model", req.Model)
	}
	if req.Context > 0 {
		args = append(args, "--context", core.Sprintf("%d", req.Context))
	}
	if req.Profile != "" {
		args = append(args, "--profile", req.Profile)
	}
	if req.NoAutoProfile {
		args = append(args, "--no-auto-profile")
	}
	return args
}

// resolveDriverBinary finds a driver binary the way the desktop crew resolves
// its sidecars, so a crew-spawned or bundled lthn-ai agrees on which binary
// runs: an explicit override dir (CORE_AI_DRIVER_DIR) → the lthn-ai executable's
// own directory (a packaged .app's Contents/MacOS, or the crew's build/.../bin —
// the driver is a sibling) → the per-user ~/Lethean/bin install → PATH. The
// PATH fallback also covers the bundle (Contents/MacOS is on PATH).
func resolveDriverBinary(name string) string {
	var dirs []string
	if override := core.Trim(core.Getenv("CORE_AI_DRIVER_DIR")); override != "" {
		dirs = append(dirs, override)
	}
	if args := core.Args(); len(args) > 0 && args[0] != "" {
		dirs = append(dirs, core.PathDir(args[0]))
	}
	if home := core.UserHomeDir(); home.OK {
		dirs = append(dirs, core.PathJoin(home.Value.(string), "Lethean", "bin"))
	}
	for _, d := range dirs {
		cand := core.PathJoin(d, name)
		if core.Stat(cand).OK {
			return cand
		}
	}
	return name // let go-process resolve via PATH
}

// waitDriverReady polls the driver's /v1/health until it answers 200 or the
// timeout elapses, returning the last failure reason on timeout. The driver
// binds its listener before loading weights, so a 200 here means "accepting
// requests"; the first inference call triggers the lazy model load.
func waitDriverReady(addr string, timeout time.Duration) (bool, string) {
	url := "http://" + addr + "/v1/health"
	deadline := time.Now().Add(timeout)
	client := &http.Client{Timeout: 2 * time.Second}
	var last string
	for time.Now().Before(deadline) {
		resp, err := client.Get(url)
		if err == nil {
			_ = resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return true, ""
			}
			last = resp.Status
		} else {
			last = err.Error()
		}
		time.Sleep(readyPollInterval)
	}
	if last == "" {
		last = "readiness timed out"
	}
	return false, last
}

// listNames returns the visible entry names in dir (dotfiles skipped), or nil
// when the directory is absent or unreadable — an empty catalogue is a valid
// answer, never an error.
func listNames(dir string) []string {
	r := core.ReadDir(core.DirFS(dir), ".")
	if !r.OK {
		return nil
	}
	entries, ok := r.Value.([]fs.DirEntry)
	if !ok {
		return nil
	}
	names := make([]string, 0, len(entries))
	for _, e := range entries {
		name := e.Name()
		if core.HasPrefix(name, ".") {
			continue
		}
		names = append(names, name)
	}
	return names
}
