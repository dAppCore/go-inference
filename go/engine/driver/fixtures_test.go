// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sync"
	"testing"
	"time"

	core "dappco.re/go"
	"github.com/gin-gonic/gin"
)

// Shared fixtures for the driver package's hermetic test suite. No models, no
// network, no real lthn-mlx/lthn-cuda/lthn-amd binary — every "driver" the
// tests spawn is a tiny compiled sleeper reached via CORE_AI_DRIVER_DIR (the
// package's own binary-resolution seam), and every "engine" HTTP surface is
// an httptest.Server. Nothing here edits the real PATH globally — PATH is
// only ever overridden per-test via t.Setenv, which os/exec-style tooling
// (and t.Cleanup) restores automatically.

func init() {
	// Quiets gin's debug-mode banner across every *_test.go file in this
	// package — CreateTestContextOnly + the route-registration tests would
	// otherwise print noise on every run.
	gin.SetMode(gin.TestMode)
}

// fakeDriverSource is a compiled stand-in for lthn-mlx/lthn-cuda/lthn-amd. It
// ignores its argv (serve --addr ... --model ... — whatever serveArgs built)
// and stays alive until killed. Signals are deliberately not trapped: the
// tests kill the process, and an untrapped default-terminate is exactly the
// behaviour spawn/Stop/Status/crash-restart assert against.
//
// It is a COMPILED BINARY, not the "#!/bin/sh\nexec /bin/sleep 600" script it
// replaces, and that is a portability property rather than a preference:
//
//   - a shebang script is not executable on Windows at all, and there is no
//     /bin/sh to interpret it
//   - an extensionless file is not executable on Windows either, so the name
//     has to carry the platform's suffix
//   - a .cmd wrapper would satisfy both and introduce a worse problem —
//     killing cmd.exe does not reliably kill the child it spawned, so a hung
//     test would replace a failing one. A single process with no children
//     cannot leak one.
//
// The old script also had to invoke /bin/sleep by absolute path, because
// isolateDriverLookup empties PATH and a bare `sleep` exited 127 instantly,
// turning every spawn into a silent crash-restart storm. A self-contained
// binary has no such dependency to get wrong.
const fakeDriverSource = `package main

import "time"

func main() { time.Sleep(10 * time.Minute) }
`

var (
	stubMu    sync.Mutex
	stubCache = map[string]string{}

	// Captured at package initialisation, which runs before any test body.
	// isolateDriverLookup deliberately points PATH at an empty directory, so
	// by the time a test asks for the stub neither the go toolchain nor
	// anything it shells out to can be found any more. Resolving both here
	// keeps the build independent of whatever a test has done to PATH.
	goToolPath, goToolErr = exec.LookPath("go")
	realPath              = os.Getenv("PATH")
)

// buildStub compiles source into a standalone executable and returns its
// path, building each distinct source at most once per test run. Compiling
// once and copying is what keeps the cost to one `go build` per stub for the
// whole package.
func buildStub(t *testing.T, source string) string {
	t.Helper()
	stubMu.Lock()
	defer stubMu.Unlock()
	if built, ok := stubCache[source]; ok {
		return built
	}
	if goToolErr != nil {
		t.Fatalf("locate go toolchain: %v", goToolErr)
	}
	dir, err := os.MkdirTemp("", "driver-stub-build")
	if err != nil {
		t.Fatalf("stub build dir: %v", err)
	}
	file := filepath.Join(dir, "main.go")
	if err := os.WriteFile(file, []byte(source), 0o600); err != nil {
		t.Fatalf("write stub source: %v", err)
	}
	out := filepath.Join(dir, "stub"+driverExeSuffix)
	// GOWORK=off and GOFLAGS cleared: each stub is a standalone main with no
	// imports beyond the standard library, and must not inherit the caller's
	// workspace or build flags. PATH is restored explicitly — see realPath.
	build := exec.Command(goToolPath, "build", "-o", out, file)
	build.Env = append(os.Environ(), "GOWORK=off", "GOFLAGS=", "PATH="+realPath)
	if output, err := build.CombinedOutput(); err != nil {
		t.Fatalf("go build stub: %v: %s", err, output)
	}
	stubCache[source] = out
	return out
}

// writeStub copies a compiled stub into dir under name plus the platform's
// executable suffix, and returns the written path.
func writeStub(t *testing.T, dir, name, source string) string {
	t.Helper()
	built, err := os.ReadFile(buildStub(t, source))
	if err != nil {
		t.Fatalf("read stub: %v", err)
	}
	path := core.PathJoin(dir, name+driverExeSuffix)
	if r := core.WriteFile(path, built, 0o755); !r.OK {
		t.Fatalf("write stub %s: %v", path, r.Value)
	}
	return path
}

// quickExitSource is a stand-in that exits 0 immediately — the "the process
// was already gone" case for Stop/Status/adminAddr. It replaces an
// extensionless "#!/bin/sh\nexit 0" script for the same reasons the sleeper
// did: neither the shebang nor the missing extension works on Windows.
const quickExitSource = `package main

func main() {}
`

// writeQuickExit drops a compiled exit-0 stand-in named name into dir.
func writeQuickExit(t *testing.T, dir, name string) string {
	t.Helper()
	return writeStub(t, dir, name, quickExitSource)
}

// driverExeSuffix is the extension an executable must carry on this platform.
// Windows resolves a command name through %PATHEXT%, so a stand-in written
// without it can never be found or run.
var driverExeSuffix = func() string {
	if runtime.GOOS == "windows" {
		return ".exe"
	}
	return ""
}()

// writeFakeDriver drops an executable fake driver named name into dir and
// returns its path — with the platform's executable suffix appended, which is
// the path resolveDriverBinary will report.
func writeFakeDriver(t *testing.T, dir, name string) string {
	t.Helper()
	return writeStub(t, dir, name, fakeDriverSource)
}

// isolateDriverLookup points every directory resolveDriverBinary consults at
// throwaway test-owned locations: CORE_AI_DRIVER_DIR at a fresh empty temp
// dir (highest-priority lookup — wins over anything a real machine has on
// PATH or in ~/Lethean/bin), HOME at a fresh temp dir (so ~/Lethean/bin can't
// see a real host install), and PATH at an empty temp dir (so the final PATH
// fallback can't accidentally resolve a real binary on a developer's
// machine). Returns the CORE_AI_DRIVER_DIR path for the caller to populate.
func isolateDriverLookup(t *testing.T) string {
	t.Helper()
	driverDir := t.TempDir()
	t.Setenv("CORE_AI_DRIVER_DIR", driverDir)
	t.Setenv("HOME", t.TempDir())
	t.Setenv("PATH", t.TempDir())
	return driverDir
}

// newHealthyDriver isolates driver lookup and writes a fake sleeper binary
// named for runtime, returning the CORE_AI_DRIVER_DIR it lives in.
func newHealthyDriver(t *testing.T, runtime string) string {
	t.Helper()
	dir := isolateDriverLookup(t)
	writeFakeDriver(t, dir, runtimeBinary[runtime])
	return dir
}

// newHealthServer starts an in-process HTTP server answering /v1/health with
// 200 (or always 503 when healthy is false) and returns its host:port — what
// waitDriverReady polls. It stands in for the driver's own health endpoint;
// the spawned fake-driver *process* and this listener are deliberately
// decoupled (the production code never checks they're the same PID), which
// is what makes a portable, hermetic Serve() test possible without writing
// an HTTP server in shell.
func newHealthServer(t *testing.T, healthy bool) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if healthy && r.URL.Path == "/v1/health" {
			w.WriteHeader(http.StatusOK)
			return
		}
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	t.Cleanup(srv.Close)
	return core.TrimPrefix(srv.URL, "http://")
}

// freeDeadAddr returns a loopback host:port that is guaranteed free (nothing
// listening) at the moment it's returned — a real ephemeral port grabbed then
// immediately released, so waitDriverReady's "connection refused" path is
// exercised against a genuinely unreachable address rather than a made-up
// port number that might collide with something else on the host.
func freeDeadAddr(t *testing.T) string {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	addr := core.TrimPrefix(srv.URL, "http://")
	srv.Close()
	return addr
}

// shrinkReadyWait temporarily lowers driverReadyTimeout/readyPollInterval so
// a spawned-but-never-ready Serve() call fails in milliseconds instead of the
// production 30s, restoring the originals on test cleanup. Safe only because
// the driver package's tests never run in parallel (t.Parallel is never
// used here) — these are process-wide vars.
func shrinkReadyWait(t *testing.T, timeout, poll time.Duration) {
	t.Helper()
	origTimeout, origPoll := driverReadyTimeout, readyPollInterval
	driverReadyTimeout, readyPollInterval = timeout, poll
	t.Cleanup(func() { driverReadyTimeout, readyPollInterval = origTimeout, origPoll })
}

// waitUntil polls cond every step until it returns true or timeout elapses.
// The "poll a condition with a deadline" pattern for the one genuinely-async
// assertion in this suite (crash-restart) instead of a synchronisation sleep.
func waitUntil(timeout, step time.Duration, cond func() bool) bool {
	deadline := time.Now().Add(timeout)
	for {
		if cond() {
			return true
		}
		if time.Now().After(deadline) {
			return false
		}
		time.Sleep(step)
	}
}
