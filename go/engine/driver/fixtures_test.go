// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	core "dappco.re/go"
	"github.com/gin-gonic/gin"
)

// Shared fixtures for the driver package's hermetic test suite. No models, no
// network, no real lthn-mlx/lthn-cuda/lthn-amd binary — every "driver" the
// tests spawn is a tiny sleeper script reached via CORE_AI_DRIVER_DIR (the
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

// fakeDriverScript is a POSIX-sh stand-in for lthn-mlx/lthn-cuda/lthn-amd. It
// ignores its argv (serve --addr ... --model ... — whatever serveArgs built)
// and just stays alive until killed: SIGKILL can't be caught, and a plain
// `sleep` terminates on SIGTERM by default too, so no trap is needed to
// satisfy spawn/Stop/Status/crash-restart tests. The sleep is invoked by
// absolute path because isolateDriverLookup deliberately empties PATH — a
// bare `sleep` exits 127 instantly, turning every spawn into a silent
// crash-restart storm (the loop raced its own storm and only won on some
// platforms; caught on the linux CI runner).
const fakeDriverScript = "#!/bin/sh\nexec /bin/sleep 600\n"

// writeFakeDriver drops an executable fake driver binary named name into dir
// and returns its path.
func writeFakeDriver(t *testing.T, dir, name string) string {
	t.Helper()
	path := core.PathJoin(dir, name)
	if r := core.WriteFile(path, []byte(fakeDriverScript), 0o755); !r.OK {
		t.Fatalf("write fake driver %s: %v", path, r.Value)
	}
	return path
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
