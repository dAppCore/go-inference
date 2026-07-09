// SPDX-Licence-Identifier: EUPL-1.2

package lab

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"time"

	core "dappco.re/go"
)

// --- AX-7 canonical triplets ---

func TestCmd_AddLabCommands_Good(t *core.T) {
	root := core.New()
	r := AddLabCommands(root)
	cmd := root.Command("lab")

	core.AssertTrue(t, r.OK)
	core.AssertTrue(t, cmd.OK)
	core.AssertEqual(t, "lab", cmd.Value.(*core.Command).Name)
}

func TestCmd_AddLabCommands_Bad(t *core.T) {
	root := core.New()
	AddLabCommands(root)
	AddLabCommands(root)

	core.AssertLen(t, root.Commands(), 2)
	core.AssertEqual(t, "lab", root.Commands()[0])
}

func TestCmd_AddLabCommands_Ugly(t *core.T) {
	root := core.New()
	root.Command("lab", core.Command{Description: "pre-existing"})
	AddLabCommands(root)

	core.AssertLen(t, root.Commands(), 2)
	core.AssertEqual(t, "lab", root.Commands()[0])
}

func TestCmd_addServeCommand_Good(t *core.T) {
	// opts carries no "bind" key at all, so the Action closure must fall
	// back to defaultBindAddr (loopback). That is proven here because the
	// request then clears ValidateBindAddress and fails one guard later, at
	// ValidateRemoteAuth — keeping the assertion fast and deterministic
	// without ever opening a real listener.
	t.Setenv("CORE_LAB_API_TOKEN", "")
	root := core.New()
	core.RequireTrue(t, addServeCommand(root, "lab/serve").OK)
	cmd := root.Command("lab/serve")
	core.RequireTrue(t, cmd.OK)

	r := cmd.Value.(*core.Command).Run(core.NewOptions(core.Option{Key: "allow-remote", Value: true}))
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "CORE_LAB_API_TOKEN")
}

func TestCmd_addServeCommand_Bad(t *core.T) {
	// An explicit non-loopback "bind" option must pass straight through —
	// the "" fallback branch is skipped — so ValidateBindAddress itself is
	// what rejects the request.
	root := core.New()
	core.RequireTrue(t, addServeCommand(root, "lab/serve").OK)
	cmd := root.Command("lab/serve")
	core.RequireTrue(t, cmd.OK)

	r := cmd.Value.(*core.Command).Run(core.NewOptions(core.Option{Key: "bind", Value: "0.0.0.0:9"}))
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "non-loopback")
}

func TestCmd_RunServe_Good(t *core.T) {
	t.Setenv("CORE_LAB_API_TOKEN", "")
	r := RunServe(CommandOptions{Bind: "0.0.0.0:8080"})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "non-loopback")
}

func TestCmd_RunServe_Bad(t *core.T) {
	t.Setenv("CORE_LAB_API_TOKEN", "")
	r := RunServe(CommandOptions{Bind: "127.0.0.1:8080", AllowRemote: true})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "CORE_LAB_API_TOKEN")
}

func TestCmd_RunServe_Ugly(t *core.T) {
	t.Setenv("CORE_LAB_API_TOKEN", "")
	r := RunServe(CommandOptions{Bind: "not-a-host", AllowRemote: false})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "non-loopback")
}

func TestCmd_RunServe_Good_ServeAndShutdown(t *core.T) {
	// Exercises the full serve lifecycle RunServe's guard-clause tests above
	// never reach: real bind on an ephemeral loopback port, a live
	// authenticated HTTP round trip, then a clean shutdown driven through
	// the notifyServeContext seam (never a real OS signal), observed with
	// deadline-bounded polling rather than sleep-as-sync.
	t.Setenv("CORE_LAB_API_TOKEN", "serve-test-token")

	addr := freeLoopbackAddr(t)

	ctx, cancel := context.WithCancel(context.Background())
	orig := notifyServeContext
	notifyServeContext = func(context.Context) (context.Context, context.CancelFunc) {
		return ctx, cancel
	}
	defer func() { notifyServeContext = orig }()

	done := make(chan core.Result, 1)
	go func() {
		done <- RunServe(CommandOptions{Bind: addr, AllowRemote: true})
	}()

	waitForReachable(t, addr, 5*time.Second)

	unauthed, err := http.Get("http://" + addr + "/healthz")
	core.RequireNoError(t, err)
	core.AssertEqual(t, http.StatusUnauthorized, unauthed.StatusCode)
	core.ReadAll(unauthed.Body)

	req, err := http.NewRequest(http.MethodGet, "http://"+addr+"/healthz", nil)
	core.RequireNoError(t, err)
	req.Header.Set("Authorization", "Bearer serve-test-token")
	authed, err := http.DefaultClient.Do(req)
	core.RequireNoError(t, err)
	core.AssertEqual(t, http.StatusOK, authed.StatusCode)
	body := core.ReadAll(authed.Body)
	core.RequireTrue(t, body.OK)
	core.AssertContains(t, body.Value.(string), `"status":"ok"`)

	cancel()

	select {
	case r := <-done:
		core.AssertTrue(t, r.OK)
	case <-time.After(5 * time.Second):
		t.Fatal("RunServe did not shut down within deadline after cancel")
	}
}

func TestCmd_RunServe_Bad_ListenAddrInUse(t *core.T) {
	// Occupies the target port for the whole test so ListenAndServe fails
	// immediately, driving RunServe's `case err := <-errc` branch instead
	// of the ctx.Done() shutdown branch.
	t.Setenv("CORE_LAB_API_TOKEN", "")

	occupied, err := net.Listen("tcp", "127.0.0.1:0")
	core.RequireNoError(t, err)
	defer occupied.Close()

	r := RunServe(CommandOptions{Bind: occupied.Addr().String()})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "in use")
}

func TestCmd_newServeMux_Good(t *core.T) {
	mux := newServeMux("")
	srv := httptest.NewServer(mux)
	defer srv.Close()

	for _, path := range []string{"/", "/health", "/healthz"} {
		resp, err := http.Get(srv.URL + path)
		core.RequireNoError(t, err)
		core.AssertEqual(t, http.StatusOK, resp.StatusCode, path)
		core.ReadAll(resp.Body)
	}
}

func TestCmd_newServeMux_Bad(t *core.T) {
	// "GET /" is registered as a subtree catch-all, so an unregistered path
	// falls through to index() rather than 404 — that is genuine routing
	// behaviour worth pinning down. What the mux does reject is the wrong
	// method: every registered pattern here is GET-only, so any other verb
	// must get 405, whether the path is a named route or the catch-all.
	mux := newServeMux("")
	srv := httptest.NewServer(mux)
	defer srv.Close()

	fallthroughGet, err := http.Get(srv.URL + "/nope")
	core.RequireNoError(t, err)
	core.AssertEqual(t, http.StatusOK, fallthroughGet.StatusCode)
	core.ReadAll(fallthroughGet.Body)

	wrongMethodHealth, err := http.Post(srv.URL+"/health", "text/plain", nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, http.StatusMethodNotAllowed, wrongMethodHealth.StatusCode)
	core.ReadAll(wrongMethodHealth.Body)

	wrongMethodRoot, err := http.Post(srv.URL+"/", "text/plain", nil)
	core.RequireNoError(t, err)
	core.AssertEqual(t, http.StatusMethodNotAllowed, wrongMethodRoot.StatusCode)
	core.ReadAll(wrongMethodRoot.Body)
}

func TestCmd_newServeMux_Ugly(t *core.T) {
	// A configured token must gate every registered route uniformly, not
	// just a subset.
	mux := newServeMux("secret-token")
	srv := httptest.NewServer(mux)
	defer srv.Close()

	for _, path := range []string{"/", "/health", "/healthz"} {
		resp, err := http.Get(srv.URL + path)
		core.RequireNoError(t, err)
		core.AssertEqual(t, http.StatusUnauthorized, resp.StatusCode, path)
		core.ReadAll(resp.Body)
	}
}

func TestCmd_index_Good(t *core.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/", nil)

	index(w, r)

	core.AssertEqual(t, http.StatusOK, w.Code)
	core.AssertEqual(t, "text/plain; charset=utf-8", w.Header().Get("Content-Type"))
	core.AssertEqual(t, "the inference stack lab\n", w.Body.String())
}

func TestCmd_healthz_Good(t *core.T) {
	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/healthz", nil)

	healthz(w, r)

	core.AssertEqual(t, http.StatusOK, w.Code)
	core.AssertEqual(t, "application/json", w.Header().Get("Content-Type"))
	core.AssertEqual(t, "{\"status\":\"ok\"}\n", w.Body.String())
}

func TestCmd_ValidateBindAddress_Good(t *core.T) {
	r := ValidateBindAddress("127.0.0.1:8080", false)
	got := IsLoopbackBindAddress("127.0.0.1:8080")
	want := true

	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, want, got)
}

func TestCmd_ValidateBindAddress_Bad(t *core.T) {
	r := ValidateBindAddress("0.0.0.0:8080", false)
	got := r.Error()
	want := "non-loopback"

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, want)
}

func TestCmd_ValidateBindAddress_Ugly(t *core.T) {
	r := ValidateBindAddress(":8080", true)
	got := IsLoopbackBindAddress(":8080")
	want := false

	core.AssertTrue(t, r.OK)
	core.AssertEqual(t, want, got)
}

func TestCmd_IsLoopbackBindAddress_Good(t *core.T) {
	got := IsLoopbackBindAddress("localhost:8080")
	ipv4 := IsLoopbackBindAddress("127.0.0.1:8080")
	ipv6 := IsLoopbackBindAddress("[::1]:8080")

	core.AssertTrue(t, got)
	core.AssertTrue(t, ipv4)
	core.AssertTrue(t, ipv6)
}

func TestCmd_IsLoopbackBindAddress_Bad(t *core.T) {
	got := IsLoopbackBindAddress("0.0.0.0:8080")
	wildcard := IsLoopbackBindAddress(":8080")
	remote := IsLoopbackBindAddress("example.com:8080")

	core.AssertFalse(t, got)
	core.AssertFalse(t, wildcard)
	core.AssertFalse(t, remote)
}

func TestCmd_IsLoopbackBindAddress_Ugly(t *core.T) {
	empty := IsLoopbackBindAddress("")
	malformed := IsLoopbackBindAddress("::notanaddr:8080")
	missingPort := IsLoopbackBindAddress("localhost")

	core.AssertFalse(t, empty)
	core.AssertFalse(t, malformed)
	core.AssertFalse(t, missingPort)
}

func TestCmd_requireAuth_Good(t *core.T) {
	called := false
	inner := func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}
	h := requireAuth(inner, "secret")

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/health", nil)
	r.Header.Set("Authorization", "Bearer secret")

	h(w, r)

	core.AssertTrue(t, called)
	core.AssertEqual(t, http.StatusOK, w.Code)
}

func TestCmd_requireAuth_Bad(t *core.T) {
	// Same-length-but-wrong token exercises the ConstantTimeCompare mismatch
	// arm rather than only the length short-circuit.
	called := false
	inner := func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}
	h := requireAuth(inner, "secret")

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/health", nil)
	r.Header.Set("Authorization", "Bearer wrong!")

	h(w, r)

	core.AssertFalse(t, called)
	core.AssertEqual(t, http.StatusUnauthorized, w.Code)
}

func TestCmd_requireAuth_Ugly(t *core.T) {
	// Empty configured token means auth is off — passthrough even with no
	// Authorization header at all.
	called := false
	inner := func(w http.ResponseWriter, r *http.Request) {
		called = true
		w.WriteHeader(http.StatusOK)
	}
	h := requireAuth(inner, "")

	w := httptest.NewRecorder()
	r := httptest.NewRequest(http.MethodGet, "/health", nil)

	h(w, r)

	core.AssertTrue(t, called)
	core.AssertEqual(t, http.StatusOK, w.Code)
}

func TestCmd_ValidateRemoteAuth_Good(t *core.T) {
	r := ValidateRemoteAuth(false, "")
	remote := ValidateRemoteAuth(true, "token")
	want := true

	core.AssertTrue(t, r.OK)
	core.AssertTrue(t, remote.OK)
	core.AssertTrue(t, want)
}

func TestCmd_ValidateRemoteAuth_Bad(t *core.T) {
	r := ValidateRemoteAuth(true, "")
	got := r.Error()
	want := "CORE_LAB_API_TOKEN"

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, want)
}

func TestCmd_ValidateRemoteAuth_Ugly(t *core.T) {
	r := ValidateRemoteAuth(true, "  ")
	got := r.Error()
	want := "--allow-remote"

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, want)
}

// --- test helpers ---

// freeLoopbackAddr reserves an ephemeral loopback port, releases it
// immediately, and returns "host:port" for a caller (e.g. RunServe) to bind
// moments later. The tiny bind-then-close race is the standard, accepted
// trade-off for learning a free port ahead of time in tests.
func freeLoopbackAddr(t *core.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	core.RequireNoError(t, err)
	addr := l.Addr().String()
	core.RequireNoError(t, l.Close())
	return addr
}

// waitForReachable polls addr with a bounded deadline until a TCP dial
// succeeds, failing the test if the deadline elapses first. Used instead of
// a blind sleep so the test proceeds the instant RunServe's listener is
// actually accepting connections.
func waitForReachable(t *core.T, addr string, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		conn, err := net.DialTimeout("tcp", addr, 100*time.Millisecond)
		if err == nil {
			_ = conn.Close()
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s to become reachable", addr)
}
