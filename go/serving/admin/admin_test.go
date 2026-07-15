// SPDX-Licence-Identifier: EUPL-1.2

package admin

import (
	"net/http"
	"net/http/httptest"
	"runtime"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// fakeReloader records the path (and load options) it was asked to swap in.
// err, when set, makes ReloadModel fail without mutating current/gotPath —
// simulating a load failure that must not corrupt the resolver's state.
type fakeReloader struct {
	current string
	gotPath string
	gotOpts []inference.LoadOption
	err     error
}

func (f *fakeReloader) CurrentPath() string { return f.current }
func (f *fakeReloader) ReloadModel(newPath string, opts []inference.LoadOption) (string, string, error) {
	if f.err != nil {
		return "", "", f.err
	}
	prev := f.current
	f.gotPath = newPath
	f.gotOpts = opts
	f.current = newPath
	return prev, newPath, nil
}

// seedModel writes a minimal verified model pack (config.json + .sha256 sidecar)
// under a temp HOME's ~/Lethean/lem/models/<name> and returns its path.
func seedModel(t *testing.T, name string) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	dir := core.PathJoin(home, "Lethean", "lem", "models", name)
	if r := core.MkdirAll(dir, 0o755); !r.OK {
		t.Fatalf("mkdir model dir: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{"model_type":"gemma4"}`), 0o644); !r.OK {
		t.Fatalf("write config: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(dir, shaManifestFilename), []byte("abc123  config.json\n"), 0o600); !r.OK {
		t.Fatalf("write sha: %v", r.Value)
	}
	return dir
}

// TestMachineHash_Stable_Good proves the machine identity is deterministic —
// the reload confirm gate depends on GET /machine and the gate computing the
// same value.
func TestMachineHash_Stable_Good(t *testing.T) {
	// Two separate calls compared for equality — the thing under test is
	// call-to-call stability, so this must stay two calls, not a copy-paste dupe.
	if MachineHash() != MachineHash() {
		t.Fatal("MachineHash is not stable across calls")
	}
	if !strings.HasPrefix(MachineHash(), "lem-") {
		t.Fatalf("MachineHash = %q, want lem- prefix", MachineHash())
	}
}

// TestMachineHandler_MethodRejection_Bad proves a non-GET /v1/admin/machine is
// rejected before any identity work.
func TestMachineHandler_MethodRejection_Bad(t *testing.T) {
	mux := NewMux(Config{})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathMachine, nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("POST /v1/admin/machine = %d, want 405", rec.Code)
	}
}

// TestMachineHandler_HappyPath_Good proves GET /v1/admin/machine reports the
// same hash MachineHash() computes, plus the runtime/Go/OS/arch identity
// fields the reload confirm gate and operator tooling both read.
func TestMachineHandler_HappyPath_Good(t *testing.T) {
	mux := NewMux(Config{})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, PathMachine, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("GET /v1/admin/machine = %d, want 200", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, MachineHash()) {
		t.Fatalf("machine body = %s, want it to carry MachineHash() = %s", body, MachineHash())
	}
	if !strings.Contains(body, "go-inference") || !strings.Contains(body, runtime.GOOS) {
		t.Fatalf("machine body = %s, want runtime + GOOS fields", body)
	}
}

// TestNotImplementedHandler_Good proves the placeholder mounted for
// PathReload when NewMux is built without a Reloader answers 501 and names
// the blocker rather than 404ing silently.
func TestNotImplementedHandler_Good(t *testing.T) {
	mux := NewMux(Config{})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathReload, strings.NewReader(`{}`)))
	if rec.Code != http.StatusNotImplemented {
		t.Fatalf("reload with no Reloader = %d, want 501", rec.Code)
	}
	body := rec.Body.String()
	if !strings.Contains(body, "serve/reload") || !strings.Contains(body, "no resolver wired") {
		t.Fatalf("not-implemented body = %s, want endpoint name + blocker", body)
	}
}

// TestWriteJSON_Good pins the success path: status code set, content-type
// application/json, and the marshalled value in the body.
func TestWriteJSON_Good(t *testing.T) {
	rec := httptest.NewRecorder()
	writeJSON(rec, http.StatusCreated, map[string]string{"k": "v"})
	if rec.Code != http.StatusCreated {
		t.Fatalf("status = %d, want 201", rec.Code)
	}
	if got := rec.Header().Get("content-type"); got != "application/json" {
		t.Fatalf("content-type = %q, want application/json", got)
	}
	if !strings.Contains(rec.Body.String(), `"k":"v"`) {
		t.Fatalf("body = %s, want the marshalled map", rec.Body.String())
	}
}

// TestWriteJSON_MarshalFails_Bad proves a value JSON cannot encode (a channel)
// degrades to a 500 with a fixed error body instead of panicking or writing a
// truncated response.
func TestWriteJSON_MarshalFails_Bad(t *testing.T) {
	rec := httptest.NewRecorder()
	writeJSON(rec, http.StatusOK, make(chan int))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500 on marshal failure", rec.Code)
	}
	if got, want := rec.Body.String(), `{"error":"marshal failed"}`; got != want {
		t.Fatalf("body = %s, want %s", got, want)
	}
}

// TestReadJSONBody_Good proves a well-formed, under-cap body decodes into the
// target.
func TestReadJSONBody_Good(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(`{"model":"m"}`))
	var target ReloadRequest
	if err := readJSONBody(req, &target); err != nil {
		t.Fatalf("readJSONBody: %v", err)
	}
	if target.Model != "m" {
		t.Fatalf("target.Model = %q, want %q", target.Model, "m")
	}
}

// TestReadJSONBody_MalformedJSON_Bad proves invalid JSON surfaces as an error
// rather than a zero-valued target.
func TestReadJSONBody_MalformedJSON_Bad(t *testing.T) {
	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(`{not json`))
	var target ReloadRequest
	if err := readJSONBody(req, &target); err == nil {
		t.Fatal("readJSONBody should error on malformed JSON")
	}
}

// TestReadJSONBody_TooLarge_Bad proves a body over the 64KB cap is refused
// (the MaxBytesReader DoS guard) rather than buffered in full.
func TestReadJSONBody_TooLarge_Bad(t *testing.T) {
	huge := `{"model":"` + strings.Repeat("a", 70*1024) + `"}`
	req := httptest.NewRequest(http.MethodPost, "/x", strings.NewReader(huge))
	var target ReloadRequest
	if err := readJSONBody(req, &target); err == nil {
		t.Fatal("readJSONBody should reject a body over the 64KB cap")
	}
}

// TestPrintAudit_NilWriter_Good proves a nil writer silences the audit line
// instead of panicking — the "no logger wired" default used across the
// package.
func TestPrintAudit_NilWriter_Good(t *testing.T) {
	printAudit(nil, "unreachable %d", 1)
}

// TestPrintAudit_Good proves a real writer receives the formatted line.
func TestPrintAudit_Good(t *testing.T) {
	var buf strings.Builder
	printAudit(&buf, "audit %s=%d", "count", 3)
	if got, want := buf.String(), "audit count=3\n"; got != want {
		t.Fatalf("printAudit wrote %q, want %q", got, want)
	}
}

// TestHostname_Good proves hostname() never panics and, on a normal test
// host, returns the same value core.Hostname() reports.
func TestHostname_Good(t *testing.T) {
	got := hostname()
	want := ""
	if r := core.Hostname(); r.OK {
		if h, ok := r.Value.(string); ok {
			want = h
		}
	}
	if got != want {
		t.Fatalf("hostname() = %q, want %q", got, want)
	}
}

// TestReloadHandler_ValidSwap_Good proves a confirmed reload of a verified model
// resolves the path and drives the resolver's ReloadModel seam.
func TestReloadHandler_ValidSwap_Good(t *testing.T) {
	dir := seedModel(t, "mymodel")
	rl := &fakeReloader{current: "/models/old"}
	mux := NewMux(Config{Reloader: rl})

	body := `{"model":"mymodel","confirm_machine":"` + MachineHash() + `"}`
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathReload, strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("reload status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	// The handler symlink-resolves the path (the containment-security behaviour);
	// resolve the expected dir the same way before comparing (macOS /var →
	// /private/var).
	wantDir := dir
	if r := core.PathEvalSymlinks(dir); r.OK {
		wantDir = r.Value.(string)
	}
	if rl.gotPath != wantDir {
		t.Fatalf("resolver got path %q, want %q", rl.gotPath, wantDir)
	}
}

// TestReloadHandler_WrongConfirm_Bad proves a confirm_machine mismatch is denied
// before any swap (the confused-deputy gate).
func TestReloadHandler_WrongConfirm_Bad(t *testing.T) {
	seedModel(t, "mymodel")
	rl := &fakeReloader{current: "/models/old"}
	mux := NewMux(Config{Reloader: rl})

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathReload, strings.NewReader(`{"model":"mymodel","confirm_machine":"wrong"}`)))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("wrong-confirm status = %d, want 400", rec.Code)
	}
	if rl.gotPath != "" {
		t.Fatalf("resolver was called on a denied reload: %q", rl.gotPath)
	}
}

// TestReloadHandler_NoManifest_Ugly proves a model dir without a .sha256 sidecar
// is refused even with a valid confirmation (integrity gate).
func TestReloadHandler_NoManifest_Ugly(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home)
	dir := core.PathJoin(home, "Lethean", "lem", "models", "nomanifest")
	if r := core.MkdirAll(dir, 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	rl := &fakeReloader{current: "/models/old"}
	mux := NewMux(Config{Reloader: rl})

	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathReload, strings.NewReader(`{"model":"nomanifest","confirm_machine":"`+MachineHash()+`"}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("no-manifest status = %d, want 400", rec.Code)
	}
	if rl.gotPath != "" {
		t.Fatalf("resolver was called on an unverified model: %q", rl.gotPath)
	}
}

// TestReloadHandler_MethodRejection_Bad proves a non-POST /v1/admin/serve/reload
// is rejected before any body work.
func TestReloadHandler_MethodRejection_Bad(t *testing.T) {
	rl := &fakeReloader{current: "/models/old"}
	mux := NewMux(Config{Reloader: rl})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, PathReload, nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("GET reload = %d, want 405", rec.Code)
	}
}

// TestReloadHandler_BadBody_Ugly proves a malformed JSON body is a 400, not a
// panic or a 500.
func TestReloadHandler_BadBody_Ugly(t *testing.T) {
	rl := &fakeReloader{current: "/models/old"}
	mux := NewMux(Config{Reloader: rl})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathReload, strings.NewReader(`{not json`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("malformed body = %d, want 400", rec.Code)
	}
}

// TestReloadHandler_MissingModelAndPath_Bad proves a request naming neither
// model nor model_path is denied before the confirmation gate.
func TestReloadHandler_MissingModelAndPath_Bad(t *testing.T) {
	rl := &fakeReloader{current: "/models/old"}
	mux := NewMux(Config{Reloader: rl})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathReload, strings.NewReader(`{"confirm_machine":"`+MachineHash()+`"}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("no model/model_path = %d, want 400", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "model or model_path required") {
		t.Fatalf("body = %s, want the model-required reason", rec.Body.String())
	}
}

// TestReloadHandler_MissingConfirmation_Bad proves an omitted confirm_machine
// (as opposed to a wrong one) is denied with its own reason.
func TestReloadHandler_MissingConfirmation_Bad(t *testing.T) {
	seedModel(t, "mymodel")
	rl := &fakeReloader{current: "/models/old"}
	mux := NewMux(Config{Reloader: rl})
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathReload, strings.NewReader(`{"model":"mymodel"}`)))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("missing confirmation = %d, want 400", rec.Code)
	}
	if !strings.Contains(rec.Body.String(), "confirm_machine required") {
		t.Fatalf("body = %s, want the confirmation-required reason", rec.Body.String())
	}
}

// TestReloadHandler_ModelPath_Good proves the model_path (preferred, absolute)
// field takes the bindModelPathToStandardDir route, and that context_length +
// adapter_path both reach the resolver as load options.
func TestReloadHandler_ModelPath_Good(t *testing.T) {
	dir := seedModel(t, "mymodel")
	rl := &fakeReloader{current: "/models/old"}
	mux := NewMux(Config{Reloader: rl})

	body := `{"model_path":"` + dir + `","confirm_machine":"` + MachineHash() +
		`","context_length":4096,"adapter_path":"/adapters/lora"}`
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathReload, strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("model_path reload status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	wantDir := dir
	if r := core.PathEvalSymlinks(dir); r.OK {
		wantDir = r.Value.(string)
	}
	if rl.gotPath != wantDir {
		t.Fatalf("resolver got path %q, want %q", rl.gotPath, wantDir)
	}
	if len(rl.gotOpts) != 2 {
		t.Fatalf("resolver got %d load options, want 2 (context_length + adapter_path)", len(rl.gotOpts))
	}
}

// TestReloadHandler_ModelPathEscapesDir_Bad proves a model_path outside
// standardModelDir() is refused even with a valid confirmation.
func TestReloadHandler_ModelPathEscapesDir_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	outside := t.TempDir()
	if r := core.WriteFile(core.PathJoin(outside, shaManifestFilename), []byte("x"), 0o600); !r.OK {
		t.Fatalf("seed sha sidecar: %v", r.Value)
	}
	rl := &fakeReloader{current: "/models/old"}
	mux := NewMux(Config{Reloader: rl})

	body := `{"model_path":"` + outside + `","confirm_machine":"` + MachineHash() + `"}`
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathReload, strings.NewReader(body)))

	if rec.Code != http.StatusBadRequest {
		t.Fatalf("escaping model_path status = %d, want 400", rec.Code)
	}
	if rl.gotPath != "" {
		t.Fatalf("resolver was called on an escaping model_path: %q", rl.gotPath)
	}
}

// TestReloadHandler_ReloadModelFails_Ugly proves a resolver-side load failure
// surfaces as a 500 (not swallowed, not a 200).
func TestReloadHandler_ReloadModelFails_Ugly(t *testing.T) {
	seedModel(t, "mymodel")
	rl := &fakeReloader{current: "/models/old", err: core.NewError("engine boom")}
	mux := NewMux(Config{Reloader: rl})

	body := `{"model":"mymodel","confirm_machine":"` + MachineHash() + `"}`
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathReload, strings.NewReader(body)))

	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("load-failure status = %d, want 500; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), "engine boom") {
		t.Fatalf("body = %s, want the underlying load error", rec.Body.String())
	}
}

// TestReloadHandler_FirstLoad_Good proves a reload from the pre-first-load
// state (CurrentPath == "") reports "from_model_path":"" rather than a stale
// value.
func TestReloadHandler_FirstLoad_Good(t *testing.T) {
	seedModel(t, "mymodel")
	rl := &fakeReloader{current: ""}
	mux := NewMux(Config{Reloader: rl})

	body := `{"model":"mymodel","confirm_machine":"` + MachineHash() + `"}`
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathReload, strings.NewReader(body)))

	if rec.Code != http.StatusOK {
		t.Fatalf("first-load reload status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if !strings.Contains(rec.Body.String(), `"from_model_path":""`) {
		t.Fatalf("body = %s, want an empty from_model_path on first load", rec.Body.String())
	}
}

// TestBindModelPathToStandardDir_Empty_Bad proves an empty path is refused
// directly (the guard reload's own "" check upstream already prevents in
// practice, but the helper defends itself too).
func TestBindModelPathToStandardDir_Empty_Bad(t *testing.T) {
	if _, err := bindModelPathToStandardDir(""); err == nil {
		t.Fatal("bindModelPathToStandardDir(\"\") should error")
	}
}

// TestBindModelPathToStandardDir_NotFound_Bad proves a path that doesn't
// exist on disk is refused rather than silently bound.
func TestBindModelPathToStandardDir_NotFound_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if _, err := bindModelPathToStandardDir("/does/not/exist"); err == nil {
		t.Fatal("bindModelPathToStandardDir(missing) should error")
	}
}

// TestBindModelPathToStandardDir_NoManifest_Bad proves a real, in-tree model
// dir missing the .sha256 sidecar is refused — the same integrity gate
// resolveModelNameToPath enforces for the legacy basename route.
func TestBindModelPathToStandardDir_NoManifest_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	dir := core.PathJoin(standardModelDir(), "nomanifest")
	if r := core.MkdirAll(dir, 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	if _, err := bindModelPathToStandardDir(dir); err == nil {
		t.Fatal("bindModelPathToStandardDir should refuse a dir with no .sha256 sidecar")
	}
}

// TestResolveModelNameToPath_Empty_Bad proves an empty basename is refused
// with its own "required" reason (distinct from the traversal-shape checks).
func TestResolveModelNameToPath_Empty_Bad(t *testing.T) {
	if _, err := resolveModelNameToPath(""); err == nil {
		t.Fatal("resolveModelNameToPath(\"\") should error")
	}
}

// TestResolveModelNameToPath_DirNotFound_Bad proves a syntactically valid
// basename that doesn't exist under standardModelDir() is refused.
func TestResolveModelNameToPath_DirNotFound_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if _, err := resolveModelNameToPath("ghost"); err == nil {
		t.Fatal("resolveModelNameToPath(missing dir) should error")
	}
}

// TestResolveModelNameToPath_SymlinkEscape_Bad proves a basename that
// resolves (via a symlink) to a target outside standardModelDir() is refused
// — a literal-string basename check alone would miss this.
func TestResolveModelNameToPath_SymlinkEscape_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	outside := t.TempDir()
	if r := core.WriteFile(core.PathJoin(outside, shaManifestFilename), []byte("x"), 0o600); !r.OK {
		t.Fatalf("seed sha sidecar outside: %v", r.Value)
	}
	if r := core.MkdirAll(standardModelDir(), 0o755); !r.OK {
		t.Fatalf("mkdir models root: %v", r.Value)
	}
	link := core.PathJoin(standardModelDir(), "escaper")
	if r := core.Symlink(outside, link); !r.OK {
		t.Fatalf("symlink: %v", r.Value)
	}
	if _, err := resolveModelNameToPath("escaper"); err == nil {
		t.Fatal("resolveModelNameToPath should refuse a symlink escaping the models dir")
	}
}

// TestListKnownModels_Good proves only subdirs carrying a .sha256 sidecar are
// reported — a bare dir (partial download, no sidecar yet) is excluded, and a
// plain file sitting in the models root (not a dir at all) is skipped rather
// than mis-treated as a model.
func TestListKnownModels_Good(t *testing.T) {
	seedModel(t, "verified-a")
	dir := standardModelDir()
	if r := core.MkdirAll(core.PathJoin(dir, "unverified"), 0o755); !r.OK {
		t.Fatalf("mkdir unverified: %v", r.Value)
	}
	if r := core.MkdirAll(core.PathJoin(dir, "verified-b"), 0o755); !r.OK {
		t.Fatalf("mkdir verified-b: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(dir, "verified-b", shaManifestFilename), []byte("x"), 0o600); !r.OK {
		t.Fatalf("seed verified-b sha: %v", r.Value)
	}
	if r := core.WriteFile(core.PathJoin(dir, "README.txt"), []byte("not a model"), 0o644); !r.OK {
		t.Fatalf("seed stray file: %v", r.Value)
	}

	got := ListKnownModels()
	want := map[string]bool{"verified-a": true, "verified-b": true}
	if len(got) != len(want) {
		t.Fatalf("ListKnownModels() = %v, want exactly %v", got, want)
	}
	for _, name := range got {
		if !want[name] {
			t.Fatalf("ListKnownModels() included unexpected %q (unverified dirs must be excluded)", name)
		}
	}
}

// TestListKnownModels_NoDir_Bad proves a missing models root (nothing
// downloaded yet) reports an empty list, not an error or a panic.
func TestListKnownModels_NoDir_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if got := ListKnownModels(); len(got) != 0 {
		t.Fatalf("ListKnownModels() on an absent root = %v, want empty", got)
	}
}

// TestPathWithinDir_Containment_Good covers the child / sibling / escape /
// exact-match / trailing-slash-equivalence cases of the models-dir
// containment gate.
func TestPathWithinDir_Containment_Good(t *testing.T) {
	if !pathWithinDir("/m/models", "/m/models/gemma") {
		t.Error("child dir should be contained")
	}
	if pathWithinDir("/m/models", "/m/models-evil") {
		t.Error("sibling dir must not be contained")
	}
	if pathWithinDir("/m/models", "/etc/passwd") {
		t.Error("unrelated absolute path must not be contained")
	}
	if !pathWithinDir("/m/models", "/m/models") {
		t.Error("an identical path should be contained (literal fast path)")
	}
	if !pathWithinDir("/m/models", "/m/models/") {
		t.Error("a trailing-slash variant of the root should be contained")
	}
}

// TestPathWithinDir_MismatchedPathKinds_Bad proves a rootResolved/resolved
// pair PathRel cannot relate (one relative, one absolute) fails closed rather
// than panicking or reporting containment.
func TestPathWithinDir_MismatchedPathKinds_Bad(t *testing.T) {
	if pathWithinDir("relative/root", "/abs/target") {
		t.Error("an unrelatable path pair must not be reported as contained")
	}
}

// TestResolveModelNameToPath_RejectsTraversal_Bad proves basenames with path
// separators or traversal are refused outright.
func TestResolveModelNameToPath_RejectsTraversal_Bad(t *testing.T) {
	for _, name := range []string{"../etc", "a/b", ".hidden"} {
		if _, err := resolveModelNameToPath(name); err == nil {
			t.Errorf("name %q should be rejected", name)
		}
	}
}
