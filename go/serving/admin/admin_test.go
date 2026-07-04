// SPDX-Licence-Identifier: EUPL-1.2

package admin

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// fakeReloader records the path it was asked to swap in.
type fakeReloader struct {
	current string
	gotPath string
}

func (f *fakeReloader) CurrentPath() string { return f.current }
func (f *fakeReloader) ReloadModel(newPath string, _ []inference.LoadOption) (string, string, error) {
	prev := f.current
	f.gotPath = newPath
	f.current = newPath
	return prev, newPath, nil
}

// seedModel writes a minimal verified model pack (config.json + .sha256 sidecar)
// under a temp HOME's ~/Lethean/data/models/<name> and returns its path.
func seedModel(t *testing.T, name string) string {
	t.Helper()
	home := t.TempDir()
	t.Setenv("HOME", home)
	dir := core.PathJoin(home, "Lethean", "data", "models", name)
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
	if MachineHash() != MachineHash() {
		t.Fatal("MachineHash is not stable across calls")
	}
	if !strings.HasPrefix(MachineHash(), "lem-") {
		t.Fatalf("MachineHash = %q, want lem- prefix", MachineHash())
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
	dir := core.PathJoin(home, "Lethean", "data", "models", "nomanifest")
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

// TestPathWithinDir_Containment_Good covers the child / sibling / escape cases
// of the models-dir containment gate.
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
