// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	coreprocess "dappco.re/go/process"
)

func TestCanonicalRepoDir_Good(t *testing.T) {
	if got := CanonicalRepoDir("mlx-community/gemma-4-e2b-it-4bit"); got != "mlx-community__gemma-4-e2b-it-4bit" {
		t.Fatalf("CanonicalRepoDir = %q, want the engine's org__name form", got)
	}
}

func TestAllowRepo_CreatesAppendsAndIdempotent_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	if r := AllowRepo("mlx-community/gemma-3-1b-it-4bit"); !r.OK {
		t.Fatalf("AllowRepo(first) failed: %v", r.Value)
	}
	if r := AllowRepo("openai/gpt-oss-20b"); !r.OK {
		t.Fatalf("AllowRepo(second) failed: %v", r.Value)
	}
	// Idempotent — re-allowing must not duplicate.
	r := AllowRepo("openai/gpt-oss-20b")
	if !r.OK {
		t.Fatalf("AllowRepo(repeat) failed: %v", r.Value)
	}
	allowed := r.Value.([]string)
	if len(allowed) != 2 || allowed[0] != "mlx-community/gemma-3-1b-it-4bit" || allowed[1] != "openai/gpt-oss-20b" {
		t.Fatalf("allowlist = %v, want both repos exactly once", allowed)
	}

	// The file is the engine's exact shape: {"repos": [...]}.
	data := core.ReadFile(allowedModelsPath())
	if !data.OK {
		t.Fatal("allowed-models.json not written")
	}
	var onDisk allowedModelsFile
	if r := core.JSONUnmarshal(data.Value.([]byte), &onDisk); !r.OK || len(onDisk.Repos) != 2 {
		t.Fatalf("on-disk allowlist = %v (parse ok=%t), want 2 repos under the engine's key", onDisk.Repos, r.OK)
	}
}

func TestAllowRepo_PreservesExistingEngineFile_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	path := allowedModelsPath()
	if r := core.MkdirAll(core.PathDir(path), 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	seed := `{"repos":["lthn/LEM-Gemma3-1B","openai/gpt-oss-20b"]}`
	if r := core.WriteFile(path, []byte(seed), 0o600); !r.OK {
		t.Fatalf("seed: %v", r.Value)
	}

	r := AllowRepo("mlx-community/gemma-3-1b-it-4bit")
	if !r.OK {
		t.Fatalf("AllowRepo over real engine file failed: %v", r.Value)
	}
	repos := r.Value.([]string)
	if len(repos) != 3 || repos[0] != "lthn/LEM-Gemma3-1B" || repos[2] != "mlx-community/gemma-3-1b-it-4bit" {
		t.Fatalf("repos = %v, want existing entries preserved + new appended", repos)
	}
}

func TestAllowRepo_EmptyRepo_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if r := AllowRepo("  "); r.OK {
		t.Fatal("AllowRepo(blank) succeeded, want refusal")
	}
}

func TestAllowRepo_CorruptFile_Ugly(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	path := allowedModelsPath()
	if r := core.MkdirAll(core.PathDir(path), 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	if r := core.WriteFile(path, []byte(`not json at all`), 0o600); !r.OK {
		t.Fatalf("seed corrupt file: %v", r.Value)
	}
	// A corrupt allowlist must refuse loudly, never silently overwrite the
	// operator's file.
	if r := AllowRepo("mlx-community/gemma-3-1b-it-4bit"); r.OK {
		t.Fatal("AllowRepo over corrupt file succeeded, want refusal")
	}
}

func TestAdmin_DownloadModel_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	svc := &Service{}
	if r := svc.DownloadModel(RuntimeMLX, "org/repo", "main"); r.OK {
		t.Fatal("DownloadModel with no running engine succeeded, want refusal")
	}
	if r := svc.DownloadJobStatus(RuntimeMLX, "job-1"); r.OK {
		t.Fatal("DownloadJobStatus with no running engine succeeded, want refusal")
	}
}

// seedAdminToken writes the engine-managed admin.token file under the
// current (test-scoped) HOME, mirroring what the engine does on first boot.
func seedAdminToken(t *testing.T, token string) {
	t.Helper()
	path := adminTokenPath()
	if r := core.MkdirAll(core.PathDir(path), 0o755); !r.OK {
		t.Fatalf("mkdir admin token dir: %v", r.Value)
	}
	if r := core.WriteFile(path, []byte(token), 0o600); !r.OK {
		t.Fatalf("seed admin token: %v", r.Value)
	}
}

func TestAdmin_ReadAdminToken_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	seedAdminToken(t, "secret-token\n")

	token, err := readAdminToken()
	if err != nil {
		t.Fatalf("readAdminToken failed: %v", err)
	}
	if token != "secret-token" {
		t.Fatalf("readAdminToken = %q, want the trimmed token", token)
	}
}

func TestAdmin_ReadAdminToken_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // resolves fine, but the engine never wrote a token
	if _, err := readAdminToken(); err == nil {
		t.Fatal("readAdminToken succeeded with no token file, want failure")
	}
}

func TestAdmin_ReadAdminToken_Ugly(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	seedAdminToken(t, "   \n")
	if _, err := readAdminToken(); err == nil {
		t.Fatal("readAdminToken succeeded over a whitespace-only token file, want failure")
	}
}

func TestAdmin_AdminAddr_Good(t *testing.T) {
	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: pid, Addr: "127.0.0.1:5555", Running: true},
	}}
	addr, err := s.adminAddr(RuntimeMLX)
	if err != nil {
		t.Fatalf("adminAddr failed: %v", err)
	}
	if addr != "127.0.0.1:5555" {
		t.Fatalf("adminAddr = %q, want the tracked address", addr)
	}
}

// TestAdmin_AdminAddr_Bad covers a tracked runtime whose process has already
// exited — adminAddr must refuse rather than hand back a dead address.
func TestAdmin_AdminAddr_Bad(t *testing.T) {
	dir := t.TempDir()
	quick := core.PathJoin(dir, "quick")
	if r := core.WriteFile(quick, []byte("#!/bin/sh\nexit 0\n"), 0o755); !r.OK {
		t.Fatalf("write quick-exit script: %v", r.Value)
	}
	proc := benchProcSvc(t)
	sr := proc.StartWithOptions(core.Background(), coreprocess.RunOptions{Command: quick, Detach: true, KillGroup: true})
	if !sr.OK {
		t.Fatalf("spawn quick-exit script: %v", sr.Value)
	}
	p := sr.Value.(*coreprocess.Process)
	_ = proc.Wait(p.ID)

	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: p.ID, Addr: "127.0.0.1:5555", Running: true},
	}}
	if _, err := s.adminAddr(RuntimeMLX); err == nil {
		t.Fatal("adminAddr succeeded for a runtime whose process already exited, want refusal")
	}
}

func TestAdmin_AdminRoundTrip_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	seedAdminToken(t, "tok")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer tok" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"job-1","status":"done"}`))
	}))
	t.Cleanup(srv.Close)

	r := adminRoundTrip(http.MethodGet, srv.URL+"/v1/admin/models/download?job=job-1", nil)
	if !r.OK {
		t.Fatalf("adminRoundTrip failed: %v", r.Value)
	}
	job, ok := r.Value.(DownloadJob)
	if !ok || job.ID != "job-1" || job.Status != "done" {
		t.Fatalf("adminRoundTrip = %+v, want the decoded job", r.Value)
	}
}

func TestAdmin_AdminRoundTrip_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	seedAdminToken(t, "tok")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusForbidden)
		_, _ = w.Write([]byte("nope"))
	}))
	t.Cleanup(srv.Close)

	r := adminRoundTrip(http.MethodGet, srv.URL+"/v1/admin/models/download?job=job-1", nil)
	if r.OK {
		t.Fatal("adminRoundTrip against a refusing engine succeeded, want failure")
	}
	if !core.Contains(r.Error(), "engine refused") {
		t.Fatalf("adminRoundTrip error = %q, want it naming the refusal", r.Error())
	}
}

func TestAdmin_AdminRoundTrip_Ugly(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	seedAdminToken(t, "tok")
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("not json"))
	}))
	t.Cleanup(srv.Close)

	r := adminRoundTrip(http.MethodGet, srv.URL+"/v1/admin/models/download?job=job-1", nil)
	if r.OK {
		t.Fatal("adminRoundTrip over an invalid JSON body succeeded, want a decode failure")
	}
	if !core.Contains(r.Error(), "decode job reply") {
		t.Fatalf("adminRoundTrip error = %q, want it naming the decode failure", r.Error())
	}
}

// TestAdmin_DownloadModel_Good walks the full authenticated path: allowlist
// is irrelevant here (that's the engine's own job), but the driver-side
// plumbing — resolve the running engine's address, default the revision,
// authenticate, decode the reply — all has to line up, and DownloadJobStatus
// against the same fake engine proves the polling half too.
func TestAdmin_DownloadModel_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	seedAdminToken(t, "tok")

	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"id":"job-9","status":"pending","repo":"org/repo","revision":"main"}`))
	}))
	t.Cleanup(srv.Close)
	addr := core.TrimPrefix(srv.URL, "http://")

	proc := benchProcSvc(t)
	pid := benchSleepProc(t, proc)
	s := &Service{proc: proc, served: map[string]*Served{
		RuntimeMLX: {Runtime: RuntimeMLX, ProcessID: pid, Addr: addr, Running: true},
	}}

	r := s.DownloadModel(RuntimeMLX, "org/repo", "") // empty revision must default to "main"
	if !r.OK {
		t.Fatalf("DownloadModel failed: %v", r.Value)
	}
	job, ok := r.Value.(DownloadJob)
	if !ok || job.ID != "job-9" {
		t.Fatalf("DownloadModel job = %+v, want the decoded job", r.Value)
	}
	if !core.Contains(string(gotBody), `"revision":"main"`) {
		t.Fatalf("DownloadModel request body = %s, want the defaulted revision", gotBody)
	}

	r2 := s.DownloadJobStatus(RuntimeMLX, "job-9")
	if !r2.OK {
		t.Fatalf("DownloadJobStatus failed: %v", r2.Value)
	}
	if job2, ok := r2.Value.(DownloadJob); !ok || job2.ID != "job-9" {
		t.Fatalf("DownloadJobStatus job = %+v, want job-9", r2.Value)
	}
}
