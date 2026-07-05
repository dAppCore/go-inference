// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"bytes"
	"io"
	"net/http"
	"time"

	core "dappco.re/go"
)

// Engine admin client — the driver-side counterpart of a running LEM
// Engine's /v1/admin surface (model downloads today). The host app IS the
// engine's operator: the download allowlist
// (~/Lethean/data/allowed-models.json) and the Bearer token
// (~/Lethean/data/admin.token) are engine-managed files the host curates
// and reads — writing a curated repo into the allowlist before requesting
// its download is the intended operator path, not a policy bypass.

// adminHTTPTimeout bounds admin round-trips. Downloads run as engine-side
// jobs — the POST returns a job id immediately; the polling GET is quick.
const adminHTTPTimeout = 30 * time.Second

// DownloadJob mirrors the engine's admin download job JSON (go-mlx
// adminDownloadJob): status pending → running → done | failed. BytesDone /
// BytesTotal drive progress; DestPath is where the weights land
// (~/Lethean/data/models/<org__name>/<revision>).
type DownloadJob struct {
	ID         string `json:"id"`
	Status     string `json:"status"`
	Repo       string `json:"repo"`
	Revision   string `json:"revision"`
	DestPath   string `json:"dest_path,omitempty"`
	BytesTotal int64  `json:"bytes_total,omitempty"`
	BytesDone  int64  `json:"bytes_done,omitempty"`
	FileCount  int    `json:"file_count,omitempty"`
	Error      string `json:"error,omitempty"`
}

// CanonicalRepoDir mirrors the engine's canonicaliseRepoName: the directory
// a downloaded repo lands under ~/Lethean/data/models. Used to match
// catalogue scans against curated repos.
//
//	driver.CanonicalRepoDir("mlx-community/gemma-4-e2b-it-4bit")
//	// → "mlx-community__gemma-4-e2b-it-4bit"
func CanonicalRepoDir(repo string) string {
	return core.Replace(repo, "/", "__")
}

func allowedModelsPath() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "data", "allowed-models.json")
}

func adminTokenPath() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "data", "admin.token")
}

// allowedModelsFile mirrors the engine's allowlist shape (go-mlx
// admin_download.go loadAllowedModels): {"repos": ["org/name", …]}. The
// field-exercise run caught the first draft of this client assuming a bare
// array — the engine's parser is the contract, not a guess.
type allowedModelsFile struct {
	Repos []string `json:"repos"`
}

// AllowRepo ensures repo is in the engine's download allowlist —
// read-modify-write of allowed-models.json (created when absent, 0600 to
// match the engine's posture for its data/ siblings). Idempotent; returns
// the resulting repo list. Unparseable JSON refuses loudly — never
// silently overwrite the operator's file.
//
//	driver.AllowRepo("mlx-community/gemma-4-e2b-it-4bit")
func AllowRepo(repo string) core.Result {
	repo = core.Trim(repo)
	if repo == "" {
		return core.Fail(core.E("driver.AllowRepo", "repo required", nil))
	}
	path := allowedModelsPath()
	var f allowedModelsFile
	if data := core.ReadFile(path); data.OK {
		raw, _ := data.Value.([]byte)
		if len(raw) > 0 {
			if r := core.JSONUnmarshal(raw, &f); !r.OK {
				return core.Fail(core.E("driver.AllowRepo",
					"allowed-models.json did not parse — fix or remove it", nil))
			}
		}
	}
	for _, a := range f.Repos {
		if a == repo {
			return core.Ok(f.Repos)
		}
	}
	f.Repos = append(f.Repos, repo)
	encoded := core.JSONMarshalIndent(f, "", "  ")
	if !encoded.OK {
		return core.Fail(core.E("driver.AllowRepo", "encode allowlist", nil))
	}
	if r := core.MkdirAll(core.PathDir(path), 0o755); !r.OK {
		return core.Fail(core.E("driver.AllowRepo", "create data dir", nil))
	}
	raw, _ := encoded.Value.([]byte)
	if r := core.WriteFile(path, raw, 0o600); !r.OK {
		return core.Fail(core.E("driver.AllowRepo", "write allowlist", nil))
	}
	return core.Ok(f.Repos)
}

// adminAddr resolves the live listen address for runtime, requiring a
// running driver — admin routes only exist on a bound engine.
func (s *Service) adminAddr(runtime string) (string, error) {
	for _, sv := range s.Status() {
		if sv.Runtime != runtime {
			continue
		}
		if !sv.Running || sv.Addr == "" {
			return "", core.E("driver.admin", "engine not running — start it first", nil)
		}
		return sv.Addr, nil
	}
	return "", core.E("driver.admin", "runtime not supervised — start the engine first", nil)
}

// readAdminToken reads the engine-managed Bearer token. The engine writes
// it on first serve boot, so "absent" means the engine has never run.
func readAdminToken() (string, error) {
	data := core.ReadFile(adminTokenPath())
	if !data.OK {
		return "", core.E("driver.admin",
			"admin token absent — the engine writes it on first start", nil)
	}
	raw, _ := data.Value.([]byte)
	token := core.Trim(string(raw))
	if token == "" {
		return "", core.E("driver.admin", "admin token file is empty", nil)
	}
	return token, nil
}

// DownloadModel kicks an engine-side HuggingFace download job and returns
// the engine's DownloadJob snapshot (poll with DownloadJobStatus). The repo
// must already be allowlisted (AllowRepo) — this call never widens policy.
//
//	r := svc.DownloadModel(driver.RuntimeMLX, "mlx-community/gemma-4-e2b-it-4bit", "main")
//	if r.OK { job := r.Value.(driver.DownloadJob) }
func (s *Service) DownloadModel(runtime, repo, revision string) core.Result {
	if core.Trim(repo) == "" {
		return core.Fail(core.E("driver.DownloadModel", "repo required", nil))
	}
	if revision == "" {
		revision = "main"
	}
	addr, err := s.adminAddr(runtime)
	if err != nil {
		return core.Fail(err)
	}
	body := core.JSONMarshal(map[string]string{"repo": repo, "revision": revision})
	if !body.OK {
		return core.Fail(core.E("driver.DownloadModel", "encode request", nil))
	}
	raw, _ := body.Value.([]byte)
	return adminRoundTrip(http.MethodPost, "http://"+addr+"/v1/admin/models/download", raw)
}

// DownloadJobStatus polls an engine-side download job by id.
//
//	r := svc.DownloadJobStatus(driver.RuntimeMLX, jobID)
func (s *Service) DownloadJobStatus(runtime, jobID string) core.Result {
	if core.Trim(jobID) == "" {
		return core.Fail(core.E("driver.DownloadJobStatus", "job id required", nil))
	}
	addr, err := s.adminAddr(runtime)
	if err != nil {
		return core.Fail(err)
	}
	return adminRoundTrip(http.MethodGet, "http://"+addr+"/v1/admin/models/download?job="+jobID, nil)
}

// adminRoundTrip performs one authenticated admin call and decodes the
// engine's DownloadJob reply. Non-2xx bodies surface verbatim — the
// engine's deny reasons (allowlist, busy) are operator-readable.
func adminRoundTrip(method, url string, body []byte) core.Result {
	var reader io.Reader
	if body != nil {
		reader = bytes.NewReader(body)
	}
	req, err := http.NewRequest(method, url, reader)
	if err != nil {
		return core.Fail(core.E("driver.admin", "build request", err))
	}
	token, err := readAdminToken()
	if err != nil {
		return core.Fail(err)
	}
	req.Header.Set("Authorization", "Bearer "+token)
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	client := &http.Client{Timeout: adminHTTPTimeout}
	resp, err := client.Do(req)
	if err != nil {
		return core.Fail(core.E("driver.admin", "engine unreachable", err))
	}
	defer func() { _ = resp.Body.Close() }()
	payload, err := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	if err != nil {
		return core.Fail(core.E("driver.admin", "read reply", err))
	}
	if resp.StatusCode < 200 || resp.StatusCode > 299 {
		return core.Fail(core.E("driver.admin",
			core.Sprintf("engine refused (%d): %s", resp.StatusCode, core.Trim(string(payload))), nil))
	}
	var job DownloadJob
	if r := core.JSONUnmarshal(payload, &job); !r.OK {
		return core.Fail(core.E("driver.admin", "decode job reply", nil))
	}
	return core.Ok(job)
}
