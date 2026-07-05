// SPDX-Licence-Identifier: EUPL-1.2

package admin

import (
	"io"
	"io/fs"
	"net/http"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// /v1/admin/serve/reload — hot-swap the loaded model. CRITICAL-class: any caller
// who can flip the model owns every subsequent /v1/chat/completions response.
// The handler gates the verb before the swap:
//
//  1. Model NAME (basename) or a path that resolves under ~/Lethean/lem/models/.
//  2. The resolved dir carries a .sha256 sidecar (integrity contract — refuse
//     "whatever's on disk").
//  3. Confirmation == the machine hash from /v1/admin/machine (confused-deputy
//     defence).
//  4. Bearer auth on the path-prefix (the serving library's admin wall).
//
// Drain policy: in-flight Generate/Chat calls complete on old weights (the
// resolver hands back the active model at resolve time; the caller's reference
// keeps it alive through GC); new requests get new weights. Audit-emit on every
// attempt + outcome.

// ReloadRequest is the body shape for POST /v1/admin/serve/reload.
type ReloadRequest struct {
	Model          string `json:"model,omitempty"`           // basename under the models dir (legacy)
	ModelPath      string `json:"model_path,omitempty"`      // absolute path under the models dir (preferred)
	Confirmation   string `json:"confirmation,omitempty"`    // legacy name for confirm_machine
	ConfirmMachine string `json:"confirm_machine,omitempty"` // machine hash from /v1/admin/machine
	AdapterPath    string `json:"adapter_path,omitempty"`    // optional LoRA adapter overlay
	ContextLength  int    `json:"context_length,omitempty"`  // optional context-length override
}

// ReloadResponse names the swap; from/to feed audit + the "weights changed
// mid-conversation" client notice.
type ReloadResponse struct {
	Status   string `json:"status"`
	From     string `json:"from_model_path"`
	To       string `json:"to_model_path"`
	LoadedAt int64  `json:"loaded_at_unix"`
}

// shaManifestFilename is the sidecar the downloader writes (one digest per file,
// "<sha256>  <filename>", shasum -a 256 format). Reload refuses any model dir
// missing it — no hot-swap to unverified-integrity models.
const shaManifestFilename = ".sha256"

// standardModelDir returns ~/Lethean/lem/models/ — the root reload + download
// bound against.
func standardModelDir() string {
	return core.PathJoin(core.Env("HOME"), "Lethean", "lem", "models")
}

// reloadHandler answers POST /v1/admin/serve/reload against resolver. It
// audit-emits the attempt BEFORE the gate checks so brute-force confirmation
// guesses are visible even when refused.
func reloadHandler(resolver Reloader, log io.Writer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req ReloadRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "invalid body: "+err.Error(), http.StatusBadRequest)
			return
		}
		modelName := core.Trim(req.Model)
		modelPath := core.Trim(req.ModelPath)
		confirmation := core.Trim(req.ConfirmMachine)
		if confirmation == "" {
			confirmation = core.Trim(req.Confirmation)
		}
		from := resolver.CurrentPath()

		auditTarget := modelName
		if modelPath != "" {
			auditTarget = modelPath
		}
		printAudit(log, "admin: serve_reload attempt requester=%s from=%s to=%s adapter=%s",
			r.RemoteAddr, from, auditTarget, req.AdapterPath)

		if modelName == "" && modelPath == "" {
			reloadDeny(w, log, from, auditTarget, "model or model_path required")
			return
		}
		if confirmation == "" {
			reloadDeny(w, log, from, auditTarget, "confirm_machine required (machine_hash from /v1/admin/machine)")
			return
		}
		// Gate: confirmation matches the live machine hash.
		if confirmation != MachineHash() {
			reloadDeny(w, log, from, auditTarget, "confirm_machine mismatch")
			return
		}

		// Gate: resolve target → on-disk path, bound under the models dir + sha
		// sidecar present.
		var toPath string
		var err error
		if modelPath != "" {
			toPath, err = bindModelPathToStandardDir(modelPath)
		} else {
			toPath, err = resolveModelNameToPath(modelName)
		}
		if err != nil {
			reloadDeny(w, log, from, auditTarget, err.Error())
			return
		}

		var opts []inference.LoadOption
		if req.ContextLength > 0 {
			opts = append(opts, inference.WithContextLen(req.ContextLength))
		}
		if core.Trim(req.AdapterPath) != "" {
			opts = append(opts, inference.WithAdapterPath(req.AdapterPath))
		}

		prevPath, newPath, err := resolver.ReloadModel(toPath, opts)
		if err != nil {
			reloadFail(w, log, from, auditTarget, "load failed: "+err.Error(), http.StatusInternalServerError)
			return
		}
		if prevPath == "" {
			prevPath = from
		}
		printAudit(log, "admin: serve_reload success requester=%s from=%s to=%s", r.RemoteAddr, prevPath, newPath)
		writeJSON(w, http.StatusOK, ReloadResponse{Status: "ok", From: prevPath, To: newPath, LoadedAt: time.Now().Unix()})
	}
}

func reloadDeny(w http.ResponseWriter, log io.Writer, from, target, reason string) {
	printAudit(log, "admin: serve_reload deny from=%s to=%s reason=%s", from, target, reason)
	http.Error(w, reason, http.StatusBadRequest)
}

func reloadFail(w http.ResponseWriter, log io.Writer, from, target, reason string, status int) {
	printAudit(log, "admin: serve_reload fail from=%s to=%s reason=%s", from, target, reason)
	http.Error(w, reason, status)
}

// pathWithinDir reports whether resolved lives inside rootResolved, via a
// PathRel containment test (not a raw string prefix — a case-insensitive
// filesystem can hand back a different casing that a byte-prefix check would
// falsely reject).
func pathWithinDir(rootResolved, resolved string) bool {
	if resolved == rootResolved {
		return true
	}
	rel := core.PathRel(rootResolved, resolved)
	if !rel.OK {
		return false
	}
	r, _ := rel.Value.(string)
	if r == "" || r == "." {
		return true
	}
	if r == ".." || core.HasPrefix(r, "../") || core.PathIsAbs(r) {
		return false
	}
	return true
}

// bindModelPathToStandardDir accepts an absolute model path and verifies it
// canonicalises to a child of standardModelDir() with a sha sidecar present.
func bindModelPathToStandardDir(path string) (string, error) {
	if path == "" {
		return "", core.NewError("model_path required")
	}
	root := standardModelDir()
	rootResolved := root
	if r := core.PathEvalSymlinks(root); r.OK {
		rootResolved = r.Value.(string)
	}
	resolved := path
	if r := core.PathEvalSymlinks(path); r.OK {
		resolved = r.Value.(string)
	} else {
		return "", core.NewError("model dir not found: " + path)
	}
	if !pathWithinDir(rootResolved, resolved) {
		return "", core.NewError("model path escapes models dir")
	}
	if r := core.PathEvalSymlinks(core.PathJoin(resolved, shaManifestFilename)); !r.OK {
		return "", core.NewError("model has no sha manifest: " + path)
	}
	return resolved, nil
}

// resolveModelNameToPath maps a basename to its on-disk dir under
// standardModelDir(), refusing any name that escapes the tree or lacks a sha
// sidecar.
func resolveModelNameToPath(name string) (string, error) {
	if core.Contains(name, "/") || core.Contains(name, "..") || core.HasPrefix(name, ".") {
		return "", core.NewError("model name must be a basename (no /, no .., no leading .)")
	}
	if name == "" {
		return "", core.NewError("model name required")
	}
	root := standardModelDir()
	candidate := core.PathJoin(root, name)
	rootResolved := root
	if r := core.PathEvalSymlinks(root); r.OK {
		rootResolved = r.Value.(string)
	}
	resolved := candidate
	if r := core.PathEvalSymlinks(candidate); r.OK {
		resolved = r.Value.(string)
	} else {
		return "", core.NewError("model dir not found: " + name)
	}
	if !pathWithinDir(rootResolved, resolved) {
		return "", core.NewError("model path escapes models dir")
	}
	if r := core.PathEvalSymlinks(core.PathJoin(resolved, shaManifestFilename)); !r.OK {
		return "", core.NewError("model lacks " + shaManifestFilename + " sidecar — refuse hot-swap to unverified-integrity model")
	}
	return resolved, nil
}

// ListKnownModels returns the basenames of subdirs under standardModelDir() that
// carry a .sha256 sidecar — the set reload will accept.
func ListKnownModels() []string {
	root := standardModelDir()
	entries := core.ReadDir(core.DirFS(root), ".")
	if !entries.OK {
		return nil
	}
	dirEntries, ok := entries.Value.([]fs.DirEntry)
	if !ok {
		return nil
	}
	out := []string{}
	for _, e := range dirEntries {
		if !e.IsDir() {
			continue
		}
		if r := core.PathEvalSymlinks(core.PathJoin(root, e.Name(), shaManifestFilename)); r.OK {
			out = append(out, e.Name())
		}
	}
	return out
}
