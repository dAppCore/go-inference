// SPDX-Licence-Identifier: EUPL-1.2

package admin

import (
	"io"
	"net/http"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// The multi-model control plane under /v1/admin/models*. It grows the admin
// subsystem from the single-model hot-swap (serve/reload) to an oMLX-parity
// registry: list what is configured/resident, load a model on demand, unload one
// to free memory, and pin/unpin it against eviction.
//
// load is CRITICAL-class exactly like serve/reload — a caller who can load
// weights owns what the server serves — so it carries the same gate: the target
// binds under ~/Lethean/lem/models/ with a .sha256 sidecar (no loading "whatever
// is on disk"), and confirm_machine must equal /v1/admin/machine's hash (the
// confused-deputy defence). unload + pin operate only on already-registered,
// already-vetted models (they introduce no new weights), so the Bearer wall on
// /v1/admin/* is their gate; every attempt is still audited.

// Model control-plane route paths.
const (
	PathModels      = "/v1/admin/models"
	PathModelLoad   = "/v1/admin/models/load"
	PathModelUnload = "/v1/admin/models/unload"
	PathModelPin    = "/v1/admin/models/pin"
)

// ModelStatus is one model's snapshot in the /v1/admin/models list. It is the
// admin wire shape; the serving resolver supplies it via an adapter so this
// package need not import serving (no cycle).
type ModelStatus struct {
	ID           string   `json:"id"`
	Path         string   `json:"path"`
	Resident     bool     `json:"resident"`
	Pinned       bool     `json:"pinned"`
	EstBytes     uint64   `json:"est_bytes"`
	Profiles     []string `json:"profiles,omitempty"`
	LastUsedUnix int64    `json:"last_used_unix,omitempty"`
}

// ModelController is the multi-model registry seam the admin routes drive. The
// serving multi-model resolver satisfies it through a thin adapter, so this
// package stays serving-free.
type ModelController interface {
	// ListModels returns every configured model's residency snapshot.
	ListModels() []ModelStatus
	// LoadModel registers (if new) and loads id at the already-validated path,
	// optionally pinning it, and returns the canonical loaded id.
	LoadModel(id, path string, opts []inference.LoadOption, pinned bool) (string, error)
	// UnloadModel force-evicts a model by id, freeing its memory.
	UnloadModel(id string) error
	// SetPinned toggles a model's eviction exemption.
	SetPinned(id string, pinned bool) error
}

// LoadModelRequest is the body for POST /v1/admin/models/load.
type LoadModelRequest struct {
	ID             string `json:"id,omitempty"`              // registry id; "" derives from the path basename
	Model          string `json:"model,omitempty"`           // basename under the models dir
	ModelPath      string `json:"model_path,omitempty"`      // absolute path under the models dir (preferred)
	ConfirmMachine string `json:"confirm_machine,omitempty"` // machine hash from /v1/admin/machine
	ContextLength  int    `json:"context_length,omitempty"`  // optional context-length override
	AdapterPath    string `json:"adapter_path,omitempty"`    // optional LoRA adapter overlay
	Pin            bool   `json:"pin,omitempty"`             // pin the loaded model against eviction
}

// ModelActionResponse names the outcome of a load/unload/pin action.
type ModelActionResponse struct {
	Status   string `json:"status"`
	ID       string `json:"id,omitempty"`
	Path     string `json:"path,omitempty"`
	Pinned   bool   `json:"pinned,omitempty"`
	LoadedAt int64  `json:"loaded_at_unix,omitempty"`
}

// ModelIDRequest is the body for unload — names a registered model.
type ModelIDRequest struct {
	ID    string `json:"id,omitempty"`
	Model string `json:"model,omitempty"` // accepted as an alias for id
}

// PinRequest is the body for POST /v1/admin/models/pin.
type PinRequest struct {
	ID     string `json:"id,omitempty"`
	Model  string `json:"model,omitempty"`
	Pinned bool   `json:"pinned"`
}

// ModelsListResponse is the GET /v1/admin/models body.
type ModelsListResponse struct {
	Object string        `json:"object"`
	Data   []ModelStatus `json:"data"`
}

// mountModelRoutes wires the multi-model control plane onto mux when a controller
// is present. Called from NewMux; a nil controller leaves the routes unmounted
// (the single-model serve does not advertise them).
func mountModelRoutes(mux *http.ServeMux, ctrl ModelController, log io.Writer) {
	if ctrl == nil {
		return
	}
	mux.HandleFunc(PathModels, listModelsHandler(ctrl))
	mux.HandleFunc(PathModelLoad, loadModelHandler(ctrl, log))
	mux.HandleFunc(PathModelUnload, unloadModelHandler(ctrl, log))
	mux.HandleFunc(PathModelPin, pinModelHandler(ctrl, log))
}

// listModelsHandler answers GET /v1/admin/models with the registry snapshot.
func listModelsHandler(ctrl ModelController) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		writeJSON(w, http.StatusOK, ModelsListResponse{Object: "list", Data: ctrl.ListModels()})
	}
}

// loadModelHandler answers POST /v1/admin/models/load. It gates exactly like
// serve/reload: audit the attempt first, require confirm_machine, bind the
// target under the models dir with a sha sidecar, then load.
func loadModelHandler(ctrl ModelController, log io.Writer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req LoadModelRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "invalid body: "+err.Error(), http.StatusBadRequest)
			return
		}
		modelName := core.Trim(req.Model)
		modelPath := core.Trim(req.ModelPath)
		target := modelName
		if modelPath != "" {
			target = modelPath
		}
		printAudit(log, "admin: model_load attempt requester=%s target=%s pin=%v adapter=%s", r.RemoteAddr, target, req.Pin, req.AdapterPath)

		if modelName == "" && modelPath == "" {
			modelDeny(w, log, "load", target, "model or model_path required")
			return
		}
		if core.Trim(req.ConfirmMachine) != MachineHash() {
			modelDeny(w, log, "load", target, "confirm_machine mismatch (machine_hash from /v1/admin/machine)")
			return
		}

		var toPath string
		var err error
		if modelPath != "" {
			toPath, err = bindModelPathToStandardDir(modelPath)
		} else {
			toPath, err = resolveModelNameToPath(modelName)
		}
		if err != nil {
			modelDeny(w, log, "load", target, err.Error())
			return
		}

		var opts []inference.LoadOption
		if req.ContextLength > 0 {
			opts = append(opts, inference.WithContextLen(req.ContextLength))
		}
		if core.Trim(req.AdapterPath) != "" {
			opts = append(opts, inference.WithAdapterPath(req.AdapterPath))
		}

		loadedID, err := ctrl.LoadModel(core.Trim(req.ID), toPath, opts, req.Pin)
		if err != nil {
			modelFail(w, log, "load", target, "load failed: "+err.Error())
			return
		}
		printAudit(log, "admin: model_load success requester=%s id=%s path=%s pin=%v", r.RemoteAddr, loadedID, toPath, req.Pin)
		writeJSON(w, http.StatusOK, ModelActionResponse{Status: "ok", ID: loadedID, Path: toPath, Pinned: req.Pin, LoadedAt: time.Now().Unix()})
	}
}

// unloadModelHandler answers POST /v1/admin/models/unload — free a model's memory
// by id. It introduces no weights, so the Bearer wall is its gate.
func unloadModelHandler(ctrl ModelController, log io.Writer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req ModelIDRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "invalid body: "+err.Error(), http.StatusBadRequest)
			return
		}
		id := modelIDOf(req.ID, req.Model)
		printAudit(log, "admin: model_unload attempt requester=%s id=%s", r.RemoteAddr, id)
		if id == "" {
			modelDeny(w, log, "unload", id, "id or model required")
			return
		}
		if err := ctrl.UnloadModel(id); err != nil {
			modelFail(w, log, "unload", id, err.Error())
			return
		}
		printAudit(log, "admin: model_unload success requester=%s id=%s", r.RemoteAddr, id)
		writeJSON(w, http.StatusOK, ModelActionResponse{Status: "ok", ID: id})
	}
}

// pinModelHandler answers POST /v1/admin/models/pin — toggle a model's eviction
// exemption. Bearer-gated like unload.
func pinModelHandler(ctrl ModelController, log io.Writer) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var req PinRequest
		if err := readJSONBody(r, &req); err != nil {
			http.Error(w, "invalid body: "+err.Error(), http.StatusBadRequest)
			return
		}
		id := modelIDOf(req.ID, req.Model)
		printAudit(log, "admin: model_pin attempt requester=%s id=%s pinned=%v", r.RemoteAddr, id, req.Pinned)
		if id == "" {
			modelDeny(w, log, "pin", id, "id or model required")
			return
		}
		if err := ctrl.SetPinned(id, req.Pinned); err != nil {
			modelFail(w, log, "pin", id, err.Error())
			return
		}
		printAudit(log, "admin: model_pin success requester=%s id=%s pinned=%v", r.RemoteAddr, id, req.Pinned)
		writeJSON(w, http.StatusOK, ModelActionResponse{Status: "ok", ID: id, Pinned: req.Pinned})
	}
}

// modelIDOf resolves the id from the id field, falling back to the model alias.
func modelIDOf(id, model string) string {
	if v := core.Trim(id); v != "" {
		return v
	}
	return core.Trim(model)
}

// modelDeny audits + 400s a rejected control-plane request.
func modelDeny(w http.ResponseWriter, log io.Writer, action, target, reason string) {
	printAudit(log, "admin: model_%s deny target=%s reason=%s", action, target, reason)
	http.Error(w, reason, http.StatusBadRequest)
}

// modelFail audits + 500s a control-plane action that failed after its gates.
func modelFail(w http.ResponseWriter, log io.Writer, action, target, reason string) {
	printAudit(log, "admin: model_%s fail target=%s reason=%s", action, target, reason)
	http.Error(w, reason, http.StatusInternalServerError)
}
