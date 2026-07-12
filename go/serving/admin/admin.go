// SPDX-Licence-Identifier: EUPL-1.2

// Package admin is the /v1/admin/* control plane rescued out of lthn-mlx's
// cmd/mlx admin subsystem so the business logic lives in a go-inference library.
// It mounts the machine-identity, serve-status and hot-swap-reload handlers over
// a Reloader (the serving hot-swap resolver); serving.RunServe composes this mux
// at /v1/admin/ behind the Bearer wall.
//
// The heavier admin subsystems lthn-mlx also carried — HF model download (its
// own tree-API + verified-fetch client) and SFT (native LoRA training) — are NOT
// here: download is a self-contained subsystem go-inference's hf package does not
// yet expose, and SFT belongs with the training rescue. They mount onto the same
// NewMux seam when their libraries land.
//
//	mux := admin.NewMux(admin.Config{Reloader: resolver, ServeStatus: st, Log: os.Stderr})
//	// serving.RunServe mounts mux at /v1/admin/ behind the Bearer wall
package admin

import (
	"crypto/sha256"
	"encoding/hex"
	"io"
	"net/http"
	"runtime"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// Admin route paths under /v1/admin/*.
const (
	PathMachine     = "/v1/admin/machine"
	PathServeStatus = "/v1/admin/serve/status"
	PathReload      = "/v1/admin/serve/reload"
)

// Reloader is the hot-swap seam the reload handler drives. The serving
// hot-swap resolver satisfies it structurally, so this package does not import
// serving (no cycle: serving imports admin to compose the mux).
type Reloader interface {
	// CurrentPath is the active model path (or the boot path pre-first-load).
	CurrentPath() string
	// ReloadModel swaps in newPath and returns the previous + new active paths.
	ReloadModel(newPath string, newOpts []inference.LoadOption) (prevPath, newActive string, err error)
}

// Config bundles the dependencies NewMux needs.
type Config struct {
	Reloader    Reloader
	ServeStatus ServeStatus
	Log         io.Writer // admin audit lines (auth denies, reload attempts)
	// ModelController, when set, mounts the multi-model control plane
	// (/v1/admin/models*: list/load/unload/pin). nil leaves those routes
	// unmounted — the single-model serve keeps only serve/reload.
	ModelController ModelController
}

// NewMux mounts the /v1/admin/* control-plane handlers. Returns a Handler that
// only knows the admin paths — serving.RunServe composes it at /v1/admin/ behind
// the Bearer wall.
func NewMux(cfg Config) http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc(PathMachine, machineHandler)
	var currentPath func() string
	if cfg.Reloader != nil {
		currentPath = cfg.Reloader.CurrentPath
	}
	mux.HandleFunc(PathServeStatus, statusHandler(cfg.ServeStatus, currentPath))
	if cfg.Reloader != nil {
		mux.HandleFunc(PathReload, reloadHandler(cfg.Reloader, cfg.Log))
	} else {
		mux.HandleFunc(PathReload, notImplementedHandler("serve/reload", "no resolver wired — admin mux built without a Reloader"))
	}
	mountModelRoutes(mux, cfg.ModelController, cfg.Log)
	return mux
}

// MachineInfo is the response shape for GET /v1/admin/machine.
type MachineInfo struct {
	Hash      string `json:"hash"`
	Hostname  string `json:"hostname,omitempty"`
	Runtime   string `json:"runtime"`
	GoVersion string `json:"go_version,omitempty"`
	OS        string `json:"os,omitempty"`
	Arch      string `json:"arch,omitempty"`
	Time      int64  `json:"time"`
}

// MachineHash returns this machine's stable identity token — the value a reload
// caller must echo as confirm_machine (proving it did a /v1/admin/machine GET
// first: the confused-deputy defence). It is an engine-neutral hash of
// hostname + GOOS + GOARCH; lthn-mlx derived it from the metal device probe,
// which go-inference does not expose engine-neutrally, so the identity is
// host-derived here. Consistent between GET /machine and the reload gate — which
// is all the defence needs.
func MachineHash() string {
	host := hostname()
	sum := sha256.Sum256([]byte(host + "|" + runtime.GOOS + "|" + runtime.GOARCH))
	return "lem-" + hex.EncodeToString(sum[:12])
}

func hostname() string {
	if r := core.Hostname(); r.OK {
		if h, ok := r.Value.(string); ok {
			return h
		}
	}
	return ""
}

// machineHandler answers GET /v1/admin/machine with the machine identity used to
// decide which tuned profiles belong here and to confirm reloads.
func machineHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, MachineInfo{
		Hash:      MachineHash(),
		Hostname:  hostname(),
		Runtime:   "go-inference",
		GoVersion: runtime.Version(),
		OS:        runtime.GOOS,
		Arch:      runtime.GOARCH,
		Time:      time.Now().Unix(),
	})
}

// notImplementedHandler is the 501 placeholder naming what's blocking a route.
func notImplementedHandler(name, blocker string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusNotImplemented, map[string]string{
			"endpoint": name,
			"status":   "not implemented",
			"blocker":  blocker,
		})
	}
}

// writeJSON marshals v and writes it with the given status.
func writeJSON(w http.ResponseWriter, status int, v any) {
	encoded := core.JSONMarshal(v)
	w.Header().Set("content-type", "application/json")
	if !encoded.OK {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte(`{"error":"marshal failed"}`))
		return
	}
	w.WriteHeader(status)
	_, _ = w.Write(encoded.Value.([]byte))
}

// readJSONBody decodes the request body into target. The body is capped at 64KB
// — legitimate admin payloads serialise to <1KB; the cap prevents a
// memory-exhaustion DoS via an adversarial multi-GB POST.
func readJSONBody(r *http.Request, target any) error {
	defer r.Body.Close()
	body, err := io.ReadAll(http.MaxBytesReader(nil, r.Body, 64*1024))
	if err != nil {
		return err
	}
	res := core.JSONUnmarshal(body, target)
	if !res.OK {
		return res.Value.(error)
	}
	return nil
}

// printAudit writes an admin audit line to w (nil silences it).
func printAudit(w io.Writer, format string, args ...any) {
	if w == nil {
		return
	}
	core.Print(w, format, args...)
}
