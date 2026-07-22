// SPDX-Licence-Identifier: EUPL-1.2

package admin

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// fakeController records what the admin model routes drove it with.
type fakeController struct {
	statuses     []ModelStatus
	loadedID     string
	loadedPath   string
	loadedOpts   []inference.LoadOption
	loadedPinned bool
	loadErr      error
	unloaded     string
	unloadErr    error
	pinnedID     string
	pinnedVal    bool
	pinErr       error
}

func (f *fakeController) ListModels() []ModelStatus { return f.statuses }

func (f *fakeController) LoadModel(id, path string, opts []inference.LoadOption, pinned bool) (string, error) {
	if f.loadErr != nil {
		return "", f.loadErr
	}
	if id == "" {
		id = core.PathBase(path)
	}
	f.loadedID, f.loadedPath, f.loadedOpts, f.loadedPinned = id, path, opts, pinned
	return id, nil
}

func (f *fakeController) UnloadModel(id string) error { f.unloaded = id; return f.unloadErr }

func (f *fakeController) SetPinned(id string, pinned bool) error {
	f.pinnedID, f.pinnedVal = id, pinned
	return f.pinErr
}

// jsonBody marshals v to a request body reader.
func jsonBody(t *testing.T, v any) io.Reader {
	t.Helper()
	res := core.JSONMarshal(v)
	if !res.OK {
		t.Fatalf("marshal body: %v", res.Value)
	}
	return bytes.NewReader(res.Value.([]byte))
}

// modelMux builds an admin mux wired only with the fake controller.
func modelMux(ctrl ModelController) http.Handler {
	return NewMux(Config{ModelController: ctrl, Log: io.Discard})
}

// TestListModelsHandler_Good proves GET /v1/admin/models returns the controller's
// snapshot in OpenAI list shape.
func TestListModelsHandler_Good(t *testing.T) {
	fc := &fakeController{statuses: []ModelStatus{
		{ID: "qwen3", Path: "/m/qwen3", Resident: true, Pinned: true, EstBytes: 100, Profiles: []string{"creative"}},
		{ID: "bge", Path: "/m/bge"},
	}}
	rec := httptest.NewRecorder()
	modelMux(fc).ServeHTTP(rec, httptest.NewRequest(http.MethodGet, PathModels, nil))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	var got ModelsListResponse
	if r := core.JSONUnmarshal(rec.Body.Bytes(), &got); !r.OK {
		t.Fatalf("unmarshal: %v", r.Value)
	}
	if got.Object != "list" || len(got.Data) != 2 || got.Data[0].ID != "qwen3" || !got.Data[0].Resident {
		t.Fatalf("list body = %+v, want the two seeded statuses", got)
	}
}

// TestListModelsHandler_MethodRejection_Bad proves the list route is GET-only.
func TestListModelsHandler_MethodRejection_StatusMethodNotAllowed_Bad(t *testing.T) {
	rec := httptest.NewRecorder()
	modelMux(&fakeController{}).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathModels, nil))
	if rec.Code != http.StatusMethodNotAllowed {
		t.Fatalf("status = %d, want 405", rec.Code)
	}
}

// TestLoadModelHandler_Good proves a seeded, sha-verified model under the models
// dir loads with confirm_machine, and the controller receives the resolved path
// + pin.
func TestLoadModelHandler_Good(t *testing.T) {
	dir := seedModel(t, "qwen3")
	fc := &fakeController{}
	body := jsonBody(t, LoadModelRequest{ModelPath: dir, ConfirmMachine: MachineHash(), ContextLength: 4096, Pin: true})
	rec := httptest.NewRecorder()
	modelMux(fc).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathModelLoad, body))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	// The handler binds + canonicalises the path (symlink-resolved), so compare
	// against the eval-symlinks form of the seeded dir rather than the raw path.
	wantPath := dir
	if r := core.PathEvalSymlinks(dir); r.OK {
		wantPath = r.Value.(string)
	}
	if fc.loadedPath != wantPath {
		t.Fatalf("controller loaded path %q, want %q", fc.loadedPath, wantPath)
	}
	if !fc.loadedPinned {
		t.Fatal("pin flag not forwarded to the controller")
	}
	if len(fc.loadedOpts) != 1 {
		t.Fatalf("load opts = %d, want 1 (context length)", len(fc.loadedOpts))
	}
}

// TestLoadModelHandler_ConfirmMismatch_Bad proves the confused-deputy gate: a
// wrong confirm_machine is refused before any load.
func TestLoadModelHandler_ConfirmMismatch_ConfirmMachine_Bad(t *testing.T) {
	dir := seedModel(t, "qwen3")
	fc := &fakeController{}
	body := jsonBody(t, LoadModelRequest{ModelPath: dir, ConfirmMachine: "lem-wronghash"})
	rec := httptest.NewRecorder()
	modelMux(fc).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathModelLoad, body))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	if fc.loadedPath != "" {
		t.Fatal("a confirm_machine mismatch must not reach the controller")
	}
}

// TestLoadModelHandler_MissingTarget_ModelPath_Bad proves a body with neither
// model nor model_path is refused — both fields are explicit zero values so
// the omission this test pins is visible in the request literal, not implicit.
func TestLoadModelHandler_MissingTarget_ModelPath_Bad(t *testing.T) {
	fc := &fakeController{}
	body := jsonBody(t, LoadModelRequest{Model: "", ModelPath: "", ConfirmMachine: MachineHash()})
	rec := httptest.NewRecorder()
	modelMux(fc).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathModelLoad, body))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
}

// TestLoadModelHandler_UnverifiedModel_Bad proves a name that does not resolve to
// a sha-verified dir under the models tree is refused (no loading "whatever is on
// disk").
func TestLoadModelHandler_UnverifiedModel_SeedModel_Bad(t *testing.T) {
	seedModel(t, "qwen3") // sets HOME to a temp models dir
	fc := &fakeController{}
	body := jsonBody(t, LoadModelRequest{Model: "nonexistent", ConfirmMachine: MachineHash()})
	rec := httptest.NewRecorder()
	modelMux(fc).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathModelLoad, body))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400; body=%s", rec.Code, rec.Body.String())
	}
	if fc.loadedPath != "" {
		t.Fatal("an unverified model must not reach the controller")
	}
}

// TestLoadModelHandler_ControllerError_Fail proves a loader failure after the
// gates surfaces as 500.
func TestLoadModelHandler_ControllerError_Fail(t *testing.T) {
	dir := seedModel(t, "qwen3")
	fc := &fakeController{loadErr: core.NewError("out of memory")}
	body := jsonBody(t, LoadModelRequest{ModelPath: dir, ConfirmMachine: MachineHash()})
	rec := httptest.NewRecorder()
	modelMux(fc).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathModelLoad, body))
	if rec.Code != http.StatusInternalServerError {
		t.Fatalf("status = %d, want 500", rec.Code)
	}
}

// TestUnloadModelHandler_Good proves unload forwards the id to the controller.
func TestUnloadModelHandler_Good(t *testing.T) {
	fc := &fakeController{}
	body := jsonBody(t, ModelIDRequest{Model: "qwen3"})
	rec := httptest.NewRecorder()
	modelMux(fc).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathModelUnload, body))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if fc.unloaded != "qwen3" {
		t.Fatalf("controller unloaded %q, want qwen3", fc.unloaded)
	}
}

// TestUnloadModelHandler_MissingID_Bad proves an unload with no id is refused.
func TestUnloadModelHandler_MissingID_ModelIDRequest_Bad(t *testing.T) {
	fc := &fakeController{}
	rec := httptest.NewRecorder()
	modelMux(fc).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathModelUnload, jsonBody(t, ModelIDRequest{})))
	if rec.Code != http.StatusBadRequest {
		t.Fatalf("status = %d, want 400", rec.Code)
	}
	if fc.unloaded != "" {
		t.Fatal("an id-less unload must not reach the controller")
	}
}

// TestPinModelHandler_Good proves pin forwards the id + pinned flag.
func TestPinModelHandler_Good(t *testing.T) {
	fc := &fakeController{}
	body := jsonBody(t, PinRequest{ID: "qwen3", Pinned: true})
	rec := httptest.NewRecorder()
	modelMux(fc).ServeHTTP(rec, httptest.NewRequest(http.MethodPost, PathModelPin, body))
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200; body=%s", rec.Code, rec.Body.String())
	}
	if fc.pinnedID != "qwen3" || !fc.pinnedVal {
		t.Fatalf("controller pin = (%q,%v), want (qwen3,true)", fc.pinnedID, fc.pinnedVal)
	}
}

// TestModelRoutes_Unmounted_WhenNoController proves the multi-model routes are
// absent (404) on a single-model admin mux — no controller, no advertised
// surface.
func TestModelRoutes_Unmounted_WhenNoController(t *testing.T) {
	mux := NewMux(Config{Log: io.Discard}) // no ModelController
	rec := httptest.NewRecorder()
	mux.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, PathModels, nil))
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d, want 404 (route unmounted)", rec.Code)
	}
}
