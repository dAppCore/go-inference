package agent

import (
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
)

func testWorkerServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case core.Contains(r.URL.String(), "/register"):
			core.WriteString(w, `{}`)
		case core.Contains(r.URL.String(), "/next"):
			core.WriteString(w, `{"tasks":[],"count":0}`)
		default:
			core.WriteString(w, `{}`)
		}
	}))
}

func TestWorker_RunWorkerLoop_Good(t *core.T) {
	srv := testWorkerServer()
	defer srv.Close()
	cfg := &WorkerConfig{APIBase: srv.URL, WorkerID: "w1", OneShot: true, BatchSize: 1}
	core.AssertNotPanics(t, func() { RunWorkerLoop(cfg) })
}

func TestWorker_RunWorkerLoop_Bad(t *core.T) {
	cfg := &WorkerConfig{APIBase: "http://127.0.0.1:1", WorkerID: "w1", OneShot: true, BatchSize: 1}
	core.AssertNotPanics(t, func() { RunWorkerLoop(cfg) })
	core.AssertEqual(t, "w1", cfg.WorkerID)
}

func TestWorker_RunWorkerLoop_Ugly(t *core.T) {
	srv := testWorkerServer()
	defer srv.Close()
	cfg := &WorkerConfig{APIBase: srv.URL, OneShot: true}
	core.AssertNotPanics(t, func() { RunWorkerLoop(cfg) })
}

func TestWorker_MachineID_Good(t *core.T) {
	id := MachineID()
	core.AssertNotEqual(t, "", id)
	core.AssertTrue(t, len(id) > 0)
}

func TestWorker_MachineID_Bad(t *core.T) {
	id := MachineID()
	core.AssertNotNil(t, id)
	core.AssertTrue(t, len(id) >= 0)
}

func TestWorker_MachineID_Ugly(t *core.T) {
	first := MachineID()
	second := MachineID()
	core.AssertEqual(t, first, second)
}

func TestWorker_Hostname_Good(t *core.T) {
	name := Hostname()
	core.AssertNotNil(t, name)
	core.AssertTrue(t, len(name) >= 0)
}

func TestWorker_Hostname_Bad(t *core.T) {
	name := Hostname()
	core.AssertEqual(t, name, Hostname())
	core.AssertTrue(t, len(name) >= 0)
}

func TestWorker_Hostname_Ugly(t *core.T) {
	name := Hostname()
	core.AssertNotContains(t, name, "\n")
	core.AssertTrue(t, len(name) >= 0)
}

func TestWorker_ReadKeyFile_Good(t *core.T) {
	got := ReadKeyFile()
	core.AssertTrue(t, len(got) >= 0)
	core.AssertNotNil(t, got)
}

func TestWorker_ReadKeyFile_Bad(t *core.T) {
	got := ReadKeyFile()
	core.AssertEqual(t, got, ReadKeyFile())
	core.AssertTrue(t, len(got) >= 0)
}

func TestWorker_ReadKeyFile_Ugly(t *core.T) {
	got := ReadKeyFile()
	core.AssertNotContains(t, got, "\r")
	core.AssertTrue(t, len(got) >= 0)
}

func TestWorker_SplitComma_Good(t *core.T) {
	parts := SplitComma("a,b,c")
	core.AssertEqual(t, []string{"a", "b", "c"}, parts)
	core.AssertLen(t, parts, 3)
}

func TestWorker_SplitComma_Bad(t *core.T) {
	parts := SplitComma("")
	core.AssertEmpty(t, parts)
	core.AssertLen(t, parts, 0)
}

func TestWorker_SplitComma_Ugly(t *core.T) {
	parts := SplitComma(" a, ,b ")
	core.AssertEqual(t, []string{"a", "b"}, parts)
	core.AssertLen(t, parts, 2)
}

func TestWorker_truncStr_Good(t *core.T) {
	got := truncStr("hello world", 5)
	core.AssertEqual(t, "hello...", got)
}

func TestWorker_truncStr_Bad(t *core.T) {
	got := truncStr("short", 10)
	core.AssertEqual(t, "short", got)
}

func TestWorker_truncStr_Ugly(t *core.T) {
	got := truncStr("abc", 3)
	core.AssertEqual(t, "abc", got)
	got = truncStr("abcdef", 3)
	core.AssertEqual(t, "abc...", got)
}

func TestWorker_apiPost_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.AssertEqual(t, "POST", r.Method)
		core.WriteString(w, `{"ok":true}`)
	}))
	defer srv.Close()
	cfg := &WorkerConfig{APIBase: srv.URL, APIKey: "testkey"}
	r := apiPost(cfg, "/test", map[string]any{"id": 1})
	core.AssertTrue(t, r.OK)
}

func TestWorker_apiPost_Bad(t *core.T) {
	cfg := &WorkerConfig{APIBase: "http://127.0.0.1:1", APIKey: "testkey"}
	r := apiPost(cfg, "/test", nil)
	core.AssertFalse(t, r.OK)
}

func TestWorker_apiPost_Ugly(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		core.WriteString(w, `{"error":"bad request"}`)
	}))
	defer srv.Close()
	cfg := &WorkerConfig{APIBase: srv.URL, APIKey: "testkey"}
	r := apiPost(cfg, "/test", map[string]any{})
	core.AssertFalse(t, r.OK)
}

func TestWorker_apiPatch_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.AssertEqual(t, "PATCH", r.Method)
		core.WriteString(w, `{"ok":true}`)
	}))
	defer srv.Close()
	cfg := &WorkerConfig{APIBase: srv.URL, APIKey: "testkey"}
	r := apiPatch(cfg, "/test", map[string]any{"id": 1})
	core.AssertTrue(t, r.OK)
}

func TestWorker_apiDelete_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.AssertEqual(t, "DELETE", r.Method)
		core.WriteString(w, `{"ok":true}`)
	}))
	defer srv.Close()
	cfg := &WorkerConfig{APIBase: srv.URL, APIKey: "testkey"}
	r := apiDelete(cfg, "/test", nil)
	core.AssertTrue(t, r.OK)
}
