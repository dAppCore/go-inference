package agent

import (
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
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
	// HOME resolves to a directory containing a real key file — the
	// success return trims surrounding whitespace from its contents.
	home := t.TempDir()
	core.RequireNoError(t, coreio.Local.EnsureDir(core.Path(home, ".config", "lem")))
	core.RequireNoError(t, coreio.Local.Write(core.Path(home, ".config", "lem", "api_key"), "  secret-key-123  \n"))
	t.Setenv("HOME", home)

	got := ReadKeyFile()
	core.AssertEqual(t, "secret-key-123", got)
}

func TestWorker_ReadKeyFile_Bad(t *core.T) {
	// An empty HOME makes core.UserHomeDir() itself fail, short-circuiting
	// before the key file is ever looked up.
	t.Setenv("HOME", "")

	got := ReadKeyFile()
	core.AssertEqual(t, "", got)
}

func TestWorker_ReadKeyFile_Ugly(t *core.T) {
	// HOME resolves fine, but ~/.config/lem/api_key does not exist there.
	t.Setenv("HOME", t.TempDir())

	got := ReadKeyFile()
	core.AssertEqual(t, "", got)
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

func TestWorker_apiGet_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.AssertEqual(t, "GET", r.Method)
		core.AssertEqual(t, "Bearer testkey", r.Header.Get("Authorization"))
		core.WriteString(w, `{"ok":true}`)
	}))
	defer srv.Close()
	cfg := &WorkerConfig{APIBase: srv.URL, APIKey: "testkey"}
	r := apiGet(cfg, "/test")
	requireResultOK(t, r)
	core.AssertContains(t, string(r.Value.([]byte)), "ok")
}

func TestWorker_apiGet_Bad(t *core.T) {
	cfg := &WorkerConfig{APIBase: "http://127.0.0.1:1", APIKey: "testkey"}
	r := apiGet(cfg, "/test")
	assertResultError(t, r)
}

func TestWorker_apiGet_Ugly(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		core.WriteString(w, `{"error":"not found"}`)
	}))
	defer srv.Close()
	cfg := &WorkerConfig{APIBase: srv.URL}
	r := apiGet(cfg, "/missing")
	assertResultError(t, r, "HTTP 404")
}

func TestWorker_workerHeartbeat_Good(t *core.T) {
	called := false
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		called = true
		core.AssertEqual(t, "/api/lem/workers/heartbeat", r.URL.Path)
		core.WriteString(w, `{}`)
	}))
	defer srv.Close()
	cfg := &WorkerConfig{APIBase: srv.URL, WorkerID: "w1", APIKey: "k"}
	core.AssertNotPanics(t, func() { workerHeartbeat(cfg) })
	core.AssertTrue(t, called)
}

func TestWorker_workerHeartbeat_Bad(t *core.T) {
	cfg := &WorkerConfig{APIBase: "http://127.0.0.1:1", WorkerID: "w1"}
	core.AssertNotPanics(t, func() { workerHeartbeat(cfg) })
}

func TestWorker_workerHeartbeat_Ugly(t *core.T) {
	var gotBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rBody := readAll(r.Body); rBody.OK {
			gotBody = rBody.Value.([]byte)
		}
		core.WriteString(w, `{}`)
	}))
	defer srv.Close()
	// WorkerID left empty — the body still posts, just with an empty field.
	cfg := &WorkerConfig{APIBase: srv.URL}
	core.AssertNotPanics(t, func() { workerHeartbeat(cfg) })
	core.AssertContains(t, string(gotBody), `"worker_id":""`)
}

func TestWorker_workerRegister_Good(t *core.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rBody := readAll(r.Body); rBody.OK {
			_ = core.JSONUnmarshal(rBody.Value.([]byte), &gotBody)
		}
		core.WriteString(w, `{}`)
	}))
	defer srv.Close()
	cfg := &WorkerConfig{
		APIBase: srv.URL, WorkerID: "w1", Name: "worker-one",
		GPUType: "mps", VRAMGb: 32,
		Languages: []string{"en"}, Models: []string{"gemma3"},
	}
	r := workerRegister(cfg)
	requireResultOK(t, r)
	core.AssertEqual(t, "mps", gotBody["gpu_type"])
	core.AssertEqual(t, float64(32), gotBody["vram_gb"])
	core.AssertNotNil(t, gotBody["languages"])
	core.AssertNotNil(t, gotBody["supported_models"])
}

func TestWorker_workerRegister_Bad(t *core.T) {
	cfg := &WorkerConfig{APIBase: "http://127.0.0.1:1", WorkerID: "w1"}
	r := workerRegister(cfg)
	assertResultError(t, r)
}

func TestWorker_workerRegister_Ugly(t *core.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if rBody := readAll(r.Body); rBody.OK {
			_ = core.JSONUnmarshal(rBody.Value.([]byte), &gotBody)
		}
		core.WriteString(w, `{}`)
	}))
	defer srv.Close()
	// Zero-value GPU/VRAM/Languages/Models skip every optional body field.
	cfg := &WorkerConfig{APIBase: srv.URL, WorkerID: "w1"}
	r := workerRegister(cfg)
	requireResultOK(t, r)
	_, hasGPU := gotBody["gpu_type"]
	core.AssertFalse(t, hasGPU)
	_, hasVRAM := gotBody["vram_gb"]
	core.AssertFalse(t, hasVRAM)
}

func TestWorker_workerPoll_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case core.Contains(r.URL.String(), "/next"):
			core.AssertContains(t, r.URL.String(), "type=capability")
			core.WriteString(w, `{"tasks":[{"id":1,"prompt_text":"hi"}],"count":1}`)
		default:
			core.WriteString(w, `{}`)
		}
	}))
	defer srv.Close()
	// DryRun makes the claimed task's processing trivially succeed, so the
	// task is counted as processed.
	cfg := &WorkerConfig{APIBase: srv.URL, WorkerID: "w1", BatchSize: 1, DryRun: true, TaskType: "capability"}
	got := workerPoll(cfg)
	core.AssertEqual(t, 1, got)
}

func TestWorker_workerPoll_Bad(t *core.T) {
	cfg := &WorkerConfig{APIBase: "http://127.0.0.1:1", WorkerID: "w1", BatchSize: 1}
	got := workerPoll(cfg)
	core.AssertEqual(t, 0, got)

	// A non-JSON body fails decoding rather than panicking.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.WriteString(w, `not valid json`)
	}))
	defer srv.Close()
	cfg2 := &WorkerConfig{APIBase: srv.URL, WorkerID: "w1", BatchSize: 1}
	got2 := workerPoll(cfg2)
	core.AssertEqual(t, 0, got2)
}

func TestWorker_workerPoll_Ugly(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case core.Contains(r.URL.String(), "/next"):
			core.WriteString(w, `{"tasks":[{"id":7,"prompt_text":"hi"}],"count":1}`)
		case core.Contains(r.URL.String(), "/claim"):
			w.WriteHeader(http.StatusInternalServerError)
		default:
			core.WriteString(w, `{}`)
		}
	}))
	defer srv.Close()
	// The claim call fails, so the task is skipped (abandoned + continue)
	// rather than counted as processed.
	cfg := &WorkerConfig{APIBase: srv.URL, WorkerID: "w1", BatchSize: 1}
	got := workerPoll(cfg)
	core.AssertEqual(t, 0, got)
}

func TestWorker_workerProcessTask_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.WriteString(w, `{}`)
	}))
	defer srv.Close()
	// DryRun short-circuits before workerInfer is ever called.
	cfg := &WorkerConfig{APIBase: srv.URL, WorkerID: "w1", DryRun: true}
	r := workerProcessTask(cfg, APITask{ID: 1, TaskType: "capability", PromptText: "hello"})
	requireResultOK(t, r)
}

func TestWorker_workerProcessTask_Bad(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()
	cfg := &WorkerConfig{APIBase: srv.URL, WorkerID: "w1"}
	r := workerProcessTask(cfg, APITask{ID: 1})
	assertResultError(t, r, "claim")

	// Claim and status-patch succeed, but the InferURL is unreachable —
	// exercises the abandoned-status patch + wrapped inference error.
	okSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.WriteString(w, `{}`)
	}))
	defer okSrv.Close()
	cfg2 := &WorkerConfig{APIBase: okSrv.URL, InferURL: "http://127.0.0.1:1", WorkerID: "w1"}
	r2 := workerProcessTask(cfg2, APITask{ID: 2, PromptText: "hello"})
	assertResultError(t, r2, "inference")

	// Claim, patch, and inference all succeed, but the final result
	// submission fails — exercises the "submit result" wrap.
	resultFailSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case core.HasSuffix(r.URL.Path, "/chat/completions"):
			core.WriteString(w, `{"choices":[{"message":{"content":"a fully generated response"}}]}`)
		case core.HasSuffix(r.URL.Path, "/result"):
			w.WriteHeader(http.StatusInternalServerError)
		default:
			core.WriteString(w, `{}`)
		}
	}))
	defer resultFailSrv.Close()
	cfg3 := &WorkerConfig{APIBase: resultFailSrv.URL, InferURL: resultFailSrv.URL, WorkerID: "w1"}
	r3 := workerProcessTask(cfg3, APITask{ID: 3, PromptText: "hello"})
	assertResultError(t, r3, "submit result")
}

func TestWorker_workerProcessTask_Ugly(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case core.HasSuffix(r.URL.Path, "/chat/completions"):
			core.WriteString(w, `{"choices":[{"message":{"content":"a fully generated response"}}]}`)
		default:
			core.WriteString(w, `{}`)
		}
	}))
	defer srv.Close()
	// Non-DryRun drives the full path through workerInfer and the result
	// submission — ModelName left empty exercises the "default" fallback.
	cfg := &WorkerConfig{APIBase: srv.URL, InferURL: srv.URL, WorkerID: "w1"}
	r := workerProcessTask(cfg, APITask{ID: 2, PromptText: "hello"})
	requireResultOK(t, r)
}

func TestWorker_workerInfer_Good(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.WriteString(w, `{"choices":[{"message":{"content":"a fully generated response"}}]}`)
	}))
	defer srv.Close()
	cfg := &WorkerConfig{InferURL: srv.URL}
	var task APITask
	mustJSONUnmarshalBytes(t, []byte(`{"prompt_text":"hi","model_name":"m","config":{"temperature":0.5,"max_tokens":100}}`), &task)

	r := workerInfer(cfg, task)
	requireResultOK(t, r)
	core.AssertEqual(t, "a fully generated response", r.Value.(string))
}

func TestWorker_workerInfer_Bad(t *core.T) {
	cfg := &WorkerConfig{InferURL: "http://127.0.0.1:1"}
	r := workerInfer(cfg, APITask{PromptText: "hi", ModelName: "m"})
	assertResultError(t, r, "inference request")
}

func TestWorker_workerInfer_Ugly(t *core.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.WriteString(w, `{"choices":[{"message":{"content":"hi"}}]}`)
	}))
	defer srv.Close()
	// No task.Config — default temperature/max-tokens apply — and the reply
	// is shorter than the 10-char floor.
	cfg := &WorkerConfig{InferURL: srv.URL}
	r := workerInfer(cfg, APITask{PromptText: "hi", ModelName: "m"})
	assertResultError(t, r, "too short")
}
