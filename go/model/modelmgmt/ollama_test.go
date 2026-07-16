package modelmgmt

import (
	"net/http"
	"net/http/httptest"

	"dappco.re/go"
	coreio "dappco.re/go/io"
)

func ollamaTestServer(t *core.T, createBody string, deleteStatus int) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case core.Contains(r.URL.Path, "/api/blobs/") && r.Method == http.MethodHead:
			w.WriteHeader(http.StatusNotFound)
		case core.Contains(r.URL.Path, "/api/blobs/") && r.Method == http.MethodPost:
			w.WriteHeader(http.StatusCreated)
		case r.URL.Path == "/api/create":
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(createBody))
		case r.URL.Path == "/api/delete":
			if deleteStatus == 0 {
				w.WriteHeader(http.StatusOK)
				return
			}
			w.WriteHeader(deleteStatus)
			_, _ = w.Write([]byte("delete failed"))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
}

func writePeftDir(t *core.T) string {
	t.Helper()
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "adapter_model.safetensors"), "model"))
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "adapter_config.json"), "{}"))
	return dir
}

// ollamaBlobServer serves the blob HEAD/POST endpoints with configurable
// status codes so ollamaUploadBlob's branches can be driven directly.
func ollamaBlobServer(t *core.T, headStatus, postStatus int) *httptest.Server {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch {
		case core.Contains(r.URL.Path, "/api/blobs/") && r.Method == http.MethodHead:
			w.WriteHeader(headStatus)
		case core.Contains(r.URL.Path, "/api/blobs/") && r.Method == http.MethodPost:
			w.WriteHeader(postStatus)
			if postStatus >= 300 {
				_, _ = w.Write([]byte("blob rejected"))
			}
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

func TestOllama_ollamaUploadBlob_Good(t *core.T) {
	// HEAD 200 means the blob already exists; upload is skipped.
	server := ollamaBlobServer(t, http.StatusOK, http.StatusCreated)
	dir := t.TempDir()
	file := core.JoinPath(dir, "blob.bin")
	core.RequireNoError(t, coreio.Local.Write(file, "payload"))
	r := ollamaUploadBlob(server.URL, file)
	requireResultOK(t, r)
	core.AssertContains(t, r.Value.(string), "sha256:")
}

// TestOllama_ollamaUploadBlob_Bad covers two distinct local-read failures:
// a missing file and a path that is a directory, not a file.
func TestOllama_ollamaUploadBlob_Bad(t *core.T) {
	server := ollamaBlobServer(t, http.StatusNotFound, http.StatusCreated)
	assertResultError(t, ollamaUploadBlob(server.URL, core.JoinPath(t.TempDir(), "missing.bin")), "read")
	assertResultError(t, ollamaUploadBlob(server.URL, t.TempDir()), "read")
}

func TestOllama_ollamaUploadBlob_Ugly(t *core.T) {
	// HEAD 404 forces the POST, which the server then rejects.
	server := ollamaBlobServer(t, http.StatusNotFound, http.StatusInternalServerError)
	dir := t.TempDir()
	file := core.JoinPath(dir, "blob.bin")
	core.RequireNoError(t, coreio.Local.Write(file, "payload"))
	assertResultError(t, ollamaUploadBlob(server.URL, file), "HTTP 500")
}

func TestOllama_OllamaCreateModel_Good(t *core.T) {
	server := ollamaTestServer(t, `{"status":"success"}`+"\n", 0)
	defer server.Close()
	assertResultOK(t, OllamaCreateModel(server.URL, "tmp-model", "base", writePeftDir(t)))
}

func TestOllama_OllamaCreateModel_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	assertResultError(t, OllamaCreateModel("http://127.0.0.1:1", "tmp-model", "base", t.TempDir()))
}

func TestOllama_OllamaCreateModel_Ugly(t *core.T) {
	server := ollamaTestServer(t, `{"error":"create failed"}`+"\n", 0)
	defer server.Close()
	assertResultError(t, OllamaCreateModel(server.URL, "tmp-model", "base", writePeftDir(t)))
}

func TestOllama_OllamaCreateModel_DecodeError(t *core.T) {
	// A create stream line that is not valid JSON surfaces a decode error.
	server := ollamaTestServer(t, `{not json}`+"\n", 0)
	defer server.Close()
	assertResultError(t, OllamaCreateModel(server.URL, "tmp-model", "base", writePeftDir(t)), "decode")
}

func TestOllama_OllamaCreateModel_CfgUploadFail(t *core.T) {
	// Only the safetensors blob exists, so the adapter-config upload fails its
	// read after the first upload succeeds.
	server := ollamaTestServer(t, `{"status":"success"}`+"\n", 0)
	defer server.Close()
	dir := t.TempDir()
	core.RequireNoError(t, coreio.Local.Write(core.JoinPath(dir, "adapter_model.safetensors"), "model"))
	assertResultError(t, OllamaCreateModel(server.URL, "tmp-model", "base", dir), "adapter config")
}

// TestOllama_OllamaCreateModel_Unreachable covers two refused-connection
// hosts, both asserting the wrapping message names the safetensors upload
// step specifically — disambiguating this failure mode from the
// "adapter config"/"decode" failures covered elsewhere in this file.
func TestOllama_OllamaCreateModel_Unreachable(t *core.T) {
	r := OllamaCreateModel("http://127.0.0.1:1", "tmp-model", "base", writePeftDir(t))
	assertResultError(t, r, "upload adapter safetensors")
	core.AssertFalse(t, r.OK)
	assertResultError(t, OllamaCreateModel("http://127.0.0.1:2", "tmp-model", "base", writePeftDir(t)), "upload adapter safetensors")
}

func TestOllama_OllamaDeleteModel_Good(t *core.T) {
	server := ollamaTestServer(t, `{"status":"success"}`+"\n", 0)
	defer server.Close()
	assertResultOK(t, OllamaDeleteModel(server.URL, "tmp-model"))
}

func TestOllama_OllamaDeleteModel_Bad(t *core.T) {
	stubName := t.Name()
	core.AssertNotEmpty(t, stubName)
	assertResultError(t, OllamaDeleteModel("http://127.0.0.1:1", "tmp-model"))
}

func TestOllama_OllamaDeleteModel_Ugly(t *core.T) {
	server := ollamaTestServer(t, `{"status":"success"}`+"\n", http.StatusInternalServerError)
	defer server.Close()
	assertResultError(t, OllamaDeleteModel(server.URL, "tmp-model"))
}

func TestOllama_HFBaseModelMap_Good(t *core.T) {
	// gemma-4-e2b is the one Gemma 4 size with a verified base HF id
	// (engine/hip/model/gemma4.OfficialE2BTargetModelID).
	core.AssertEqual(t, "google/gemma-4-E2B-it", HFBaseModelMap["gemma-4-e2b"])
	core.AssertEqual(t, "google/gemma-3-1b-it", HFBaseModelMap["gemma-3-1b"])
	core.AssertEqual(t, "google/gemma-3-27b-it", HFBaseModelMap["gemma-3-27b"])
}

func TestOllama_HFBaseModelMap_Bad(t *core.T) {
	// The other Gemma 4 sizes only appear in this repo as mlx-community
	// pre-quantized derivatives, never a confirmed base "google/..." repo —
	// they stay absent rather than guessed.
	for _, tag := range []string{"gemma-4-12b", "gemma-4-26b-a4b", "gemma-4-31b"} {
		_, ok := HFBaseModelMap[tag]
		core.AssertFalse(t, ok, tag)
	}
}

func TestOllama_OllamaBaseModelMap_Ugly(t *core.T) {
	// Nothing in this repo confirms an Ollama library name for Gemma 4 or
	// Qwen 3.5/3.6 — they stay absent rather than guessed.
	for _, tag := range []string{"gemma-4-e2b", "gemma-4-12b", "qwen-3-5"} {
		_, ok := OllamaBaseModelMap[tag]
		core.AssertFalse(t, ok, tag)
	}
}
