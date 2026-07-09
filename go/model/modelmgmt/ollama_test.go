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
