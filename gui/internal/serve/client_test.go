// SPDX-Licence-Identifier: EUPL-1.2

package serve

import (
	"context"
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
	"dappco.re/go/inference/serving/admin"
)

// statusServer returns an httptest server answering the admin status path with
// body, and rejecting any request whose Bearer token != wantToken (empty
// wantToken accepts any token).
func statusServer(body, wantToken string) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != admin.PathServeStatus {
			http.Error(w, "not found", http.StatusNotFound)
			return
		}
		if wantToken != "" && r.Header.Get("Authorization") != "Bearer "+wantToken {
			http.Error(w, "unauthorised", http.StatusUnauthorized)
			return
		}
		w.Header().Set("content-type", "application/json")
		core.WriteString(w, body)
	}))
}

func TestClient_NewClient_Good(t *core.T) {
	client := NewClient("http://127.0.0.1:36911", func() string { return "tok" })

	core.AssertNotNil(t, client)
	core.AssertEqual(t, "http://127.0.0.1:36911", client.baseURL)
}

func TestClient_NewClient_Bad(t *core.T) {
	client := NewClient("http://127.0.0.1:36911/", nil)

	core.AssertNotNil(t, client)
	core.AssertEqual(t, "http://127.0.0.1:36911", client.baseURL)
}

func TestClient_NewClient_Ugly(t *core.T) {
	client := NewClient("", func() string { return "" })

	core.AssertNotNil(t, client)
	core.AssertEqual(t, "", client.baseURL)
}

func TestClient_Client_Status_Good(t *core.T) {
	srv := statusServer(`{"model_path":"/models/gemma","runtime":"go-inference","loaded_at_unix":1700000000}`, "test-token")
	defer srv.Close()

	client := NewClient(srv.URL, func() string { return "test-token" })
	st, err := client.Status(context.Background())

	core.AssertNoError(t, err)
	core.AssertTrue(t, st.Up)
	core.AssertEqual(t, "/models/gemma", st.ModelPath)
	core.AssertEqual(t, "go-inference", st.Runtime)
	core.AssertEqual(t, int64(1700000000), st.LoadedAt)
}

func TestClient_Client_Status_Bad(t *core.T) {
	// A server that is created then closed leaves nothing listening — the
	// daemon-down path must resolve to Up:false with a nil error.
	srv := statusServer(`{}`, "")
	url := srv.URL
	srv.Close()

	client := NewClient(url, func() string { return "" })
	st, err := client.Status(context.Background())

	core.AssertNoError(t, err)
	core.AssertFalse(t, st.Up)
}

func TestClient_Client_Status_Ugly(t *core.T) {
	// Reachable but rejects the token — a real error, distinct from "down".
	srv := statusServer(`{}`, "right-token")
	defer srv.Close()

	client := NewClient(srv.URL, func() string { return "wrong-token" })
	st, err := client.Status(context.Background())

	core.AssertError(t, err)
	core.AssertFalse(t, st.Up)
	core.AssertContains(t, core.ErrorMessage(err), "401")
}

func TestClient_Client_Status_ModelLess(t *core.T) {
	// serve started model-less: Up is true, ModelPath empty.
	srv := statusServer(`{"model_path":"","runtime":"go-inference"}`, "")
	defer srv.Close()

	client := NewClient(srv.URL, nil)
	st, err := client.Status(context.Background())

	core.AssertNoError(t, err)
	core.AssertTrue(t, st.Up)
	core.AssertEqual(t, "", st.ModelPath)
}
