// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"io"
	"net/http"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// TestRunServe_MultiModel_ListsModelsAndAdmin_Good is the end-to-end wiring
// proof: a multi-model RunServe advertises every configured model plus its
// profile combinations on /v1/models, and the /v1/admin/models control plane
// (behind the Bearer wall) reports the registry — confirming RunServe routes the
// multi-model branch, hostServe mounts the model control plane, and the compat
// mux's /v1/models list is fed from the registry.
func TestRunServe_MultiModel_ListsModelsAndAdmin_Good(t *testing.T) {
	addr := freeListenAddr(t)
	log := core.NewBuffer()
	loader := func(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
		return &mockTextModel{modelType: core.PathBase(path)}, nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		errCh <- RunServe(ctx, ServeConfig{
			Addr:       addr,
			Log:        log,
			AdminToken: "test-token",
			Loader:     loader,
			Models: []ModelSpec{
				{ID: "qwen3", Path: "/m/qwen3", Aliases: []string{"qwen"}, Profiles: map[string]ProfileConfig{"creative": {Temperature: ptrFloat32(0.9)}}},
				{ID: "bge", Path: "/m/bge", Pinned: true},
			},
			MemoryCeiling: 1 << 30,
		})
	}()

	resp := waitForHTTPUp(t, "http://"+addr+"/v1/health")
	resp.Body.Close()

	// /v1/models — no auth; lists residents + profiles.
	modelsBody := httpGetBody(t, "http://"+addr+"/v1/models", "")
	for _, want := range []string{`"qwen3"`, `"qwen3:creative"`, `"bge"`} {
		if !core.Contains(modelsBody, want) {
			t.Fatalf("/v1/models body missing %s:\n%s", want, modelsBody)
		}
	}

	// /v1/admin/models — Bearer-gated; reports the registry snapshot.
	adminBody := httpGetBody(t, "http://"+addr+"/v1/admin/models", "test-token")
	for _, want := range []string{`"id":"qwen3"`, `"id":"bge"`, `"pinned":true`} {
		if !core.Contains(adminBody, want) {
			t.Fatalf("/v1/admin/models body missing %s:\n%s", want, adminBody)
		}
	}

	// The boot log names the multi-model mode.
	if !core.Contains(log.String(), "multi-model") {
		t.Fatalf("boot log = %q, want a multi-model notice", log.String())
	}

	cancel()
	select {
	case err := <-errCh:
		if err != nil {
			t.Fatalf("RunServe returned %v after cancel, want nil", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("RunServe did not shut down within 3s of cancel")
	}
}

// TestRunServe_MultiModel_AdminUnmounted_WhenSingle_Good proves the multi-model
// control plane stays absent on a single-model serve (Models empty) — the
// zero-behaviour-change guarantee at the route level.
func TestRunServe_MultiModel_AdminUnmounted_WhenSingle_Good(t *testing.T) {
	addr := freeListenAddr(t)
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		errCh <- RunServe(ctx, ServeConfig{Addr: addr, AdminToken: "test-token", Log: core.NewBuffer()})
	}()
	resp := waitForHTTPUp(t, "http://"+addr+"/v1/health")
	resp.Body.Close()

	req, _ := http.NewRequest(http.MethodGet, "http://"+addr+"/v1/admin/models", nil)
	req.Header.Set("Authorization", "Bearer test-token")
	got, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET /v1/admin/models: %v", err)
	}
	got.Body.Close()
	if got.StatusCode != http.StatusNotFound {
		t.Fatalf("single-model /v1/admin/models = %d, want 404 (route unmounted)", got.StatusCode)
	}

	cancel()
	<-errCh
}

// httpGetBody GETs url (optionally with a Bearer token) and returns the body.
func httpGetBody(t *testing.T, url, token string) string {
	t.Helper()
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		t.Fatalf("new request %s: %v", url, err)
	}
	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("GET %s: %v", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("GET %s = %d, want 200", url, resp.StatusCode)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read %s body: %v", url, err)
	}
	return string(body)
}
