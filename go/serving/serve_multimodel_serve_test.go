// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"bytes"
	"context"
	"io"
	"net/http"
	"testing"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference"
	openai "dappco.re/go/inference/serving/provider/openai"
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
	// -scheduler is unset on this ServeConfig — the flag-unset contract (#35)
	// says NO scheduler is ever built, so the admin snapshot must carry no
	// scheduler fields at all.
	if core.Contains(adminBody, "scheduler_mode") || core.Contains(adminBody, "scheduler_stats") {
		t.Fatalf("/v1/admin/models body carries scheduler fields with -scheduler unset:\n%s", adminBody)
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

// TestRunServe_MultiModel_Scheduler_RoutesThroughPerModelInstance_Good is the
// full-stack proof for #35: a multi-model RunServe with BOTH -models-config
// (Models) AND -scheduler set builds a scheduler instance per resident model,
// a real chat-completions request routes through it end to end (real HTTP +
// JSON, not just the resolver seam), and the /v1/admin/models snapshot names
// the mode plus the routed model's own stats. Per-instance isolation (a
// DIFFERENT model's stats staying untouched) is pinned more precisely at the
// resolver level by TestMultiModelResolver_Scheduler_PerModelInstance_Good;
// this test's unique value is proving the wiring survives the real
// HTTP-handler → resolver → scheduler → admin-JSON round trip.
func TestRunServe_MultiModel_Scheduler_RoutesThroughPerModelInstance_Good(t *testing.T) {
	addr := freeListenAddr(t)
	log := core.NewBuffer()
	loader := func(path string, _ ...inference.LoadOption) (inference.TextModel, error) {
		return &mockTextModel{modelType: core.PathBase(path), tokens: []inference.Token{{Text: "hi"}}}, nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() {
		errCh <- RunServe(ctx, ServeConfig{
			Addr:       addr,
			Log:        log,
			AdminToken: "test-token",
			Loader:     loader,
			Scheduler:  "serial",
			Models: []ModelSpec{
				{ID: "qwen3", Path: "/m/qwen3"},
				{ID: "bge", Path: "/m/bge", Pinned: true},
			},
			MemoryCeiling: 1 << 30,
		})
	}()

	resp := waitForHTTPUp(t, "http://"+addr+"/v1/health")
	resp.Body.Close()

	if !core.Contains(log.String(), "scheduler serial") {
		t.Fatalf("boot log = %q, want a scheduler-mode notice", log.String())
	}

	// Route a real chat-completions request at qwen3 specifically.
	body := `{"model":"qwen3","messages":[{"role":"user","content":"hi"}]}`
	req, err := http.NewRequest(http.MethodPost, "http://"+addr+openai.DefaultChatCompletionsPath, bytes.NewReader([]byte(body)))
	if err != nil {
		t.Fatalf("new request: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	chatResp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("POST chat/completions: %v", err)
	}
	chatBody, _ := io.ReadAll(chatResp.Body)
	chatResp.Body.Close()
	if chatResp.StatusCode != http.StatusOK {
		t.Fatalf("chat completions for qwen3 = %d, want 200 (body: %s)", chatResp.StatusCode, chatBody)
	}

	// The admin snapshot names the scheduler mode and qwen3's own stats show
	// the routed request. bge is never addressed so it never loads at all
	// (lazy load — Pinned only exempts a RESIDENT model from eviction, it does
	// not force an eager load), hence no scheduler_stats entry for it either;
	// per-model isolation while BOTH are resident is pinned precisely by
	// TestMultiModelResolver_Scheduler_PerModelInstance_Good.
	adminBody := httpGetBody(t, "http://"+addr+"/v1/admin/models", "test-token")
	if !core.Contains(adminBody, `"scheduler_mode":"serial"`) {
		t.Fatalf("/v1/admin/models missing scheduler_mode:\n%s", adminBody)
	}
	if !core.Contains(adminBody, `"submitted":1`) {
		t.Fatalf("/v1/admin/models missing a submitted:1 scheduler_stats entry for the routed model:\n%s", adminBody)
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
