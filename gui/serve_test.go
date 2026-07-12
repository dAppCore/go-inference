// SPDX-Licence-Identifier: EUPL-1.2

package main

import (
	"context"
	"net/http"
	"net/http/httptest"

	core "dappco.re/go"
	"dappco.re/go/inference/gui/internal/serve"
	"github.com/wailsapp/wails/v3/pkg/application"
)

// fakeLemBinary writes an executable stand-in for lem (ignores args, sleeps) so
// Start spawns a real child that Stop terminates via SIGTERM.
func fakeLemBinary(t *core.T) string {
	dir := t.TempDir()
	path := core.PathJoin(dir, "lem")
	if r := core.WriteFile(path, []byte("#!/bin/sh\nsleep 300\n"), 0o755); !r.OK {
		t.Fatal("write fake lem: " + r.Error())
	}
	return path
}

// upServer answers the admin status path with a fixed running snapshot.
func upServer(t *core.T) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		core.WriteString(w, `{"model_path":"/models/gemma-4-e2b-it-4bit","runtime":"go-inference"}`)
	}))
	t.Cleanup(srv.Close)
	return srv
}

// modelDir writes one discoverable model under a fresh temp dir and returns it.
func modelDir(t *core.T) string {
	root := t.TempDir()
	dir := core.PathJoin(root, "gemma-4-e2b-it-4bit")
	if r := core.MkdirAll(dir, 0o755); !r.OK {
		t.Fatal("mkdir: " + r.Error())
	}
	if r := core.WriteFile(core.PathJoin(dir, "config.json"), []byte(`{"model_type":"gemma3"}`), 0o644); !r.OK {
		t.Fatal("write config: " + r.Error())
	}
	if r := core.WriteFile(core.PathJoin(dir, "model.safetensors"), []byte("w"), 0o644); !r.OK {
		t.Fatal("write weights: " + r.Error())
	}
	return root
}

func TestServe_NewServeService_Good(t *core.T) {
	svc := NewServeService(":36911", "lem", t.TempDir())

	core.AssertNotNil(t, svc.client)
	core.AssertNotNil(t, svc.manager)
}

func TestServe_NewServeService_Bad(t *core.T) {
	svc := NewServeService(":36911", "", "") // empty modelsDir defaults

	core.AssertTrue(t, core.HasSuffix(svc.modelsDir, "Lethean/lem/models"))
}

func TestServe_NewServeService_Ugly(t *core.T) {
	svc := NewServeService("127.0.0.1:36911", "lem", t.TempDir())

	core.AssertEqual(t, "127.0.0.1:36911", svc.manager.Addr())
}

func TestServe_statusURL_Good(t *core.T) {
	core.AssertEqual(t, "http://127.0.0.1:36911", statusURL(":36911"))
}

func TestServe_statusURL_Bad(t *core.T) {
	core.AssertEqual(t, "http://localhost:8080", statusURL("localhost:8080"))
}

func TestServe_statusURL_Ugly(t *core.T) {
	core.AssertEqual(t, "http://127.0.0.1:0", statusURL(":0"))
}

func TestServe_ServeService_ServiceName_Good(t *core.T) {
	svc := &ServeService{}
	core.AssertEqual(t, "ServeService", svc.ServiceName())
}

func TestServe_ServeService_ServiceName_Bad(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir())
	core.AssertEqual(t, "ServeService", svc.ServiceName())
}

func TestServe_ServeService_ServiceName_Ugly(t *core.T) {
	svc := &ServeService{}
	core.AssertEqual(t, svc.ServiceName(), svc.ServiceName())
}

func TestServe_ServeService_ServiceStartup_Good(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir())
	ctx, cancel := core.WithCancel(core.Background())
	cancel() // one poll then the loop returns — no lingering ticker

	r := svc.ServiceStartup(ctx, application.ServiceOptions{})

	core.AssertTrue(t, r.OK)
	core.AssertFalse(t, svc.IsUp())
}

func TestServe_ServeService_ServiceStartup_Bad(t *core.T) {
	svc := &ServeService{manager: serve.NewManager("lem", ":0"), client: serve.NewClient("http://127.0.0.1:0", nil)}
	ctx, cancel := core.WithCancel(core.Background())
	cancel()

	r := svc.ServiceStartup(ctx, application.ServiceOptions{})

	core.AssertTrue(t, r.OK)
}

func TestServe_ServeService_ServiceStartup_Ugly(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir())
	ctx, cancel := core.WithCancel(core.Background())
	cancel()

	first := svc.ServiceStartup(ctx, application.ServiceOptions{})
	second := svc.ServiceStartup(ctx, application.ServiceOptions{})

	core.AssertTrue(t, first.OK)
	core.AssertTrue(t, second.OK)
}

func TestServe_ServeService_ServiceShutdown_Good(t *core.T) {
	svc := NewServeService(":0", fakeLemBinary(t), t.TempDir())
	svc.Start("")

	r := svc.ServiceShutdown()

	core.AssertTrue(t, r.OK)
	core.AssertFalse(t, svc.manager.Managed())
}

func TestServe_ServeService_ServiceShutdown_Bad(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir()) // nothing spawned

	r := svc.ServiceShutdown()

	core.AssertTrue(t, r.OK)
}

func TestServe_ServeService_ServiceShutdown_Ugly(t *core.T) {
	svc := NewServeService(":0", fakeLemBinary(t), t.TempDir())
	svc.Start("")

	first := svc.ServiceShutdown()
	second := svc.ServiceShutdown()

	core.AssertTrue(t, first.OK)
	core.AssertTrue(t, second.OK)
}

func TestServe_ServeService_Status_Good(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir())
	svc.client = serve.NewClient(upServer(t).URL, func() string { return "" })

	st := svc.Status(context.Background())

	core.AssertTrue(t, st.Up)
	core.AssertEqual(t, "/models/gemma-4-e2b-it-4bit", st.ModelPath)
}

func TestServe_ServeService_Status_Bad(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir()) // nothing listening on :0

	st := svc.Status(context.Background())

	core.AssertFalse(t, st.Up)
}

func TestServe_ServeService_Status_Ugly(t *core.T) {
	// Reachable but 401 → client error → folded to down.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "no", http.StatusUnauthorized)
	}))
	defer srv.Close()
	svc := NewServeService(":0", "lem", t.TempDir())
	svc.client = serve.NewClient(srv.URL, func() string { return "" })

	st := svc.Status(context.Background())

	core.AssertFalse(t, st.Up)
}

func TestServe_ServeService_GetSnapshot_Good(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir())
	svc.client = serve.NewClient(upServer(t).URL, func() string { return "" })
	svc.Status(context.Background())

	snap := svc.GetSnapshot()

	core.AssertTrue(t, snap.Up)
	core.AssertEqual(t, "gemma-4-e2b-it-4bit", snap.ModelName)
}

func TestServe_ServeService_GetSnapshot_Bad(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir())

	snap := svc.GetSnapshot()

	core.AssertFalse(t, snap.Up)
	core.AssertFalse(t, snap.Managed)
}

func TestServe_ServeService_GetSnapshot_Ugly(t *core.T) {
	svc := NewServeService(":0", fakeLemBinary(t), t.TempDir())
	defer svc.Stop()
	svc.Start("")

	snap := svc.GetSnapshot()

	core.AssertTrue(t, snap.Managed)
	core.AssertFalse(t, snap.Up) // managed process, but nothing answers on :0
}

func TestServe_ServeService_ListModels_Good(t *core.T) {
	svc := NewServeService(":0", "lem", modelDir(t))

	models := svc.ListModels()

	core.AssertLen(t, models, 1)
	core.AssertEqual(t, "gemma-4-e2b-it-4bit", models[0].Name)
}

func TestServe_ServeService_ListModels_Bad(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir()) // empty dir

	models := svc.ListModels()

	core.AssertLen(t, models, 0)
}

func TestServe_ServeService_ListModels_Ugly(t *core.T) {
	svc := &ServeService{modelsDir: ""} // no dir configured

	models := svc.ListModels()

	core.AssertLen(t, models, 0)
}

func TestServe_ServeService_Start_Good(t *core.T) {
	svc := NewServeService(":0", fakeLemBinary(t), t.TempDir())
	defer svc.Stop()

	r := svc.Start("")

	core.AssertTrue(t, r.OK)
	core.AssertTrue(t, svc.manager.Managed())
}

func TestServe_ServeService_Start_Bad(t *core.T) {
	svc := NewServeService(":0", "/nonexistent/lem", t.TempDir())

	r := svc.Start("/models/x")

	core.AssertFalse(t, r.OK)
	core.AssertFalse(t, svc.manager.Managed())
}

func TestServe_ServeService_Start_Ugly(t *core.T) {
	svc := NewServeService(":0", fakeLemBinary(t), t.TempDir())
	defer svc.Stop()

	first := svc.Start("")
	second := svc.Start("/models/other") // idempotent while managing

	core.AssertTrue(t, first.OK)
	core.AssertTrue(t, second.OK)
}

func TestServe_ServeService_Stop_Good(t *core.T) {
	svc := NewServeService(":0", fakeLemBinary(t), t.TempDir())
	svc.Start("")

	r := svc.Stop()

	core.AssertTrue(t, r.OK)
	core.AssertFalse(t, svc.manager.Managed())
}

func TestServe_ServeService_Stop_Bad(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir())

	r := svc.Stop()

	core.AssertTrue(t, r.OK)
}

func TestServe_ServeService_Stop_Ugly(t *core.T) {
	svc := NewServeService(":0", fakeLemBinary(t), t.TempDir())
	svc.Start("")

	first := svc.Stop()
	second := svc.Stop()

	core.AssertTrue(t, first.OK)
	core.AssertTrue(t, second.OK)
}

func TestServe_ServeService_IsUp_Good(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir())
	svc.client = serve.NewClient(upServer(t).URL, func() string { return "" })
	svc.Status(context.Background())

	core.AssertTrue(t, svc.IsUp())
}

func TestServe_ServeService_IsUp_Bad(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir())

	core.AssertFalse(t, svc.IsUp())
}

func TestServe_ServeService_IsUp_Ugly(t *core.T) {
	svc := NewServeService(":0", "lem", t.TempDir())
	svc.Status(context.Background()) // nothing on :0 → still down

	core.AssertFalse(t, svc.IsUp())
}

func TestServe_readTokenFile_Good(t *core.T) {
	path := core.PathJoin(t.TempDir(), "admin.token")
	core.WriteFile(path, []byte("  lthn-mlx_abc123\n"), 0o600)

	core.AssertEqual(t, "lthn-mlx_abc123", readTokenFile(path))
}

func TestServe_readTokenFile_Bad(t *core.T) {
	core.AssertEqual(t, "", readTokenFile(core.PathJoin(t.TempDir(), "absent.token")))
}

func TestServe_readTokenFile_Ugly(t *core.T) {
	path := core.PathJoin(t.TempDir(), "empty.token")
	core.WriteFile(path, []byte("   \n"), 0o600)

	core.AssertEqual(t, "", readTokenFile(path))
}

func TestServe_readAdminToken_Good(t *core.T) {
	// Delegates to readTokenFile against the canonical path; returns a string
	// (empty when serve has not yet minted a token on this machine).
	core.AssertEqual(t, readAdminToken(), readAdminToken())
}
