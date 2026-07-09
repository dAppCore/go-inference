// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"time"

	core "dappco.re/go"
	"dappco.re/go/inference/serving/compat"
)

// ExampleWithReadHeaderTimeout shows the option applied to a raw serveConfig.
func ExampleWithReadHeaderTimeout() {
	var cfg serveConfig
	WithReadHeaderTimeout(10 * time.Second)(&cfg)
	fmt.Println(cfg.readHeaderTimeout)
	// Output:
	// 10s
}

// ExampleWithWriteTimeout shows the option applied to a raw serveConfig.
func ExampleWithWriteTimeout() {
	var cfg serveConfig
	WithWriteTimeout(90 * time.Second)(&cfg)
	fmt.Println(cfg.writeTimeout)
	// Output:
	// 1m30s
}

// ExampleWithShutdownTimeout shows the option applied to a raw serveConfig.
func ExampleWithShutdownTimeout() {
	var cfg serveConfig
	WithShutdownTimeout(20 * time.Second)(&cfg)
	fmt.Println(cfg.shutdownTimeout)
	// Output:
	// 20s
}

// ExampleWithAdminToken shows the Bearer token landing on the config that
// Serve uses to decide whether to raise the /v1/admin/* auth wall.
func ExampleWithAdminToken() {
	var cfg serveConfig
	WithAdminToken("s3cret")(&cfg)
	fmt.Println(cfg.adminToken)
	// Output:
	// s3cret
}

// ExampleWithAdminHandler shows a custom handler mounted at /v1/admin/.
func ExampleWithAdminHandler() {
	var cfg serveConfig
	WithAdminHandler(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))(&cfg)
	rec := httptest.NewRecorder()
	cfg.adminHandler.ServeHTTP(rec, httptest.NewRequest(http.MethodGet, "/", nil))
	fmt.Println(rec.Code)
	// Output:
	// 200
}

// ExampleWithAdminConfig shows the host-owned Models callback wired through
// to the compatibility mux's admin config.
func ExampleWithAdminConfig() {
	var cfg serveConfig
	WithAdminConfig(compat.AdminConfig{Models: func() []string { return []string{"gemma3"} }})(&cfg)
	fmt.Println(cfg.admin.Models())
	// Output:
	// [gemma3]
}

// ExampleWithAuditLog shows an admin auth-deny line landing on the
// configured writer.
func ExampleWithAuditLog() {
	var cfg serveConfig
	buf := core.NewBuffer()
	WithAuditLog(buf)(&cfg)
	core.Print(cfg.audit, "serve admin: auth deny path=%s", "/v1/admin/machine")
	fmt.Print(buf.String())
	// Output:
	// serve admin: auth deny path=/v1/admin/machine
}

// ExampleServe shows a boot against an already-cancelled context: Serve binds
// the listener, immediately observes ctx.Done, and shuts down cleanly — the
// same graceful path a live serve takes on SIGINT, just collapsed to run
// instantly for the doc example.
func ExampleServe() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := Serve(ctx, "127.0.0.1:0", noModelResolver)
	fmt.Println(err)
	// Output:
	// <nil>
}

// ExampleRunServe shows the same graceful already-cancelled boot through the
// full cmd/lem composition, model-less (no --model given, so no engine load
// is required for the example to run standalone).
func ExampleRunServe() {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := RunServe(ctx, ServeConfig{Addr: "127.0.0.1:0"})
	fmt.Println(err)
	// Output:
	// <nil>
}
