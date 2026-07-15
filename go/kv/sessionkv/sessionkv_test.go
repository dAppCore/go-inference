// SPDX-License-Identifier: EUPL-1.2

package sessionkv

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/state"
	"github.com/gin-gonic/gin"
)

func TestOpenCreateReopenPersists(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "session.kv")
	ctx := context.Background()

	host, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open (create): %v", err)
	}
	if got := host.store.ChunkCount(); got != 0 {
		t.Fatalf("fresh store ChunkCount = %d, want 0", got)
	}
	if _, err := host.store.Put(ctx, "remembered", state.PutOptions{Kind: "note"}); err != nil {
		t.Fatalf("Put: %v", err)
	}
	if got := host.store.ChunkCount(); got != 1 {
		t.Fatalf("after Put ChunkCount = %d, want 1", got)
	}
	if err := host.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Reopen the same path — chunks persist (open-or-create reopens, never
	// truncates an existing store).
	reopened, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open (reopen): %v", err)
	}
	defer reopened.Close()
	if got := reopened.store.ChunkCount(); got != 1 {
		t.Fatalf("reopened ChunkCount = %d, want 1 (chunk should persist)", got)
	}
}

func TestOpenEmptyPath(t *testing.T) {
	if _, err := Open(context.Background(), ""); err == nil {
		t.Fatal("Open(\"\") should error (path required), got nil")
	}
}

func TestStatusAndChunkRefRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	path := core.PathJoin(t.TempDir(), "session.kv")
	ctx := context.Background()

	host, err := Open(ctx, path)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer host.Close()
	if _, err := host.store.Put(ctx, "remembered", state.PutOptions{}); err != nil {
		t.Fatalf("Put: %v", err)
	}

	r := gin.New()
	host.RegisterRoutes(r.Group(host.BasePath()))

	// status → 200, names the store path
	if code, body := doGet(r, "/v1/state/status"); code != http.StatusOK || !core.Contains(body, "session.kv") {
		t.Fatalf("status: code=%d body=%q", code, body)
	}
	// known chunk → 200 with its ref metadata (never content)
	if code, body := doGet(r, "/v1/state/chunks/1"); code != http.StatusOK || !core.Contains(body, "chunk_id") {
		t.Fatalf("chunks/1: code=%d body=%q", code, body)
	}
	// unknown chunk → 404
	if code, _ := doGet(r, "/v1/state/chunks/999"); code != http.StatusNotFound {
		t.Fatalf("chunks/999: code=%d, want 404", code)
	}
	// non-integer id → 400
	if code, _ := doGet(r, "/v1/state/chunks/abc"); code != http.StatusBadRequest {
		t.Fatalf("chunks/abc: code=%d, want 400", code)
	}
}

// TestDescribeMatchesRegisteredRoutes gates route-description drift: every route
// RegisterRoutes mounts must have a matching Describe entry and every Describe
// entry must name a registered route. A route added to one but not the other (a
// new endpoint left out of the OpenAPI spec, or a documented route that was
// never wired) fails here instead of shipping a lying spec.
func TestDescribeMatchesRegisteredRoutes(t *testing.T) {
	gin.SetMode(gin.TestMode)
	host, err := Open(context.Background(), core.PathJoin(t.TempDir(), "session.kv"))
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	defer host.Close()

	r := gin.New()
	host.RegisterRoutes(r.Group(host.BasePath()))

	// registered: the gin engine's actually-mounted routes, keyed by their full
	// path (BasePath + relative).
	registered := map[string]bool{}
	for _, ri := range r.Routes() {
		registered[ri.Method+" "+ri.Path] = false
	}
	if len(registered) == 0 {
		t.Fatal("RegisterRoutes mounted no routes")
	}

	// described: each Describe entry prefixed with BasePath to match gin's
	// full-path form — concatenation, not trimming, so the comparison is exact
	// in both directions.
	described := host.Describe()
	if len(described) == 0 {
		t.Fatal("Describe returned no route descriptions")
	}
	for _, d := range described {
		key := d.Method + " " + host.BasePath() + d.Path
		if _, ok := registered[key]; !ok {
			t.Errorf("Describe advertises %q but no route is registered for it", key)
			continue
		}
		registered[key] = true
	}
	for key, matched := range registered {
		if !matched {
			t.Errorf("route %q is registered but Describe does not advertise it", key)
		}
	}
}

func doGet(r *gin.Engine, path string) (int, string) {
	req := httptest.NewRequest(http.MethodGet, path, nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	return w.Code, w.Body.String()
}
