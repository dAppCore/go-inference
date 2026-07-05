// SPDX-License-Identifier: EUPL-1.2

package sessionkv

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/state"
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

func doGet(r *gin.Engine, path string) (int, string) {
	req := httptest.NewRequest(http.MethodGet, path, nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	return w.Code, w.Body.String()
}
