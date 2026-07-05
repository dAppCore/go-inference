// SPDX-License-Identifier: EUPL-1.2

// Package sessionkv hosts the durable session.kv (State) store for lthn-ai —
// the on-disk home for model memory: KV-cache bundles, knowledge-pack chunks,
// and book state. It owns a filestore-backed state.Store and exposes a small
// read-only inspection surface at /v1/state so an operator can see what the
// host holds without waking a model.
//
// The model reaches chunk *content* in-process at line speed (the Librarian
// token protocol, Wake/Sleep). This HTTP surface is for inspection only: it
// returns chunk metadata (refs) and counts, never chunk content, and binds
// wherever lthn-ai binds (loopback by default).
package sessionkv

import (
	"context"
	"net/http"
	"strconv"

	core "dappco.re/go"
	coreapi "dappco.re/go/api"
	"dappco.re/go/inference/state"
	"dappco.re/go/inference/state/filestore"
	"github.com/gin-gonic/gin"
)

// Host owns the durable State store and serves its inspection routes. It
// implements coreapi.RouteGroup so lthn-ai mounts it on the engine.
type Host struct {
	store *filestore.Store
	path  string
}

var _ coreapi.RouteGroup = (*Host)(nil)

// Open opens the session.kv store at path, creating it (and its parent dirs) on
// first run and reopening it otherwise. The store is an append-only state
// file-log (codec state/file-log).
//
//	host, err := sessionkv.Open(ctx, "/Users/me/Lethean/data/state/session.kv")
//	if err != nil {
//		return err
//	}
//	defer host.Close()
func Open(ctx context.Context, path string) (*Host, error) {
	if core.Trim(path) == "" {
		return nil, core.E("sessionkv.Open", "state store path is required", nil)
	}
	var (
		store *filestore.Store
		err   error
	)
	// Create truncates, so only Create when the file genuinely doesn't exist;
	// reopen an existing store to preserve its chunks.
	if core.Stat(path).OK {
		store, err = filestore.Open(ctx, path)
	} else {
		store, err = filestore.Create(ctx, path)
	}
	if err != nil {
		return nil, core.E("sessionkv.Open", "open state store", err)
	}
	return &Host{store: store, path: path}, nil
}

// Close releases the underlying store. Safe on a nil Host.
func (h *Host) Close() error {
	if h == nil || h.store == nil {
		return nil
	}
	return h.store.Close()
}

// Name implements coreapi.RouteGroup.
func (h *Host) Name() string { return "session-kv" }

// BasePath implements coreapi.RouteGroup.
func (h *Host) BasePath() string { return "/v1/state" }

// RegisterRoutes implements coreapi.RouteGroup.
func (h *Host) RegisterRoutes(rg *gin.RouterGroup) {
	if h == nil || rg == nil {
		return
	}
	rg.GET("/status", h.status)
	rg.GET("/chunks/:id", h.chunkRef)
}

// Describe implements coreapi.Describable for OpenAPI generation.
func (h *Host) Describe() []coreapi.RouteDescription {
	return []coreapi.RouteDescription{
		{Method: http.MethodGet, Path: "/status", Summary: "session.kv store status (path, codec, chunk count)", Tags: []string{"state"}},
		{Method: http.MethodGet, Path: "/chunks/:id", Summary: "Chunk metadata (ref) by id — never content", Tags: []string{"state"}},
	}
}

// statusResponse is the JSON body for GET /status. A typed struct avoids the
// per-request gin.H map allocation (header + bucket + per-value interface
// boxing); fields are declared in encoding/json's sorted-key order so the bytes
// on the wire are byte-for-byte identical to the map it replaced.
type statusResponse struct {
	Chunks int    `json:"chunks"`
	Codec  string `json:"codec"`
	Open   bool   `json:"open"`
	Path   string `json:"path"`
}

// status reports the store's location, codec, and chunk count — enough to
// confirm the memory host is live and how much it holds, with no content.
func (h *Host) status(c *gin.Context) {
	c.JSON(http.StatusOK, statusResponse{
		Chunks: h.store.ChunkCount(),
		Codec:  filestore.CodecFile,
		Open:   h.store != nil,
		Path:   h.path,
	})
}

// chunkRefResponse is the JSON body for a resolved chunk — its ref metadata
// only, never content. A typed struct avoids the per-request gin.H map alloc.
type chunkRefResponse struct {
	Ref state.ChunkRef `json:"ref"`
}

// chunkError is the JSON body for the chunkRef error branches. ID is omitted
// when unset (the bad-id 400), reproducing the gin.H maps it replaced
// byte-for-byte while dropping their per-request map allocation.
type chunkError struct {
	Error string `json:"error"`
	ID    int    `json:"id,omitempty"`
}

// chunkRef returns the metadata (ref) for one stored chunk — id, codec,
// segment, frame offset — never the chunk's content, which the model reaches
// in-process. A non-integer id is 400; an unknown id is 404.
func (h *Host) chunkRef(c *gin.Context) {
	id, err := strconv.Atoi(c.Param("id"))
	if err != nil || id < 1 {
		c.JSON(http.StatusBadRequest, chunkError{Error: "chunk id must be a positive integer"})
		return
	}
	chunk, rerr := h.store.Resolve(c.Request.Context(), id)
	if rerr != nil {
		c.JSON(http.StatusNotFound, chunkError{Error: "chunk not found", ID: id})
		return
	}
	c.JSON(http.StatusOK, chunkRefResponse{Ref: chunk.Ref})
}
