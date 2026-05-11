// SPDX-Licence-Identifier: EUPL-1.2

// Package state defines portable model-state storage and lifecycle contracts.
package state

import (
	"context"
	stdio "io"

	core "dappco.re/go"
)

var ErrChunkNotFound = core.NewError("memvid chunk not found")

const (
	CodecMemory  = "memory/plaintext"
	CodecQRVideo = "memvid/qr-video"
)

type Store interface {
	Get(ctx context.Context, chunkID int) (string, error)
}

type Resolver interface {
	Resolve(ctx context.Context, chunkID int) (Chunk, error)
}

type URIResolver interface {
	ResolveURI(ctx context.Context, uri string) (Chunk, error)
}

type Writer interface {
	Put(ctx context.Context, text string, opts PutOptions) (ChunkRef, error)
}

type BinaryResolver interface {
	ResolveBytes(ctx context.Context, chunkID int) (Chunk, error)
}

type RefBinaryResolver interface {
	ResolveRefBytes(ctx context.Context, ref ChunkRef) (Chunk, error)
}

type BinaryWriter interface {
	PutBytes(ctx context.Context, data []byte, opts PutOptions) (ChunkRef, error)
}

type BinaryStreamWriter interface {
	PutBytesStream(ctx context.Context, payloadSize int, opts PutOptions, write func(stdio.Writer) error) (ChunkRef, error)
}

type PutOptions struct {
	URI    string            `json:"uri,omitempty"`
	Title  string            `json:"title,omitempty"`
	Kind   string            `json:"kind,omitempty"`
	Track  string            `json:"track,omitempty"`
	Tags   map[string]string `json:"tags,omitempty"`
	Labels []string          `json:"labels,omitempty"`
}

type Chunk struct {
	Ref  ChunkRef `json:"ref"`
	Text string   `json:"text"`
	Data []byte   `json:"data,omitempty"`
}

type ChunkRef struct {
	ChunkID        int    `json:"chunk_id"`
	FrameOffset    uint64 `json:"frame_offset,omitempty"`
	HasFrameOffset bool   `json:"has_frame_offset,omitempty"`
	Codec          string `json:"codec,omitempty"`
	Segment        string `json:"segment,omitempty"`
}

type ChunkNotFoundError struct {
	ID int
}

func (e *ChunkNotFoundError) Error() string {
	return core.Sprintf("memvid chunk %d not found", e.ID)
}

func (e *ChunkNotFoundError) Unwrap() error {
	return ErrChunkNotFound
}

type URIChunkNotFoundError struct {
	URI string
}

func (e *URIChunkNotFoundError) Error() string {
	if e.URI == "" {
		return "memvid chunk URI not found"
	}
	return core.Sprintf("memvid chunk URI %q not found", e.URI)
}

func (e *URIChunkNotFoundError) Unwrap() error {
	return ErrChunkNotFound
}

func Resolve(ctx context.Context, store Store, chunkID int) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	if resolver, ok := store.(Resolver); ok {
		return resolver.Resolve(ctx, chunkID)
	}
	text, err := store.Get(ctx, chunkID)
	if err != nil {
		return Chunk{}, err
	}
	return Chunk{
		Ref:  ChunkRef{ChunkID: chunkID},
		Text: text,
	}, nil
}

func ResolveBytes(ctx context.Context, store Store, chunkID int) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return Chunk{}, &ChunkNotFoundError{ID: chunkID}
	}
	if resolver, ok := store.(BinaryResolver); ok {
		chunk, err := resolver.ResolveBytes(ctx, chunkID)
		if err != nil {
			return Chunk{}, err
		}
		if len(chunk.Data) == 0 && chunk.Text != "" {
			chunk.Data = []byte(chunk.Text)
		}
		return chunk, nil
	}
	chunk, err := Resolve(ctx, store, chunkID)
	if err != nil {
		return Chunk{}, err
	}
	if len(chunk.Data) == 0 && chunk.Text != "" {
		chunk.Data = []byte(chunk.Text)
	}
	return chunk, nil
}

func ResolveRefBytes(ctx context.Context, store Store, ref ChunkRef) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return Chunk{}, &ChunkNotFoundError{ID: ref.ChunkID}
	}
	if resolver, ok := store.(RefBinaryResolver); ok {
		chunk, err := resolver.ResolveRefBytes(ctx, ref)
		if err != nil {
			return Chunk{}, err
		}
		if len(chunk.Data) == 0 && chunk.Text != "" {
			chunk.Data = []byte(chunk.Text)
		}
		return chunk, nil
	}
	if ref.ChunkID == 0 {
		return Chunk{}, &ChunkNotFoundError{ID: ref.ChunkID}
	}
	return ResolveBytes(ctx, store, ref.ChunkID)
}

func ResolveURI(ctx context.Context, store Store, uri string) (Chunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil || core.Trim(uri) == "" {
		return Chunk{}, &URIChunkNotFoundError{URI: uri}
	}
	if resolver, ok := store.(URIResolver); ok {
		return resolver.ResolveURI(ctx, uri)
	}
	return Chunk{}, &URIChunkNotFoundError{URI: uri}
}

func MergeRef(base, overlay ChunkRef) ChunkRef {
	out := base
	if overlay.ChunkID != 0 || base.ChunkID == 0 {
		out.ChunkID = overlay.ChunkID
	}
	if overlay.HasFrameOffset {
		out.FrameOffset = overlay.FrameOffset
		out.HasFrameOffset = true
	}
	if overlay.Codec != "" {
		out.Codec = overlay.Codec
	}
	if overlay.Segment != "" {
		out.Segment = overlay.Segment
	}
	return out
}
