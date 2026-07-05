// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"maps"

	core "dappco.re/go"
	state "dappco.re/go/inference/model/state"
)

const (
	// KVSnapshotStateKind identifies State chunks containing go-mlx KV state.
	KVSnapshotStateKind = "go-mlx/kv-snapshot"
	// KVSnapshotStateVersion is the JSON envelope schema version.
	KVSnapshotStateVersion = 1
	// KVSnapshotMemvidKind identifies old memvid-named chunks containing
	// go-mlx KV state.
	//
	// Deprecated: use KVSnapshotStateKind.
	KVSnapshotMemvidKind = KVSnapshotStateKind
	// KVSnapshotMemvidVersion is the JSON envelope schema version.
	//
	// Deprecated: use KVSnapshotStateVersion.
	KVSnapshotMemvidVersion = KVSnapshotStateVersion
)

// Constant validation errors hoisted to package vars.
// errStateStoreNil and errSnapshotNil are defined in blocks.go (same package).
var (
	errUnsupportedStateKVSnapshotVersion  = core.NewError("mlx: unsupported State KV snapshot version")
	errUnsupportedStateKVSnapshotEncoding = core.NewError("mlx: unsupported State KV snapshot binary encoding")
	errStateKVSnapshotHash                = core.NewError("mlx: State KV snapshot hash mismatch")
	errStateKVPayloadLen                  = core.NewError("mlx: State KV payload length mismatch")
	errStateKVPayloadNonByte              = core.NewError("mlx: State KV payload decoded to non-byte data")
	errStateKVSnapshotKind                = core.NewError("mlx: invalid State KV snapshot kind")
)

// StateOptions controls how KV snapshots are stored in State.
type StateOptions struct {
	KVEncoding Encoding
	URI        string
	Title      string
	Kind       string
	Track      string
	Tags       map[string]string
	Labels     []string
}

// MemvidOptions controls how KV snapshots are stored in the old memvid-named
// State store.
//
// Deprecated: use StateOptions.
type MemvidOptions = StateOptions

type kvSnapshotStateEnvelope struct {
	Version          int    `json:"version"`
	Kind             string `json:"kind"`
	KVVersion        int    `json:"kv_version"`
	KVEncoding       string `json:"kv_encoding,omitempty"`
	BinaryEncoding   string `json:"binary_encoding"`
	KVHash           string `json:"kv_hash"`
	Architecture     string `json:"architecture,omitempty"`
	TokenCount       int    `json:"token_count,omitempty"`
	TokenOffset      int    `json:"token_offset,omitempty"`
	GeneratedTokens  int    `json:"generated_tokens,omitempty"`
	NumLayers        int    `json:"num_layers,omitempty"`
	NumHeads         int    `json:"num_heads,omitempty"`
	SeqLen           int    `json:"seq_len,omitempty"`
	HeadDim          int    `json:"head_dim,omitempty"`
	NumQueryHeads    int    `json:"num_query_heads,omitempty"`
	PayloadByteCount int    `json:"payload_byte_count,omitempty"`
	Data             string `json:"data"`
}

// SaveState writes this KV snapshot to a State cold store. The payload is the
// same binary format used by Save, base64 wrapped so text-oriented State stores
// and QR-video backends can carry it without lossy conversion.
func (s *Snapshot) SaveState(ctx context.Context, store state.Writer, opts StateOptions) (state.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil {
		return state.ChunkRef{}, errSnapshotNil
	}
	if store == nil {
		return state.ChunkRef{}, errStateStoreNil
	}
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return state.ChunkRef{}, err
	}
	data, err := s.bytesWithOptions(SaveOptions{KVEncoding: encoding})
	if err != nil {
		return state.ChunkRef{}, err
	}
	envelope := kvSnapshotStateEnvelope{
		Version:          KVSnapshotStateVersion,
		Kind:             KVSnapshotStateKind,
		KVVersion:        effectiveVersion(s, encoding),
		KVEncoding:       string(encoding),
		BinaryEncoding:   "base64",
		KVHash:           core.SHA256Hex(data),
		Architecture:     s.Architecture,
		TokenCount:       len(s.Tokens),
		TokenOffset:      EffectiveTokenOffset(s),
		GeneratedTokens:  len(s.Generated),
		NumLayers:        s.NumLayers,
		NumHeads:         s.NumHeads,
		SeqLen:           s.SeqLen,
		HeadDim:          s.HeadDim,
		NumQueryHeads:    s.NumQueryHeads,
		PayloadByteCount: len(data),
		Data:             core.Base64Encode(data),
	}
	ref, err := store.Put(ctx, core.JSONMarshalString(envelope), kvSnapshotStatePutOptions(s, opts, envelope))
	if err != nil {
		return state.ChunkRef{}, core.E("Snapshot.SaveState", "write State chunk", err)
	}
	return ref, nil
}

// SaveMemvid writes this KV snapshot to the old memvid-named State store.
//
// Deprecated: use SaveState.
func (s *Snapshot) SaveMemvid(ctx context.Context, store state.Writer, opts MemvidOptions) (state.ChunkRef, error) {
	return s.SaveState(ctx, store, opts)
}

// LoadFromState resolves and decodes a KV snapshot from a State chunk ref.
func LoadFromState(ctx context.Context, store state.Store, ref state.ChunkRef) (*Snapshot, error) {
	return LoadFromStateWithOptions(ctx, store, ref, LoadOptions{})
}

// LoadFromStateWithOptions resolves and decodes a KV snapshot from a State
// chunk ref with explicit decode options.
func LoadFromStateWithOptions(ctx context.Context, store state.Store, ref state.ChunkRef, opts LoadOptions) (*Snapshot, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, errStateStoreNil
	}
	chunk, err := state.Resolve(ctx, store, ref.ChunkID)
	if err != nil {
		return nil, core.E("LoadFromState", "resolve State chunk", err)
	}
	var envelope kvSnapshotStateEnvelope
	if result := core.JSONUnmarshalString(chunk.Text, &envelope); !result.OK {
		return nil, core.E("LoadFromState", "parse State envelope", result.Err())
	}
	data, err := decodeKVSnapshotStateEnvelope(envelope)
	if err != nil {
		return nil, err
	}
	return parseKVSnapshotWithOptions(data, opts)
}

// LoadFromMemvid resolves and decodes a KV snapshot from an old memvid-named
// State chunk ref.
//
// Deprecated: use LoadFromState.
func LoadFromMemvid(ctx context.Context, store state.Store, ref state.ChunkRef) (*Snapshot, error) {
	return LoadFromState(ctx, store, ref)
}

// LoadFromMemvidWithOptions resolves and decodes a KV snapshot from an old
// memvid-named State chunk ref with explicit decode options.
//
// Deprecated: use LoadFromStateWithOptions.
func LoadFromMemvidWithOptions(ctx context.Context, store state.Store, ref state.ChunkRef, opts LoadOptions) (*Snapshot, error) {
	return LoadFromStateWithOptions(ctx, store, ref, opts)
}

func decodeKVSnapshotStateEnvelope(envelope kvSnapshotStateEnvelope) ([]byte, error) {
	if envelope.Version <= 0 || envelope.Version > KVSnapshotStateVersion {
		return nil, errUnsupportedStateKVSnapshotVersion
	}
	if envelope.Kind != KVSnapshotStateKind {
		return nil, errStateKVSnapshotKind
	}
	if envelope.BinaryEncoding != "base64" {
		return nil, errUnsupportedStateKVSnapshotEncoding
	}
	decoded := core.Base64Decode(envelope.Data)
	if !decoded.OK {
		return nil, core.E("LoadFromState", "decode State KV payload", decoded.Err())
	}
	data, ok := decoded.Value.([]byte)
	if !ok {
		return nil, errStateKVPayloadNonByte
	}
	if envelope.PayloadByteCount > 0 && len(data) != envelope.PayloadByteCount {
		return nil, errStateKVPayloadLen
	}
	if envelope.KVHash != "" && core.SHA256Hex(data) != envelope.KVHash {
		return nil, errStateKVSnapshotHash
	}
	return data, nil
}

func kvSnapshotStatePutOptions(snapshot *Snapshot, opts StateOptions, envelope kvSnapshotStateEnvelope) state.PutOptions {
	kind := opts.Kind
	if kind == "" {
		kind = KVSnapshotStateKind
	}
	track := opts.Track
	if track == "" {
		track = "session-kv"
	}
	tags := cloneKVSnapshotStateTags(opts.Tags)
	tags["kv_hash"] = envelope.KVHash
	tags["kv_encoding"] = envelope.KVEncoding
	tags["architecture"] = envelope.Architecture
	tags["token_count"] = core.Itoa(envelope.TokenCount)
	tags["payload_bytes"] = core.Itoa(envelope.PayloadByteCount)
	// Pre-size for the deterministic 2 appended labels — avoids the
	// geometric-grow path on every State KV save.
	labels := make([]string, len(opts.Labels), len(opts.Labels)+2)
	copy(labels, opts.Labels)
	labels = append(labels, "go-mlx", "kv-snapshot")
	// Skip the "mlx://kv-snapshot/" + KVHash concat when opts.URI is
	// already set — the previous firstNonEmpty call materialised it
	// unconditionally.
	uri := opts.URI
	if uri == "" {
		uri = "mlx://kv-snapshot/" + envelope.KVHash
	}
	return state.PutOptions{
		URI:    uri,
		Title:  firstNonEmpty(opts.Title, "go-mlx KV snapshot"),
		Kind:   kind,
		Track:  track,
		Tags:   tags,
		Labels: labels,
	}
}

func cloneKVSnapshotStateTags(input map[string]string) map[string]string {
	// Caller always writes up to 6 additional bookkeeping tags after the
	// clone (kv_hash, kv_encoding, payload_encoding, block_index,
	// token_start, token_count) — size against input+6 so the map never
	// grows mid-insert on the per-block-save path.
	if len(input) == 0 {
		return make(map[string]string, 6)
	}
	out := make(map[string]string, len(input)+6)
	maps.Copy(out, input)
	return out
}

func effectiveVersion(snapshot *Snapshot, encoding Encoding) int {
	version := snapshot.Version
	if version == 0 {
		version = SnapshotVersion
	}
	if encoding != KVSnapshotEncodingFloat32 && version < 3 {
		version = 3
	}
	if snapshotHasLayerNativeTensors(snapshot) && version < 4 {
		version = 4
	}
	if snapshotHasLayerCompressedPayloads(snapshot) && version < 5 {
		version = 5
	}
	if snapshotHasLayerMaxSize(snapshot) && version < 6 {
		version = 6
	}
	return version
}

func snapshotHasLayerMaxSize(snapshot *Snapshot) bool {
	if snapshot == nil {
		return false
	}
	for i := range snapshot.Layers {
		if snapshot.Layers[i].MaxSize > 0 {
			return true
		}
	}
	return false
}

func snapshotHasLayerCompressedPayloads(snapshot *Snapshot) bool {
	if snapshot == nil {
		return false
	}
	for i := range snapshot.Layers {
		layer := &snapshot.Layers[i]
		if layer.CacheMode != "" || len(layer.TurboQuantPayloads) > 0 {
			return true
		}
	}
	return false
}

func EffectiveTokenOffset(snapshot *Snapshot) int {
	if snapshot == nil {
		return 0
	}
	if snapshot.TokenOffset != 0 {
		return snapshot.TokenOffset
	}
	return len(snapshot.Tokens)
}
