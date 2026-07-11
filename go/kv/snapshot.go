// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"crypto/sha256"
	"hash"
	"sync"

	core "dappco.re/go"
)

const (
	// SnapshotVersion is the on-disk binary format version for KV snapshots.
	// v6 records each layer's source-cache MaxSize (window/rotation clamp) so
	// wake restores carry the slept geometry instead of trusting wake-era
	// model templates.
	SnapshotVersion = 6

	kvSnapshotMagic = "MLXKV001"
)

// Constant validation errors hoisted to package vars — each previously
// allocated a fresh core.NewError on the (rare but hot under churn)
// failure path. errSnapshotNil is defined in blocks.go (same package).
var (
	errRawTensorNeedsNative       = core.NewError("mlx: KV snapshot raw tensor requires native encoding")
	errUnsupportedNativeDtype     = core.NewError("mlx: unsupported KV native tensor dtype")
	errStateTokenBlockTokenCount  = core.NewError("mlx: State token block token count is invalid")
	errNativeByteLenMismatch      = core.NewError("mlx: KV native tensor byte length mismatch")
	errUnsupportedTensorEncoding  = core.NewError("mlx: unsupported KV tensor encoding")
	errUnsupportedSnapshotVersion = core.NewError("mlx: unsupported KV snapshot version")
	errUnsupportedNativeTensor    = core.NewError("mlx: unsupported KV snapshot native tensor dtype")
	errTruncatedSnapshot          = core.NewError("mlx: truncated KV snapshot")
	errNativeElementCount         = core.NewError("mlx: KV native tensor element count mismatch")
	errInvalidSnapshotMagic       = core.NewError("mlx: invalid KV snapshot magic")
	errTurboQuantPayloadMode      = core.NewError("mlx: TurboQuant KV payload requires turboquant cache mode")
	errTurboQuantPayloadMissing   = core.NewError("mlx: turboquant cache mode requires TurboQuant KV payload")
)

// Encoding controls how K/V tensors are represented on disk.
type Encoding string

const (
	// KVSnapshotEncodingFloat32 preserves exact float32 K/V cache tensors.
	KVSnapshotEncodingFloat32 Encoding = "float32"
	// EncodingQ8 stores K/V cache tensors as symmetric int8 plus scale.
	EncodingQ8 Encoding = "q8"
	// EncodingNative stores K/V tensors in their captured dtype when
	// native dtype bytes are present, falling back to float32 otherwise.
	EncodingNative Encoding = "native"
)

// SaveOptions controls the portable binary snapshot encoding.
type SaveOptions struct {
	KVEncoding Encoding
}

// LoadOptions controls how portable binary snapshots are decoded.
type LoadOptions struct {
	// RawKVOnly preserves native K/V tensor bytes without decoding float32
	// side slices. Float32 and Q8 snapshot encodings still decode to float32.
	RawKVOnly bool
}

// CaptureOptions controls native K/V capture.
type CaptureOptions struct {
	// RawKVOnly captures native K/V dtype bytes without retaining float32
	// key/value slices when the native backend can provide raw tensors.
	RawKVOnly bool
	// BlockStartToken skips capture of blocks ending at or before this token
	// (the trusted-prefix sleep lane — see StateBlockOptions.ReusePrefixTrusted).
	BlockStartToken int
}

// Snapshot is a CPU-readable copy of model key/value cache tensors.
type Snapshot struct {
	Version       int
	Architecture  string
	Tokens        []int32
	Generated     []int32
	TokenOffset   int
	NumLayers     int
	NumHeads      int
	SeqLen        int
	HeadDim       int
	NumQueryHeads int
	LogitShape    []int32
	Logits        []float32
	Layers        []LayerSnapshot
}

// LayerSnapshot contains cache tensors for a logical transformer layer.
type LayerSnapshot struct {
	Layer      int
	CacheIndex int
	CacheMode  string
	// MaxSize is the source cache's window/rotation clamp at capture time
	// (0 = unclamped or pre-v6 snapshot; restore falls back to the model
	// template's geometry).
	MaxSize            int
	TurboQuantPayloads [][]byte
	KeyDType           string
	KeyBytes           []byte
	KeyShape           []int32
	ValueDType         string
	ValueBytes         []byte
	ValueShape         []int32
	Heads              []HeadSnapshot
}

// HeadSnapshot contains flattened key/value tensors for one KV head.
type HeadSnapshot struct {
	Key        []float32
	KeyDType   string
	KeyBytes   []byte
	Value      []float32
	ValueDType string
	ValueBytes []byte
}

// Head returns a defensive copy of the key/value tensors for layer and head.
func (s *Snapshot) Head(layer, head int) (HeadSnapshot, bool) {
	if s == nil || layer < 0 || head < 0 {
		return HeadSnapshot{}, false
	}
	layerSnapshot, ok := s.layer(layer)
	if !ok || head >= len(layerSnapshot.Heads) {
		return HeadSnapshot{}, false
	}
	return cloneKVHead(layerSnapshot.Heads[head]), true
}

func (s *Snapshot) layer(layer int) (LayerSnapshot, bool) {
	if layer < len(s.Layers) && s.Layers[layer].Layer == layer {
		return s.Layers[layer], true
	}
	for _, snapshot := range s.Layers {
		if snapshot.Layer == layer {
			return snapshot, true
		}
	}
	if layer < len(s.Layers) && s.Layers[layer].Layer == 0 {
		return s.Layers[layer], true
	}
	return LayerSnapshot{}, false
}

// Clone returns a deep copy of the snapshot.
func (s *Snapshot) Clone() *Snapshot {
	if s == nil {
		return nil
	}
	cloned := &Snapshot{
		Version:       s.Version,
		Architecture:  s.Architecture,
		Tokens:        core.SliceClone(s.Tokens),
		Generated:     core.SliceClone(s.Generated),
		TokenOffset:   s.TokenOffset,
		NumLayers:     s.NumLayers,
		NumHeads:      s.NumHeads,
		SeqLen:        s.SeqLen,
		HeadDim:       s.HeadDim,
		NumQueryHeads: s.NumQueryHeads,
		LogitShape:    core.SliceClone(s.LogitShape),
		Logits:        core.SliceClone(s.Logits),
		Layers:        cloneKVLayers(s.Layers),
	}
	return cloned
}

func cloneKVLayers(src []LayerSnapshot) []LayerSnapshot {
	if len(src) == 0 {
		return nil
	}
	cloned := make([]LayerSnapshot, len(src))
	for i, layer := range src {
		cloned[i] = LayerSnapshot{
			Layer:              layer.Layer,
			CacheIndex:         layer.CacheIndex,
			CacheMode:          layer.CacheMode,
			MaxSize:            layer.MaxSize,
			TurboQuantPayloads: cloneKVByteSlices(layer.TurboQuantPayloads),
			KeyDType:           layer.KeyDType,
			KeyBytes:           core.SliceClone(layer.KeyBytes),
			KeyShape:           core.SliceClone(layer.KeyShape),
			ValueDType:         layer.ValueDType,
			ValueBytes:         core.SliceClone(layer.ValueBytes),
			ValueShape:         core.SliceClone(layer.ValueShape),
			Heads:              cloneKVHeads(layer.Heads),
		}
	}
	return cloned
}

func cloneKVByteSlices(src [][]byte) [][]byte {
	if len(src) == 0 {
		return nil
	}
	cloned := make([][]byte, len(src))
	for i := range src {
		cloned[i] = core.SliceClone(src[i])
	}
	return cloned
}

func cloneKVHeads(src []HeadSnapshot) []HeadSnapshot {
	if len(src) == 0 {
		return nil
	}
	cloned := make([]HeadSnapshot, len(src))
	for i, head := range src {
		cloned[i] = cloneKVHead(head)
	}
	return cloned
}

func cloneKVHead(src HeadSnapshot) HeadSnapshot {
	return HeadSnapshot{
		Key:        core.SliceClone(src.Key),
		KeyDType:   src.KeyDType,
		KeyBytes:   core.SliceClone(src.KeyBytes),
		Value:      core.SliceClone(src.Value),
		ValueDType: src.ValueDType,
		ValueBytes: core.SliceClone(src.ValueBytes),
	}
}

func DropFloat32(snapshot *Snapshot) {
	if snapshot == nil {
		return
	}
	for layerIndex := range snapshot.Layers {
		for headIndex := range snapshot.Layers[layerIndex].Heads {
			head := &snapshot.Layers[layerIndex].Heads[headIndex]
			if len(head.KeyBytes) > 0 {
				head.Key = nil
			}
			if len(head.ValueBytes) > 0 {
				head.Value = nil
			}
		}
	}
}

const defaultCacheBlockSize = 512

const kvSnapshotTurboQuantCacheMode = "turboquant"

func normalizeSnapshot(snapshot *Snapshot) {
	if snapshot == nil {
		return
	}
	if snapshot.Version == 0 {
		snapshot.Version = SnapshotVersion
	}
	if snapshot.TokenOffset == 0 {
		snapshot.TokenOffset = len(snapshot.Tokens)
	}
}

func requiresNativeEncoding(snapshot *Snapshot) bool {
	if snapshot == nil {
		return false
	}
	if snapshotHasLayerNativeTensors(snapshot) {
		return true
	}
	for _, layer := range snapshot.Layers {
		for _, head := range layer.Heads {
			if len(head.Key) == 0 && len(head.KeyBytes) > 0 {
				return true
			}
			if len(head.Value) == 0 && len(head.ValueBytes) > 0 {
				return true
			}
		}
	}
	return false
}

func snapshotHasLayerNativeTensors(snapshot *Snapshot) bool {
	if snapshot == nil {
		return false
	}
	for _, layer := range snapshot.Layers {
		if len(layer.KeyBytes) > 0 || len(layer.ValueBytes) > 0 {
			return true
		}
	}
	return false
}

// snapshotHasher bundles a sha256 stream and its digest scratch so both
// recycle across calls as one pooled heap unit. sum lives as a field, not
// a per-call [sha256.Size]byte: h.Sum(sum[:0]) passes the buffer through
// the hash.Hash interface, so escape analysis (flow-insensitive across the
// interface boundary) forces a fresh local array to the heap on every
// call — a field on the already-heap-resident pooled object costs zero.
// Same shape as blockcache's blockCacheHasher.
type snapshotHasher struct {
	h   hash.Hash
	sum [sha256.Size]byte
}

// snapshotHashPool recycles snapshotHasher units across the KV snapshot
// hash paths (HashSnapshot, the per-block reuse-check and streamed-save
// lanes). Each borrow Resets before use; sha256 is deterministic so
// pooling never alters the digest. Package-level (not a Snapshot field)
// on purpose: per-block saves hash concurrently, so a shared field would
// race — the same rationale as blockcache's hasher pool. Reclaims both
// the sha256.New allocation (the digest struct escaping through the
// hash.Hash interface) and the per-call sum-scratch escape.
var snapshotHashPool = sync.Pool{New: func() any {
	return &snapshotHasher{h: sha256.New()}
}}

func acquireSnapshotHasher() *snapshotHasher {
	scratch := snapshotHashPool.Get().(*snapshotHasher)
	scratch.h.Reset()
	return scratch
}

func releaseSnapshotHasher(scratch *snapshotHasher) {
	snapshotHashPool.Put(scratch)
}

// hashSnapshotEncoded streams snapshot through a pooled sha256 with the
// given save options and returns the lowercase-hex digest. Shared by
// HashSnapshot and hashStateBlockPayload so the hasher + sum escape are
// pooled once across the per-block hash paths.
//
// core.HexEncode returns the 64-char string with a single alloc (its
// zero-copy AsString aliases the freshly-encoded buffer, versus
// hex.EncodeToString's return-side copy) — the sole remaining allocation.
func hashSnapshotEncoded(snapshot *Snapshot, opts SaveOptions) (string, error) {
	scratch := acquireSnapshotHasher()
	defer releaseSnapshotHasher(scratch)
	if err := snapshot.writeWithOptions(scratch.h, opts); err != nil {
		return "", err
	}
	return core.HexEncode(scratch.h.Sum(scratch.sum[:0])), nil
}

// HashSnapshot computes a stable hash of a normalised Snapshot for use as
// a content-addressed identifier.
//
//	hash, err := kv.HashSnapshot(snap)
func HashSnapshot(snapshot *Snapshot) (string, error) {
	if snapshot == nil {
		return "", errSnapshotNil
	}
	// Stream the encoded bytes straight into sha256 — skips the
	// bytesWithOptions intermediate []byte alloc (~50KB for 2048-token
	// snapshots). bytesWithOptions is read-only over the snapshot, so
	// the stream-encoder produces identical bytes.
	opts := SaveOptions{}
	if requiresNativeEncoding(snapshot) {
		opts.KVEncoding = EncodingNative
	}
	return hashSnapshotEncoded(snapshot, opts)
}
