// SPDX-Licence-Identifier: EUPL-1.2

package agent

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"hash"
	"strconv"
	"sync"

	core "dappco.re/go"
	"dappco.re/go/inference/kv"
	"dappco.re/go/inference/kv/memory"
	"dappco.re/go/inference/model/bundle"
	state "dappco.re/go/inference/model/state"
)

// hashBufPool reuses bytes.Buffer instances used while assembling the
// canonical input for indexEntryHash. The Buffer backing slice never
// escapes (we hash-and-discard before Reset), so pooling is safe and
// collapses ~1000 per-Validate Builder allocs into 1 reused buffer.
var hashBufPool = sync.Pool{
	New: func() any {
		// 384 covers the typical rich-entry input (~250 bytes) with
		// headroom for long URIs / extra labels; smaller starting
		// caps would force a grow on the common path.
		buf := make([]byte, 0, 384)
		return bytes.NewBuffer(buf)
	},
}

const (
	// StateIndexKind identifies a State-stored lookup index
	// for named spans inside one or more KV block bundles.
	StateIndexKind = "go-mlx/kv-snapshot-bundle-index"
	// KVSnapshotStateBundleIndexVersion is the bundle-index schema version.
	KVSnapshotStateBundleIndexVersion = 1
	// MemvidIndexKind identifies an old memvid-named lookup index for named
	// spans inside one or more KV block bundles.
	//
	// Deprecated: use StateIndexKind.
	MemvidIndexKind = StateIndexKind
	// KVSnapshotMemvidBundleIndexVersion is the bundle-index schema version.
	//
	// Deprecated: use KVSnapshotStateBundleIndexVersion.
	KVSnapshotMemvidBundleIndexVersion = KVSnapshotStateBundleIndexVersion
)

// seenURIPool reuses the duplicate-detection set Validate builds for
// indexes above validateLinearScanThreshold. The map never escapes
// validate (membership-only, discarded before return), so pooling is
// safe; clear() empties it without freeing buckets, so after warmup a
// repeated Validate over a large index allocates ~0 bytes for the set
// instead of a fresh len(Entries)-sized map each call. Validate fires
// per load, per save, and at the tail of every NewStateIndex, so the
// large-index path hit this on every round.
var seenURIPool = sync.Pool{
	New: func() any {
		m := make(map[string]struct{}, validateLinearScanThreshold*2)
		return &m
	},
}

// stateIndexPutLabels is the canonical label set attached to every
// SaveStateIndex Put call. Package-scoped so each call shares one backing
// array instead of allocating a fresh slice literal per save.
var stateIndexPutLabels = []string{"go-mlx", "kv-snapshot-bundle-index"}

// Sentinel validation errors hoisted to package scope. Each previously
// triggered a fresh core.NewError allocation per error-path hit; the
// hot Validate path returns one of these on every bad entry, and
// keeping them as singletons collapses N allocs → 0 on the failure
// branches and also lets callers errors.Is them.
var (
	errStateIndexNil                  = core.NewError("mlx: State index is nil")
	errStateIndexUnsupportedVersion   = core.NewError("mlx: unsupported State index version")
	errStateIndexInvalidKind          = core.NewError("mlx: invalid State index kind")
	errStateIndexEmptyTokenCount      = core.NewError("mlx: State index token count is empty")
	errStateIndexNoEntries            = core.NewError("mlx: State index has no entries")
	errStateIndexDuplicateURI         = core.NewError("mlx: duplicate State index URI")
	errStateIndexHashMismatch         = core.NewError("mlx: State index hash mismatch")
	errStateIndexEntryURIRequired     = core.NewError("mlx: State index entry URI is required")
	errStateIndexEntryBundleRequired  = core.NewError("mlx: State index entry bundle URI is required")
	errStateIndexEntryTokenStart      = core.NewError("mlx: State index entry token start is invalid")
	errStateIndexEntryTokenCount      = core.NewError("mlx: State index entry token count is empty")
	errStateIndexEntryExceedsBundle   = core.NewError("mlx: State index entry exceeds bundle token count")
	errStateIndexEntryByteSpan        = core.NewError("mlx: State index entry byte span is invalid")
	errStateIndexEntryHashMismatch    = core.NewError("mlx: State index entry hash mismatch")
	errStateIndexEntryNotFound        = core.NewError("mlx: State index entry not found")
	errStateIndexPrefixInvalid        = core.NewError("mlx: State index prefix is invalid")
	errStateStoreNil                  = core.NewError("mlx: state store is nil")
	errStateIndexURIRequired          = core.NewError("mlx: State index URI is required")
	errStateIndexArchitectureMismatch = core.NewError("mlx: State index model architecture mismatch")
	errStateIndexLayerMismatch        = core.NewError("mlx: State index model layer mismatch")
	errStateIndexQuantMismatch        = core.NewError("mlx: State index model quantization mismatch")
	errStateIndexModelHashMismatch    = core.NewError("mlx: State index model hash mismatch")
	errStateIndexExceedsContext       = core.NewError("mlx: State index exceeds model context length")
	errStateIndexTokenizerMismatch    = core.NewError("mlx: State index tokenizer hash mismatch")
	errStateIndexChatTemplateMismatch = core.NewError("mlx: State index chat template hash mismatch")
	errStateURIRequired               = core.NewError("mlx: State URI is required")
)

// StateIndexOptions configures a durable index for named State
// spans such as chapters, sections, or checkpointed agent states.
type StateIndexOptions struct {
	BundleURI string
	Title     string
	Model     string
	ModelPath string
	ModelInfo memory.ModelInfo
	Tokenizer bundle.Tokenizer
	Entries   []StateIndexEntry
}

// MemvidIndexOptions configures a durable index for old memvid-named KV
// bundle spans such as chapters, sections, or checkpointed agent states.
//
// Deprecated: use StateIndexOptions.
type MemvidIndexOptions = StateIndexOptions

// StateIndex records model identity and named token spans for restoring
// partial prefixes from a larger durable State block bundle.
type StateIndex struct {
	Version      int               `json:"version"`
	Kind         string            `json:"kind"`
	BundleURI    string            `json:"bundle_uri,omitempty"`
	SnapshotHash string            `json:"snapshot_hash,omitempty"`
	KVEncoding   kv.Encoding       `json:"kv_encoding,omitempty"`
	TokenCount   int               `json:"token_count,omitempty"`
	BlockSize    int               `json:"block_size,omitempty"`
	Model        bundle.Model      `json:"model"`
	Tokenizer    bundle.Tokenizer  `json:"tokenizer"`
	Entries      []StateIndexEntry `json:"entries,omitempty"`
	Hash         string            `json:"hash,omitempty"`
}

// MemvidIndex records model identity and named token spans for restoring
// partial prefixes from a larger old memvid-named KV block bundle.
//
// Deprecated: use StateIndex.
type MemvidIndex = StateIndex

// StateIndexEntry names one logical span in a State bundle. The current wake
// path restores the prefix ending at TokenStart+TokenCount.
type StateIndexEntry struct {
	URI        string            `json:"uri"`
	BundleURI  string            `json:"bundle_uri,omitempty"`
	Title      string            `json:"title,omitempty"`
	TokenStart int               `json:"token_start"`
	TokenCount int               `json:"token_count"`
	ByteStart  int64             `json:"byte_start,omitempty"`
	ByteCount  int64             `json:"byte_count,omitempty"`
	Hash       string            `json:"hash,omitempty"`
	Labels     []string          `json:"labels,omitempty"`
	Meta       map[string]string `json:"meta,omitempty"`
}

// MemvidIndexEntry names one logical span in an old memvid-named KV bundle.
//
// Deprecated: use StateIndexEntry.
type MemvidIndexEntry = StateIndexEntry

// NewStateIndex builds an index around a durable State block bundle. When no
// entries are supplied, it creates one full-bundle entry.
func NewStateIndex(bundle *kv.StateBlockBundle, opts StateIndexOptions) (*StateIndex, error) {
	if err := kv.ValidateStateBlockBundle(bundle); err != nil {
		return nil, err
	}
	index := &StateIndex{
		Version:      KVSnapshotStateBundleIndexVersion,
		Kind:         StateIndexKind,
		BundleURI:    core.Trim(opts.BundleURI),
		SnapshotHash: bundle.SnapshotHash,
		KVEncoding:   bundle.KVEncoding,
		TokenCount:   bundle.TokenCount,
		BlockSize:    bundle.BlockSize,
		Model:        indexModel(bundle, opts),
		Tokenizer:    stateBundleTokenizer(opts.Tokenizer),
		Entries:      cloneIndexEntries(opts.Entries),
	}
	if len(index.Entries) == 0 {
		index.Entries = []StateIndexEntry{{
			URI:        firstNonEmpty(index.BundleURI, "mlx://kv/full"),
			BundleURI:  index.BundleURI,
			Title:      firstNonEmpty(opts.Title, "full bundle"),
			TokenStart: 0,
			TokenCount: bundle.TokenCount,
		}}
	}
	sortedBlocks := stateBlockRefsSortedByTokenStart(bundle.Blocks)
	// Pass 1: default the per-entry BundleURI and resolve byte spans.
	// Byte-span fill must complete before any entry is hashed because the
	// canonical entry hash covers ByteStart/ByteCount, so hashing is split
	// into pass 2 below. Count the entries whose Hash must be computed so
	// the batch buffer can be sized exactly.
	emptyHashes := 0
	for i := range index.Entries {
		if index.Entries[i].BundleURI == "" {
			index.Entries[i].BundleURI = index.BundleURI
		}
		if sortedBlocks {
			fillIndexEntryByteSpanSorted(&index.Entries[i], bundle)
		} else {
			fillIndexEntryByteSpan(&index.Entries[i], bundle)
		}
		if index.Entries[i].Hash == "" {
			emptyHashes++
		}
	}
	// Pass 2: hash. The previous loop assigned each computed hash via
	// core.HexEncode, allocating a fresh 64-byte string per entry — one
	// heap object per entry (1006 of 1006 allocs on a 1000-entry index,
	// measured). Batch all computed digests into ONE 64*emptyHashes hex
	// buffer instead: hex.Encode each digest into its window, then alias
	// each window into the entry Hash. AsString aliases the buffer
	// (never written again after its window is filled and never reused),
	// so the substrings keep it live for the index's lifetime — N entry
	// hashes collapse to a single allocation, byte-identical to N
	// separate HexEncode calls. Pre-set hashes use the zero-alloc
	// indexEntryHashEquals to verify rather than materialising hex.
	const hexLen = sha256.Size * 2
	var (
		hexBuf []byte
		hexPos int
	)
	if emptyHashes > 0 {
		hexBuf = make([]byte, hexLen*emptyHashes)
	}
	for i := range index.Entries {
		if index.Entries[i].Hash == "" {
			window := hexBuf[hexPos : hexPos+hexLen]
			sum := indexEntryHashBytes(&index.Entries[i])
			hex.Encode(window, sum[:])
			index.Entries[i].Hash = core.AsString(window)
			hexPos += hexLen
		} else if !indexEntryHashEquals(&index.Entries[i], index.Entries[i].Hash) {
			return nil, errStateIndexEntryHashMismatch
		}
	}
	index.Hash = indexHash(index)
	if err := index.validate(false); err != nil {
		return nil, err
	}
	return index, nil
}

// NewMemvidIndex builds an index around an old memvid-named KV block bundle. When no
// entries are supplied, it creates one full-bundle entry.
//
// Deprecated: use NewStateIndex.
func NewMemvidIndex(bundle *kv.MemvidBlockBundle, opts MemvidIndexOptions) (*MemvidIndex, error) {
	return NewStateIndex(bundle, opts)
}

// Validate checks schema, model identity, and indexed span bounds.
func (index *StateIndex) Validate() error {
	return index.validate(true)
}

// validateLinearScanThreshold is the entry count below which Validate
// uses an O(N²) linear scan over previously-seen URIs instead of
// allocating a hash-set. Measured on M3 Ultra: for N ≤ 32 a string-eq
// scan dominates map setup + bucket allocation. Above that, the map's
// O(N) scaling pays back. Typical session/chapter indexes sit well
// under the threshold so this collapses the seen-map alloc to zero on
// the common path.
const validateLinearScanThreshold = 32

func (index *StateIndex) validate(checkHashes bool) error {
	if index == nil {
		return errStateIndexNil
	}
	if index.Version <= 0 || index.Version > KVSnapshotStateBundleIndexVersion {
		return errStateIndexUnsupportedVersion
	}
	if index.Kind != StateIndexKind {
		return errStateIndexInvalidKind
	}
	if index.TokenCount <= 0 {
		return errStateIndexEmptyTokenCount
	}
	if len(index.Entries) == 0 {
		return errStateIndexNoEntries
	}
	indexBundleURIEmpty := core.Trim(index.BundleURI) == ""
	if len(index.Entries) <= validateLinearScanThreshold {
		for i := range index.Entries {
			entry := &index.Entries[i]
			if err := index.validateEntry(entry, checkHashes, indexBundleURIEmpty); err != nil {
				return err
			}
			uri := entry.URI
			for j := range i {
				if index.Entries[j].URI == uri {
					return errStateIndexDuplicateURI
				}
			}
		}
	} else {
		// Pooled membership set — cleared on checkout so buckets are
		// reused across calls instead of allocating a fresh
		// len(Entries)-sized map every Validate. The set never escapes
		// this branch; defer Put returns it on every exit path.
		seenPtr := seenURIPool.Get().(*map[string]struct{})
		seen := *seenPtr
		clear(seen)
		defer seenURIPool.Put(seenPtr)
		for i := range index.Entries {
			entry := &index.Entries[i]
			if err := index.validateEntry(entry, checkHashes, indexBundleURIEmpty); err != nil {
				return err
			}
			if _, ok := seen[entry.URI]; ok {
				return errStateIndexDuplicateURI
			}
			seen[entry.URI] = struct{}{}
		}
	}
	if checkHashes && index.Hash != "" && !indexHashEquals(index, index.Hash) {
		return errStateIndexHashMismatch
	}
	return nil
}

func (index *StateIndex) validateEntry(entry *StateIndexEntry, checkHash, indexBundleURIEmpty bool) error {
	if core.Trim(entry.URI) == "" {
		return errStateIndexEntryURIRequired
	}
	if indexBundleURIEmpty && core.Trim(entry.BundleURI) == "" {
		return errStateIndexEntryBundleRequired
	}
	if entry.TokenStart < 0 {
		return errStateIndexEntryTokenStart
	}
	if entry.TokenCount <= 0 {
		return errStateIndexEntryTokenCount
	}
	if entry.TokenStart+entry.TokenCount > index.TokenCount {
		return errStateIndexEntryExceedsBundle
	}
	if entry.ByteStart < 0 || entry.ByteCount < 0 {
		return errStateIndexEntryByteSpan
	}
	if checkHash && entry.Hash != "" && !indexEntryHashEquals(entry, entry.Hash) {
		return errStateIndexEntryHashMismatch
	}
	return nil
}

// Entry returns a defensive copy of the entry with URI.
func (index *StateIndex) Entry(uri string) (StateIndexEntry, bool) {
	if index == nil {
		return StateIndexEntry{}, false
	}
	for i := range index.Entries {
		if index.Entries[i].URI == uri {
			return cloneIndexEntry(index.Entries[i]), true
		}
	}
	return StateIndexEntry{}, false
}

// RequiredContextLength reports the largest prefix length needed by any entry.
func (index *StateIndex) RequiredContextLength() int {
	if index == nil {
		return 0
	}
	required := 0
	for i := range index.Entries {
		if end := index.Entries[i].PrefixTokens(); end > required {
			required = end
		}
	}
	return required
}

// PrefixTokens reports the prefix length needed to restore this entry.
func (entry StateIndexEntry) PrefixTokens() int {
	return entry.TokenStart + entry.TokenCount
}

// SaveStateIndex stores the index JSON in the same State store as its
// referenced bundle manifests.
func SaveStateIndex(ctx context.Context, store state.Writer, index *StateIndex, uri string) (state.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return state.ChunkRef{}, errStateStoreNil
	}
	if core.Trim(uri) == "" {
		return state.ChunkRef{}, errStateIndexURIRequired
	}
	if err := index.Validate(); err != nil {
		return state.ChunkRef{}, err
	}
	ref, err := store.Put(ctx, core.JSONMarshalString(index), state.PutOptions{
		URI:    uri,
		Title:  "go-mlx State index",
		Kind:   StateIndexKind,
		Track:  "session-kv-index",
		Labels: stateIndexPutLabels,
	})
	if err != nil {
		return state.ChunkRef{}, core.E("kv.Snapshot.SaveStateIndex", "write State index", err)
	}
	return ref, nil
}

// SaveMemvidIndex stores the index JSON in the same old memvid-named store as its
// referenced bundle manifests.
//
// Deprecated: use SaveStateIndex.
func SaveMemvidIndex(ctx context.Context, store state.Writer, index *MemvidIndex, uri string) (state.ChunkRef, error) {
	return SaveStateIndex(ctx, store, index, uri)
}

// LoadStateIndex restores an index by URI from a State store.
func LoadStateIndex(ctx context.Context, store state.Store, uri string) (*StateIndex, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, errStateStoreNil
	}
	if core.Trim(uri) == "" {
		return nil, errStateIndexURIRequired
	}
	chunk, err := state.ResolveURI(ctx, store, uri)
	if err != nil {
		return nil, core.E("LoadStateIndex", "resolve State index", err)
	}
	var index StateIndex
	if result := core.JSONUnmarshalString(chunk.Text, &index); !result.OK {
		return nil, core.E("LoadStateIndex", "parse State index", kv.ResultError(result))
	}
	if err := index.Validate(); err != nil {
		return nil, err
	}
	return &index, nil
}

// LoadMemvidIndex restores an index by URI from an old memvid-named store.
//
// Deprecated: use LoadStateIndex.
func LoadMemvidIndex(ctx context.Context, store state.Store, uri string) (*MemvidIndex, error) {
	return LoadStateIndex(ctx, store, uri)
}

// LoadPrefixFromStateIndex resolves entryURI through index,
// loads its referenced block bundle, and restores only the prefix required by
// that entry.
func LoadPrefixFromStateIndex(ctx context.Context, store state.Store, index *StateIndex, entryURI string, opts kv.LoadOptions) (*kv.Snapshot, StateIndexEntry, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, StateIndexEntry{}, errStateStoreNil
	}
	if err := index.Validate(); err != nil {
		return nil, StateIndexEntry{}, err
	}
	entry, ok := index.Entry(entryURI)
	if !ok {
		return nil, StateIndexEntry{}, errStateIndexEntryNotFound
	}
	bundleURI := entry.BundleURI
	if bundleURI == "" {
		bundleURI = index.BundleURI
	}
	bundle, err := kv.LoadStateBlockBundle(ctx, store, bundleURI)
	if err != nil {
		return nil, StateIndexEntry{}, err
	}
	prefixTokens := entry.PrefixTokens()
	if prefixTokens <= 0 || prefixTokens > bundle.TokenCount {
		return nil, StateIndexEntry{}, errStateIndexPrefixInvalid
	}
	snapshot, err := kv.LoadPrefixFromStateBlocksWithOptions(ctx, store, bundle, prefixTokens, opts)
	if err != nil {
		return nil, StateIndexEntry{}, err
	}
	return snapshot, entry, nil
}

// LoadPrefixFromMemvidIndex resolves entryURI through index, loads its
// referenced block bundle, and restores only the prefix required by that entry.
//
// Deprecated: use LoadPrefixFromStateIndex.
func LoadPrefixFromMemvidIndex(ctx context.Context, store state.Store, index *MemvidIndex, entryURI string, opts kv.LoadOptions) (*kv.Snapshot, MemvidIndexEntry, error) {
	return LoadPrefixFromStateIndex(ctx, store, index, entryURI, opts)
}

// CheckStateIndexCompatibility verifies model and tokenizer identity before
// restoring indexed State into a loaded model.
func CheckStateIndexCompatibility(info memory.ModelInfo, tokenizer bundle.Tokenizer, index *StateIndex) error {
	if err := index.Validate(); err != nil {
		return err
	}
	if index.Model.Architecture != "" && info.Architecture != "" && index.Model.Architecture != info.Architecture {
		return errStateIndexArchitectureMismatch
	}
	if index.Model.NumLayers > 0 && info.NumLayers > 0 && index.Model.NumLayers != info.NumLayers {
		return errStateIndexLayerMismatch
	}
	if index.Model.QuantBits > 0 && info.QuantBits > 0 && index.Model.QuantBits != info.QuantBits {
		return errStateIndexQuantMismatch
	}
	if index.Model.Hash != "" && index.Model.Name == "" && index.Model.Path == "" && modelHashComparable(info, index.Model) {
		active := indexModel(nil, StateIndexOptions{ModelInfo: info})
		if active.Hash != "" && active.Hash != index.Model.Hash {
			return errStateIndexModelHashMismatch
		}
	}
	if info.ContextLength > 0 && index.RequiredContextLength() > info.ContextLength {
		return errStateIndexExceedsContext
	}
	if index.Tokenizer.Hash != "" && tokenizer.Hash != "" && index.Tokenizer.Hash != tokenizer.Hash {
		return errStateIndexTokenizerMismatch
	}
	if index.Tokenizer.ChatTemplateHash != "" && tokenizer.ChatTemplateHash != "" && index.Tokenizer.ChatTemplateHash != tokenizer.ChatTemplateHash {
		return errStateIndexChatTemplateMismatch
	}
	return nil
}

// CheckMemvidIndexCompatibility verifies model and tokenizer
// identity before restoring indexed KV state into a loaded model.
//
// Deprecated: use CheckStateIndexCompatibility.
func CheckMemvidIndexCompatibility(info memory.ModelInfo, tokenizer bundle.Tokenizer, index *MemvidIndex) error {
	return CheckStateIndexCompatibility(info, tokenizer, index)
}

func modelHashComparable(info memory.ModelInfo, model bundle.Model) bool {
	if model.Architecture != "" && info.Architecture == "" {
		return false
	}
	if model.VocabSize > 0 && info.VocabSize == 0 {
		return false
	}
	if model.NumLayers > 0 && info.NumLayers == 0 {
		return false
	}
	if model.QuantBits > 0 && info.QuantBits == 0 {
		return false
	}
	if model.ContextLength > 0 && info.ContextLength == 0 {
		return false
	}
	return true
}

func indexModel(blk *kv.StateBlockBundle, opts StateIndexOptions) bundle.Model {
	info := opts.ModelInfo
	if info.Architecture == "" && blk != nil {
		info.Architecture = blk.Architecture
	}
	model := bundle.Model{
		Name:          opts.Model,
		Path:          opts.ModelPath,
		Architecture:  info.Architecture,
		VocabSize:     info.VocabSize,
		NumLayers:     info.NumLayers,
		HiddenSize:    info.HiddenSize,
		QuantBits:     info.QuantBits,
		QuantGroup:    info.QuantGroup,
		ContextLength: info.ContextLength,
	}
	// Build the canonical identity input into the pooled bytes.Buffer
	// (shared with indexHash + indexEntryHash) then hash directly via
	// sha256.Sum256. Saves the *strings.Builder + Builder.String()
	// intermediate string vs the legacy `stateHash(builder.String())`
	// path — same digest input, two allocs collapsed into one (just
	// the HexEncode return string).
	buf := hashBufPool.Get().(*bytes.Buffer)
	buf.Reset()
	var intBuf [20]byte
	buf.WriteString(model.Name)
	buf.WriteByte('\n')
	buf.WriteString(model.Path)
	buf.WriteByte('\n')
	buf.WriteString(model.Architecture)
	buf.WriteByte('\n')
	buf.Write(strconv.AppendInt(intBuf[:0], int64(model.VocabSize), 10))
	buf.WriteByte('\n')
	buf.Write(strconv.AppendInt(intBuf[:0], int64(model.NumLayers), 10))
	buf.WriteByte('\n')
	buf.Write(strconv.AppendInt(intBuf[:0], int64(model.QuantBits), 10))
	buf.WriteByte('\n')
	buf.Write(strconv.AppendInt(intBuf[:0], int64(model.ContextLength), 10))
	sum := sha256.Sum256(buf.Bytes())
	hashBufPool.Put(buf)
	model.Hash = core.HexEncode(sum[:])
	return model
}

func fillIndexEntryByteSpan(entry *StateIndexEntry, bundle *kv.StateBlockBundle) {
	if entry == nil || bundle == nil || len(bundle.Blocks) == 0 {
		return
	}
	if entry.ByteStart != 0 || entry.ByteCount != 0 {
		return
	}
	spanStart := entry.TokenStart
	spanEnd := entry.TokenStart + entry.TokenCount
	if spanEnd <= spanStart {
		return
	}
	var (
		byteStartSet bool
		byteStart    int64
		byteCount    int64
	)
	blocks := bundle.Blocks
	for i := range blocks {
		refStart := blocks[i].TokenStart
		refEnd := refStart + blocks[i].TokenCount
		if refEnd <= spanStart || refStart >= spanEnd {
			continue
		}
		chunk := kv.StateBlockChunkRef(blocks[i])
		if !byteStartSet && chunk.HasFrameOffset && chunk.FrameOffset <= uint64(1<<63-1) {
			byteStart = int64(chunk.FrameOffset)
			byteStartSet = true
		}
		if blocks[i].PayloadByteCount > 0 {
			byteCount += int64(blocks[i].PayloadByteCount)
		}
	}
	if entry.ByteStart == 0 && byteStartSet {
		entry.ByteStart = byteStart
	}
	if entry.ByteCount == 0 && byteCount > 0 {
		entry.ByteCount = byteCount
	}
}

func fillIndexEntryByteSpanSorted(entry *StateIndexEntry, bundle *kv.StateBlockBundle) {
	if entry == nil || bundle == nil || len(bundle.Blocks) == 0 {
		return
	}
	if entry.ByteStart != 0 || entry.ByteCount != 0 {
		return
	}
	spanStart := entry.TokenStart
	spanEnd := entry.TokenStart + entry.TokenCount
	if spanEnd <= spanStart {
		return
	}
	blocks := bundle.Blocks
	lo, hi := 0, len(blocks)
	for lo < hi {
		mid := lo + (hi-lo)/2
		if blocks[mid].TokenStart+blocks[mid].TokenCount <= spanStart {
			lo = mid + 1
		} else {
			hi = mid
		}
	}
	var (
		byteStartSet bool
		byteStart    int64
		byteCount    int64
	)
	for i := lo; i < len(blocks); i++ {
		if blocks[i].TokenStart >= spanEnd {
			break
		}
		chunk := kv.StateBlockChunkRef(blocks[i])
		if !byteStartSet && chunk.HasFrameOffset && chunk.FrameOffset <= uint64(1<<63-1) {
			byteStart = int64(chunk.FrameOffset)
			byteStartSet = true
		}
		if blocks[i].PayloadByteCount > 0 {
			byteCount += int64(blocks[i].PayloadByteCount)
		}
	}
	if entry.ByteStart == 0 && byteStartSet {
		entry.ByteStart = byteStart
	}
	if entry.ByteCount == 0 && byteCount > 0 {
		entry.ByteCount = byteCount
	}
}

func stateBlockRefsSortedByTokenStart(blocks []kv.StateBlockRef) bool {
	for i := 1; i < len(blocks); i++ {
		prevStart := blocks[i-1].TokenStart
		curStart := blocks[i].TokenStart
		if curStart < prevStart {
			return false
		}
		if curStart == prevStart && blocks[i].Index < blocks[i-1].Index {
			return false
		}
	}
	return true
}

// indexHashBytes streams the canonical input into a sha256 hasher and
// returns the binary digest in a stack-allocated array. The bounded
// header (Kind|BundleURI|...|ChatTemplateHash) is pre-built in a
// pooled bytes.Buffer so the two int writes don't escape their digit
// buffer to the heap through hash.Hash's interface dispatch; the
// per-entry tail then streams pipe+entry-hash pairs straight to
// sha256 because Builder-batching the entry tail loses at scale —
// the doubling backing slice grows into hundreds of KB on a 1000-
// entry index (measured 25 µs streaming vs 57 µs full-builder).
//
// Returns the zero array when index is nil so the hex wrapper can
// emit "" without an extra branch.
func indexHashBytes(index *StateIndex) [sha256.Size]byte {
	var zero [sha256.Size]byte
	if index == nil {
		return zero
	}
	header := hashBufPool.Get().(*bytes.Buffer)
	header.Reset()
	var intBuf [20]byte
	header.WriteString(index.Kind)
	header.WriteByte('|')
	header.WriteString(index.BundleURI)
	header.WriteByte('|')
	header.WriteString(index.SnapshotHash)
	header.WriteByte('|')
	header.WriteString(string(index.KVEncoding))
	header.WriteByte('|')
	header.Write(strconv.AppendInt(intBuf[:0], int64(index.TokenCount), 10))
	header.WriteByte('|')
	header.Write(strconv.AppendInt(intBuf[:0], int64(index.BlockSize), 10))
	header.WriteByte('|')
	header.WriteString(index.Model.Hash)
	header.WriteByte('|')
	header.WriteString(index.Tokenizer.Hash)
	header.WriteByte('|')
	header.WriteString(index.Tokenizer.ChatTemplateHash)
	h := sha256.New()
	h.Write(header.Bytes())
	hashBufPool.Put(header)
	for i := range index.Entries {
		writeIndexHashString(h, "|")
		entryHash := index.Entries[i].Hash
		if entryHash == "" {
			entryHash = indexEntryHash(&index.Entries[i])
		}
		writeIndexHashString(h, entryHash)
	}
	// Sum into a stack-allocated [32]byte rather than passing nil
	// (which heap-allocates the digest slice).
	var sumBuf [sha256.Size]byte
	digest := h.Sum(sumBuf[:0])
	var out [sha256.Size]byte
	copy(out[:], digest)
	return out
}

func indexHash(index *StateIndex) string {
	if index == nil {
		return ""
	}
	sum := indexHashBytes(index)
	return core.HexEncode(sum[:])
}

// indexHashEquals reports whether expectedHex matches the
// freshly-computed canonical hash of index. Avoids the HexEncode
// alloc by decoding expectedHex into a stack [32]byte and comparing
// arrays. Used by Validate's tail check so the index-hash recompute
// path adds zero allocs.
func indexHashEquals(index *StateIndex, expectedHex string) bool {
	if len(expectedHex) != sha256.Size*2 {
		return false
	}
	sum := indexHashBytes(index)
	var expected [sha256.Size]byte
	if _, err := hex.Decode(expected[:], core.AsBytes(expectedHex)); err != nil {
		return false
	}
	return sum == expected
}

// indexEntryHashBytes writes the canonical entry input into the shared
// hashBufPool and returns the binary SHA-256 digest in a stack-allocated
// array. The hex wrapper builds on this; validate() reuses the binary
// form to compare against the stored hex without allocating the
// computed hex string.
func indexEntryHashBytes(entry *StateIndexEntry) [sha256.Size]byte {
	b := hashBufPool.Get().(*bytes.Buffer)
	b.Reset()
	var intBuf [20]byte
	b.WriteString(entry.URI)
	b.WriteByte('|')
	b.WriteString(entry.BundleURI)
	b.WriteByte('|')
	b.WriteString(entry.Title)
	b.WriteByte('|')
	b.Write(strconv.AppendInt(intBuf[:0], int64(entry.TokenStart), 10))
	b.WriteByte('|')
	b.Write(strconv.AppendInt(intBuf[:0], int64(entry.TokenCount), 10))
	b.WriteByte('|')
	b.Write(strconv.AppendInt(intBuf[:0], entry.ByteStart, 10))
	b.WriteByte('|')
	b.Write(strconv.AppendInt(intBuf[:0], entry.ByteCount, 10))
	for _, label := range entry.Labels {
		b.WriteByte('|')
		b.WriteString(label)
	}
	if len(entry.Meta) == 1 {
		for key, value := range entry.Meta {
			b.WriteByte('|')
			b.WriteString(key)
			b.WriteByte('=')
			b.WriteString(value)
		}
	} else if len(entry.Meta) > 1 {
		// Stack-rooted small-buffer for the common 2-8 meta-key case
		// (sleepEntryMeta produces 0-3 parent_* keys + caller-supplied
		// session id / agent name). For larger Meta append spills to
		// heap on the second grow — accepted floor for the rare path.
		var stackKeys [8]string
		keys := stackKeys[:0]
		for key := range entry.Meta {
			keys = append(keys, key)
		}
		core.SliceSort(keys)
		for _, key := range keys {
			b.WriteByte('|')
			b.WriteString(key)
			b.WriteByte('=')
			b.WriteString(entry.Meta[key])
		}
	}
	sum := sha256.Sum256(b.Bytes())
	hashBufPool.Put(b)
	return sum
}

func indexEntryHash(entry *StateIndexEntry) string {
	sum := indexEntryHashBytes(entry)
	return core.HexEncode(sum[:])
}

// indexEntryHashEquals reports whether expectedHex (a 64-char SHA-256
// hex string) matches the freshly-computed canonical hash of entry.
// Avoids the HexEncode alloc of indexEntryHash by decoding the
// expected hex into a stack [32]byte and comparing arrays. Hit per
// entry on every Validate(checkHashes=true) — N alloc savings for
// N-entry indexes.
func indexEntryHashEquals(entry *StateIndexEntry, expectedHex string) bool {
	if len(expectedHex) != sha256.Size*2 {
		return false
	}
	sum := indexEntryHashBytes(entry)
	var expected [sha256.Size]byte
	if _, err := hex.Decode(expected[:], core.AsBytes(expectedHex)); err != nil {
		return false
	}
	return sum == expected
}

// writeIndexHashString is the only remaining hash.Hash helper —
// used inside indexHash's per-entry tail to stream pipe + hex
// separator/value pairs. The Int / Int64 helpers were removed when
// indexHash moved its integer fields into the header Builder
// (strconv.AppendInt into a concrete *bytes.Buffer avoids the
// hash.Hash-interface escape they used to incur).
func writeIndexHashString(h hash.Hash, value string) {
	h.Write(core.AsBytes(value))
}

// cloneIndexEntries returns a shallow per-entry copy of entries: one
// slice allocation, zero per-entry allocations. The construction path
// in NewStateIndex mutates only scalar entry fields (BundleURI, Hash)
// on the returned slice, so aliasing the caller's Labels/Meta reference
// fields is intentional and safe — the caller's input is never mutated
// through these copies. Callers that hand in retained Labels/Meta and
// later mutate them in place would stale the stored entry Hash, which
// Validate(checkHashes) catches as errStateIndexEntryHashMismatch; the
// sole in-repo caller (NewSleepIndex) hands fresh, unretained values.
// Deep per-entry isolation lives in cloneIndexEntry, used by Entry()
// where a defensive copy is part of the documented contract.
//
//	clones := cloneIndexEntries(opts.Entries)
func cloneIndexEntries(entries []StateIndexEntry) []StateIndexEntry {
	if len(entries) == 0 {
		return nil
	}
	out := make([]StateIndexEntry, len(entries))
	copy(out, entries)
	return out
}

func cloneIndexEntry(entry StateIndexEntry) StateIndexEntry {
	entry.Labels = core.SliceClone(entry.Labels)
	entry.Meta = core.MapClone(entry.Meta)
	return entry
}
