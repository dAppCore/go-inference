// SPDX-Licence-Identifier: EUPL-1.2

package kv

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	stdio "io"
	"strconv"

	core "dappco.re/go"
	state "dappco.re/go/inference/model/state"
)

// SaveStateBlocks stores each KV block as a separate State chunk and returns a
// manifest.
func (s *Snapshot) SaveStateBlocks(ctx context.Context, store state.Writer, opts StateBlockOptions) (*StateBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if s == nil {
		return nil, errSnapshotNil
	}
	if store == nil {
		return nil, errStateStoreNil
	}
	blockSize := opts.BlockSize
	if blockSize <= 0 {
		blockSize = defaultCacheBlockSize
	}
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return nil, err
	}
	// Pre-size block-tracking slices against the expected block count —
	// SaveStateBlocks walks blockSize-aligned ranges, so the count is
	// known within a layer-window adjustment of (seqLen + blockSize - 1) /
	// blockSize. Saves the geometric-grow append cycle per block.
	expectedBlocks := 1
	if blockSize > 0 && len(s.Tokens) > 0 {
		expectedBlocks = (len(s.Tokens) + blockSize - 1) / blockSize
	}
	bundle := &StateBlockBundle{
		Version:      StateBlockVersion,
		Kind:         StateBlockBundleKind,
		KVEncoding:   encoding,
		Architecture: s.Architecture,
		TokenCount:   len(s.Tokens),
		TokenOffset:  EffectiveTokenOffset(s),
		BlockSize:    blockSize,
		NumLayers:    s.NumLayers,
		NumHeads:     s.NumHeads,
		SeqLen:       EffectiveSeqLen(s),
		HeadDim:      s.HeadDim,
		Blocks:       make([]StateBlockRef, 0, expectedBlocks),
	}
	err = s.walkBlocks(blockSize, false, func(block Block) (bool, error) {
		ref, hash, payloadEncoding, payloadByteCount, reused, err := saveOrReuseKVSnapshotStateBlock(ctx, store, block, opts, encoding)
		if err != nil {
			return false, err
		}
		if reused {
			bundle.ReusedBlocks++
		}
		bundle.Blocks = append(bundle.Blocks, StateBlockRef{
			Index:            block.Index,
			TokenStart:       block.TokenStart,
			TokenCount:       block.TokenCount,
			KVHash:           hash,
			PayloadEncoding:  payloadEncoding,
			PayloadByteCount: payloadByteCount,
			State:            ref,
			Memvid:           ref,
		})
		return true, nil
	})
	if err != nil {
		return nil, err
	}
	bundle.SnapshotHash = kvSnapshotStateBlockBundleHash(bundle)
	return bundle, nil
}

// SaveMemvidBlocks stores each KV block as a separate memvid chunk and returns
// a manifest.
//
// Deprecated: use SaveStateBlocks.
func (s *Snapshot) SaveMemvidBlocks(ctx context.Context, store state.Writer, opts StateBlockOptions) (*StateBlockBundle, error) {
	return s.SaveStateBlocks(ctx, store, opts)
}

// SaveStateBlocksFromStream stores streamed KV blocks into a durable State
// bundle without retaining all sliced blocks in memory.
func SaveStateBlocksFromStream(ctx context.Context, store state.Writer, opts StateBlockOptions, stream func(func(Block) (bool, error)) error) (*StateBlockBundle, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return nil, errStateStoreNil
	}
	if stream == nil {
		return nil, errBlockStreamNil
	}
	blockSize := opts.BlockSize
	if blockSize <= 0 {
		blockSize = defaultCacheBlockSize
	}
	encoding, err := normalizeKVSnapshotEncoding(opts.KVEncoding)
	if err != nil {
		return nil, err
	}
	bundle := &StateBlockBundle{
		Version:    StateBlockVersion,
		Kind:       StateBlockBundleKind,
		KVEncoding: encoding,
		BlockSize:  blockSize,
		Blocks:     []StateBlockRef{},
	}
	// Trusted-prefix graft: adopt the parent's whole blocks below the
	// boundary by reference. The capture side skips the same range
	// (CaptureOptions.BlockStartToken), so the stream below begins at the
	// boundary and the indexes tile contiguously.
	if boundary := TrustedReuseBoundary(opts, blockSize); boundary > 0 {
		parent := opts.ReusePrefix
		for _, ref := range parent.Blocks {
			if ref.TokenStart+ref.TokenCount > boundary {
				break
			}
			grafted := ref
			grafted.Index = len(bundle.Blocks)
			bundle.Blocks = append(bundle.Blocks, grafted)
			bundle.ReusedBlocks++
		}
		if bundle.SeqLen < boundary {
			bundle.SeqLen = boundary
		}
		if bundle.TokenCount < boundary {
			bundle.TokenCount = boundary
		}
		if bundle.Architecture == "" {
			bundle.Architecture = parent.Architecture
		}
		if bundle.NumLayers == 0 {
			bundle.NumLayers = parent.NumLayers
		}
		if bundle.NumHeads == 0 {
			bundle.NumHeads = parent.NumHeads
		}
		if bundle.HeadDim == 0 {
			bundle.HeadDim = parent.HeadDim
		}
	}
	err = stream(func(block Block) (bool, error) {
		if err := ctx.Err(); err != nil {
			return false, err
		}
		if block.Snapshot == nil {
			return false, errStreamedBlockNil
		}
		ref, hash, payloadEncoding, payloadByteCount, reused, err := saveOrReuseKVSnapshotStateBlock(ctx, store, block, opts, encoding)
		if err != nil {
			return false, err
		}
		if reused {
			bundle.ReusedBlocks++
		}
		applyKVSnapshotStateBundleBlock(bundle, block)
		bundle.Blocks = append(bundle.Blocks, StateBlockRef{
			Index:            block.Index,
			TokenStart:       block.TokenStart,
			TokenCount:       block.TokenCount,
			KVHash:           hash,
			PayloadEncoding:  payloadEncoding,
			PayloadByteCount: payloadByteCount,
			State:            ref,
			Memvid:           ref,
		})
		return true, nil
	})
	if err != nil {
		return nil, err
	}
	if err := ValidateStateBlockBundle(bundle); err != nil {
		return nil, err
	}
	bundle.SnapshotHash = kvSnapshotStateBlockBundleHash(bundle)
	return bundle, nil
}

// SaveMemvidBlocksFromStream stores streamed KV blocks in a memvid-backed
// bundle without retaining all sliced blocks in memory.
//
// Deprecated: use SaveStateBlocksFromStream.
func SaveMemvidBlocksFromStream(ctx context.Context, store state.Writer, opts StateBlockOptions, stream func(func(Block) (bool, error)) error) (*StateBlockBundle, error) {
	return SaveStateBlocksFromStream(ctx, store, opts, stream)
}

func applyKVSnapshotStateBundleBlock(bundle *StateBlockBundle, block Block) {
	if bundle == nil || block.Snapshot == nil {
		return
	}
	snapshot := block.Snapshot
	if bundle.Architecture == "" {
		bundle.Architecture = snapshot.Architecture
	}
	if bundle.NumLayers == 0 {
		bundle.NumLayers = snapshot.NumLayers
	}
	if bundle.NumHeads == 0 {
		bundle.NumHeads = snapshot.NumHeads
	}
	if bundle.HeadDim == 0 {
		bundle.HeadDim = snapshot.HeadDim
	}
	if bundle.SeqLen < block.TokenStart+block.TokenCount {
		bundle.SeqLen = block.TokenStart + block.TokenCount
	}
	if bundle.TokenCount < block.TokenStart+block.TokenCount {
		bundle.TokenCount = block.TokenStart + block.TokenCount
	}
	if snapshot.TokenOffset > bundle.TokenOffset {
		bundle.TokenOffset = snapshot.TokenOffset
	}
}

func kvSnapshotStateBlockBundleHash(bundle *StateBlockBundle) string {
	if bundle == nil {
		return ""
	}
	builder := core.NewBuilder()
	// Pre-size to the exact final length so Builder never resizes mid-write.
	// Each block hash is 64 hex chars + 1 separator; the head fields run ~80
	// chars typical (architecture + 3 ints + encoding + 5 separators).
	size := len(bundle.Architecture) + len(string(bundle.KVEncoding)) + 5*1 + 30
	for _, ref := range bundle.Blocks {
		size += 1 + len(ref.KVHash)
	}
	builder.Grow(size)
	builder.WriteString(bundle.Architecture)
	builder.WriteString("|")
	builder.WriteString(string(bundle.KVEncoding))
	builder.WriteString("|")
	// strconv.AppendInt writes directly into the builder's growing
	// internal buffer; skips the three intermediate strings core.Itoa
	// would mint per call.
	var scratch [20]byte
	builder.Write(strconv.AppendInt(scratch[:0], int64(bundle.TokenCount), 10))
	builder.WriteString("|")
	builder.Write(strconv.AppendInt(scratch[:0], int64(bundle.TokenOffset), 10))
	builder.WriteString("|")
	builder.Write(strconv.AppendInt(scratch[:0], int64(bundle.BlockSize), 10))
	for _, ref := range bundle.Blocks {
		builder.WriteString("|")
		builder.WriteString(ref.KVHash)
	}
	// SHA256HexString uses core.AsBytes under the hood — skips the
	// []byte copy of the Builder.String() roundtrip on every block-
	// bundle hash computation.
	return core.SHA256HexString(builder.String())
}

func saveOrReuseKVSnapshotStateBlock(ctx context.Context, store state.Writer, block Block, opts StateBlockOptions, encoding Encoding) (state.ChunkRef, string, string, int, bool, error) {
	if reused, hash, ok, err := reusableKVSnapshotStateBlockRef(block, opts, encoding); err != nil {
		return state.ChunkRef{}, "", "", 0, false, err
	} else if ok {
		return stateBlockChunkRef(reused), hash, reused.PayloadEncoding, reused.PayloadByteCount, true, nil
	}
	ref, hash, payloadEncoding, payloadByteCount, err := saveKVSnapshotStateBlock(ctx, store, block, opts, encoding)
	return ref, hash, payloadEncoding, payloadByteCount, false, err
}

func reusableKVSnapshotStateBlockRef(block Block, opts StateBlockOptions, encoding Encoding) (StateBlockRef, string, bool, error) {
	parent := opts.ReusePrefix
	if parent == nil || len(parent.Blocks) == 0 {
		return StateBlockRef{}, "", false, nil
	}
	if parent.KVEncoding != "" && parent.KVEncoding != encoding {
		return StateBlockRef{}, "", false, nil
	}
	reuseLimit := opts.ReusePrefixTokens
	if reuseLimit <= 0 {
		reuseLimit = parent.TokenCount
	}
	if block.TokenStart < 0 || block.TokenCount <= 0 || block.TokenStart+block.TokenCount > reuseLimit {
		return StateBlockRef{}, "", false, nil
	}
	// Trusted parents match by RANGE alone — the prefix is identical by
	// construction, so serialising + hashing the captured block just to
	// decide reuse is the cost this lane exists to avoid.
	if opts.ReusePrefixTrusted {
		for _, ref := range parent.Blocks {
			if ref.TokenStart != block.TokenStart || ref.TokenCount != block.TokenCount {
				continue
			}
			reused := ref
			reused.Index = block.Index
			return reused, ref.KVHash, true, nil
		}
	}
	hash, err := hashStateBlockPayload(block, encoding)
	if err != nil {
		return StateBlockRef{}, "", false, err
	}
	for _, ref := range parent.Blocks {
		if ref.TokenStart != block.TokenStart || ref.TokenCount != block.TokenCount {
			continue
		}
		if ref.KVHash != "" && ref.KVHash != hash {
			continue
		}
		reused := ref
		reused.Index = block.Index
		reused.TokenStart = block.TokenStart
		reused.TokenCount = block.TokenCount
		reused.KVHash = hash
		return reused, hash, true, nil
	}
	return StateBlockRef{}, hash, false, nil
}

// TrustedReuseBoundary resolves the token boundary below which the parent
// bundle's blocks are adopted by reference for a trusted-prefix sleep: the
// largest run of contiguous, full, in-limit parent blocks from token zero.
// Zero when the options do not describe a trusted parent (untrusted reuse,
// missing parent, or a block-size mismatch — grafts must tile exactly).
func TrustedReuseBoundary(opts StateBlockOptions, blockSize int) int {
	parent := opts.ReusePrefix
	if !opts.ReusePrefixTrusted || parent == nil || len(parent.Blocks) == 0 {
		return 0
	}
	if parent.BlockSize != blockSize {
		return 0
	}
	reuseLimit := opts.ReusePrefixTokens
	if reuseLimit <= 0 {
		reuseLimit = parent.TokenCount
	}
	boundary := 0
	for _, ref := range parent.Blocks {
		if ref.TokenStart != boundary || ref.TokenCount != blockSize || boundary+blockSize > reuseLimit {
			break
		}
		boundary += blockSize
	}
	return boundary
}

func hashStateBlockPayload(block Block, encoding Encoding) (string, error) {
	if block.Snapshot == nil {
		return "", errBlockNil
	}
	hash := sha256.New()
	if err := block.Snapshot.writeWithOptions(hash, SaveOptions{KVEncoding: encoding}); err != nil {
		return "", err
	}
	var sum [sha256.Size]byte
	return hex.EncodeToString(hash.Sum(sum[:0])), nil
}

func saveKVSnapshotStateBlock(ctx context.Context, store state.Writer, block Block, opts StateBlockOptions, encoding Encoding) (state.ChunkRef, string, string, int, error) {
	if streamStore, ok := store.(state.BinaryStreamWriter); ok {
		payloadSize, err := block.Snapshot.encodedSizeWithOptions(SaveOptions{KVEncoding: encoding})
		if err != nil {
			return state.ChunkRef{}, "", "", 0, err
		}
		hash := sha256.New()
		ref, err := streamStore.PutBytesStream(ctx, payloadSize, kvSnapshotStateBlockPutOptions(block, opts, "", string(encoding), kvSnapshotStatePayloadRaw), func(writer stdio.Writer) error {
			return block.Snapshot.writeWithOptions(stdio.MultiWriter(writer, hash), SaveOptions{KVEncoding: encoding})
		})
		if err != nil {
			return state.ChunkRef{}, "", "", 0, core.E("Snapshot.SaveStateBlocks", "stream raw State block", err)
		}
		var sum [sha256.Size]byte
		return ref, hex.EncodeToString(hash.Sum(sum[:0])), kvSnapshotStatePayloadRaw, payloadSize, nil
	}
	data, err := block.Snapshot.bytesWithOptions(SaveOptions{KVEncoding: encoding})
	if err != nil {
		return state.ChunkRef{}, "", "", 0, err
	}
	hash := core.SHA256Hex(data)
	if binaryStore, ok := store.(state.BinaryWriter); ok {
		ref, err := binaryStore.PutBytes(ctx, data, kvSnapshotStateBlockPutOptions(block, opts, hash, string(encoding), kvSnapshotStatePayloadRaw))
		if err != nil {
			return state.ChunkRef{}, "", "", 0, core.E("Snapshot.SaveStateBlocks", "write raw State block", err)
		}
		return ref, hash, kvSnapshotStatePayloadRaw, len(data), nil
	}
	envelope := kvSnapshotStateBlockEnvelope{
		Version:          StateBlockVersion,
		Kind:             KVSnapshotStateBlockKind,
		BlockIndex:       block.Index,
		TokenStart:       block.TokenStart,
		TokenCount:       block.TokenCount,
		KVHash:           hash,
		KVEncoding:       string(encoding),
		BinaryEncoding:   "base64",
		PayloadByteCount: len(data),
		Data:             core.Base64Encode(data),
	}
	ref, err := store.Put(ctx, core.JSONMarshalString(envelope), kvSnapshotStateBlockPutOptions(block, opts, hash, string(encoding), kvSnapshotStatePayloadJSONBase64))
	if err != nil {
		return state.ChunkRef{}, "", "", 0, core.E("Snapshot.SaveStateBlocks", "write State block", err)
	}
	return ref, hash, kvSnapshotStatePayloadJSONBase64, len(data), nil
}

// SaveStateBlockBundle stores the KV block manifest in the same
// State store as its referenced blocks.
func SaveStateBlockBundle(ctx context.Context, store state.Writer, bundle *StateBlockBundle, uri string) (state.ChunkRef, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	if store == nil {
		return state.ChunkRef{}, errStateStoreNil
	}
	if core.Trim(uri) == "" {
		return state.ChunkRef{}, errBundleURIRequired
	}
	if err := ValidateStateBlockBundle(bundle); err != nil {
		return state.ChunkRef{}, err
	}
	ref, err := store.Put(ctx, core.JSONMarshalString(bundle), state.PutOptions{
		URI:    uri,
		Title:  "go-mlx State block bundle",
		Kind:   StateBlockBundleKind,
		Track:  "session-kv-blocks",
		Labels: []string{"go-mlx", "kv-snapshot-block-bundle"},
	})
	if err != nil {
		return state.ChunkRef{}, core.E("Snapshot.SaveStateBlockBundle", "write State bundle", err)
	}
	return ref, nil
}

// SaveMemvidBlockBundle stores the KV block manifest in the same
// old memvid-named store as its referenced blocks.
//
// Deprecated: use SaveStateBlockBundle.
func SaveMemvidBlockBundle(ctx context.Context, store state.Writer, bundle *MemvidBlockBundle, uri string) (state.ChunkRef, error) {
	return SaveStateBlockBundle(ctx, store, bundle, uri)
}

func kvSnapshotStateBlockPutOptions(block Block, opts StateBlockOptions, hash, kvEncoding, payloadEncoding string) state.PutOptions {
	kind := opts.Kind
	if kind == "" {
		kind = KVSnapshotStateBlockKind
	}
	track := opts.Track
	if track == "" {
		track = "session-kv-blocks"
	}
	tags := cloneKVSnapshotStateTags(opts.Tags)
	if hash != "" {
		tags["kv_hash"] = hash
	}
	tags["kv_encoding"] = kvEncoding
	tags["payload_encoding"] = payloadEncoding
	// Compute the index string once and reuse — block.Index is used in
	// tags, URI, and the default Title. The previous code minted three
	// separate copies via core.Itoa.
	indexStr := core.Itoa(block.Index)
	tags["block_index"] = indexStr
	tags["token_start"] = core.Itoa(block.TokenStart)
	tags["token_count"] = core.Itoa(block.TokenCount)
	// Skip the per-block labels make when the caller supplied no extra
	// labels — the default two-element pair is identical across blocks,
	// share a single package-global slice. State stores treat Labels as
	// read-only input; mutating the returned PutOptions is contract-
	// violating already.
	var labels []string
	if len(opts.Labels) == 0 {
		labels = kvSnapshotStateBlockDefaultLabels
	} else {
		// Pre-size for the deterministic 2 appended labels — avoids the
		// geometric-grow path on every per-block State save.
		labels = make([]string, len(opts.Labels), len(opts.Labels)+2)
		copy(labels, opts.Labels)
		labels = append(labels, "go-mlx", "kv-snapshot-block")
	}
	baseURI := core.FirstNonBlank(opts.URI, "mlx://kv-snapshot-blocks")
	// Direct string concatenation skips the fmt.Sprintf parse + format
	// state machinery on every per-block save (~SaveStateBlocks fires once
	// per checkpointed block during prefill). Avoid materialising the
	// default title when opts.Title is non-empty — the previous code
	// concatenated "go-mlx KV block " + indexStr unconditionally.
	title := opts.Title
	if title == "" {
		title = "go-mlx KV block " + indexStr
	}
	return state.PutOptions{
		URI:    baseURI + "/block/" + indexStr,
		Title:  title,
		Kind:   kind,
		Track:  track,
		Tags:   tags,
		Labels: labels,
	}
}
