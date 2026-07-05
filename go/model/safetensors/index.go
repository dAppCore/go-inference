// SPDX-Licence-Identifier: EUPL-1.2

// index.go is the bounded-memory safetensors index/streaming layer. It
// never materialises a whole tensor file in memory: ReadIndex reads only
// the JSON header (via a hand-rolled zero-allocation parser in
// header_parse.go), TensorRef records each tensor's absolute byte offsets,
// and TensorReader/ShardCache/WriteSubset stream individual tensors'
// payloads through ReadAt/chunked writes sized independently of file size.
// Float decode (DecodeFloatData) uses unsafe reinterpret-casts for F32/F64
// and, on darwin/arm64, a NEON FCVTL hardware loop for F16 — see
// float16_neon_darwin_arm64.go.
//
// The whole-file codec in safetensors.go (ReadSafetensors/WriteSafetensors/
// GetTensorData/DecodeFloat32/EncodeFloat32: eager read, reflection-driven
// JSON, no chunking) suits small files with no streaming or perf
// constraint. Every symbol here exists because the whole-file shape
// doesn't fit, either because the source is a multi-gigabyte model shard or
// because the decode runs in a per-element hot loop worth hardware
// acceleration.
//
//	idx, err := safetensors.ReadIndex("/models/model-00001.safetensors")
//	if err != nil { return err }
//	ref := idx.Tensors["model.embed_tokens.weight"]
//	values, err := safetensors.ReadRefValues(ref)

package safetensors

import (
	"context"
	"encoding/binary"
	stdio "io"
	"math"
	"unsafe"

	core "dappco.re/go"
)

// Sentinel errors hoisted to package vars — see W9-Y in header_parse.go
// for context. These are static-message errors fired on validation
// failure paths inside the read/decode hot paths. Lifting them avoids
// the per-fire core.NewError alloc and lets errors.Is comparison work
// against typed sentinels (e.g. callers wanting to distinguish "chunk
// truncated" from "chunk out of bounds" without text-matching).
var (
	errChunkOutOfBounds   = core.NewError("mlx: safetensors tensor chunk exceeds tensor bounds")
	errChunkTruncated     = core.NewError("mlx: safetensors tensor chunk is truncated")
	errF32PayloadMismatch = core.NewError("F32 payload length does not match tensor shape")
	errF16PayloadMismatch = core.NewError("F16 payload length does not match tensor shape")
	errBF16PayloadMatch   = core.NewError("BF16 payload length does not match tensor shape")
	errF64PayloadMismatch = core.NewError("F64 payload length does not match tensor shape")
)

// HeaderEntry is one tensor entry in the safetensors JSON header.
type HeaderEntry struct {
	DType       string  `json:"dtype"`
	Shape       []int64 `json:"shape"`
	DataOffsets []int64 `json:"data_offsets"`
}

type Index struct {
	Path    string
	Tensors map[string]TensorRef
	Names   []string
}

type TensorRef struct {
	Name      string
	Path      string
	DType     string
	Shape     []uint64
	Elements  int
	DataStart int64
	ByteLen   int64
}

type TensorReader struct {
	ref             TensorRef
	file            *core.OSFile
	bytesPerElement int
}

func IndexFiles(paths []string) (Index, error) {
	if len(paths) == 0 {
		return Index{Tensors: map[string]TensorRef{}}, nil
	}
	// Reuse the first shard's map + Names slice as the merged
	// accumulator — saves one empty-map alloc and lets us size the
	// merged Names slice based on the first shard's count × shard
	// count (close enough for uniform safetensors splits). Subsequent
	// shards merge their entries in-place.
	first, err := ReadIndex(paths[0])
	if err != nil {
		return Index{}, err
	}
	if len(paths) == 1 {
		core.SliceSort(first.Names)
		first.Path = ""
		return first, nil
	}
	// Estimate the merged total: assume each remaining shard has at
	// least as many tensors as the first. Over-allocate by 1.5x to
	// absorb non-uniform splits without re-growing.
	estTotal := max(len(first.Names)*len(paths), len(first.Names)+len(first.Names))
	merged := Index{Tensors: first.Tensors, Path: ""}
	if cap(first.Names) < estTotal {
		grown := make([]string, len(first.Names), estTotal)
		copy(grown, first.Names)
		merged.Names = grown
	} else {
		merged.Names = first.Names
	}
	for _, path := range paths[1:] {
		shard, err := ReadIndex(path)
		if err != nil {
			return Index{}, err
		}
		if cap(merged.Names) < len(merged.Names)+len(shard.Names) {
			grown := make([]string, len(merged.Names), len(merged.Names)+len(shard.Names))
			copy(grown, merged.Names)
			merged.Names = grown
		}
		for _, name := range shard.Names {
			if _, ok := merged.Tensors[name]; ok {
				return Index{}, core.NewError("mlx: duplicate tensor in safetensors shards: " + name)
			}
			merged.Tensors[name] = shard.Tensors[name]
			merged.Names = append(merged.Names, name)
		}
	}
	core.SliceSort(merged.Names)
	return merged, nil
}

func ReadIndex(path string) (Index, error) {
	opened := core.Open(path)
	if !opened.OK {
		return Index{}, opened.Err()
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close()

	var headerLenBuf [8]byte
	if _, err := stdio.ReadFull(file, headerLenBuf[:]); err != nil {
		return Index{}, err
	}
	headerLen := binary.LittleEndian.Uint64(headerLenBuf[:])
	headerBytes := make([]byte, int(headerLen))
	if _, err := stdio.ReadFull(file, headerBytes); err != nil {
		return Index{}, err
	}
	return ParseHeaderRefs(path, headerBytes, int64(8+headerLen))
}

// ParseHeaderRefs walks an already-read safetensors header bytes blob
// and emits one TensorRef per non-metadata tensor into a returned
// Index. dataStart is the absolute byte offset in the source file
// where tensor payloads begin (typically 8 + len(headerBytes), the
// position right after the 8-byte little-endian header length).
//
// Callers that have already validated the header length (e.g.
// pkg/metal/minimax_m2 which enforces a per-pack size cap before
// reading) can use this to share the hand-rolled walker — see Wave 8
// W8-K — without re-opening the file. The walker is the same one
// ReadIndex drives internally: zero-alloc string spans into the
// header arena, interned canonical dtype strings, one shared shape
// slab per Index. Per-tensor cost lands at ~1 alloc once the arena
// is in scope.
func ParseHeaderRefs(path string, headerBytes []byte, dataStart int64) (Index, error) {
	// First pass — count tensors + total shape dims so the map, Names
	// slice and shape slab each take one sized allocation. The walker
	// then runs a hand-rolled JSON parse over the header bytes,
	// emitting one TensorRef per tensor directly (no HeaderEntry,
	// no per-tensor Shape/DataOffsets slice allocs). This replaces the
	// reflection-driven json.Unmarshal that dominated the alloc count
	// on model-load (see Wave 8 W8-I profile).
	tensors, totalDims := countTensorsAndDims(headerBytes)
	if tensors < 0 {
		// Fall back to a conservative initial size — the parser will
		// surface any structural error encountered on the live pass.
		tensors = 0
		totalDims = 0
	}
	index := Index{
		Path:    path,
		Tensors: make(map[string]TensorRef, tensors),
		Names:   make([]string, 0, tensors),
	}
	shapeSlab := make([]uint64, 0, totalDims)
	if err := parseHeaderInto(path, headerBytes, dataStart, &index, &shapeSlab); err != nil {
		return Index{}, err
	}
	core.SliceSort(index.Names)
	return index, nil
}

// refFromHeaderSlab is the index-local variant of RefFromHeader that
// carves each tensor's Shape slice out of a shared uint64 slab. Callers
// guarantee the slab has enough capacity (sized by the prior header
// scan). Public RefFromHeader retains its standalone allocation form.
func RefFromHeader(path, name string, entry HeaderEntry, dataStart int64) (TensorRef, error) {
	if len(entry.DataOffsets) != 2 {
		return TensorRef{}, core.NewError("mlx: safetensors tensor has invalid data_offsets: " + name)
	}
	begin := entry.DataOffsets[0]
	end := entry.DataOffsets[1]
	if begin < 0 || end < begin {
		return TensorRef{}, core.NewError("mlx: safetensors tensor offsets are invalid: " + name)
	}
	shape := make([]uint64, len(entry.Shape))
	elements := 1
	for i, dim := range entry.Shape {
		if dim <= 0 {
			return TensorRef{}, core.NewError("mlx: safetensors tensor has invalid shape: " + name)
		}
		shape[i] = uint64(dim)
		elements *= int(dim)
	}
	return TensorRef{
		Name:      name,
		Path:      path,
		DType:     core.Upper(entry.DType),
		Shape:     shape,
		Elements:  elements,
		DataStart: dataStart + begin,
		ByteLen:   end - begin,
	}, nil
}

func ReadRefValues(ref TensorRef) ([]float32, error) {
	opened := core.Open(ref.Path)
	if !opened.OK {
		return nil, opened.Err()
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close()
	return readRefValuesFrom(file, ref)
}

// readRefValuesFrom is the open-handle core of ReadRefValues: it reads the
// ref's payload from an already-open shard handle via ReadAt and decodes
// it. ReadRefValues opens the file and delegates here; ShardCache-bound
// readers (TensorReader.ReadValues) call it over a handle opened ONCE for
// the whole load, so many refs from one shard skip the per-ref core.Open +
// path→C-string syscall.ByteSliceFromString alloc. ReadAt is offset-
// addressed, so the shared handle is safe for any number of refs. The
// decode path is byte-for-byte the one ReadRefValues always ran.
func readRefValuesFrom(file *core.OSFile, ref TensorRef) ([]float32, error) {
	raw := make([]byte, int(ref.ByteLen))
	n, err := file.ReadAt(raw, ref.DataStart)
	if err != nil && !(err == stdio.EOF && n == len(raw)) {
		return nil, err
	}
	return DecodeFloatData(ref.DType, raw, ref.Elements)
}

func WriteRefFloat32Chunks(ctx context.Context, file *core.OSFile, ref TensorRef, chunkElements int) error {
	if chunkElements <= 0 {
		chunkElements = defaultChunkElements
	}
	reader, err := OpenReader(ref)
	if err != nil {
		return err
	}
	defer reader.Close()
	// Reuse three scratch buffers across chunked writes:
	//   raw       — the byte payload read from the source file
	//   values    — the decoded float32 slice
	//   writeBuf  — the re-encoded bytes the writer flushes
	// Each chunk previously allocated all three; now they grow once
	// to chunkElements (or chunkElements*bytesPerElement / 4) and are
	// reused for every subsequent chunk on the same tensor.
	var (
		rawScratch    []byte
		valuesScratch []float32
		writeScratch  []byte
	)
	for offset := 0; offset < ref.Elements; offset += chunkElements {
		if err := ctx.Err(); err != nil {
			return err
		}
		count := min(chunkElements, ref.Elements-offset)
		var values []float32
		rawScratch, valuesScratch, values, err = reader.readFloat32ChunkInto(offset, count, rawScratch, valuesScratch)
		if err != nil {
			return err
		}
		writeScratch, err = writeFloat32ValuesScratch(file, values, writeScratch)
		if err != nil {
			return err
		}
	}
	return nil
}

func ReadRefFloat32Chunk(ref TensorRef, offset, count int) ([]float32, error) {
	reader, err := OpenReader(ref)
	if err != nil {
		return nil, err
	}
	defer reader.Close()
	return reader.ReadFloat32Chunk(offset, count)
}

func OpenReaders(refs []TensorRef) ([]TensorReader, error) {
	readers := make([]TensorReader, 0, len(refs))
	for _, ref := range refs {
		reader, err := OpenReader(ref)
		if err != nil {
			CloseReaders(readers)
			return nil, err
		}
		readers = append(readers, reader)
	}
	return readers, nil
}

func OpenReader(ref TensorRef) (TensorReader, error) {
	bytesPerElement, err := DTypeByteSize(ref.DType)
	if err != nil {
		return TensorReader{}, err
	}
	opened := core.Open(ref.Path)
	if !opened.OK {
		return TensorReader{}, opened.Err()
	}
	return TensorReader{
		ref:             ref,
		file:            opened.Value.(*core.OSFile),
		bytesPerElement: bytesPerElement,
	}, nil
}

// NewFileReader binds an already-open file to a tensor ref without
// opening anything. Reads use ReadAt at the ref's absolute DataStart, so
// many refs sharing one shard file can each get a reader over the same
// *core.OSFile — a caller iterating tensors in one pack opens each shard
// once and reuses the handle instead of paying os.Open + the path→C-string
// syscall.ByteSliceFromString allocation per tensor. The returned reader
// does NOT own the file: its Close is a no-op concern for the caller, who
// owns the handle's lifetime and must close it exactly once.
func NewFileReader(file *core.OSFile, ref TensorRef) (TensorReader, error) {
	bytesPerElement, err := DTypeByteSize(ref.DType)
	if err != nil {
		return TensorReader{}, err
	}
	return TensorReader{
		ref:             ref,
		file:            file,
		bytesPerElement: bytesPerElement,
	}, nil
}

func CloseReaders(readers []TensorReader) {
	for _, reader := range readers {
		reader.Close()
	}
}

// ShardCache lazily opens safetensors shard files and caches the handle by
// Path, so a load reading many tensors from one shard opens it ONCE. The
// leaf ReadRefValues/ReadRefRaw each core.Open(ref.Path) per ref, so an
// 8-expert MiniMax M2 LoadPackedExperts reopened its shard ~70 times —
// os.newFile + the path→C-string syscall.ByteSliceFromString alloc once
// per reopen, the FLAT-dominant cost of that load by alloc_objects. The
// cache opens each distinct shard once and hands ReadAt-based readers
// (NewFileReader) over the shared handle; ReadAt is offset-addressed so
// any number of readers safely share one handle. Reads through the bound
// reader (ReadValues / ReadRaw) are byte-identical to the leaf functions —
// only the open count differs.
//
// The caller owns the cache lifetime and MUST Close it exactly once (a
// deferred Close right after NewShardCache closes every handle even on a
// mid-load error). The cache is read-only (ReadAt), so Close is a
// best-effort flush of shared read handles.
//
//	cache := safetensors.NewShardCache()
//	defer cache.Close()
//	reader, err := cache.Reader(ref)   // opens ref.Path once, reuses after
//	values, err := reader.ReadValues() // same bytes as ReadRefValues(ref)
type ShardCache struct {
	files map[string]*core.OSFile
}

// NewShardCache returns an empty Path-keyed shard-handle cache. It pre-sizes
// for a single shard (the common case — one model split lands most tensors
// in the same file); additional distinct paths grow the map lazily.
func NewShardCache() *ShardCache {
	return &ShardCache{files: make(map[string]*core.OSFile, 1)}
}

// Reader returns a TensorReader bound to ref over a cache-owned handle for
// ref.Path, opening that shard once on first use and reusing the handle for
// every subsequent ref in the same file. The returned reader does NOT own
// the handle — its Close is a no-op concern; the cache closes all handles
// on ShardCache.Close.
func (c *ShardCache) Reader(ref TensorRef) (TensorReader, error) {
	file, err := c.handle(ref.Path)
	if err != nil {
		return TensorReader{}, err
	}
	return NewFileReader(file, ref)
}

// handle returns the cached *core.OSFile for path, opening (and caching) it
// on first request. Shared by Reader and the direct read helpers so every
// path goes through exactly one open per distinct shard.
func (c *ShardCache) handle(path string) (*core.OSFile, error) {
	if file, ok := c.files[path]; ok {
		return file, nil
	}
	opened := core.Open(path)
	if !opened.OK {
		return nil, opened.Err()
	}
	file := opened.Value.(*core.OSFile)
	c.files[path] = file
	return file, nil
}

// ReadRefValues reads and decodes ref to float32 over a cache-owned handle,
// opening ref.Path once and reusing it for later refs in the same shard.
// Output is byte-identical to the package-level ReadRefValues(ref) — same
// ReadAt offset, same DecodeFloatData — for callers that prefer the
// ref-in / values-out shape over holding a TensorReader.
func (c *ShardCache) ReadRefValues(ref TensorRef) ([]float32, error) {
	file, err := c.handle(ref.Path)
	if err != nil {
		return nil, err
	}
	return readRefValuesFrom(file, ref)
}

// ReadRefRaw reads ref's raw payload over a cache-owned handle, opening
// ref.Path once and reusing it for later refs in the same shard. Output is
// byte-identical to the package-level ReadRefRaw(ref), including the
// ByteLen validation and truncation check.
func (c *ShardCache) ReadRefRaw(ref TensorRef) ([]byte, error) {
	if ref.ByteLen < 0 || ref.ByteLen > int64(maxIntValue()) {
		return nil, core.NewError("mlx: safetensors tensor byte length is invalid: " + ref.Name)
	}
	file, err := c.handle(ref.Path)
	if err != nil {
		return nil, err
	}
	return readRefRawFrom(file, ref)
}

// Close closes every cached shard handle. Best-effort: the cache is
// read-only (ReadAt) so a close error has nothing to flush and no caller to
// inform. Safe to call once; the caller owns the lifetime.
func (c *ShardCache) Close() {
	for _, file := range c.files {
		file.Close()
	}
	clear(c.files)
}

func (r TensorReader) Close() {
	if r.file != nil {
		_ = r.file.Close()
	}
}

func (r TensorReader) ReadFloat32Chunk(offset, count int) ([]float32, error) {
	if offset < 0 || count < 0 || offset+count > r.ref.Elements {
		return nil, errChunkOutOfBounds
	}
	raw := make([]byte, count*r.bytesPerElement)
	start := r.ref.DataStart + int64(offset*r.bytesPerElement)
	n, err := r.file.ReadAt(raw, start)
	if err != nil && !(err == stdio.EOF && n == len(raw)) {
		return nil, err
	}
	if n != len(raw) {
		return nil, errChunkTruncated
	}
	return DecodeFloatData(r.ref.DType, raw, count)
}

// ReadFloat32ChunkInto is the exported scratch-aware variant of
// ReadFloat32Chunk. A caller looping over chunks (the merge linear/SLERP
// write paths, ComparePacks' per-tensor scan) hands back the prior
// chunk's byte + float32 buffers so each subsequent chunk reuses the
// backing arrays instead of allocating a fresh raw []byte and decoded
// []float32 every iteration. The returned values slice is the decoded
// chunk (length count); the first two returns carry the possibly-grown
// scratch buffers forward for the next call. Decoded output is
// byte-identical to ReadFloat32Chunk — same DecodeFloatData path.
//
// Reader-with-its-own-scratch is the required pattern when two readers'
// chunks are live simultaneously (base vs tuned, SLERP a vs b): give
// each reader an independent valuesScratch so reading the second does
// not clobber the first.
func (r TensorReader) ReadFloat32ChunkInto(offset, count int, rawScratch []byte, valuesScratch []float32) ([]byte, []float32, []float32, error) {
	return r.readFloat32ChunkInto(offset, count, rawScratch, valuesScratch)
}

// readFloat32ChunkInto is the scratch-aware variant of ReadFloat32Chunk.
// It accepts (and returns) byte + float32 scratch buffers so a caller
// in a chunked loop (WriteRefFloat32Chunks) can avoid allocating fresh
// buffers per chunk. The returned values slice always equals the
// (possibly grown) valuesScratch sliced to count.
func (r TensorReader) readFloat32ChunkInto(offset, count int, rawScratch []byte, valuesScratch []float32) ([]byte, []float32, []float32, error) {
	if offset < 0 || count < 0 || offset+count > r.ref.Elements {
		return rawScratch, valuesScratch, nil, errChunkOutOfBounds
	}
	rawNeed := count * r.bytesPerElement
	if cap(rawScratch) < rawNeed {
		rawScratch = make([]byte, rawNeed)
	} else {
		rawScratch = rawScratch[:rawNeed]
	}
	start := r.ref.DataStart + int64(offset*r.bytesPerElement)
	n, err := r.file.ReadAt(rawScratch, start)
	if err != nil && !(err == stdio.EOF && n == len(rawScratch)) {
		return rawScratch, valuesScratch, nil, err
	}
	if n != len(rawScratch) {
		return rawScratch, valuesScratch, nil, errChunkTruncated
	}
	values, err := decodeFloatDataInto(r.ref.DType, rawScratch, count, valuesScratch)
	if err != nil {
		return rawScratch, valuesScratch, nil, err
	}
	if cap(values) > cap(valuesScratch) {
		valuesScratch = values
	}
	return rawScratch, valuesScratch, values, nil
}

func DTypeByteSize(dtype string) (int, error) {
	// Canonical fast path covers the four supported dtypes by exact
	// match (the common case after RefFromHeader has normalised
	// entry.DType through core.Upper).
	switch dtype {
	case "F16", "BF16":
		return 2, nil
	case "F32":
		return 4, nil
	case "F64":
		return 8, nil
	}
	// Non-canonical input (callers handing us lowercase / mixed case).
	// Branch by length so we never call core.Upper — that path was
	// dominating the 26 ns / 1 alloc on lowercase "bf16". Each branch
	// is a single direct byte compare for the ASCII letters.
	switch len(dtype) {
	case 3:
		// F16, F32, F64.
		if (dtype[0] == 'F' || dtype[0] == 'f') && dtype[1] == '1' && dtype[2] == '6' {
			return 2, nil
		}
		if (dtype[0] == 'F' || dtype[0] == 'f') && dtype[1] == '3' && dtype[2] == '2' {
			return 4, nil
		}
		if (dtype[0] == 'F' || dtype[0] == 'f') && dtype[1] == '6' && dtype[2] == '4' {
			return 8, nil
		}
	case 4:
		// BF16.
		if (dtype[0] == 'B' || dtype[0] == 'b') && (dtype[1] == 'F' || dtype[1] == 'f') && dtype[2] == '1' && dtype[3] == '6' {
			return 2, nil
		}
	}
	return 0, core.NewError("unsupported dense safetensors dtype: " + dtype)
}

func maxIntValue() int { return int(^uint(0) >> 1) }

func ReadRefRaw(ref TensorRef) ([]byte, error) {
	if ref.ByteLen < 0 || ref.ByteLen > int64(maxIntValue()) {
		return nil, core.NewError("mlx: safetensors tensor byte length is invalid: " + ref.Name)
	}
	opened := core.Open(ref.Path)
	if !opened.OK {
		return nil, opened.Err()
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close()
	return readRefRawFrom(file, ref)
}

// readRefRawFrom is the open-handle core of ReadRefRaw: it reads the ref's
// raw payload from an already-open shard handle via ReadAt. ReadRefRaw
// opens the file and delegates here; ShardCache-bound readers
// (TensorReader.ReadRaw) call it over a handle opened ONCE for the load,
// so many refs from one shard skip the per-ref core.Open. The caller of
// ReadRefRaw has already validated ByteLen; cache readers reach this only
// via a ref whose ByteLen the index built, identical bytes either way.
func readRefRawFrom(file *core.OSFile, ref TensorRef) ([]byte, error) {
	raw := make([]byte, int(ref.ByteLen))
	n, err := file.ReadAt(raw, ref.DataStart)
	if err != nil && !(err == stdio.EOF && n == len(raw)) {
		return nil, err
	}
	if n != len(raw) {
		return nil, core.NewError("mlx: safetensors tensor payload is truncated: " + ref.Name)
	}
	return raw, nil
}

// ReadValues reads and decodes the bound ref's full payload to float32
// over this reader's shard handle. It is the open-handle equivalent of
// ReadRefValues — byte-identical output, same DecodeFloatData path — for a
// caller that opened the shard once (e.g. via ShardCache) and reads many
// refs from it without reopening per ref.
func (r TensorReader) ReadValues() ([]float32, error) {
	return readRefValuesFrom(r.file, r.ref)
}

// ReadRaw reads the bound ref's raw payload over this reader's shard
// handle. It is the open-handle equivalent of ReadRefRaw — byte-identical
// bytes, same truncation check — for a caller reading many refs from one
// already-open shard. ByteLen is validated by the index build that
// produced the ref; ShardCache callers pass index-built refs.
func (r TensorReader) ReadRaw() ([]byte, error) {
	if r.ref.ByteLen < 0 || r.ref.ByteLen > int64(maxIntValue()) {
		return nil, core.NewError("mlx: safetensors tensor byte length is invalid: " + r.ref.Name)
	}
	return readRefRawFrom(r.file, r.ref)
}

const defaultChunkElements = 1 << 20

// writeFloat32ValuesScratch reuses a caller-supplied byte buffer for
// the F32 encode. The buffer is grown when too small and returned so
// the caller (WriteRefFloat32Chunks) can reuse it across chunks.
func writeFloat32ValuesScratch(file *core.OSFile, values []float32, scratch []byte) ([]byte, error) {
	need := len(values) * 4
	if cap(scratch) < need {
		scratch = make([]byte, need)
	} else {
		scratch = scratch[:need]
	}
	for i, value := range values {
		binary.LittleEndian.PutUint32(scratch[i*4:], math.Float32bits(value))
	}
	_, err := file.Write(scratch)
	return scratch, err
}

func DecodeFloatData(dtype string, raw []byte, elements int) ([]float32, error) {
	return decodeFloatDataInto(dtype, raw, elements, nil)
}

// decodeFloatDataInto is the scratch-aware variant of DecodeFloatData.
// Callers that decode in a loop (WriteRefFloat32Chunks) can hand back
// the prior chunk's slice to avoid re-allocating.
func decodeFloatDataInto(dtype string, raw []byte, elements int, scratch []float32) ([]float32, error) {
	var values []float32
	if cap(scratch) < elements {
		values = make([]float32, elements)
	} else {
		values = scratch[:elements]
	}
	switch dtype {
	case "F32":
		if len(raw) != elements*4 {
			return nil, errF32PayloadMismatch
		}
		// Reinterpret-cast: float32 storage is little-endian on both
		// Go-supported architectures (arm64 + amd64), so the safetensors
		// on-disk byte view of an F32 tensor matches []float32 verbatim.
		// One memcpy replaces N × (LittleEndian.Uint32 + Float32frombits +
		// per-iter raw[i*4:] re-slice). Same pattern as kv/snapshot.go
		// decodeKVSnapshotNativeTensor.
		dst := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(values))), elements*4)
		copy(dst, raw)
	case "F16":
		if len(raw) != elements*2 {
			return nil, errF16PayloadMismatch
		}
		// Reinterpret-cast raw as []uint16. fp16 storage is little-endian
		// on both supported architectures, so bytes-on-disk match the
		// uint16 layout exactly. This eliminates the per-iter byte pair
		// combine + raw[i*2:] re-slice. On darwin/arm64 the conversion is
		// then vectorised via a NEON FCVTL V.4S, V.4H inner loop (cgo) —
		// see float16_neon_darwin_arm64.go. All other platforms fall
		// through to the scalar Float16ToFloat32 path via
		// float16_scalar.go. Output is bit-identical across builds.
		src16 := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(raw))), elements)
		float16SliceToFloat32(src16, values, elements)
	case "BF16":
		if len(raw) != elements*2 {
			return nil, errBF16PayloadMatch
		}
		// Same unsafe-uint16-slice pattern as F16. BF16 → F32 is just
		// "uint16 → uint32 → shift 16 → Float32frombits" which is itself
		// the high-half bit pattern of the target float32 — but Go's
		// Float32frombits is unavoidable to preserve NaN payloads.
		// The unsafe-slice cast still skips the per-iter byte combine.
		src16 := unsafe.Slice((*uint16)(unsafe.Pointer(unsafe.SliceData(raw))), elements)
		for i, v := range src16 {
			values[i] = math.Float32frombits(uint32(v) << 16)
		}
	case "F64":
		if len(raw) != elements*8 {
			return nil, errF64PayloadMismatch
		}
		// Reinterpret-cast raw to []float64 in place, then downcast each
		// element to float32. float64 storage is little-endian on both
		// supported architectures (arm64 + amd64) so this is bit-exact
		// vs binary.LittleEndian.Uint64+Float64frombits, but skips both
		// the per-iter raw[i*8:] re-slice bounds check and the
		// Uint64+Float64frombits dance — the compiler emits a direct
		// LDR + FCVT pair on arm64.
		src64 := unsafe.Slice((*float64)(unsafe.Pointer(unsafe.SliceData(raw))), elements)
		for i, v := range src64 {
			values[i] = float32(v)
		}
	default:
		return nil, core.NewError("unsupported dense safetensors dtype: " + dtype)
	}
	return values, nil
}
