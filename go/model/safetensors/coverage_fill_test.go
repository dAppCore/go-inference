// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"context"
	"testing"
	"time"

	core "dappco.re/go"
)

// errOnceContext is a context.Context whose Err() returns nil on its first
// call and a fixed error on every call after. It lets a test slip past a
// function's entry-guard ctx.Err() check and then trip the in-loop ctx.Err()
// check on the next call — exercising the in-loop cancellation return that a
// plain pre-cancelled context cannot reach (the entry guard catches that one
// first). The function under test only consumes ctx through the Err() seam.
type errOnceContext struct{ calls int }

func (c *errOnceContext) Deadline() (time.Time, bool) { return time.Time{}, false }
func (c *errOnceContext) Done() <-chan struct{}       { return nil }
func (c *errOnceContext) Value(any) any               { return nil }
func (c *errOnceContext) Err() error {
	c.calls++
	if c.calls <= 1 {
		return nil
	}
	return context.Canceled
}

// This file lifts statement coverage on the error / boundary branches that
// the Good/Bad/Ugly suites in safetensors_test.go and write_test.go leave
// open. Each test drives one real fault through a genuine seam:
//
//   - a negative DataStart makes *os.File.ReadAt return a non-EOF error
//     ("negative offset"), exercising the read-error short-circuits;
//   - a destination file opened O_RDONLY makes Write return EBADF, driving
//     the write-failure paths;
//   - a path whose parent is a regular file makes MkdirAll fail, and a path
//     that is itself a directory makes the create OpenFile fail;
//   - asymmetric and empty leading shards drive IndexFiles' merge-grow and
//     Names-reuse branches; bad shard paths drive its read-error returns.
//
// No production code is changed and no fault injection beyond what a real
// *os.File / filesystem already produces is used.

// negOffsetRef returns a ref over a real on-disk payload but with DataStart
// forced negative, so the next ReadAt against it fails with a non-EOF
// "negative offset" error rather than reaching EOF.
func negOffsetRef(t *testing.T, dir string) TensorRef {
	t.Helper()
	path := core.PathJoin(dir, "neg.safetensors")
	writeF32Safetensors(t, path, map[string][]float32{"weight": {1, 2, 3, 4}})
	index, err := ReadIndex(path)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["weight"]
	ref.DataStart = -1 // any ReadAt now errors with "negative offset"
	return ref
}

// roDestFile creates a destination file and re-opens it read-only, so any
// Write against the returned handle fails with EBADF. The handle is closed
// via t.Cleanup.
func roDestFile(t *testing.T, dir, name string) *core.OSFile {
	t.Helper()
	path := core.PathJoin(dir, name)
	if r := core.WriteFile(path, []byte{}, 0o644); !r.OK {
		t.Fatalf("WriteFile(seed dst): %v", r.Value)
	}
	opened := core.OpenFile(path, core.O_RDONLY, 0o644)
	if !opened.OK {
		t.Fatalf("OpenFile(O_RDONLY): %v", opened.Value)
	}
	f := opened.Value.(*core.OSFile)
	t.Cleanup(func() { _ = f.Close() })
	return f
}

// --- ReadRefValues: ReadAt non-EOF error path (safetensors.go ~L217) ---

func TestSafetensors_ReadRefValues_ReadError(t *testing.T) {
	ref := negOffsetRef(t, t.TempDir())
	if _, err := ReadRefValues(ref); err == nil {
		t.Fatal("ReadRefValues(negative DataStart) error = nil")
	}
}

// --- ReadRefRaw: open error + ReadAt non-EOF error (safetensors.go L443, L451) ---

func TestSafetensors_ReadRefRaw_OpenError(t *testing.T) {
	// Valid (non-negative) ByteLen so the length guard passes, but the
	// path does not exist → core.Open fails → resultError branch.
	ref := TensorRef{
		Name:    "x",
		Path:    core.PathJoin(t.TempDir(), "absent.safetensors"),
		ByteLen: 4,
	}
	if _, err := ReadRefRaw(ref); err == nil {
		t.Fatal("ReadRefRaw(missing file) error = nil")
	}
}

func TestSafetensors_ReadRefRaw_ReadError(t *testing.T) {
	ref := negOffsetRef(t, t.TempDir())
	if _, err := ReadRefRaw(ref); err == nil {
		t.Fatal("ReadRefRaw(negative DataStart) error = nil")
	}
}

// --- ReadFloat32Chunk: ReadAt non-EOF error path (safetensors.go ~L339) ---

func TestSafetensors_TensorReader_ReadFloat32Chunk_ReadError(t *testing.T) {
	dir := t.TempDir()
	ref := negOffsetRef(t, dir)
	reader, err := OpenReader(ref)
	if err != nil {
		t.Fatalf("OpenReader: %v", err)
	}
	defer reader.Close()
	// offset+count within Elements (4) so the bounds check passes and the
	// failing ReadAt at the negative DataStart is reached.
	if _, err := reader.ReadFloat32Chunk(0, 4); err == nil {
		t.Fatal("ReadFloat32Chunk(negative DataStart) error = nil")
	}
}

// --- readFloat32ChunkInto: ReadAt non-EOF error + decode error
// (safetensors.go ~L383 and ~L390) ---

func TestSafetensors_TensorReader_ReadFloat32ChunkInto_ReadError(t *testing.T) {
	dir := t.TempDir()
	ref := negOffsetRef(t, dir)
	reader, err := OpenReader(ref)
	if err != nil {
		t.Fatalf("OpenReader: %v", err)
	}
	defer reader.Close()
	if _, _, _, err := reader.ReadFloat32ChunkInto(0, 4, nil, nil); err == nil {
		t.Fatal("ReadFloat32ChunkInto(negative DataStart) error = nil")
	}
}

func TestSafetensors_TensorReader_ReadFloat32ChunkInto_DecodeError(t *testing.T) {
	// The decode-error return in readFloat32ChunkInto is unreachable through
	// the public constructors: OpenReader / NewFileReader set bytesPerElement
	// from DTypeByteSize, so the raw read length (count*bytesPerElement)
	// always equals the decoder's expected payload length (count*stride) for
	// every one of the four supported dtypes — a successful read therefore
	// never yields a decode mismatch. decodeFloatDataInto's own mismatch and
	// unsupported-dtype branches are covered directly by
	// TestSafetensors_DecodeFloatData_Bad in safetensors_test.go.
	t.Skip("decodeFloatDataInto error from readFloat32ChunkInto is structurally unreachable")
}

// --- WriteRefFloat32Chunks: OpenReader error + read error + write error
// (safetensors.go ~L228, ~L251, ~L255) ---

func TestSafetensors_WriteRefFloat32Chunks_OpenError(t *testing.T) {
	// Ref points at a missing file → OpenReader's core.Open fails before any
	// chunk loop runs.
	dir := t.TempDir()
	ref := TensorRef{
		Name:     "x",
		Path:     core.PathJoin(dir, "absent.safetensors"),
		DType:    "F32",
		Elements: 4,
		ByteLen:  16,
	}
	out := roDestFile(t, dir, "out.f32") // never written to on this path
	if err := WriteRefFloat32Chunks(context.Background(), out, ref, 2); err == nil {
		t.Fatal("WriteRefFloat32Chunks(missing source) error = nil")
	}
}

func TestSafetensors_WriteRefFloat32Chunks_ReadError(t *testing.T) {
	// Source ref reads fine through OpenReader but its negative DataStart
	// makes the per-chunk ReadAt fail inside readFloat32ChunkInto.
	dir := t.TempDir()
	ref := negOffsetRef(t, dir)
	out := roDestFile(t, dir, "out.f32")
	if err := WriteRefFloat32Chunks(context.Background(), out, ref, 2); err == nil {
		t.Fatal("WriteRefFloat32Chunks(read error) error = nil")
	}
}

func TestSafetensors_WriteRefFloat32Chunks_WriteError(t *testing.T) {
	// Source reads cleanly; the destination is read-only so the first
	// writeFloat32ValuesScratch flush fails with EBADF.
	dir := t.TempDir()
	src := core.PathJoin(dir, "src.safetensors")
	writeF32Safetensors(t, src, map[string][]float32{"weight": {1, 2, 3, 4}})
	index, err := ReadIndex(src)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	out := roDestFile(t, dir, "out.f32")
	if err := WriteRefFloat32Chunks(context.Background(), out, index.Tensors["weight"], 2); err == nil {
		t.Fatal("WriteRefFloat32Chunks(read-only dest) error = nil")
	}
}

// --- ReadRefFloat32Chunk: OpenReader error (safetensors.go ~L264) ---

func TestSafetensors_ReadRefFloat32Chunk_OpenError(t *testing.T) {
	ref := TensorRef{
		Name:     "x",
		Path:     core.PathJoin(t.TempDir(), "absent.safetensors"),
		DType:    "F32",
		Elements: 4,
		ByteLen:  16,
	}
	if _, err := ReadRefFloat32Chunk(ref, 0, 4); err == nil {
		t.Fatal("ReadRefFloat32Chunk(missing file) error = nil")
	}
}

// --- IndexFiles: shard-0 read error, later-shard read error, empty-leading
// shard Names reuse, and asymmetric merge grow (safetensors.go L70, L87, L92, L95) ---

func TestSafetensors_IndexFiles_FirstShardError(t *testing.T) {
	// Two paths so the single-shard fast path is skipped; the first does not
	// exist → ReadIndex(paths[0]) errors.
	dir := t.TempDir()
	missing := core.PathJoin(dir, "missing-0.safetensors")
	p2 := core.PathJoin(dir, "shard-1.safetensors")
	writeRawSafetensors(t, p2, map[string][]byte{"a": {1}})
	if _, err := IndexFiles([]string{missing, p2}); err == nil {
		t.Fatal("IndexFiles(missing first shard) error = nil")
	}
}

func TestSafetensors_IndexFiles_LaterShardError(t *testing.T) {
	// First shard is valid; a later shard path is missing → the in-loop
	// ReadIndex error return fires.
	dir := t.TempDir()
	p1 := core.PathJoin(dir, "shard-0.safetensors")
	missing := core.PathJoin(dir, "missing-1.safetensors")
	writeRawSafetensors(t, p1, map[string][]byte{"a": {1}})
	if _, err := IndexFiles([]string{p1, missing}); err == nil {
		t.Fatal("IndexFiles(missing later shard) error = nil")
	}
}

func TestSafetensors_IndexFiles_MergeBranches(t *testing.T) {
	t.Run("empty_leading_shard", func(t *testing.T) {
		// An empty ("{}") first shard yields zero Names, so estTotal is 0 and
		// the Names-reuse else branch (cap(first.Names) >= estTotal) is taken
		// instead of the grow branch.
		dir := t.TempDir()
		empty := core.PathJoin(dir, "empty.safetensors")
		real := core.PathJoin(dir, "real.safetensors")
		writeRawSafetensors(t, empty, map[string][]byte{})
		writeRawSafetensors(t, real, map[string][]byte{"a": {1}, "b": {2}})
		index, err := IndexFiles([]string{empty, real})
		if err != nil {
			t.Fatalf("IndexFiles(empty+real): %v", err)
		}
		if len(index.Names) != 2 {
			t.Fatalf("Names = %v, want [a b]", index.Names)
		}
	})

	t.Run("asymmetric_grows_in_loop", func(t *testing.T) {
		// A tiny leading shard (1 tensor) followed by a much larger one
		// (5 tensors) overflows the merged Names capacity mid-loop, driving
		// the in-loop grow branch.
		dir := t.TempDir()
		p1 := core.PathJoin(dir, "small.safetensors")
		p2 := core.PathJoin(dir, "big.safetensors")
		writeRawSafetensors(t, p1, map[string][]byte{"a": {1}})
		writeRawSafetensors(t, p2, map[string][]byte{
			"b": {2}, "c": {3}, "d": {4}, "e": {5}, "f": {6},
		})
		index, err := IndexFiles([]string{p1, p2})
		if err != nil {
			t.Fatalf("IndexFiles(asymmetric): %v", err)
		}
		if len(index.Names) != 6 {
			t.Fatalf("Names = %v, want 6 tensors", index.Names)
		}
	})
}

// --- WriteSubset: nil ctx, mkdir failure, create failure, header-write
// failure, in-loop cancellation, payload-copy failure (write.go L29, L48, L52,
// L60/L63, L74, L78) ---

func TestWrite_WriteSubset_NilContext(t *testing.T) {
	// A nil ctx must be tolerated (replaced by context.Background) and the
	// write must still succeed.
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	target := core.PathJoin(dir, "out.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{"x": {1, 2, 3, 4}})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	//nolint:staticcheck // intentionally passing a nil context to exercise the guard.
	if err := WriteSubset(nil, target, []TensorRef{index.Tensors["x"]}); err != nil {
		t.Fatalf("WriteSubset(nil ctx): %v", err)
	}
	if _, err := ReadIndex(target); err != nil {
		t.Fatalf("ReadIndex(target): %v", err)
	}
}

func TestWrite_WriteSubset_MkdirError(t *testing.T) {
	// Make the target's parent a regular file: MkdirAll(parent) then fails
	// with "not a directory".
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{"x": {1, 2, 3, 4}})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	// "blocker" is a file; target sits "under" it, so MkdirAll(blocker/sub)
	// cannot create the directory.
	blocker := core.PathJoin(dir, "blocker")
	if r := core.WriteFile(blocker, []byte{0}, 0o644); !r.OK {
		t.Fatalf("WriteFile(blocker): %v", r.Value)
	}
	target := core.PathJoin(blocker, "sub", "out.safetensors")
	if err := WriteSubset(context.Background(), target, []TensorRef{index.Tensors["x"]}); err == nil {
		t.Fatal("WriteSubset(mkdir under file) error = nil")
	}
}

func TestWrite_WriteSubset_CreateError(t *testing.T) {
	// Point the target at an existing directory: the create OpenFile fails
	// with "is a directory" while MkdirAll(parent) succeeds.
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{"x": {1, 2, 3, 4}})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	targetDir := core.PathJoin(dir, "iam-a-dir")
	if r := core.MkdirAll(targetDir, 0o755); !r.OK {
		t.Fatalf("MkdirAll(targetDir): %v", r.Value)
	}
	if err := WriteSubset(context.Background(), targetDir, []TensorRef{index.Tensors["x"]}); err == nil {
		t.Fatal("WriteSubset(target is a directory) error = nil")
	}
}

func TestWrite_WriteSubset_PayloadCopyError(t *testing.T) {
	// The header writes fine, but the source ref's negative DataStart makes
	// the per-ref chunked payload copy fail on its first ReadAt.
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	target := core.PathJoin(dir, "out.safetensors")
	writeF32Safetensors(t, source, map[string][]float32{"x": {1, 2, 3, 4}})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["x"]
	ref.DataStart = -1 // genuine non-EOF ReadAt failure during the copy
	if err := WriteSubset(context.Background(), target, []TensorRef{ref}); err == nil {
		t.Fatal("WriteSubset(payload copy error) error = nil")
	}
}

// --- writeRefRawChunksScratch: source open error + truncated payload
// (write.go L255, L278) ---

func TestSafetensors_writeRefRawChunksScratch_OpenError(t *testing.T) {
	// ctx not cancelled; source path missing → in.Open fails.
	dir := t.TempDir()
	out := roDestFile(t, dir, "out.bin")
	ref := TensorRef{
		Name:    "x",
		Path:    core.PathJoin(dir, "absent.safetensors"),
		ByteLen: 4,
	}
	if _, err := writeRefRawChunksScratch(context.Background(), out, ref, defaultRawChunkBytes, nil); err == nil {
		t.Fatal("writeRefRawChunksScratch(missing source) error = nil")
	}
}

func TestSafetensors_writeRefRawChunksScratch_Truncated(t *testing.T) {
	// The source file is real but the ref claims more bytes than it holds, so
	// the chunk ReadAt returns a short count and the truncated-payload guard
	// fires. Write to a real (writable) destination so the failure is the
	// read truncation, not a write error.
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{"x": {1, 2, 3, 4}})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ref := index.Tensors["x"]
	ref.ByteLen += 4096 // demand well past EOF → short read → truncated

	dstPath := core.PathJoin(dir, "out.bin")
	created := core.OpenFile(dstPath, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		t.Fatalf("OpenFile(dst): %v", created.Value)
	}
	out := created.Value.(*core.OSFile)
	defer out.Close()
	if _, err := writeRefRawChunksScratch(context.Background(), out, ref, defaultRawChunkBytes, nil); err == nil {
		t.Fatal("writeRefRawChunksScratch(truncated source) error = nil")
	}
}

func TestSafetensors_writeRefRawChunksScratch_WriteError(t *testing.T) {
	// Source reads cleanly; the destination is read-only so writeAll fails.
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{"x": {1, 2, 3, 4}})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	out := roDestFile(t, dir, "out.bin")
	if _, err := writeRefRawChunksScratch(context.Background(), out, index.Tensors["x"], defaultRawChunkBytes, nil); err == nil {
		t.Fatal("writeRefRawChunksScratch(read-only dest) error = nil")
	}
}

// --- writeAll: the no-progress guard and the Write-error return
// (write.go L293, L296) ---

func TestSafetensors_writeAll_WriteError(t *testing.T) {
	dir := t.TempDir()
	out := roDestFile(t, dir, "ro.bin")
	if err := writeAll(out, []byte{1, 2, 3}); err == nil {
		t.Fatal("writeAll(read-only file) error = nil")
	}
}

// --- countTensorsAndDims: truncated-mid-object fallback (header_parse.go L889) ---

func TestSafetensors_countTensorsAndDims_TruncatedObject(t *testing.T) {
	// A header that opens an object then ends inside it (just whitespace, no
	// closing brace) makes the pre-scan run off the end and return its
	// sentinel; ParseHeaderRefs then falls back to a zero-sized estimate and
	// surfaces the structural error from the live walk.
	header := []byte("{   ")
	if _, err := ParseHeaderRefs("synthetic", header, int64(8+len(header))); err == nil {
		t.Fatal("ParseHeaderRefs(truncated object header) error = nil")
	}
}

// --- WriteSubset: in-loop context cancellation (write.go L74) ---

func TestWrite_WriteSubset_CancelInLoop(t *testing.T) {
	// errOnceContext passes the entry-guard Err() check (call 1 → nil) and
	// then fails the per-ref loop Err() check (call 2 → Canceled), so the
	// in-loop cancellation return fires after the header has been written.
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	target := core.PathJoin(dir, "out.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{"x": {1, 2, 3, 4}})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	ctx := &errOnceContext{}
	if err := WriteSubset(ctx, target, []TensorRef{index.Tensors["x"]}); err == nil {
		t.Fatal("WriteSubset(cancel in loop) error = nil")
	}
	if ctx.calls < 2 {
		t.Fatalf("Err() called %d times, want the loop check to run", ctx.calls)
	}
}

// --- writeRefRawChunksScratch: chunkBytes<=0 default + in-loop cancellation
// (write.go L251, L270) ---

func TestSafetensors_writeRefRawChunksScratch_DefaultChunkAndCancel(t *testing.T) {
	dir := t.TempDir()
	source := core.PathJoin(dir, "source.safetensors")
	writeRawSafetensors(t, source, map[string][]byte{"x": {1, 2, 3, 4}})
	index, err := ReadIndex(source)
	if err != nil {
		t.Fatalf("ReadIndex: %v", err)
	}
	out := roDestFile(t, dir, "out.bin")
	// chunkBytes <= 0 selects the defaultRawChunkBytes fallback; a
	// pre-cancelled context then trips the first per-chunk Err() check before
	// any read/write occurs (so the read-only dest is never reached).
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := writeRefRawChunksScratch(ctx, out, index.Tensors["x"], 0, nil); err == nil {
		t.Fatal("writeRefRawChunksScratch(cancelled, default chunk) error = nil")
	}
}

// --- subsetHeaderEncoded: invalid byte length, duplicate name, oversized
// shape (write.go L106, L109, L146) ---

func TestSafetensors_subsetHeaderEncoded_Errors(t *testing.T) {
	t.Run("negative_byte_length", func(t *testing.T) {
		_, _, err := subsetHeaderEncoded([]TensorRef{
			{Name: "x", DType: "F32", Shape: []uint64{1}, ByteLen: -1},
		})
		if err == nil {
			t.Fatal("subsetHeaderEncoded(negative ByteLen) error = nil")
		}
	})
	t.Run("duplicate_name", func(t *testing.T) {
		_, _, err := subsetHeaderEncoded([]TensorRef{
			{Name: "dup", DType: "F32", Shape: []uint64{1}, ByteLen: 4},
			{Name: "dup", DType: "F32", Shape: []uint64{1}, ByteLen: 4},
		})
		if err == nil {
			t.Fatal("subsetHeaderEncoded(duplicate name) error = nil")
		}
	})
	t.Run("shape_dim_too_large", func(t *testing.T) {
		// A shape dim above math.MaxInt64 (as a uint64) trips the
		// overflow guard inside the dim-emit loop.
		_, _, err := subsetHeaderEncoded([]TensorRef{
			{Name: "x", DType: "F32", Shape: []uint64{uint64(maxInt64Value()) + 1}, ByteLen: 4},
		})
		if err == nil {
			t.Fatal("subsetHeaderEncoded(oversized shape dim) error = nil")
		}
	})
}

// --- appendJSONString + hexNibble: the remaining escape arms and the
// high-nibble hex digit (write.go L187-196, L214) ---

func TestSafetensors_appendJSONString_AllEscapes(t *testing.T) {
	// Drive every escape branch in one string: backspace, form-feed,
	// carriage-return and tab take their dedicated \X forms, while 0x0f (low
	// nibble 15 → 'f', exercising hexNibble's >=10 arm) and 0x1f (0x1 high,
	// 0xf low) take the default \u00XX form. The double-quote and backslash
	// fast escapes are included too.
	in := "a\bb\fc\rd\te\x0ff\x1f\"g\\h"
	got := string(appendJSONString(nil, in))
	// 0x0f and 0x1f have no dedicated escape so they take the default
	// \u00XX form; the nibble 'f' (15) exercises hexNibble's >=10 arm.
	want := "\"a\\bb\\fc\\rd\\te\\u000ff\\u001f\\\"g\\\\h\""
	if got != want {
		t.Fatalf("appendJSONString:\n got=%s\nwant=%s", got, want)
	}
}

// --- appendJSONInt64: the negative-value branch (write.go L229, L239) ---

func TestSafetensors_appendJSONInt64_Negative(t *testing.T) {
	// subsetHeaderEncoded only ever emits non-negative offsets/dims, so the
	// signed branch is exercised directly here.
	cases := []struct {
		v    int64
		want string
	}{
		{0, "0"},
		{-1, "-1"},
		{-9223372036854775808, "-9223372036854775808"}, // math.MinInt64
		{12345, "12345"},
	}
	for _, tc := range cases {
		if got := string(appendJSONInt64(nil, tc.v)); got != tc.want {
			t.Errorf("appendJSONInt64(%d) = %q, want %q", tc.v, got, tc.want)
		}
	}
}

// --- minInt64: the b<=a branch (write.go L307) ---

func TestSafetensors_minInt64_Branches(t *testing.T) {
	if got := minInt64(2, 5); got != 2 { // a < b
		t.Errorf("minInt64(2,5) = %d, want 2", got)
	}
	if got := minInt64(5, 2); got != 2 { // a >= b → the else return
		t.Errorf("minInt64(5,2) = %d, want 2", got)
	}
	if got := minInt64(4, 4); got != 4 {
		t.Errorf("minInt64(4,4) = %d, want 4", got)
	}
}
