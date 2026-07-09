// SPDX-Licence-Identifier: EUPL-1.2

package safetensors_test

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"

	core "dappco.re/go"
	"dappco.re/go/inference/model/safetensors"
)

// exampleFs is the filesystem handle the examples use for temp dirs.
// Examples cannot take a *testing.T, so they cannot call t.TempDir();
// core's Fs.TempDir is the established example idiom (see core's own
// ExampleData_New).
var exampleFs = (&core.Fs{}).New("/")

// mkTempDir returns a fresh temp directory for an example fixture. The
// second return is the cleanup closure callers defer.
func mkTempDir() (string, func()) {
	dir := exampleFs.TempDir("safetensors-example").Value.(string)
	return dir, func() { exampleFs.DeleteAll(dir) }
}

// buildSafetensors assembles a complete safetensors file on disk from a
// raw JSON header string and a tensor payload blob. It is the in-situ
// fixture builder the round-trip examples below use: the 8-byte
// little-endian header length, then the header JSON, then the payload —
// exactly the on-disk layout ReadIndex expects. Hand-rolling the header
// (rather than marshalling a map[string]HeaderEntry) lets the examples
// exercise headers a real writer emits but the test marshaller cannot —
// a __metadata__ block and the full dtype vocabulary.
func buildSafetensors(path, header string, payload []byte) error {
	out := make([]byte, 8+len(header)+len(payload))
	binary.LittleEndian.PutUint64(out[:8], uint64(len(header)))
	copy(out[8:], header)
	copy(out[8+len(header):], payload)
	if result := core.WriteFile(path, out, 0o644); !result.OK {
		if err, ok := result.Value.(error); ok {
			return err
		}
		return core.NewError("write failed")
	}
	return nil
}

// ExampleDTypeByteSize shows how to resolve the on-disk byte width of a
// supported dense safetensors dtype. The dtype string is matched
// case-insensitively, so a header carrying "bf16" resolves the same as
// the canonical "BF16".
func ExampleDTypeByteSize() {
	for _, dtype := range []string{"F16", "BF16", "F32", "F64", "bf16"} {
		size, err := safetensors.DTypeByteSize(dtype)
		if err != nil {
			fmt.Printf("%s: %v\n", dtype, err)
			continue
		}
		fmt.Printf("%s -> %d bytes/element\n", dtype, size)
	}
	// Output:
	// F16 -> 2 bytes/element
	// BF16 -> 2 bytes/element
	// F32 -> 4 bytes/element
	// F64 -> 8 bytes/element
	// bf16 -> 2 bytes/element
}

// ExampleDecodeFloatData decodes a little-endian F32 payload (the raw
// bytes as they appear in a safetensors tensor blob) into a []float32.
// The element count is supplied by the caller from the tensor's shape;
// DecodeFloatData validates that the payload length matches.
func ExampleDecodeFloatData() {
	// Three F32 values: 1.0, 2.5, -3.0 laid out little-endian.
	raw := make([]byte, 3*4)
	binary.LittleEndian.PutUint32(raw[0:], math.Float32bits(1.0))
	binary.LittleEndian.PutUint32(raw[4:], math.Float32bits(2.5))
	binary.LittleEndian.PutUint32(raw[8:], math.Float32bits(-3.0))

	values, err := safetensors.DecodeFloatData("F32", raw, 3)
	if err != nil {
		fmt.Println("decode:", err)
		return
	}
	fmt.Println(values)
	// Output:
	// [1 2.5 -3]
}

// ExampleFloat16ToFloat32 converts a single IEEE-754 half-precision bit
// pattern to float32. 0x3c00 is the half-precision encoding of 1.0 and
// 0xc000 is -2.0.
func ExampleFloat16ToFloat32() {
	fmt.Println(safetensors.Float16ToFloat32(0x3c00))
	fmt.Println(safetensors.Float16ToFloat32(0xc000))
	// Output:
	// 1
	// -2
}

// ExampleRefFromHeader builds a TensorRef from a parsed header entry. The
// returned ref carries the absolute byte range of the tensor payload in
// the source file (DataStart is dataStart + the entry's begin offset) and
// the element count derived from the shape.
func ExampleRefFromHeader() {
	entry := safetensors.HeaderEntry{
		DType:       "F32",
		Shape:       []int64{2, 3},
		DataOffsets: []int64{0, 24},
	}
	ref, err := safetensors.RefFromHeader("model.safetensors", "weight", entry, 128)
	if err != nil {
		fmt.Println("ref:", err)
		return
	}
	fmt.Printf("name=%s dtype=%s elements=%d start=%d len=%d\n",
		ref.Name, ref.DType, ref.Elements, ref.DataStart, ref.ByteLen)
	// Output:
	// name=weight dtype=F32 elements=6 start=128 len=24
}

// Example_writeReadRoundTrip is the end-to-end read/write usage: build an
// index over a source file, write a subset of its tensors to a fresh
// safetensors file with WriteSubset, then read that file back and confirm
// the float values survive bit-exact. This is how the merge / shard tools
// stream tensors between files without loading a whole model — the values
// are copied through bounded chunks, never fully materialised.
func Example_writeReadRoundTrip() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	src := core.PathJoin(dir, "src.safetensors")
	dst := core.PathJoin(dir, "subset.safetensors")

	// Source file: two F32 tensors laid out little-endian, header sorted
	// by name (alpha < beta) exactly as a real writer emits.
	alpha := []float32{1, 2, 3}
	beta := []float32{-4.5, 1024.25}
	payload := make([]byte, 0, (len(alpha)+len(beta))*4)
	for _, v := range alpha {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	for _, v := range beta {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	header := `{"alpha":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},` +
		`"beta":{"dtype":"F32","shape":[2],"data_offsets":[12,20]}}`
	if err := buildSafetensors(src, header, payload); err != nil {
		fmt.Println("build:", err)
		return
	}

	index, err := safetensors.ReadIndex(src)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}

	// Write just "alpha" to a new file, then read it straight back.
	if err := safetensors.WriteSubset(context.Background(), dst, []safetensors.TensorRef{index.Tensors["alpha"]}); err != nil {
		fmt.Println("write subset:", err)
		return
	}
	back, err := safetensors.ReadIndex(dst)
	if err != nil {
		fmt.Println("read back:", err)
		return
	}
	values, err := safetensors.ReadRefValues(back.Tensors["alpha"])
	if err != nil {
		fmt.Println("read values:", err)
		return
	}
	fmt.Printf("subset tensors: %v\n", back.Names)
	fmt.Printf("alpha values: %v\n", values)
	// Output:
	// subset tensors: [alpha]
	// alpha values: [1 2 3]
}

// Example_readChunked shows reading a tensor in element-range chunks via a
// TensorReader, the pattern the chunked-write and pack-compare paths use to
// bound peak memory. OpenReader binds the ref to its file; each
// ReadFloat32Chunk decodes only the requested element window.
func Example_readChunked() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	path := core.PathJoin(dir, "chunked.safetensors")

	want := []float32{10, 11, 12, 13, 14, 15}
	payload := make([]byte, 0, len(want)*4)
	for _, v := range want {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	header := fmt.Sprintf(`{"vec":{"dtype":"F32","shape":[%d],"data_offsets":[0,%d]}}`, len(want), len(payload))
	if err := buildSafetensors(path, header, payload); err != nil {
		fmt.Println("build:", err)
		return
	}

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	reader, err := safetensors.OpenReader(index.Tensors["vec"])
	if err != nil {
		fmt.Println("open reader:", err)
		return
	}
	defer reader.Close()

	// Read elements [2,5) — a middle window, not the whole tensor.
	chunk, err := reader.ReadFloat32Chunk(2, 3)
	if err != nil {
		fmt.Println("read chunk:", err)
		return
	}
	fmt.Printf("elements[2:5] = %v\n", chunk)
	// Output:
	// elements[2:5] = [12 13 14]
}

// ExampleReadIndex_metadata shows that the __metadata__ entry safetensors
// writers prepend (a free-form JSON object of housekeeping fields) is
// indexed past and dropped — only real tensors appear in the returned
// Index. The metadata object here deliberately carries the full spread of
// JSON value kinds (string, array, bool, null, nested object) that the
// header walker must skip over to reach the tensor that follows it.
func ExampleReadIndex_metadata() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	path := core.PathJoin(dir, "with_meta.safetensors")

	// One F32 scalar payload (value 1.0) preceded by a rich metadata block.
	payload := binary.LittleEndian.AppendUint32(nil, math.Float32bits(1.0))
	header := `{"__metadata__":{` +
		`"format":"pt",` + // string value
		`"shape_hint":[1,2,3],` + // array value (skipArray)
		`"trained":true,` + // bool literal (skipLiteral)
		`"notes":null,` + // null literal (skipLiteral)
		`"extra":{"nested":"object","deep":[false]}` + // nested object (skipObject)
		`},"scalar":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`
	if err := buildSafetensors(path, header, payload); err != nil {
		fmt.Println("build:", err)
		return
	}

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	// Print via the sorted Names slice — ranging the Tensors map directly
	// would make the output order non-deterministic.
	fmt.Printf("tensors: %v\n", index.Names)
	fmt.Printf("scalar dtype: %s\n", index.Tensors["scalar"].DType)
	// Output:
	// tensors: [scalar]
	// scalar dtype: F32
}

// ExampleReadIndex_dtypes shows that ReadIndex catalogues a tensor of any
// declared dtype — indexing reads the header only, so dtypes go-mlx cannot
// yet decode (the integer, boolean and 8-bit-float families) still produce
// valid refs. The header carries the dtype string case the parser canon-
// icalises (lowercase "f32" → "F32"), demonstrating the case-insensitive
// dtype interning real-world headers from older writers rely on.
func ExampleReadIndex_dtypes() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	path := core.PathJoin(dir, "dtypes.safetensors")

	// Each tensor is a single element; byte widths differ by dtype so the
	// payload offsets advance accordingly. Names are pre-sorted.
	type entry struct {
		name  string
		dtype string
		bytes int
	}
	entries := []entry{
		{"a_bool", "BOOL", 1},
		{"b_f8e4", "F8_E4M3FN", 1},
		{"c_f8e5", "F8_E5M2", 1},
		{"d_i16", "I16", 2},
		{"e_i8", "I8", 1},
		{"f_lower", "f32", 4}, // lowercase — interned to F32
		{"g_u8", "U8", 1},
	}
	header := "{"
	payload := []byte{}
	offset := 0
	for i, e := range entries {
		if i > 0 {
			header += ","
		}
		header += fmt.Sprintf(`%q:{"dtype":%q,"shape":[1],"data_offsets":[%d,%d]}`,
			e.name, e.dtype, offset, offset+e.bytes)
		payload = append(payload, make([]byte, e.bytes)...)
		offset += e.bytes
	}
	header += "}"
	if err := buildSafetensors(path, header, payload); err != nil {
		fmt.Println("build:", err)
		return
	}

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	for _, name := range index.Names {
		fmt.Printf("%s -> %s\n", name, index.Tensors[name].DType)
	}
	// Output:
	// a_bool -> BOOL
	// b_f8e4 -> F8_E4M3FN
	// c_f8e5 -> F8_E5M2
	// d_i16 -> I16
	// e_i8 -> I8
	// f_lower -> F32
	// g_u8 -> U8
}

// twoTensorFile writes a fixture with two F32 tensors ("alpha"=[1 2 3],
// "beta"=[4 5]) and returns its path plus a cleanup closure. Several
// examples below share this exact layout so their Output blocks line up.
func twoTensorFile() (path string, cleanup func(), err error) {
	dir, cleanup := mkTempDir()
	path = core.PathJoin(dir, "two.safetensors")
	payload := make([]byte, 0, 5*4)
	for _, v := range []float32{1, 2, 3, 4, 5} {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	header := `{"alpha":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},` +
		`"beta":{"dtype":"F32","shape":[2],"data_offsets":[12,20]}}`
	if err = buildSafetensors(path, header, payload); err != nil {
		cleanup()
		return "", func() {}, err
	}
	return path, cleanup, nil
}

// ExampleIndexFiles merges the headers of several safetensors shards into a
// single Index whose Names are sorted across all shards and whose tensors
// each remember which shard file they came from. Only the headers are read
// — no tensor payload is loaded.
func ExampleIndexFiles() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	p1 := core.PathJoin(dir, "shard-1.safetensors")
	p2 := core.PathJoin(dir, "shard-2.safetensors")

	// shard-1 holds "wte"; shard-2 holds "lm_head". Each is one F32 scalar.
	one := binary.LittleEndian.AppendUint32(nil, math.Float32bits(1))
	if err := buildSafetensors(p1, `{"wte":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, one); err != nil {
		fmt.Println("build p1:", err)
		return
	}
	if err := buildSafetensors(p2, `{"lm_head":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}`, one); err != nil {
		fmt.Println("build p2:", err)
		return
	}

	index, err := safetensors.IndexFiles([]string{p1, p2})
	if err != nil {
		fmt.Println("index files:", err)
		return
	}
	fmt.Printf("tensors: %v\n", index.Names)
	fmt.Printf("lm_head from shard-2: %t\n", index.Tensors["lm_head"].Path == p2)
	// Output:
	// tensors: [lm_head wte]
	// lm_head from shard-2: true
}

// ExampleParseHeaderRefs walks an in-memory safetensors header blob into an
// Index without opening a file — the entry point a caller uses once it has
// already read and length-validated the header bytes. dataStart is the
// absolute file offset where tensor payloads begin (8 + header length).
func ExampleParseHeaderRefs() {
	header := []byte(`{"weight":{"dtype":"bf16","shape":[2,3],"data_offsets":[0,12]}}`)
	dataStart := int64(8 + len(header))
	index, err := safetensors.ParseHeaderRefs("model.safetensors", header, dataStart)
	if err != nil {
		fmt.Println("parse:", err)
		return
	}
	ref := index.Tensors["weight"]
	// dtype lowercase "bf16" canonicalises to BF16; DataStart = dataStart+0.
	fmt.Printf("names=%v dtype=%s elements=%d start=%d\n",
		index.Names, ref.DType, ref.Elements, ref.DataStart)
	// Output:
	// names=[weight] dtype=BF16 elements=6 start=71
}

// ExampleReadRefValues reads a tensor's full payload off disk and decodes it
// to []float32 in one call — the convenience wrapper for tensors small
// enough to materialise whole. The element count comes from the indexed
// ref's shape.
func ExampleReadRefValues() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	values, err := safetensors.ReadRefValues(index.Tensors["alpha"])
	if err != nil {
		fmt.Println("read values:", err)
		return
	}
	fmt.Println(values)
	// Output:
	// [1 2 3]
}

// ExampleReadRefRaw reads a tensor's payload as the raw on-disk bytes,
// undecoded — the path the subset writer and pack tools use to copy tensor
// bytes verbatim between files. Here the 12 raw bytes of a 3-element F32
// tensor are read back and the first float32 (little-endian) is recovered.
func ExampleReadRefRaw() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	raw, err := safetensors.ReadRefRaw(index.Tensors["alpha"])
	if err != nil {
		fmt.Println("read raw:", err)
		return
	}
	first := math.Float32frombits(binary.LittleEndian.Uint32(raw[:4]))
	fmt.Printf("raw bytes: %d, first value: %v\n", len(raw), first)
	// Output:
	// raw bytes: 12, first value: 1
}

// ExampleReadRefFloat32Chunk reads a single element window of a tensor
// without opening a reader explicitly — the one-shot convenience over
// OpenReader + ReadFloat32Chunk + Close. Here elements [1,3) of "alpha" are
// read.
func ExampleReadRefFloat32Chunk() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	chunk, err := safetensors.ReadRefFloat32Chunk(index.Tensors["alpha"], 1, 2)
	if err != nil {
		fmt.Println("read chunk:", err)
		return
	}
	fmt.Println(chunk)
	// Output:
	// [2 3]
}

// ExampleWriteRefFloat32Chunks streams a tensor's values, decoded to
// float32, into an open file as a bare little-endian float32 blob (NOT a
// safetensors container) — the export path the merge tools use to write a
// raw weight stream. chunkElements bounds peak memory; here 2 forces the
// 5-element tensor out over multiple chunks.
func ExampleWriteRefFloat32Chunks() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()
	dir := core.PathDir(path)

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	// "beta" = [4 5]; write it as a raw f32 blob in 1-element chunks.
	dst := core.PathJoin(dir, "beta.f32")
	created := core.OpenFile(dst, core.O_CREATE|core.O_WRONLY|core.O_TRUNC, 0o644)
	if !created.OK {
		fmt.Println("open dst:", created.Value)
		return
	}
	out := created.Value.(*core.OSFile)
	if err := safetensors.WriteRefFloat32Chunks(context.Background(), out, index.Tensors["beta"], 1); err != nil {
		out.Close()
		fmt.Println("write chunks:", err)
		return
	}
	out.Close()

	read := core.ReadFile(dst)
	if !read.OK {
		fmt.Println("read dst:", read.Value)
		return
	}
	raw := read.Value.([]byte)
	fmt.Printf("raw bytes: %d (no 8-byte header)\n", len(raw))
	fmt.Printf("first value: %v\n", math.Float32frombits(binary.LittleEndian.Uint32(raw[:4])))
	// Output:
	// raw bytes: 8 (no 8-byte header)
	// first value: 4
}

// ExampleOpenReader binds a tensor ref to its file and reads element
// windows on demand via the returned reader — the memory-bounded read path.
// The reader owns the opened handle and must be closed.
func ExampleOpenReader() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	reader, err := safetensors.OpenReader(index.Tensors["alpha"])
	if err != nil {
		fmt.Println("open reader:", err)
		return
	}
	defer reader.Close()
	chunk, err := reader.ReadFloat32Chunk(0, 3)
	if err != nil {
		fmt.Println("read chunk:", err)
		return
	}
	fmt.Println(chunk)
	// Output:
	// [1 2 3]
}

// ExampleOpenReaders opens a batch of readers in one call — if any ref is
// bad it closes the ones already opened and returns the error, so a
// successful return means every reader is live. Close them all with
// CloseReaders.
func ExampleOpenReaders() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	readers, err := safetensors.OpenReaders([]safetensors.TensorRef{
		index.Tensors["alpha"], index.Tensors["beta"],
	})
	if err != nil {
		fmt.Println("open readers:", err)
		return
	}
	defer safetensors.CloseReaders(readers)
	beta, err := readers[1].ReadFloat32Chunk(0, 2)
	if err != nil {
		fmt.Println("read beta:", err)
		return
	}
	fmt.Printf("opened %d readers; beta = %v\n", len(readers), beta)
	// Output:
	// opened 2 readers; beta = [4 5]
}

// ExampleCloseReaders releases every handle in a batch opened by
// OpenReaders. It is safe to call on a nil or partially-filled slice, so it
// is the natural deferred cleanup for a reader batch.
func ExampleCloseReaders() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	readers, err := safetensors.OpenReaders([]safetensors.TensorRef{index.Tensors["alpha"]})
	if err != nil {
		fmt.Println("open readers:", err)
		return
	}
	safetensors.CloseReaders(readers)
	// Safe on a nil slice too.
	safetensors.CloseReaders(nil)
	fmt.Println("closed")
	// Output:
	// closed
}

// ExampleNewFileReader binds a ref to an already-open file so many tensors
// in one shard can share a single handle — the caller opens the shard once
// and owns the handle's lifetime (the reader does not close it on the
// caller's behalf in the borrow pattern shown here).
func ExampleNewFileReader() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	opened := core.Open(path)
	if !opened.OK {
		fmt.Println("open:", opened.Value)
		return
	}
	file := opened.Value.(*core.OSFile)
	defer file.Close() // caller owns the shared handle

	reader, err := safetensors.NewFileReader(file, index.Tensors["beta"])
	if err != nil {
		fmt.Println("new file reader:", err)
		return
	}
	chunk, err := reader.ReadFloat32Chunk(0, 2)
	if err != nil {
		fmt.Println("read chunk:", err)
		return
	}
	fmt.Println(chunk)
	// Output:
	// [4 5]
}

// ExampleTensorReader_ReadFloat32Chunk reads a middle element window off a
// reader — the per-window read the chunked-write and pack-compare paths
// drive to bound peak memory.
func ExampleTensorReader_ReadFloat32Chunk() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	path := core.PathJoin(dir, "vec.safetensors")

	want := []float32{10, 11, 12, 13, 14, 15}
	payload := make([]byte, 0, len(want)*4)
	for _, v := range want {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	header := fmt.Sprintf(`{"vec":{"dtype":"F32","shape":[%d],"data_offsets":[0,%d]}}`, len(want), len(payload))
	if err := buildSafetensors(path, header, payload); err != nil {
		fmt.Println("build:", err)
		return
	}

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	reader, err := safetensors.OpenReader(index.Tensors["vec"])
	if err != nil {
		fmt.Println("open reader:", err)
		return
	}
	defer reader.Close()
	chunk, err := reader.ReadFloat32Chunk(2, 3)
	if err != nil {
		fmt.Println("read chunk:", err)
		return
	}
	fmt.Printf("elements[2:5] = %v\n", chunk)
	// Output:
	// elements[2:5] = [12 13 14]
}

// ExampleTensorReader_ReadFloat32ChunkInto reads successive windows through
// caller-owned scratch buffers, reusing them across iterations so a chunked
// loop allocates its byte and float32 backing arrays once rather than per
// chunk. The decoded values are byte-identical to ReadFloat32Chunk.
func ExampleTensorReader_ReadFloat32ChunkInto() {
	dir, cleanup := mkTempDir()
	defer cleanup()
	path := core.PathJoin(dir, "vec.safetensors")

	full := []float32{0, 1, 2, 3, 4, 5}
	payload := make([]byte, 0, len(full)*4)
	for _, v := range full {
		payload = binary.LittleEndian.AppendUint32(payload, math.Float32bits(v))
	}
	header := fmt.Sprintf(`{"vec":{"dtype":"F32","shape":[%d],"data_offsets":[0,%d]}}`, len(full), len(payload))
	if err := buildSafetensors(path, header, payload); err != nil {
		fmt.Println("build:", err)
		return
	}

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	reader, err := safetensors.OpenReader(index.Tensors["vec"])
	if err != nil {
		fmt.Println("open reader:", err)
		return
	}
	defer reader.Close()

	var rawScratch []byte
	var valScratch []float32
	var sum float32
	for offset := 0; offset < len(full); offset += 2 {
		var values []float32
		rawScratch, valScratch, values, err = reader.ReadFloat32ChunkInto(offset, 2, rawScratch, valScratch)
		if err != nil {
			fmt.Println("read chunk into:", err)
			return
		}
		for _, v := range values {
			sum += v
		}
	}
	fmt.Printf("sum over reused scratch = %v\n", sum)
	// Output:
	// sum over reused scratch = 15
}

// ExampleTensorReader_Close releases the reader's underlying file handle.
// It is safe to call on the zero-value reader and tolerates being called
// more than once.
func ExampleTensorReader_Close() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	reader, err := safetensors.OpenReader(index.Tensors["alpha"])
	if err != nil {
		fmt.Println("open reader:", err)
		return
	}
	reader.Close()
	// The zero-value reader closes safely too (no handle bound).
	var zero safetensors.TensorReader
	zero.Close()
	fmt.Println("closed")
	// Output:
	// closed
}

// ExampleNewShardCache opens two tensors from the SAME shard through one cache — the
// shard is opened once and the handle is reused for every ref in that file.
func ExampleNewShardCache() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	cache := safetensors.NewShardCache()
	defer cache.Close()

	reader, err := cache.Reader(index.Tensors["alpha"])
	if err != nil {
		fmt.Println("cache reader:", err)
		return
	}
	values, err := reader.ReadValues()
	if err != nil {
		fmt.Println("read values:", err)
		return
	}
	fmt.Println(values)
	// Output:
	// [1 2 3]
}

// ExampleShardCache_Reader returns a TensorReader bound to a cache-owned handle — the
// caller reads through the reader exactly as it would one opened by OpenReader.
func ExampleShardCache_Reader() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	cache := safetensors.NewShardCache()
	defer cache.Close()

	reader, err := cache.Reader(index.Tensors["beta"])
	if err != nil {
		fmt.Println("cache reader:", err)
		return
	}
	chunk, err := reader.ReadFloat32Chunk(0, 2)
	if err != nil {
		fmt.Println("read chunk:", err)
		return
	}
	fmt.Println(chunk)
	// Output:
	// [4 5]
}

// ExampleShardCache_ReadRefValues is the ref-in/values-out shape over a cache-owned
// handle — byte-identical to the package-level ReadRefValues for callers that don't
// want to hold a TensorReader.
func ExampleShardCache_ReadRefValues() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	cache := safetensors.NewShardCache()
	defer cache.Close()

	values, err := cache.ReadRefValues(index.Tensors["alpha"])
	if err != nil {
		fmt.Println("cache read values:", err)
		return
	}
	fmt.Println(values)
	// Output:
	// [1 2 3]
}

// ExampleShardCache_ReadRefRaw reads a ref's raw payload over a cache-owned handle —
// byte-identical to the package-level ReadRefRaw.
func ExampleShardCache_ReadRefRaw() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	cache := safetensors.NewShardCache()
	defer cache.Close()

	raw, err := cache.ReadRefRaw(index.Tensors["beta"])
	if err != nil {
		fmt.Println("cache read raw:", err)
		return
	}
	fmt.Printf("%d raw bytes\n", len(raw))
	// Output:
	// 8 raw bytes
}

// ExampleShardCache_Close releases every handle the cache opened. It is safe to call on
// an empty (never-used) cache, so it is the natural deferred cleanup right after
// NewShardCache.
func ExampleShardCache_Close() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	cache := safetensors.NewShardCache()
	if _, err := cache.Reader(index.Tensors["alpha"]); err != nil {
		fmt.Println("cache reader:", err)
		return
	}
	cache.Close()
	fmt.Println("closed")
	// Output:
	// closed
}

// ExampleTensorReader_ReadValues reads and decodes the bound ref's full payload over
// the reader's own shard handle — the open-handle equivalent of ReadRefValues.
func ExampleTensorReader_ReadValues() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	reader, err := safetensors.OpenReader(index.Tensors["beta"])
	if err != nil {
		fmt.Println("open reader:", err)
		return
	}
	defer reader.Close()
	values, err := reader.ReadValues()
	if err != nil {
		fmt.Println("read values:", err)
		return
	}
	fmt.Println(values)
	// Output:
	// [4 5]
}

// ExampleTensorReader_ReadRaw reads the bound ref's raw payload over the reader's own
// shard handle — the open-handle equivalent of ReadRefRaw.
func ExampleTensorReader_ReadRaw() {
	path, cleanup, err := twoTensorFile()
	if err != nil {
		fmt.Println("fixture:", err)
		return
	}
	defer cleanup()

	index, err := safetensors.ReadIndex(path)
	if err != nil {
		fmt.Println("read index:", err)
		return
	}
	reader, err := safetensors.OpenReader(index.Tensors["alpha"])
	if err != nil {
		fmt.Println("open reader:", err)
		return
	}
	defer reader.Close()
	raw, err := reader.ReadRaw()
	if err != nil {
		fmt.Println("read raw:", err)
		return
	}
	fmt.Printf("%d raw bytes\n", len(raw))
	// Output:
	// 12 raw bytes
}
