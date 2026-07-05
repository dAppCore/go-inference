// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// TestLoadDirMmapSingle covers LoadDirMmap's single-file fallback: a directory with just
// model.safetensors (no index) maps into a one-shard DirMapping whose merged Tensors are views
// into that shard's mmap. Mirrors LoadDir's single-file path on the zero-copy side.
func TestLoadDirMmapSingle(t *testing.T) {
	dir := t.TempDir()
	x := Tensor{Dtype: "F32", Shape: []int{1}, Data: []byte{7, 0, 0, 0}}
	blob, err := Encode(map[string]Tensor{"x": x})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	writeFile(t, dir, singleName, blob)

	dm, err := LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap single: %v", err)
	}
	defer dm.Close()
	if len(dm.Shards) != 1 || len(dm.Tensors) != 1 {
		t.Fatalf("want 1 shard + 1 tensor, got %d shards %d tensors", len(dm.Shards), len(dm.Tensors))
	}
	got, ok := dm.Tensors["x"]
	if !ok {
		t.Fatal("tensor x missing")
	}
	if !bytes.Equal(got.Data, x.Data) {
		t.Fatalf("tensor x bytes mismatch: %v", got.Data)
	}
	t.Logf("LoadDirMmap: single model.safetensors mapped as a one-shard DirMapping")
}

// TestLoadDirMmapErrors drives LoadDirMmap's rejection branches: an empty dir (neither index
// nor single file), a malformed index, an empty weight_map, a shard the index names but is
// absent (exercises the closeAll cleanup on a partial map), and a tensor the index maps to a
// shard that exists but lacks it (the single-shard form makes the closeAll body deterministic).
func TestLoadDirMmapErrors(t *testing.T) {
	// neither index nor single file present
	if _, err := LoadDirMmap(t.TempDir()); err == nil {
		t.Fatal("empty dir: expected an error")
	}

	// malformed index json
	dMalformed := t.TempDir()
	writeFile(t, dMalformed, indexName, []byte(`{not json`))
	if _, err := LoadDirMmap(dMalformed); err == nil {
		t.Fatal("malformed index: expected an error")
	}

	// empty weight_map
	dEmptyMap := t.TempDir()
	writeFile(t, dEmptyMap, indexName, []byte(`{"weight_map":{}}`))
	if _, err := LoadDirMmap(dEmptyMap); err == nil {
		t.Fatal("empty weight_map: expected an error")
	}

	// shard named by the index but missing on disk (map shard fails before any shard is mapped)
	dMissingShard := t.TempDir()
	writeFile(t, dMissingShard, indexName, []byte(`{"weight_map":{"a":"missing.safetensors"}}`))
	if _, err := LoadDirMmap(dMissingShard); err == nil {
		t.Fatal("missing shard: expected an error")
	}

	// a real shard that lacks the tensor the index claims: the shard maps, THEN the lookup
	// fails, so the closeAll cleanup runs over the one already-mapped shard every time.
	dAbsentTensor := t.TempDir()
	blob, err := Encode(map[string]Tensor{"present": {Dtype: "U8", Shape: []int{1}, Data: []byte{1}}})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	writeFile(t, dAbsentTensor, "s.safetensors", blob)
	writeFile(t, dAbsentTensor, indexName, []byte(`{"weight_map":{"absent":"s.safetensors"}}`))
	if _, err := LoadDirMmap(dAbsentTensor); err == nil {
		t.Fatal("tensor absent from its shard: expected an error")
	}
	t.Logf("LoadDirMmap rejections: empty dir, malformed/empty index, missing shard, absent tensor all error")
}

// TestLoadDirMmapSingleParseError covers the single-file branch's error return: a
// model.safetensors that exists but is not a valid blob must surface LoadMmap's parse error.
func TestLoadDirMmapSingleParseError(t *testing.T) {
	dir := t.TempDir()
	// non-empty so the st.Size guard passes, but the header length overflows → Parse rejects it.
	if err := coreio.Local.Write(core.PathJoin(dir, singleName), "\xff\xff\xff\xff\xff\xff\xff\xffjunk"); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := LoadDirMmap(dir); err == nil {
		t.Fatal("single model.safetensors that won't parse: expected an error")
	}
}

// TestDirMappingCloseNil covers DirMapping.Close's nil-receiver guard: Close on a nil
// *DirMapping must be a no-op nil, and a second Close after a real one (Shards nil) too.
func TestDirMappingCloseNil(t *testing.T) {
	var nilD *DirMapping
	if err := nilD.Close(); err != nil {
		t.Fatalf("nil DirMapping Close should be nil, got %v", err)
	}

	dir := t.TempDir()
	x := Tensor{Dtype: "U8", Shape: []int{1}, Data: []byte{1}}
	blob, err := Encode(map[string]Tensor{"x": x})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	writeFile(t, dir, singleName, blob)
	dm, err := LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}
	if err := dm.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := dm.Close(); err != nil {
		t.Fatalf("second Close (Shards already nil) should be nil, got %v", err)
	}
	t.Logf("DirMapping.Close: nil receiver + double-close both safe no-ops")
}

// TestLoadDirSingleEmptyFile covers LoadDir's single-file branch surfacing a load error: a
// present-but-empty model.safetensors must error rather than return a partial map.
func TestLoadDirSingleEmptyFile(t *testing.T) {
	dir := t.TempDir()
	if err := coreio.Local.Write(core.PathJoin(dir, singleName), ""); err != nil {
		t.Fatalf("write empty: %v", err)
	}
	if _, err := LoadDir(dir); err == nil {
		t.Fatal("empty single model.safetensors: expected an error")
	}
}
