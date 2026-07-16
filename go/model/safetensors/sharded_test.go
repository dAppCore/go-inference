// SPDX-Licence-Identifier: EUPL-1.2

package safetensors

import (
	"bytes"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

func writeFile(t *testing.T, dir, name string, content []byte) {
	t.Helper()
	if err := coreio.Local.Write(core.PathJoin(dir, name), string(content)); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
}

func assertTensor(t *testing.T, m map[string]Tensor, name string, want Tensor) {
	t.Helper()
	got, ok := m[name]
	if !ok {
		t.Fatalf("merged map missing %q", name)
	}
	if got.Dtype != want.Dtype || !bytes.Equal(got.Data, want.Data) || len(got.Shape) != len(want.Shape) {
		t.Fatalf("%q: got {%s %v %v}, want {%s %v %v}", name, got.Dtype, got.Shape, got.Data, want.Dtype, want.Shape, want.Data)
	}
	for i := range got.Shape {
		if got.Shape[i] != want.Shape[i] {
			t.Fatalf("%q shape: got %v want %v", name, got.Shape, want.Shape)
		}
	}
}

// TestSharded_LoadDir_Good proves the two-shard merge: an index.json maps tensors
// across two shard files, and LoadDir returns the union with each tensor's bytes intact
// from its own shard.
func TestSharded_LoadDir_Good(t *testing.T) {
	dir := t.TempDir()
	a := Tensor{Dtype: "F32", Shape: []int{2}, Data: []byte{1, 0, 0, 0, 2, 0, 0, 0}}
	b := Tensor{Dtype: "U8", Shape: []int{3}, Data: []byte{9, 8, 7}}
	c := Tensor{Dtype: "BF16", Shape: []int{2}, Data: []byte{0xAA, 0xBB, 0xCC, 0xDD}}
	blob1, err := Encode(map[string]Tensor{"a": a, "b": b})
	if err != nil {
		t.Fatalf("Encode shard1: %v", err)
	}
	blob2, err := Encode(map[string]Tensor{"c": c})
	if err != nil {
		t.Fatalf("Encode shard2: %v", err)
	}
	writeFile(t, dir, "model-00001-of-00002.safetensors", blob1)
	writeFile(t, dir, "model-00002-of-00002.safetensors", blob2)
	writeFile(t, dir, indexName, []byte(`{"metadata":{"total_size":15},"weight_map":{
		"a":"model-00001-of-00002.safetensors",
		"b":"model-00001-of-00002.safetensors",
		"c":"model-00002-of-00002.safetensors"}}`))

	got, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("merged map has %d tensors, want 3", len(got))
	}
	assertTensor(t, got, "a", a)
	assertTensor(t, got, "b", b)
	assertTensor(t, got, "c", c)
	t.Logf("sharded: 3 tensors across 2 shards merged, bytes intact")
}

// TestLoadDirSingle proves the single-file fallback: no index, just model.safetensors.
func TestLoadDirSingle(t *testing.T) {
	dir := t.TempDir()
	x := Tensor{Dtype: "F32", Shape: []int{1}, Data: []byte{7, 0, 0, 0}}
	blob, err := Encode(map[string]Tensor{"x": x})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	writeFile(t, dir, singleName, blob)
	got, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("single map has %d tensors, want 1", len(got))
	}
	assertTensor(t, got, "x", x)
	t.Logf("single: model.safetensors loaded without an index")
}

// TestSharded_LoadDir_Bad checks the rejections: empty dir, malformed/empty index, a
// shard the index names but is missing, and a tensor the index names but its shard lacks.
func TestSharded_LoadDir_Bad(t *testing.T) {
	if _, err := LoadDir(t.TempDir()); err == nil {
		t.Fatal("empty dir: expected an error")
	}

	dMissingShard := t.TempDir()
	writeFile(t, dMissingShard, indexName, []byte(`{"weight_map":{"a":"missing.safetensors"}}`))
	if _, err := LoadDir(dMissingShard); err == nil {
		t.Fatal("missing shard file: expected an error")
	}

	dAbsentTensor := t.TempDir()
	blob, err := Encode(map[string]Tensor{"present": {Dtype: "U8", Shape: []int{1}, Data: []byte{1}}})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	writeFile(t, dAbsentTensor, "s.safetensors", blob)
	writeFile(t, dAbsentTensor, indexName, []byte(`{"weight_map":{"absent":"s.safetensors"}}`))
	if _, err := LoadDir(dAbsentTensor); err == nil {
		t.Fatal("tensor absent from its shard: expected an error")
	}

	dEmptyMap := t.TempDir()
	writeFile(t, dEmptyMap, indexName, []byte(`{"weight_map":{}}`))
	if _, err := LoadDir(dEmptyMap); err == nil {
		t.Fatal("empty weight_map: expected an error")
	}

	dMalformed := t.TempDir()
	writeFile(t, dMalformed, indexName, []byte(`{not json`))
	if _, err := LoadDir(dMalformed); err == nil {
		t.Fatal("malformed index json: expected an error")
	}
	t.Logf("rejections: empty dir, missing shard, absent tensor, empty/malformed index all error")
}

// TestSharded_LoadDir_Ugly confirms the index always wins: a directory holding BOTH
// an index.json and a stray model.safetensors (that the index doesn't reference) loads
// via the sharded path, not the single-file fallback.
func TestSharded_LoadDir_Ugly(t *testing.T) {
	dir := t.TempDir()
	a := Tensor{Dtype: "F32", Shape: []int{1}, Data: []byte{1, 0, 0, 0}}
	blobA, err := Encode(map[string]Tensor{"a": a})
	if err != nil {
		t.Fatalf("Encode shard: %v", err)
	}
	writeFile(t, dir, "shard.safetensors", blobA)
	writeFile(t, dir, indexName, []byte(`{"weight_map":{"a":"shard.safetensors"}}`))
	// A stray single-file that the index does not mention.
	strayBlob, err := Encode(map[string]Tensor{"stray": {Dtype: "U8", Shape: []int{1}, Data: []byte{9}}})
	if err != nil {
		t.Fatalf("Encode stray: %v", err)
	}
	writeFile(t, dir, singleName, strayBlob)

	got, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d tensors, want 1 (index path only, stray single-file ignored)", len(got))
	}
	assertTensor(t, got, "a", a)
}

// --- LoadDirMmap ---

// TestSharded_LoadDirMmap_Good covers the sharded layout: two shard files + an
// index.json, mapped and merged so each tensor's bytes match its source shard.
func TestSharded_LoadDirMmap_Good(t *testing.T) {
	dir := t.TempDir()
	a := Tensor{Dtype: "F32", Shape: []int{2}, Data: []byte{1, 0, 0, 0, 2, 0, 0, 0}}
	b := Tensor{Dtype: "U8", Shape: []int{3}, Data: []byte{9, 8, 7}}
	blob1, err := Encode(map[string]Tensor{"a": a})
	if err != nil {
		t.Fatalf("Encode shard1: %v", err)
	}
	blob2, err := Encode(map[string]Tensor{"b": b})
	if err != nil {
		t.Fatalf("Encode shard2: %v", err)
	}
	writeFile(t, dir, "model-00001-of-00002.safetensors", blob1)
	writeFile(t, dir, "model-00002-of-00002.safetensors", blob2)
	writeFile(t, dir, indexName, []byte(`{"weight_map":{
		"a":"model-00001-of-00002.safetensors",
		"b":"model-00002-of-00002.safetensors"}}`))

	dm, err := LoadDirMmap(dir)
	if err != nil {
		t.Fatalf("LoadDirMmap: %v", err)
	}
	defer dm.Close()
	if len(dm.Shards) != 2 || len(dm.Tensors) != 2 {
		t.Fatalf("want 2 shards + 2 tensors, got %d shards %d tensors", len(dm.Shards), len(dm.Tensors))
	}
	assertTensor(t, dm.Tensors, "a", a)
	assertTensor(t, dm.Tensors, "b", b)
}

// TestSharded_LoadDirMmap_Bad confirms an empty directory (neither an index nor a
// single model.safetensors) is rejected.
func TestSharded_LoadDirMmap_Bad(t *testing.T) {
	if _, err := LoadDirMmap(t.TempDir()); err == nil {
		t.Fatal("LoadDirMmap(empty dir) error = nil")
	}
}

// TestSharded_LoadDirMmap_Ugly covers the single-file fallback: a directory holding
// just model.safetensors (no index) maps into a one-shard DirMapping.
func TestSharded_LoadDirMmap_Ugly(t *testing.T) {
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
	assertTensor(t, dm.Tensors, "x", x)
}

// --- DirMapping.Close ---

// TestSharded_DirMapping_Close_Good confirms Close unmaps every shard and clears
// both Shards and Tensors, so a stale reference cannot be read after Close.
func TestSharded_DirMapping_Close_Good(t *testing.T) {
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
		t.Fatalf("Close: %v", err)
	}
	if dm.Shards != nil || dm.Tensors != nil {
		t.Fatalf("Shards/Tensors not cleared after Close: %v %v", dm.Shards, dm.Tensors)
	}
}

// shardCloseFaultFixture triggers a real per-shard Close failure for TestSharded_
// DirMapping_Close_Bad below. Wired only on unix, in safetensors_mmap_fault_test.go's init
// (which reuses the same badMapping fixture TestDirMappingCloseShardError already exercises) —
// this file has no build tag (DirMapping must stay usable on every platform), so it cannot
// reference the unix-only mmap syscalls directly. On non-unix builds the assigning init never
// runs, the var stays nil, and the Bad test skips: Mapping.Close is a documented no-op there
// (safetensors_mmap_other.go) and can never fail, so there is nothing to inject.
var shardCloseFaultFixture func(t *testing.T) (*Mapping, func())

// TestSharded_DirMapping_Close_Bad covers Close's per-shard error branch: when a shard's own
// Close fails, DirMapping.Close must capture that as firstErr and return it, not swallow it.
func TestSharded_DirMapping_Close_Bad(t *testing.T) {
	if shardCloseFaultFixture == nil {
		t.Skip("Mapping.Close cannot fail on this platform (documented no-op; see safetensors_mmap_other.go)")
	}
	m, cleanup := shardCloseFaultFixture(t)
	defer cleanup()
	d := &DirMapping{Shards: []*Mapping{m}, Tensors: map[string]Tensor{}}
	if err := d.Close(); err == nil {
		t.Fatal("DirMapping.Close with a shard whose Close fails: expected an error")
	}
}

// TestSharded_DirMapping_Close_Ugly confirms Close on a nil *DirMapping, and a
// second Close after a real one, are both safe no-ops.
func TestSharded_DirMapping_Close_Ugly(t *testing.T) {
	var nilD *DirMapping
	if err := nilD.Close(); err != nil {
		t.Fatalf("nil DirMapping Close: %v, want nil", err)
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
		t.Fatalf("second Close: %v, want nil", err)
	}
}
