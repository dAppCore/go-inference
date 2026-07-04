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

// TestLoadDirSharded proves the two-shard merge: an index.json maps tensors across two shard
// files, and LoadDir returns the union with each tensor's bytes intact from its own shard.
func TestLoadDirSharded(t *testing.T) {
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

// TestLoadDirErrors checks the rejections: empty dir, malformed/empty index, a shard the index
// names but is missing, and a tensor the index names but its shard lacks.
func TestLoadDirErrors(t *testing.T) {
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
