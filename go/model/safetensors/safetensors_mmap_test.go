// SPDX-Licence-Identifier: EUPL-1.2

//go:build unix

package safetensors

import (
	"bytes"
	"testing"
	"unsafe"

	coreio "dappco.re/go/io"
)

// TestSafetensors_LoadMmap_Good round-trips a synthetic checkpoint through Encode → file →
// LoadMmap and proves the key property: each Tensor.Data is a VIEW into the page-aligned
// mmap, not a heap copy. That view-into-an-aligned-base is exactly what the no-copy GPU
// buffer path needs. No model load — AX-11 synthetic.
func TestSafetensors_LoadMmap_Good(t *testing.T) {
	want := map[string]Tensor{
		"a.weight": {Dtype: "F32", Shape: []int{2, 2}, Data: []byte{0, 0, 128, 63, 0, 0, 0, 64, 0, 0, 64, 64, 0, 0, 128, 64}},
		"b.scales": {Dtype: "BF16", Shape: []int{4}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8}},
	}
	blob, err := Encode(want)
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	path := t.TempDir() + "/m.safetensors"
	if err := coreio.Local.Write(path, string(blob)); err != nil {
		t.Fatalf("write fixture: %v", err)
	}

	m, err := LoadMmap(path)
	if err != nil {
		t.Fatalf("LoadMmap: %v", err)
	}
	defer m.Close()

	if len(m.Data) != len(blob) {
		t.Fatalf("mapped %d bytes, want %d", len(m.Data), len(blob))
	}
	base := uintptr(unsafe.Pointer(&m.Data[0]))
	end := base + uintptr(len(m.Data))
	for name, w := range want {
		got, ok := m.Tensors[name]
		if !ok {
			t.Fatalf("missing tensor %s", name)
		}
		if got.Dtype != w.Dtype || !bytes.Equal(got.Data, w.Data) {
			t.Fatalf("tensor %s content mismatch", name)
		}
		// The whole point: Data must be a VIEW into the mmap, not a heap copy.
		ptr := uintptr(unsafe.Pointer(&got.Data[0]))
		if ptr < base || ptr >= end {
			t.Fatalf("tensor %s Data is a copy, not a view into the mmap — zero-copy broken", name)
		}
	}
	t.Logf("LoadMmap: %d tensors view the %d-byte page-aligned mmap (zero-copy)", len(m.Tensors), len(m.Data))
}

// TestSafetensors_LoadMmap_Bad confirms a path that does not exist surfaces the open error
// (the mmap-failure branch itself — a directory rather than a regular file — is a deeper
// syscall fault covered by TestLoadMmapMmapError in safetensors_mmap_fault_test.go).
func TestSafetensors_LoadMmap_Bad(t *testing.T) {
	if _, err := LoadMmap(t.TempDir() + "/missing.safetensors"); err == nil {
		t.Fatal("LoadMmap(missing file) error = nil")
	}
}

// TestSafetensors_LoadMmap_Ugly confirms a present-but-empty file is rejected by the
// empty-file guard rather than falling through to a zero-size mmap.
func TestSafetensors_LoadMmap_Ugly(t *testing.T) {
	path := t.TempDir() + "/empty.safetensors"
	if err := coreio.Local.Write(path, ""); err != nil {
		t.Fatalf("write empty: %v", err)
	}
	if _, err := LoadMmap(path); err == nil {
		t.Fatal("LoadMmap(empty file) error = nil")
	}
}

// TestSafetensors_Mapping_Close_Good confirms Close unmaps the file and clears Data and
// Tensors, so a stale reference cannot be read after Close.
func TestSafetensors_Mapping_Close_Good(t *testing.T) {
	blob, err := Encode(map[string]Tensor{"x": {Dtype: "U8", Shape: []int{1}, Data: []byte{1}}})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	path := t.TempDir() + "/m.safetensors"
	if err := coreio.Local.Write(path, string(blob)); err != nil {
		t.Fatalf("write: %v", err)
	}
	m, err := LoadMmap(path)
	if err != nil {
		t.Fatalf("LoadMmap: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if m.Data != nil || m.Tensors != nil {
		t.Fatalf("Data/Tensors not cleared after Close: %v %v", m.Data, m.Tensors)
	}
}

// NOTE: Mapping.Close's error path (munmap failing on a misaligned Data view) needs a real
// syscall fault to trigger — see TestMappingCloseMunmapError in safetensors_mmap_fault_test.go.

// TestSafetensors_Mapping_Close_Ugly confirms Close on a nil *Mapping, and a second Close
// after a real one, are both safe no-ops.
func TestSafetensors_Mapping_Close_Ugly(t *testing.T) {
	var nilM *Mapping
	if err := nilM.Close(); err != nil {
		t.Fatalf("nil Mapping Close: %v, want nil", err)
	}

	blob, err := Encode(map[string]Tensor{"x": {Dtype: "U8", Shape: []int{1}, Data: []byte{1}}})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	path := t.TempDir() + "/m.safetensors"
	if err := coreio.Local.Write(path, string(blob)); err != nil {
		t.Fatalf("write: %v", err)
	}
	m, err := LoadMmap(path)
	if err != nil {
		t.Fatalf("LoadMmap: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("first Close: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second Close: %v, want nil", err)
	}
}

// TestLoadDirMmapSharded covers the sharded layout: two shard files + an index.json, mapped
// and merged, with each tensor a view into ITS shard's mmap (so a backend binds one no-copy
// buffer per shard and addresses tensors by offset). No model load — AX-11 synthetic.
func TestLoadDirMmapSharded(t *testing.T) {
	dir := t.TempDir()
	s1, err := Encode(map[string]Tensor{"a.weight": {Dtype: "F32", Shape: []int{2}, Data: []byte{1, 2, 3, 4, 5, 6, 7, 8}}})
	if err != nil {
		t.Fatal(err)
	}
	s2, err := Encode(map[string]Tensor{"b.weight": {Dtype: "BF16", Shape: []int{3}, Data: []byte{9, 10, 11, 12, 13, 14}}})
	if err != nil {
		t.Fatal(err)
	}
	for f, blob := range map[string][]byte{"s1.safetensors": s1, "s2.safetensors": s2} {
		if err := coreio.Local.Write(dir+"/"+f, string(blob)); err != nil {
			t.Fatal(err)
		}
	}
	if err := coreio.Local.Write(dir+"/model.safetensors.index.json", `{"weight_map":{"a.weight":"s1.safetensors","b.weight":"s2.safetensors"}}`); err != nil {
		t.Fatal(err)
	}

	dm, err := LoadDirMmap(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer dm.Close()
	if len(dm.Shards) != 2 || len(dm.Tensors) != 2 {
		t.Fatalf("want 2 shards + 2 tensors, got %d shards %d tensors", len(dm.Shards), len(dm.Tensors))
	}
	want := map[string][]byte{"a.weight": {1, 2, 3, 4, 5, 6, 7, 8}, "b.weight": {9, 10, 11, 12, 13, 14}}
	for name, w := range want {
		got, ok := dm.Tensors[name]
		if !ok {
			t.Fatalf("missing tensor %s", name)
		}
		if !bytes.Equal(got.Data, w) {
			t.Fatalf("tensor %s content mismatch", name)
		}
		// must view ONE of the shard mmaps (zero-copy across shards)
		ptr := uintptr(unsafe.Pointer(&got.Data[0]))
		inShard := false
		for _, sh := range dm.Shards {
			base := uintptr(unsafe.Pointer(&sh.Data[0]))
			if ptr >= base && ptr < base+uintptr(len(sh.Data)) {
				inShard = true
				break
			}
		}
		if !inShard {
			t.Fatalf("tensor %s Data is not a view into any shard mmap", name)
		}
	}
	t.Logf("LoadDirMmap: %d tensors merged across %d shard mmaps (zero-copy)", len(dm.Tensors), len(dm.Shards))
}
