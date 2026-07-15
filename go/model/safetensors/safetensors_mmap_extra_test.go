// SPDX-Licence-Identifier: EUPL-1.2

//go:build unix

package safetensors

import (
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// TestLoadMmapOpenError covers LoadMmap's open failure: a path that doesn't exist must error
// (the syscall.Open branch), never returning a Mapping.
func TestLoadMmapOpenError(t *testing.T) {
	m, err := LoadMmap(core.PathJoin(t.TempDir(), "nope.safetensors"))
	if err == nil {
		t.Fatal("LoadMmap on a missing file: expected an error")
	}
	if m != nil {
		t.Fatalf("LoadMmap error must return nil Mapping, got %#v", m)
	}
}

// TestLoadMmapEmptyFile covers the st.Size <= 0 guard: a zero-byte file is rejected before any
// mmap call (mmap of length 0 is itself an error, so the guard is the real gate).
func TestLoadMmapEmptyFile(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "empty.safetensors")
	if err := coreio.Local.Write(path, ""); err != nil {
		t.Fatalf("write empty: %v", err)
	}
	if _, err := LoadMmap(path); err == nil {
		t.Fatal("LoadMmap on an empty file: expected an error")
	}
}

// TestLoadMmapParseError covers the Parse-failed-after-mmap branch (and its munmap-then-error
// cleanup): a non-empty file whose contents are not a valid safetensors blob maps fine, but
// Parse rejects it, so LoadMmap must unmap and return the error.
func TestLoadMmapParseError(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "garbage.safetensors")
	// 16 bytes: a header-length prefix far larger than the file → Parse "header length out of range".
	if err := coreio.Local.Write(path, "\xff\xff\xff\xff\xff\xff\xff\xffgarbage!"); err != nil {
		t.Fatalf("write garbage: %v", err)
	}
	if _, err := LoadMmap(path); err == nil {
		t.Fatal("LoadMmap on a non-safetensors file: expected a parse error")
	}
}

// TestMappingCloseNilAndDouble covers Mapping.Close's early-exit guard: Close on a nil *Mapping
// and a second Close (Data already nil) must both be no-op nils — never a double-munmap.
func TestMappingCloseNilAndDouble(t *testing.T) {
	var nilM *Mapping
	if err := nilM.Close(); err != nil {
		t.Fatalf("nil Mapping Close should be nil, got %v", err)
	}

	path := core.PathJoin(t.TempDir(), "m.safetensors")
	blob, err := Encode(map[string]Tensor{"a": {Dtype: "U8", Shape: []int{1}, Data: []byte{1}}})
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
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
		t.Fatalf("second Close (Data already nil) should be nil, got %v", err)
	}
	t.Logf("Mapping.Close: nil receiver + double-close both safe no-ops")
}
