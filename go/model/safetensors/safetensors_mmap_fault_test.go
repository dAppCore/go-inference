// SPDX-Licence-Identifier: EUPL-1.2

//go:build unix

package safetensors

import (
	"syscall"
	"testing"
)

// TestLoadMmapMmapError covers LoadMmap's mmap-failure branch (safetensors_mmap.go:39): a
// directory opens fine with O_RDONLY and fstat reports a non-zero size (so the empty-file
// guard passes), but mmap of a directory fails with EINVAL. That drives the one path between
// a successful fstat and Parse — a real syscall fault, no mock, no model load.
func TestLoadMmapMmapError(t *testing.T) {
	dir := t.TempDir() // a directory: open+fstat succeed, mmap fails
	m, err := LoadMmap(dir)
	if err == nil {
		t.Fatal("LoadMmap on a directory: expected an mmap error")
	}
	if m != nil {
		t.Fatalf("LoadMmap mmap-error must return a nil Mapping, got %#v", m)
	}
	t.Logf("LoadMmap: mmap of a directory fails (EINVAL) between fstat and Parse, returns nil + error")
}

// badMapping returns a *Mapping whose Data is a non-nil but MISALIGNED view (base+1) into a
// real page-aligned mmap. Munmap requires a page-aligned address, so munmap(Data) fails with
// EINVAL — exercising the Close error paths without corrupting any production loader (the
// caller owns the real base and unmaps it via the returned cleanup). Uses only the exported
// Mapping fields; no production code is modified.
func badMapping(t *testing.T) (*Mapping, func()) {
	t.Helper()
	p := t.TempDir() + "/page"
	fd, err := syscall.Open(p, syscall.O_RDWR|syscall.O_CREAT, 0o644)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	if _, err := syscall.Write(fd, make([]byte, 8192)); err != nil {
		_ = syscall.Close(fd)
		t.Fatalf("write: %v", err)
	}
	_ = syscall.Close(fd)

	rfd, err := syscall.Open(p, syscall.O_RDONLY, 0)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	data, err := syscall.Mmap(rfd, 0, 8192, syscall.PROT_READ, syscall.MAP_SHARED)
	_ = syscall.Close(rfd)
	if err != nil {
		t.Fatalf("mmap: %v", err)
	}
	// data[1:] is non-nil (so it clears Close's Data==nil guard) but misaligned, so munmap
	// of it fails; the real page-aligned base (data) is unmapped by the cleanup.
	return &Mapping{Data: data[1:], Tensors: map[string]Tensor{}}, func() { _ = syscall.Munmap(data) }
}

// TestMappingCloseMunmapError covers Mapping.Close's munmap-error branch (safetensors_mmap.go:59):
// a Mapping holding a misaligned (non-page-aligned) Data view makes the underlying munmap fail
// with EINVAL, so Close must wrap and return that error rather than swallow it. Real syscall
// fault via the exported field; no production loader is involved.
func TestMappingCloseMunmapError(t *testing.T) {
	m, cleanup := badMapping(t)
	defer cleanup()
	if err := m.Close(); err == nil {
		t.Fatal("Mapping.Close on a misaligned Data view: expected a munmap error")
	}
	t.Logf("Mapping.Close: munmap of a misaligned Data view fails (EINVAL) and is surfaced")
}

// init wires shardCloseFaultFixture (declared in sharded_test.go, which has no build tag) to
// badMapping — this is the unix side of that indirection; see the var's doc comment.
func init() {
	shardCloseFaultFixture = badMapping
}

// TestDirMappingCloseShardError covers DirMapping.Close's per-shard error branch (sharded.go:152):
// a single bad shard (misaligned Data) makes its Close fail, and DirMapping.Close must capture
// that as firstErr and return it. One shard keeps the firstErr==nil assignment deterministic.
func TestDirMappingCloseShardError(t *testing.T) {
	m, cleanup := badMapping(t)
	defer cleanup()
	d := &DirMapping{Shards: []*Mapping{m}, Tensors: map[string]Tensor{}}
	if err := d.Close(); err == nil {
		t.Fatal("DirMapping.Close with a shard whose munmap fails: expected an error")
	}
	t.Logf("DirMapping.Close: a shard's munmap failure is captured as firstErr and returned")
}
