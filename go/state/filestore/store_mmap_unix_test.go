// SPDX-Licence-Identifier: EUPL-1.2

// filestore mmap-backed borrow tests (unix): ensureMappedRegionLocked fault paths not reachable through the public Borrow* API.
//go:build darwin || linux || freebsd || netbsd || openbsd

package filestore

import (
	"context"
	"testing"

	core "dappco.re/go"
)

func TestEnsureMappedRegionLocked_Bad_StatError(t *testing.T) {
	path := core.PathJoin(t.TempDir(), "mmap-stat-fail.mvlog")
	if result := core.WriteFile(path, fileMagic, 0o600); !result.OK {
		t.Fatalf("WriteFile() error = %s", result.Error())
	}
	file := openFileOrFatal(t, path, core.O_RDONLY)
	if err := file.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	s := &Store{file: file}
	if err := s.ensureMappedRegionLocked(); err == nil {
		t.Fatal("ensureMappedRegionLocked(closed file) error = nil, want stat error")
	}
}

func TestEnsureMappedRegionLocked_Bad_SizeInvalid(t *testing.T) {
	ctx := context.Background()
	path := core.PathJoin(t.TempDir(), "mmap-zero-region.mvlog")
	store, err := Create(ctx, path)
	if err != nil {
		t.Fatalf("Create() error = %v", err)
	}
	defer store.Close()
	stat := core.Stat(path)
	if !stat.OK {
		t.Fatalf("Stat() error = %s", stat.Error())
	}
	fileSize := stat.Value.(interface{ Size() int64 }).Size()

	// baseAt pinned exactly at EOF with region left at its zero-value
	// "auto" meaning ("rest of file") makes regionSize resolve to an
	// available span of exactly 0 — unreachable through the public
	// Open/OpenRegionWithSegmentAlias API (rebuildIndex's own
	// detectHeaderLen already rejects a zero-byte region before a
	// caller could reach ensureMappedRegionLocked), so this state is
	// constructed directly.
	store.mu.Lock()
	store.baseAt = fileSize
	err = store.ensureMappedRegionLocked()
	store.mu.Unlock()
	if err == nil {
		t.Fatal("ensureMappedRegionLocked(zero-byte region) error = nil, want errMappedRegionInvalid")
	}
}
