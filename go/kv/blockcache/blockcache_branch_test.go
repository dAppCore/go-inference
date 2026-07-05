// SPDX-Licence-Identifier: EUPL-1.2

// Residual error-branch coverage for the disk-backed block cache. The
// happy paths and the chmod-driven failure paths already live in
// blockcache_disk_test.go; this file drives the last defensive arms that
// the public surface cannot reach because they sit behind a directory
// state that only a direct call to the unexported, already-locked helper
// can set up (a missing record, a read-only parent of a not-yet-created
// path, a pre-loaded zero-byte record). Every fault is injected through
// the real filesystem via the core helpers — no production seam exists,
// and per the disk layer's design none is wanted.
//
// Two adjacent arms are deliberately NOT exercised here because they are
// unreachable without mutating production code:
//
//   - readDiskRecord's `read.Value.([]byte)` type-assert failure
//     (blockcache.go ~L553): core.ReadFile always yields a []byte Value on
//     OK, so the assertion never fails through the real helper.
//   - writeDiskBlockLocked's JSONMarshal failure (blockcache.go ~L602):
//     diskRecord is composed solely of serialisable fields (int32 slices,
//     plain structs, a *state.ChunkRef), so encoding/json.Marshal cannot
//     return an error for it.
//
// Both are defensive guards on infrastructure that cannot misbehave for
// the concrete types in play; covering them would require injecting a
// fake filesystem / marshaller into production, which the brief forbids.

package blockcache

import (
	"context"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference"
)

// TestBlockcache_Service_WriteDiskBlockMkdirFailure drives the
// writeDiskBlockLocked MkdirAll-failure arm (blockcache.go ~L581): the
// configured DiskPath is nested under a regular file, so the up-front
// MkdirAll of the cache directory cannot succeed and the write surfaces a
// wrapped error before touching any record. Calling the locked helper
// directly skips the lazy load (which would fail on the same MkdirAll and
// mask this arm).
func TestBlockcache_Service_WriteDiskBlockMkdirFailure(t *testing.T) {
	parent := core.PathJoin(t.TempDir(), "afile")
	if result := core.WriteFile(parent, []byte("x"), 0o600); !result.OK {
		t.Fatalf("WriteFile(parent file) error = %s", result.Error())
	}
	// DiskPath's parent component is a regular file, so core.MkdirAll fails.
	service := New(Config{BlockSize: 2, DiskPath: core.PathJoin(parent, "blocks")})
	_, err := service.writeDiskBlockLocked(
		context.Background(),
		inference.CacheBlockRef{ID: "blk-1"},
		[]int32{1, 2},
	)
	if err == nil {
		t.Fatal("writeDiskBlockLocked(unwritable disk dir) error = nil")
	}
}

// TestBlockcache_Service_ClearDiskRecreateFailure drives the
// clearDiskLocked recreate-MkdirAll arm (blockcache.go ~L673): the DiskPath
// is a not-yet-existing child of a read-only parent directory. RemoveAll
// of the missing child returns OK (nothing to remove), so the first guard
// passes; the subsequent MkdirAll that recreates the cache directory then
// fails on the read-only parent, surfacing the recreate error.
func TestBlockcache_Service_ClearDiskRecreateFailure(t *testing.T) {
	parent := core.PathJoin(t.TempDir(), "parent")
	if result := core.MkdirAll(parent, 0o700); !result.OK {
		t.Fatalf("MkdirAll(parent) error = %s", result.Error())
	}
	// Read-only parent: RemoveAll of the missing child is a no-op success,
	// but recreating the child directory underneath it is denied.
	if result := core.Chmod(parent, 0o500); !result.OK {
		t.Fatalf("Chmod(read-only parent) error = %s", result.Error())
	}
	t.Cleanup(func() { core.Chmod(parent, 0o700) })
	// DiskPath child does not exist yet, so the RemoveAll guard succeeds and
	// control reaches the recreate MkdirAll.
	service := New(Config{DiskPath: core.PathJoin(parent, "blocks")})
	if err := service.clearDiskLocked(); err == nil {
		t.Fatal("clearDiskLocked(recreate under read-only parent) error = nil")
	}
}

// TestBlockcache_Service_RemoveDiskBlockMissingIsNil drives the
// removeDiskBlockLocked not-exist arm (blockcache.go ~L688): removing a
// block whose record file was never written fails with an IsNotExist error,
// which the helper treats as success (the record is already gone) and
// returns nil. The cache directory exists and is writable, so the Remove
// failure is specifically not-exist rather than a permission error.
func TestBlockcache_Service_RemoveDiskBlockMissingIsNil(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll(diskPath) error = %s", result.Error())
	}
	service := New(Config{DiskPath: diskPath})
	// "ghost" was never persisted, so Remove of its record path returns an
	// IsNotExist error that the helper swallows.
	if err := service.removeDiskBlockLocked("ghost"); err != nil {
		t.Fatalf("removeDiskBlockLocked(missing record) error = %v, want nil", err)
	}
}

// TestBlockcache_Service_QuarantineMissingPathSwallowed drives the
// quarantineDiskBlock not-exist arm (blockcache.go ~L703): the best-effort
// Remove of an already-vanished corrupt record fails with an IsNotExist
// error, which the helper recognises and returns from cleanly. The block is
// still accounted corrupt + evicted regardless of the missing file.
func TestBlockcache_Service_QuarantineMissingPathSwallowed(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll(diskPath) error = %s", result.Error())
	}
	service := New(Config{DiskPath: diskPath})
	before := service.evictions
	// The path does not exist, so the best-effort Remove returns IsNotExist
	// and quarantineDiskBlock takes its early return.
	service.quarantineDiskBlock(core.PathJoin(diskPath, "vanished.json"))
	if service.evictions != before+1 || service.diskCorrupt != 1 {
		t.Fatalf("evictions=%d diskCorrupt=%d, want corrupt counted despite missing file",
			service.evictions, service.diskCorrupt)
	}
}

// TestBlockcache_Service_DiskBytesReadFileFallback drives the
// diskBytesLocked Stat-then-ReadFile fallback arms (blockcache.go ~L722):
// a zero-byte *.json record reports a Stat size of 0, so the size>0 fast
// path is skipped and the byte total is taken from the ReadFile result.
// The helper is called directly with diskLoaded already set so the empty
// record is not quarantined-and-removed by a lazy load before the byte walk
// (which is why the equivalent disk_test path leaves these arms cold).
func TestBlockcache_Service_DiskBytesReadFileFallback(t *testing.T) {
	diskPath := core.PathJoin(t.TempDir(), "blocks")
	if result := core.MkdirAll(diskPath, 0o700); !result.OK {
		t.Fatalf("MkdirAll(diskPath) error = %s", result.Error())
	}
	// A zero-byte record: Stat.Size() is 0 so the size>0 branch is false and
	// the glob entry falls through to the ReadFile length (also 0).
	if result := core.WriteFile(core.PathJoin(diskPath, "empty.json"), []byte{}, 0o600); !result.OK {
		t.Fatalf("WriteFile(empty record) error = %s", result.Error())
	}
	// diskLoaded suppresses the lazy load so the empty record is not removed
	// before diskBytesLocked walks the glob.
	service := &Service{
		cfg:        Config{DiskPath: diskPath},
		blocks:     map[string]inference.CacheBlockRef{},
		diskLoaded: true,
	}
	if got := service.diskBytesLocked(); got != 0 {
		t.Fatalf("diskBytesLocked(zero-byte record) = %d, want 0", got)
	}
}

// TestBlockcache_resultError_UnknownFallback drives the resultError final
// fallback (blockcache.go ~L867): a failed Result whose Value is the empty
// string is neither an error nor OK, and its Error() text is "" (the
// string arm of core.Result.Error returns the empty string verbatim), so
// resultError reaches its last-resort synthesised error. The other three
// arms are already covered by TestBlockCacheHelpers_Good.
func TestBlockcache_resultError_UnknownFallback(t *testing.T) {
	err := resultError(core.Result{Value: "", OK: false})
	if err == nil {
		t.Fatal("resultError(empty-string failure) = nil, want synthesised error")
	}
	if err.Error() != "unknown block cache result error" {
		t.Fatalf("resultError(empty-string failure) = %q, want the synthesised fallback", err.Error())
	}
}
