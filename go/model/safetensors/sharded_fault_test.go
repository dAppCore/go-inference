// SPDX-Licence-Identifier: EUPL-1.2

//go:build unix

package safetensors

import (
	"os"
	"testing"

	core "dappco.re/go"
	coreio "dappco.re/go/io"
)

// chmodUnreadable writes content at dir/name, then strips all permission bits so the path is
// still a regular file (IsFile/Stat succeed) but cannot be opened for reading (Read/open ->
// EACCES). It registers a cleanup that restores 0644 so t.TempDir can remove the tree. Skips
// under euid 0, where the permission bits are ignored and the read would succeed.
func chmodUnreadable(t *testing.T, dir, name, content string) {
	t.Helper()
	if os.Geteuid() == 0 {
		t.Skip("running as root: 0000 perms are bypassed, so the read-failure branch is unreachable")
	}
	p := core.PathJoin(dir, name)
	if err := coreio.Local.Write(p, content); err != nil {
		t.Fatalf("write %s: %v", name, err)
	}
	if err := os.Chmod(p, 0o000); err != nil {
		t.Fatalf("chmod 000 %s: %v", name, err)
	}
	t.Cleanup(func() { _ = os.Chmod(p, 0o644) })
}

// TestLoadDirIndexReadError covers LoadDir's index-read-failure branch (sharded.go ~35): a
// present-but-unreadable model.safetensors.index.json passes the IsFile gate (it is a regular
// file) but fails the subsequent Read, so LoadDir must surface that error rather than fall
// through to the single-file path. Real FS fault (0000 perms) — no mock, no model load.
func TestLoadDirIndexReadError(t *testing.T) {
	dir := t.TempDir()
	chmodUnreadable(t, dir, indexName, `{"weight_map":{"a":"s.safetensors"}}`)
	if _, err := LoadDir(dir); err == nil {
		t.Fatal("LoadDir with an unreadable index.json: expected a read error")
	}
	t.Logf("LoadDir: an unreadable (0000) index.json that IsFile accepts surfaces a read error")
}

// TestLoadDirMmapIndexReadError covers LoadDirMmap's identical index-read-failure branch
// (sharded.go ~99): the mmap sibling must also reject a present-but-unreadable index rather
// than fall through to the single model.safetensors path.
func TestLoadDirMmapIndexReadError(t *testing.T) {
	dir := t.TempDir()
	chmodUnreadable(t, dir, indexName, `{"weight_map":{"a":"s.safetensors"}}`)
	if _, err := LoadDirMmap(dir); err == nil {
		t.Fatal("LoadDirMmap with an unreadable index.json: expected a read error")
	}
	t.Logf("LoadDirMmap: an unreadable (0000) index.json that IsFile accepts surfaces a read error")
}
