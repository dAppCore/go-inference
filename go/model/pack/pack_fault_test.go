// SPDX-Licence-Identifier: EUPL-1.2

package pack_test

import (
	"archive/tar"
	"bytes"
	"testing"

	core "dappco.re/go"
	"dappco.re/go/inference/model/pack"

	"forge.lthn.ai/Snider/Enchantrix/pkg/trix"
)

// writeMaliciousContainer encodes a .model Trix container whose payload tar
// contains a single regular-file entry named entryName. Used to feed
// Unpack a hostile archive without going through Pack (which only ever
// produces safe, relative entry names).
func writeMaliciousContainer(t *testing.T, dest, entryName string) {
	t.Helper()
	var tarBuf bytes.Buffer
	tw := tar.NewWriter(&tarBuf)
	body := []byte("pwned")
	if err := tw.WriteHeader(&tar.Header{
		Name:     entryName,
		Typeflag: tar.TypeReg,
		Mode:     0o644,
		Size:     int64(len(body)),
	}); err != nil {
		t.Fatalf("tar WriteHeader: %v", err)
	}
	if _, err := tw.Write(body); err != nil {
		t.Fatalf("tar Write: %v", err)
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("tar Close: %v", err)
	}

	encoded, err := trix.Encode(&trix.Trix{Header: map[string]any{}, Payload: tarBuf.Bytes()}, pack.Magic, nil)
	if err != nil {
		t.Fatalf("trix.Encode: %v", err)
	}
	if wr := core.WriteFile(dest, encoded, 0o644); !wr.OK {
		t.Fatalf("write container: %v", wr.Value)
	}
}

// TestPack_Unpack_TraversalEntry_Ugly proves the tar-slip guard: a container
// whose payload names a "../escape" entry is rejected by extractTar via
// safeRelPath, and nothing is written outside destDir.
func TestPack_Unpack_TraversalEntry_Ugly(t *testing.T) {
	tempRoot := t.TempDir()
	dest := core.JoinPath(tempRoot, "hostile.model")
	outDir := core.JoinPath(tempRoot, "out")

	writeMaliciousContainer(t, dest, "../escape.txt")

	if r := pack.Unpack(dest, outDir, pack.UnpackOptions{}); r.OK {
		t.Fatalf("Unpack(traversal entry): want failure, got OK")
	}
	// The escape target (sibling of outDir) must not have been created.
	escaped := core.JoinPath(tempRoot, "escape.txt")
	if (&core.Fs{}).NewUnrestricted().Exists(escaped).OK {
		t.Fatalf("traversal escaped to %q", escaped)
	}
}

// TestPack_Unpack_AbsolutePathEntry_Ugly rejects an absolute-path tar entry —
// the other half of the tar-slip guard in safeRelPath.
func TestPack_Unpack_AbsolutePathEntry_Ugly(t *testing.T) {
	tempRoot := t.TempDir()
	dest := core.JoinPath(tempRoot, "hostile.model")
	outDir := core.JoinPath(tempRoot, "out")

	writeMaliciousContainer(t, dest, "/absolute/escape.txt")

	if r := pack.Unpack(dest, outDir, pack.UnpackOptions{}); r.OK {
		t.Fatalf("Unpack(absolute entry): want failure, got OK")
	}
}

// TestPack_Unpack_MissingSource_Bad — Unpack of a src path that does not exist
// fails at the ReadFile stage.
func TestPack_Unpack_MissingSource_Bad(t *testing.T) {
	tempRoot := t.TempDir()
	if r := pack.Unpack(core.JoinPath(tempRoot, "nope.model"), core.JoinPath(tempRoot, "out"), pack.UnpackOptions{}); r.OK {
		t.Fatalf("Unpack(missing source): want failure, got OK")
	}
}

// TestPack_Unpack_CorruptContainer_Bad — bytes that are not a valid Trix
// container fail at trix.Decode.
func TestPack_Unpack_CorruptContainer_Bad(t *testing.T) {
	tempRoot := t.TempDir()
	dest := core.JoinPath(tempRoot, "corrupt.model")
	if wr := core.WriteFile(dest, []byte("not a trix container at all"), 0o644); !wr.OK {
		t.Fatalf("write corrupt: %v", wr.Value)
	}
	if r := pack.Unpack(dest, core.JoinPath(tempRoot, "out"), pack.UnpackOptions{}); r.OK {
		t.Fatalf("Unpack(corrupt container): want failure, got OK")
	}
}

// TestPack_Unpack_DestNotEmpty_Bad exercises assertDestDirWritable: a
// non-empty destDir is refused unless Overwrite is set, and accepted when it
// is. Uses a real round-trip container so the extract itself is valid.
func TestPack_Unpack_DestNotEmpty_Bad(t *testing.T) {
	tempRoot := t.TempDir()
	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")
	outDir := core.JoinPath(tempRoot, "out")

	buildFixturePack(t, srcDir)
	if r := pack.Pack(srcDir, dest, pack.PackOptions{Manifest: sampleManifest()}); !r.OK {
		t.Fatalf("Pack: %v", r.Value)
	}

	// First unpack populates outDir.
	if r := pack.Unpack(dest, outDir, pack.UnpackOptions{}); !r.OK {
		t.Fatalf("first Unpack: %v", r.Value)
	}
	// Second unpack into the now-non-empty dir without Overwrite must fail.
	if r := pack.Unpack(dest, outDir, pack.UnpackOptions{}); r.OK {
		t.Fatalf("Unpack(non-empty dest, no overwrite): want failure, got OK")
	}
	// With Overwrite it must succeed.
	if r := pack.Unpack(dest, outDir, pack.UnpackOptions{Overwrite: true}); !r.OK {
		t.Fatalf("Unpack(non-empty dest, overwrite): %v", r.Value)
	}
}

// TestPack_Unpack_DestIsFile_Bad — a destDir path that exists but is a regular
// file (not a directory) is refused by assertDestDirWritable.
func TestPack_Unpack_DestIsFile_Bad(t *testing.T) {
	tempRoot := t.TempDir()
	srcDir := core.JoinPath(tempRoot, "src")
	dest := core.JoinPath(tempRoot, "out.model")
	destFile := core.JoinPath(tempRoot, "collision")

	buildFixturePack(t, srcDir)
	if r := pack.Pack(srcDir, dest, pack.PackOptions{Manifest: sampleManifest()}); !r.OK {
		t.Fatalf("Pack: %v", r.Value)
	}
	if wr := core.WriteFile(destFile, []byte("i am a file"), 0o644); !wr.OK {
		t.Fatalf("write collision file: %v", wr.Value)
	}
	if r := pack.Unpack(dest, destFile, pack.UnpackOptions{}); r.OK {
		t.Fatalf("Unpack(dest is a file): want failure, got OK")
	}
}
