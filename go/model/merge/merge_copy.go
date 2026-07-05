// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	"crypto/sha256"

	core "dappco.re/go"
)

// SamePath reports whether a and b resolve to the same absolute path —
// the guard engines use to refuse merging a pack onto itself. Exported
// so drivers share one resolution rule instead of private copies.
//
//	if merge.SamePath(source.Root, outputRoot) { /* refuse */ }
func SamePath(a, b string) bool { return samePath(a, b) }

// SamePathResolved is the per-source-loop variant of SamePath where the
// right-hand side is already absolute — saves a resolution per iteration.
//
//	absOut, _ := core.PathAbs(outputRoot).Value.(string)
//	for _, s := range sources { if merge.SamePathResolved(s.Root, absOut) { /* refuse */ } }
func SamePathResolved(a, absB string) bool { return samePathResolved(a, absB) }

// CopyModelPackMetadata copies sourceRoot's metadata sidecars
// (config.json, tokenizer files, chat templates — the .json/.model/.txt
// set, minus weight-layout files and prior-merge provenance) into
// outputRoot. A missing or unreadable source directory is not fatal:
// the merge still produces valid weights without sibling metadata.
//
//	if err := merge.CopyModelPackMetadata(primary.Root, outputRoot); err != nil { return err }
func CopyModelPackMetadata(sourceRoot, outputRoot string) error {
	return copyModelPackMetadata(sourceRoot, outputRoot)
}

// HashFile streams path through SHA-256 and returns the hex digest —
// the tokenizer-compatibility check engines run once per source pack.
// Streamed, not whole-file-read: BPE merge tables run to several MB.
//
//	digest, err := merge.HashFile(core.PathJoin(root, "tokenizer.json"))
func HashFile(path string) (string, error) { return hashFile(path) }

// samePath reports whether a and b resolve to the same absolute path.
func samePath(a, b string) bool {
	absA := a
	if resolved := core.PathAbs(a); resolved.OK {
		absA = resolved.Value.(string)
	}
	absB := b
	if resolved := core.PathAbs(b); resolved.OK {
		absB = resolved.Value.(string)
	}
	return absA == absB
}

// samePathResolved is the per-source-loop variant where the right-hand side
// is already absolute — saves a core.PathAbs call per iteration.
func samePathResolved(a, absB string) bool {
	absA := a
	if resolved := core.PathAbs(a); resolved.OK {
		absA = resolved.Value.(string)
	}
	return absA == absB
}

// modelPackMetadataSuffixes is the canonical metadata-extension list.
// Matching is case-sensitive to mirror historical filepath.Glob("*.json"/
// "*.model"/"*.txt") behaviour (Config.JSON is not copied).
var modelPackMetadataSuffixes = [...]string{".json", ".model", ".txt"}

// copyModelPackMetadata copies sourceRoot's metadata files (config.json,
// tokenizer files, chat templates — anything matching
// modelPackMetadataSuffixes except safetensors/gguf-named siblings) into
// outputRoot. A missing/unreadable source directory is not fatal — the
// merge still produces valid weights without sibling metadata.
func copyModelPackMetadata(sourceRoot, outputRoot string) error {
	listed := core.ReadDir(core.DirFS(sourceRoot), ".")
	if !listed.OK {
		return nil
	}
	entries, ok := listed.Value.([]core.FsDirEntry)
	if !ok {
		return nil
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if !hasModelPackMetadataSuffix(name) {
			continue
		}
		if isModelWeightMetadataCopySkip(name) {
			continue
		}
		if err := copyModelPackLocalFile(core.PathJoin(sourceRoot, name), core.PathJoin(outputRoot, name)); err != nil {
			return err
		}
	}
	return nil
}

// hasModelPackMetadataSuffix reports whether name carries a metadata
// extension. Case-sensitive on purpose (see modelPackMetadataSuffixes).
func hasModelPackMetadataSuffix(name string) bool {
	for _, suffix := range modelPackMetadataSuffixes {
		if core.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}

// isModelWeightMetadataCopySkip reports whether name should be excluded
// from the metadata copy — provenance from a prior merge/adapter step, or
// anything naming a weight file (e.g. a *.safetensors.index.json shard map,
// which belongs to the source pack's own weight layout, not the merged
// output's).
func isModelWeightMetadataCopySkip(name string) bool {
	if equalFold(name, "adapter_provenance.json") {
		return true
	}
	return containsFold(name, ".safetensors") || containsFold(name, ".gguf")
}

// copyModelPackLocalFile streams sourcePath to destinationPath instead of a
// whole-file read+write — tokenizer.json can run multiple MB on real
// checkpoints, and streaming keeps this at a fixed staging buffer
// regardless of file size.
func copyModelPackLocalFile(sourcePath, destinationPath string) error {
	srcOpen := core.Open(sourcePath)
	if !srcOpen.OK {
		return srcOpen.Err()
	}
	src := srcOpen.Value.(*core.OSFile)
	defer src.Close()
	dstOpen := core.OpenFile(destinationPath, core.O_WRONLY|core.O_CREATE|core.O_TRUNC, 0o644)
	if !dstOpen.OK {
		return dstOpen.Err()
	}
	dst := dstOpen.Value.(*core.OSFile)
	if result := core.Copy(dst, src); !result.OK {
		// The copy already failed; close the partial destination on a
		// best-effort basis and surface the copy error, not the close error.
		dst.Close()
		return result.Err()
	}
	if err := dst.Close(); err != nil {
		return err
	}
	return nil
}

// hashFile streams path through a SHA-256 hasher instead of reading it
// whole — tokenizer.json's BPE merge table can be several MB on real
// checkpoints, and validatePackCompatibility hashes one per source pack.
func hashFile(path string) (string, error) {
	open := core.Open(path)
	if !open.OK {
		return "", open.Err()
	}
	file := open.Value.(*core.OSFile)
	defer file.Close()
	hasher := sha256.New()
	if result := core.Copy(hasher, file); !result.OK {
		return "", result.Err()
	}
	return core.HexEncode(hasher.Sum(nil)), nil
}
