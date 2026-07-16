// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	core "dappco.re/go"
)

// TestMergeCopy_SamePath_Good documents the reflexive case plus trailing-
// slash normalisation: both resolve through core.PathAbs to the same clean
// absolute path.
func TestMergeCopy_SamePath_Good(t *core.T) {
	a := t.TempDir()
	core.AssertTrue(t, SamePath(a, a))
	core.AssertTrue(t, SamePath(a, a+"/"))
}

// TestMergeCopy_SamePath_Bad covers two unrelated directories plus the
// prefix gotcha: a parent directory is never "the same path" as its child,
// even though the child's string carries the parent's as a prefix.
func TestMergeCopy_SamePath_Bad(t *core.T) {
	core.AssertFalse(t, SamePath(t.TempDir(), t.TempDir()))
	parent := t.TempDir()
	child := core.PathJoin(parent, "sub")
	core.AssertFalse(t, SamePath(parent, child))
}

// TestMergeCopy_SamePath_Ugly covers dot-segment traversal at two depths —
// both must resolve to the same clean absolute path as abs.
func TestMergeCopy_SamePath_Ugly(t *core.T) {
	abs := t.TempDir()
	core.AssertTrue(t, SamePath(core.PathJoin(abs, ".", "sub", ".."), abs))
	core.AssertTrue(t, SamePath(core.PathJoin(abs, "a", "..", "b", ".."), abs))
}

// TestMergeCopy_SamePathResolved_Good documents the reflexive case plus
// trailing-slash normalisation on the left-hand (resolved) side.
func TestMergeCopy_SamePathResolved_Good(t *core.T) {
	abs := t.TempDir()
	core.AssertTrue(t, SamePathResolved(abs, abs))
	core.AssertTrue(t, SamePathResolved(abs+"/", abs))
}

// TestMergeCopy_SamePathResolved_Bad covers two unrelated directories plus
// the parent/child prefix gotcha, mirroring SamePath_Bad for the resolved
// (right-hand-side-already-absolute) variant.
func TestMergeCopy_SamePathResolved_Bad(t *core.T) {
	core.AssertFalse(t, SamePathResolved(t.TempDir(), t.TempDir()))
	parent := t.TempDir()
	core.AssertFalse(t, SamePathResolved(parent, core.PathJoin(parent, "sub")))
}

func TestMergeCopy_SamePathResolved_Ugly(t *core.T) {
	// a carries dot-segments and must still be resolved via core.PathAbs
	// even though absB is assumed already absolute — SamePathResolved only
	// skips the resolution work on the right-hand side.
	abs := t.TempDir()
	rel := core.PathJoin(abs, ".", "sub", "..")
	core.AssertTrue(t, SamePathResolved(rel, abs))
}

func TestMergeCopy_CopyModelPackMetadata_Good(t *core.T) {
	src := t.TempDir()
	dst := t.TempDir()
	requireResultOK(t, core.WriteFile(core.PathJoin(src, "config.json"), []byte(`{}`), 0o644))
	requireResultOK(t, core.WriteFile(core.PathJoin(src, "tokenizer.model"), []byte("tok"), 0o644))
	requireResultOK(t, core.WriteFile(core.PathJoin(src, "model.safetensors"), []byte("weights"), 0o644))

	core.RequireNoError(t, CopyModelPackMetadata(src, dst))
	core.AssertTrue(t, coreFileExists(core.PathJoin(dst, "config.json")))
	core.AssertTrue(t, coreFileExists(core.PathJoin(dst, "tokenizer.model")))
	core.AssertFalse(t, coreFileExists(core.PathJoin(dst, "model.safetensors")))
}

// TestMergeCopy_CopyModelPackMetadata_Bad documents that a missing source
// directory is not fatal (per the CopyModelPackMetadata doc comment) and,
// critically, that nothing gets written to dst as a result.
func TestMergeCopy_CopyModelPackMetadata_Bad(t *core.T) {
	dst := t.TempDir()
	core.RequireNoError(t, CopyModelPackMetadata(core.PathJoin(t.TempDir(), "does-not-exist"), dst))
	listed := core.ReadDir(core.DirFS(dst), ".")
	requireResultOK(t, listed)
	core.AssertLen(t, listed.Value.([]core.FsDirEntry), 0)
}

func TestMergeCopy_CopyModelPackMetadata_Ugly(t *core.T) {
	src := t.TempDir()
	dst := t.TempDir()
	requireResultOK(t, core.WriteFile(core.PathJoin(src, "adapter_provenance.json"), []byte(`{}`), 0o644))

	core.RequireNoError(t, CopyModelPackMetadata(src, dst))
	core.AssertFalse(t, coreFileExists(core.PathJoin(dst, "adapter_provenance.json")))
}

func TestMergeCopy_HashFile_Good(t *core.T) {
	path := core.PathJoin(t.TempDir(), "tokenizer.json")
	requireResultOK(t, core.WriteFile(path, []byte("hello"), 0o644))
	hash, err := HashFile(path)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824", hash)
}

func TestMergeCopy_HashFile_Bad(t *core.T) {
	hash, err := HashFile(core.PathJoin(t.TempDir(), "missing.json"))
	core.AssertError(t, err)
	core.AssertEqual(t, "", hash)
}

func TestMergeCopy_HashFile_Ugly(t *core.T) {
	path := core.PathJoin(t.TempDir(), "empty.json")
	requireResultOK(t, core.WriteFile(path, nil, 0o644))
	hash, err := HashFile(path)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", hash)
}

// TestMergeCopy_HasModelPackMetadataSuffix_Good covers all three metadata
// extensions the copier recognises (.json, .model, .txt).
func TestMergeCopy_HasModelPackMetadataSuffix_Good(t *core.T) {
	core.AssertTrue(t, hasModelPackMetadataSuffix("config.json"))
	core.AssertTrue(t, hasModelPackMetadataSuffix("tokenizer.model"))
	core.AssertTrue(t, hasModelPackMetadataSuffix("special_tokens.txt"))
}

// TestMergeCopy_HasModelPackMetadataSuffix_Bad documents that suffix
// matching is case-sensitive on purpose (mirrors historical filepath.Glob
// behaviour) — an upper-cased extension is not recognised.
func TestMergeCopy_HasModelPackMetadataSuffix_Bad(t *core.T) {
	core.AssertFalse(t, hasModelPackMetadataSuffix("Config.JSON"))
	core.AssertFalse(t, hasModelPackMetadataSuffix("TOKENIZER.MODEL"))
	core.AssertFalse(t, hasModelPackMetadataSuffix("NOTES.TXT"))
}

// TestMergeCopy_HasModelPackMetadataSuffix_Ugly covers a recognised-but-
// wrong extension (safetensors and gguf) and no extension at all.
func TestMergeCopy_HasModelPackMetadataSuffix_Ugly(t *core.T) {
	core.AssertFalse(t, hasModelPackMetadataSuffix("model.safetensors"))
	core.AssertFalse(t, hasModelPackMetadataSuffix("README"))
	core.AssertFalse(t, hasModelPackMetadataSuffix("model.gguf"))
}

// TestMergeCopy_IsModelWeightMetadataCopySkip_Good covers the adapter-
// provenance skip, case-folded (equalFold-backed) at three casings.
func TestMergeCopy_IsModelWeightMetadataCopySkip_Good(t *core.T) {
	core.AssertTrue(t, isModelWeightMetadataCopySkip("adapter_provenance.json"))
	core.AssertTrue(t, isModelWeightMetadataCopySkip("Adapter_Provenance.JSON"))
	core.AssertTrue(t, isModelWeightMetadataCopySkip("ADAPTER_PROVENANCE.JSON"))
}

// TestMergeCopy_IsModelWeightMetadataCopySkip_Bad covers both weight-layout
// skip branches: a safetensors shard-index file, a gguf sidecar, and a
// plain adapter safetensors weight file.
func TestMergeCopy_IsModelWeightMetadataCopySkip_Bad(t *core.T) {
	core.AssertTrue(t, isModelWeightMetadataCopySkip("model.safetensors.index.json"))
	core.AssertTrue(t, isModelWeightMetadataCopySkip("model.gguf.json"))
	core.AssertTrue(t, isModelWeightMetadataCopySkip("adapter_model.safetensors"))
}

func TestMergeCopy_IsModelWeightMetadataCopySkip_Ugly(t *core.T) {
	core.AssertFalse(t, isModelWeightMetadataCopySkip("tokenizer_config.json"))
	core.AssertFalse(t, isModelWeightMetadataCopySkip("config.json"))
	core.AssertFalse(t, isModelWeightMetadataCopySkip("special_tokens_map.json"))
}
