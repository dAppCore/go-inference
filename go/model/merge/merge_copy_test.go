// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	core "dappco.re/go"
)

func TestMergeCopy_SamePath_Good(t *core.T) {
	a := t.TempDir()
	core.AssertTrue(t, SamePath(a, a))
}

func TestMergeCopy_SamePath_Bad(t *core.T) {
	core.AssertFalse(t, SamePath(t.TempDir(), t.TempDir()))
}

func TestMergeCopy_SamePath_Ugly(t *core.T) {
	abs := t.TempDir()
	core.AssertTrue(t, SamePath(core.PathJoin(abs, ".", "sub", ".."), abs))
}

func TestMergeCopy_SamePathResolved_Good(t *core.T) {
	abs := t.TempDir()
	core.AssertTrue(t, SamePathResolved(abs, abs))
}

func TestMergeCopy_SamePathResolved_Bad(t *core.T) {
	core.AssertFalse(t, SamePathResolved(t.TempDir(), t.TempDir()))
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

func TestMergeCopy_CopyModelPackMetadata_Bad(t *core.T) {
	dst := t.TempDir()
	core.RequireNoError(t, CopyModelPackMetadata(core.PathJoin(t.TempDir(), "does-not-exist"), dst))
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
	_, err := HashFile(core.PathJoin(t.TempDir(), "missing.json"))
	core.AssertError(t, err)
}

func TestMergeCopy_HashFile_Ugly(t *core.T) {
	path := core.PathJoin(t.TempDir(), "empty.json")
	requireResultOK(t, core.WriteFile(path, nil, 0o644))
	hash, err := HashFile(path)
	core.RequireNoError(t, err)
	core.AssertEqual(t, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", hash)
}

func TestMergeCopy_HasModelPackMetadataSuffix_Good(t *core.T) {
	core.AssertTrue(t, hasModelPackMetadataSuffix("config.json"))
}

func TestMergeCopy_HasModelPackMetadataSuffix_Bad(t *core.T) {
	core.AssertFalse(t, hasModelPackMetadataSuffix("Config.JSON"))
}

func TestMergeCopy_HasModelPackMetadataSuffix_Ugly(t *core.T) {
	core.AssertFalse(t, hasModelPackMetadataSuffix("model.safetensors"))
}

func TestMergeCopy_IsModelWeightMetadataCopySkip_Good(t *core.T) {
	core.AssertTrue(t, isModelWeightMetadataCopySkip("adapter_provenance.json"))
}

func TestMergeCopy_IsModelWeightMetadataCopySkip_Bad(t *core.T) {
	core.AssertTrue(t, isModelWeightMetadataCopySkip("model.safetensors.index.json"))
}

func TestMergeCopy_IsModelWeightMetadataCopySkip_Ugly(t *core.T) {
	core.AssertFalse(t, isModelWeightMetadataCopySkip("tokenizer_config.json"))
}
