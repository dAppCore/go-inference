// SPDX-Licence-Identifier: EUPL-1.2

package merge

import (
	core "dappco.re/go"
)

func ExampleSamePath() {
	dir, cleanup, ok := exampleMergeCopyTempDir()
	if !ok {
		return
	}
	defer cleanup()

	core.Println(SamePath(dir, dir), SamePath(dir, core.PathJoin(dir, "..")))
	// Output: true false
}

func ExampleSamePathResolved() {
	dir, cleanup, ok := exampleMergeCopyTempDir()
	if !ok {
		return
	}
	defer cleanup()

	absDir, ok := core.PathAbs(dir).Value.(string)
	if !ok {
		return
	}
	core.Println(SamePathResolved(dir, absDir))
	// Output: true
}

func ExampleCopyModelPackMetadata() {
	src, cleanupSrc, ok := exampleMergeCopyTempDir()
	if !ok {
		return
	}
	defer cleanupSrc()
	dst, cleanupDst, ok := exampleMergeCopyTempDir()
	if !ok {
		return
	}
	defer cleanupDst()

	if result := core.WriteFile(core.PathJoin(src, "config.json"), []byte(`{"model_type":"test"}`), 0o644); !result.OK {
		return
	}
	if result := core.WriteFile(core.PathJoin(src, "model.safetensors"), []byte("weights"), 0o644); !result.OK {
		return
	}

	if err := CopyModelPackMetadata(src, dst); err != nil {
		core.Println(err)
		return
	}
	configCopied := coreFileExists(core.PathJoin(dst, "config.json"))
	weightsCopied := coreFileExists(core.PathJoin(dst, "model.safetensors"))
	core.Println(configCopied, weightsCopied)
	// Output: true false
}

func ExampleHashFile() {
	dir, cleanup, ok := exampleMergeCopyTempDir()
	if !ok {
		return
	}
	defer cleanup()
	path := core.PathJoin(dir, "tokenizer.json")
	if result := core.WriteFile(path, []byte("hello"), 0o644); !result.OK {
		return
	}

	hash, err := HashFile(path)
	core.Println(err == nil, hash)
	// Output: true 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824
}

// exampleMergeCopyTempDir creates a scratch directory for an Example test
// (which has no *testing.T to call t.TempDir() with) and returns a cleanup
// func the caller must defer.
func exampleMergeCopyTempDir() (string, func(), bool) {
	dirResult := core.MkdirTemp("", "go-inference-merge-copy-example-*")
	if !dirResult.OK {
		return "", func() {}, false
	}
	dir := dirResult.Value.(string)
	return dir, func() { core.RemoveAll(dir) }, true
}
