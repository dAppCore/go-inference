// SPDX-Licence-Identifier: EUPL-1.2

// CLI tests as artefact validation (AX-10): each run* verb is exercised
// directly with t.TempDir() fixtures — pack a tiny model via the same
// pack.Pack primitive the CLI wraps, then unpack/list/inspect it back and
// assert the round trip on disk and via the pack package's own readers.
//
// main() itself is only driven for its two branches that can never call
// os.Exit — a successful verb dispatch and the "--help" case — via a
// controlled os.Args swap. Its remaining branches (missing verb, unknown
// verb, failed run*) all terminate the process directly through os.Exit,
// which would kill the whole test binary; those are deliberately left
// uncovered rather than reached by any seam.
package main

import (
	"os"
	"os/exec"
	"strings"

	core "dappco.re/go"
	"dappco.re/go/inference/model/pack"
)

// swapArgs replaces os.Args for the duration of a test and returns a
// restore func. main() reads os.Args directly with no injectable seam, so
// this is the only way to drive it — safe here because callers only ever
// point it at main()'s two non-exiting branches.
func swapArgs(t *core.T, args ...string) func() {
	t.Helper()
	orig := os.Args
	os.Args = args
	return func() { os.Args = orig }
}

func TestMain_main_Good_Dispatch(t *core.T) {
	// A verb that resolves successfully falls through main()'s final
	// `if !r.OK` unexercised, so this reaches the switch's "list" case and
	// returns normally instead of calling os.Exit.
	root := t.TempDir()
	_, modelPath := buildFixtureModel(t, root)

	restore := swapArgs(t, "lthn-model-pack", "list", modelPath)
	defer restore()

	main()
}

func TestMain_main_Good_Help(t *core.T) {
	restore := swapArgs(t, "lthn-model-pack", "--help")
	defer restore()

	main()
}

// TestMain_main_Helper is the child-process entry point for the exit paths in
// main. Those paths deliberately call os.Exit, so they cannot be exercised in
// the parent test process.
func TestMain_main_Helper(t *core.T) {
	if os.Getenv("LTHN_MODEL_PACK_MAIN_HELPER") != "1" {
		return
	}
	args := os.Getenv("LTHN_MODEL_PACK_MAIN_ARGS")
	if args == "" {
		os.Args = []string{"lthn-model-pack"}
	} else {
		os.Args = append([]string{"lthn-model-pack"}, strings.Split(args, "\x1f")...)
	}
	main()
}

// runMainProcess runs main's process-terminating branches in an isolated test
// binary and returns the observable exit status and combined terminal output.
func runMainProcess(t *core.T, args ...string) (int, string) {
	t.Helper()
	cmd := exec.Command(os.Args[0], "-test.run=^TestMain_main_Helper$")
	cmd.Env = append(os.Environ(),
		"LTHN_MODEL_PACK_MAIN_HELPER=1",
		"LTHN_MODEL_PACK_MAIN_ARGS="+strings.Join(args, "\x1f"),
	)
	out, err := cmd.CombinedOutput()
	if err == nil {
		return 0, string(out)
	}
	exit, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("run main helper: %v; output=%s", err, out)
	}
	return exit.ExitCode(), string(out)
}

func TestMain_main_Bad_MissingVerb(t *core.T) {
	code, output := runMainProcess(t)
	core.AssertEqual(t, 2, code)
	core.AssertContains(t, output, "Usage:")
}

func TestMain_main_Bad_UnknownVerb(t *core.T) {
	code, output := runMainProcess(t, "unknown")
	core.AssertEqual(t, 2, code)
	core.AssertContains(t, output, `unknown verb "unknown"`)
	core.AssertContains(t, output, "Usage:")
}

func TestMain_main_Ugly_FailedVerb(t *core.T) {
	missing := core.JoinPath(t.TempDir(), "missing.model")
	code, output := runMainProcess(t, "list", missing)
	core.AssertEqual(t, 1, code)
	core.AssertContains(t, output, "lthn-model-pack:")
}

// buildFixtureSrcDir writes a small but realistic unpacked model pack dir —
// enough for pack.Pack (and therefore runPack) to have real content to tar.
func buildFixtureSrcDir(t *core.T, dir string) {
	t.Helper()
	core.RequireTrue(t, core.MkdirAll(dir, 0o755).OK)

	files := map[string]string{
		"config.json":       `{"model_type":"gemma","hidden_size":8,"num_hidden_layers":2}`,
		"tokenizer.json":    `{"version":"1.0","bos_token":"<bos>","eos_token":"<eos>"}`,
		"model.safetensors": "fixture-tensor-bytes",
	}
	for name, content := range files {
		path := core.JoinPath(dir, name)
		core.RequireTrue(t, core.WriteFile(path, []byte(content), 0o644).OK)
	}
}

// buildFixtureModel packs a fixture src dir via runPack itself (the CLI verb
// under test), returning both the source dir and the resulting .model path
// so callers can round-trip it through unpack/list/inspect.
func buildFixtureModel(t *core.T, root string) (srcDir, modelPath string) {
	t.Helper()
	srcDir = core.JoinPath(root, "src")
	modelPath = core.JoinPath(root, "out.model")
	buildFixtureSrcDir(t, srcDir)

	r := runPack([]string{
		"-arch", "gemma",
		"-quant", "4",
		"-source", "safetensors",
		"-producer", "fixture-producer",
		srcDir, modelPath,
	})
	core.RequireTrue(t, r.OK, core.Sprintf("fixture runPack: %v", r.Value))
	return srcDir, modelPath
}

// mustReadFile reads path via core.ReadFile, failing the test on error.
func mustReadFile(t *core.T, path string) []byte {
	t.Helper()
	rr := core.ReadFile(path)
	core.RequireTrue(t, rr.OK, core.Sprintf("ReadFile %q: %v", path, rr.Value))
	return rr.Value.([]byte)
}

func TestMain_runPack_Good(t *core.T) {
	root := t.TempDir()
	srcDir := core.JoinPath(root, "src")
	dest := core.JoinPath(root, "out.model")
	buildFixtureSrcDir(t, srcDir)

	r := runPack([]string{
		"-arch", "gemma",
		"-quant", "4",
		"-source", "safetensors",
		"-producer", "fixture-producer",
		srcDir, dest,
	})

	core.AssertTrue(t, r.OK)

	data := mustReadFile(t, dest)
	core.AssertEqual(t, pack.Magic, string(data[:len(pack.Magic)]))

	manifest, _, ir := pack.Inspect(dest)
	core.RequireTrue(t, ir.OK)
	core.AssertEqual(t, "gemma", manifest.Model.Architecture)
	core.AssertEqual(t, 4, manifest.Model.QuantBits)
	core.AssertEqual(t, "safetensors", manifest.SourceFormat)
	core.AssertEqual(t, "fixture-producer", manifest.Producer.Name)
}

func TestMain_runPack_Bad(t *core.T) {
	r := runPack([]string{"only-one-positional-arg"})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "expected: pack <srcDir> <out.model>")
}

func TestMain_runPack_Ugly(t *core.T) {
	root := t.TempDir()
	missingSrc := core.JoinPath(root, "does-not-exist")
	dest := core.JoinPath(root, "out.model")

	r := runPack([]string{missingSrc, dest})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "is not a directory")
}

func TestMain_runUnpack_Good(t *core.T) {
	root := t.TempDir()
	srcDir, modelPath := buildFixtureModel(t, root)
	destDir := core.JoinPath(root, "extracted")

	r := runUnpack([]string{modelPath, destDir})

	core.AssertTrue(t, r.OK)
	want := mustReadFile(t, core.JoinPath(srcDir, "config.json"))
	got := mustReadFile(t, core.JoinPath(destDir, "config.json"))
	core.AssertEqual(t, string(want), string(got))
}

func TestMain_runUnpack_Bad(t *core.T) {
	r := runUnpack([]string{"only-one-positional-arg"})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "expected: unpack <src.model> <destDir>")
}

func TestMain_runUnpack_Ugly(t *core.T) {
	// destDir pre-populated with a conflicting file: default (no -overwrite)
	// must refuse; -overwrite must then let the same unpack through.
	root := t.TempDir()
	_, modelPath := buildFixtureModel(t, root)
	destDir := core.JoinPath(root, "extracted")
	core.RequireTrue(t, core.MkdirAll(destDir, 0o755).OK)
	core.RequireTrue(t, core.WriteFile(core.JoinPath(destDir, "pre-existing.txt"), []byte("in the way"), 0o644).OK)

	blocked := runUnpack([]string{modelPath, destDir})
	core.AssertFalse(t, blocked.OK)
	core.AssertContains(t, blocked.Error(), "not empty")

	forced := runUnpack([]string{"-overwrite", modelPath, destDir})
	core.AssertTrue(t, forced.OK)
}

func TestMain_runList_Good(t *core.T) {
	root := t.TempDir()
	_, modelPath := buildFixtureModel(t, root)

	r := runList([]string{modelPath})
	core.AssertTrue(t, r.OK)

	// Cross-check against the same artefact via the package the verb wraps:
	// the tar entries runList reported success over really are there.
	entries, manifest, lr := pack.List(modelPath)
	core.RequireTrue(t, lr.OK)
	core.AssertEqual(t, "safetensors", manifest.SourceFormat)
	var names []string
	for _, e := range entries {
		names = append(names, e.Path)
	}
	core.AssertContains(t, names, "config.json")
}

func TestMain_runList_Bad(t *core.T) {
	r := runList([]string{"a", "b"})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "expected: list <src.model>")
}

func TestMain_runList_Ugly(t *core.T) {
	root := t.TempDir()
	missing := core.JoinPath(root, "nope.model")

	r := runList([]string{missing})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "no such file")
}

func TestMain_runInspect_Good(t *core.T) {
	root := t.TempDir()
	_, modelPath := buildFixtureModel(t, root)

	r := runInspect([]string{modelPath})
	core.AssertTrue(t, r.OK)

	manifest, inspection, ir := pack.Inspect(modelPath)
	core.RequireTrue(t, ir.OK)
	core.AssertEqual(t, "gemma", manifest.Model.Architecture)
	core.AssertEqual(t, modelPath, inspection.Path)
	core.AssertNotEmpty(t, pack.Fingerprint(*manifest))
}

func TestMain_runInspect_Bad(t *core.T) {
	r := runInspect(nil)
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertContains(t, got, "expected: inspect <src.model>")
}

func TestMain_runInspect_Ugly(t *core.T) {
	// Inspect only ever decodes the Trix header, never the payload tar, so
	// truncating the tail (as Unpack/List fixtures do) would not touch it.
	// Corrupting the magic bytes breaks trix.Decode itself.
	root := t.TempDir()
	_, modelPath := buildFixtureModel(t, root)

	full := mustReadFile(t, modelPath)
	corrupt := append([]byte(nil), full...)
	corrupt[0] ^= 0xFF
	core.RequireTrue(t, core.WriteFile(modelPath, corrupt, 0o644).OK)

	r := runInspect([]string{modelPath})
	got := r.Error()

	core.AssertFalse(t, r.OK)
	core.AssertNotEmpty(t, got)
}
