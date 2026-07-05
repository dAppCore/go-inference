package modelmgmt

import (
	"syscall"

	"dappco.re/go"
	coreio "dappco.re/go/io"
)

func installConsolidateScripts(t *core.T, sshBody, scpBody string) {
	t.Helper()
	dir := t.TempDir()
	ssh := core.JoinPath(dir, "ssh")
	scp := core.JoinPath(dir, "scp")
	core.RequireNoError(t, coreio.Local.Write(ssh, sshBody))
	core.RequireNoError(t, coreio.Local.Write(scp, scpBody))
	core.RequireNoError(t, syscall.Chmod(ssh, 0o755))
	core.RequireNoError(t, syscall.Chmod(scp, 0o755))
	t.Setenv("PATH", core.Concat(dir, ":", core.Env("PATH")))
}

func TestConsolidate_Consolidate_Good(t *core.T) {
	installConsolidateScripts(t, "#!/bin/sh\nprintf '/remote/a.jsonl\\n/remote/b.jsonl\\n'\n", "#!/bin/sh\nprintf '{\"idx\":2}\\n{\"idx\":1}\\n' > \"$2\"\n")
	out := core.JoinPath(t.TempDir(), "merged.jsonl")
	err := Consolidate(ConsolidateConfig{M3Host: "m3", RemoteDir: "/remote", Pattern: "*.jsonl", OutputDir: t.TempDir(), MergedOut: out}, core.NewBuffer(nil))
	requireResultOK(t, err)
	data, readErr := coreio.Local.Read(out)
	core.RequireNoError(t, readErr)
	core.AssertContains(t, data, `"idx":1`)
}

func TestConsolidate_Consolidate_Bad(t *core.T) {
	installConsolidateScripts(t, "#!/bin/sh\nexit 1\n", "#!/bin/sh\nexit 1\n")
	err := Consolidate(ConsolidateConfig{M3Host: "m3", RemoteDir: "/remote", Pattern: "*.jsonl", OutputDir: t.TempDir()}, core.NewBuffer(nil))
	assertResultError(t, err)
}

func TestConsolidate_Consolidate_Ugly(t *core.T) {
	installConsolidateScripts(t, "#!/bin/sh\nprintf '\\n'\n", "#!/bin/sh\nexit 0\n")
	out := core.JoinPath(t.TempDir(), "empty.jsonl")
	err := Consolidate(ConsolidateConfig{M3Host: "m3", RemoteDir: "/remote", Pattern: "*.jsonl", OutputDir: t.TempDir(), MergedOut: out}, core.NewBuffer(nil))
	requireResultOK(t, err)
	data, readErr := coreio.Local.Read(out)
	core.RequireNoError(t, readErr)
	core.AssertEqual(t, "", data)
}
