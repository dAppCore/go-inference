// SPDX-Licence-Identifier: EUPL-1.2

package enginegate

import (
	"testing"

	core "dappco.re/go"
)

// HFModelPath resolves a Hugging Face repo to its local snapshot directory in
// the standard hub cache (~/.cache/huggingface/hub/models--<org>--<name>/
// snapshots/<hash>), replacing the GO_MLX_*_MODEL env vars that used to point
// tests at a pack on disk — the model is named by the test, not injected by
// process env. A trailing "*" on repo prefix-matches (for families where the
// exact pack name varies). The test is skipped when the model is not cached, so
// a checkout without the weights stays green.
//
//	target := enginegate.HFModelPath(t, "mlx-community/gemma-4-e2b-it-6bit")
//	any := enginegate.HFModelPath(t, "mlx-community/Qwen3-Next*")
func HFModelPath(t testing.TB, repo string) string {
	t.Helper()
	home := core.UserHomeDir()
	if !home.OK {
		t.Skip("Hugging Face cache unavailable: no home directory")
		return ""
	}
	hub := core.PathJoin(home.Value.(string), ".cache", "huggingface", "hub")

	want := "models--" + repo
	if parts := core.SplitN(repo, "/", 2); len(parts) == 2 {
		want = "models--" + parts[0] + "--" + parts[1]
	}
	prefix := core.HasSuffix(want, "*")
	if prefix {
		want = core.TrimSuffix(want, "*")
	}

	read := core.ReadDir(core.DirFS(hub), ".")
	entries, ok := read.Value.([]core.FsDirEntry)
	if !read.OK || !ok {
		t.Skipf("no Hugging Face cache at %s", hub)
		return ""
	}
	for _, entry := range entries {
		name := entry.Name()
		if !entry.IsDir() || (name != want && !(prefix && core.HasPrefix(name, want))) {
			continue
		}
		snapshotsDir := core.PathJoin(hub, name, "snapshots")
		snaps := core.ReadDir(core.DirFS(snapshotsDir), ".")
		snapEntries, ok := snaps.Value.([]core.FsDirEntry)
		if !snaps.OK || !ok {
			continue
		}
		for _, snap := range snapEntries {
			if snap.IsDir() {
				return core.PathJoin(snapshotsDir, snap.Name())
			}
		}
	}
	t.Skipf("model %s not in the Hugging Face cache (%s) — pull it to run this test", repo, hub)
	return ""
}
