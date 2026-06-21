// SPDX-Licence-Identifier: EUPL-1.2

package driver

import (
	"testing"

	core "dappco.re/go"
)

func TestCanonicalRepoDir_Good(t *testing.T) {
	if got := CanonicalRepoDir("mlx-community/gemma-4-e2b-it-4bit"); got != "mlx-community__gemma-4-e2b-it-4bit" {
		t.Fatalf("CanonicalRepoDir = %q, want the engine's org__name form", got)
	}
}

func TestAllowRepo_CreatesAppendsAndIdempotent_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())

	if r := AllowRepo("mlx-community/gemma-3-1b-it-4bit"); !r.OK {
		t.Fatalf("AllowRepo(first) failed: %v", r.Value)
	}
	if r := AllowRepo("openai/gpt-oss-20b"); !r.OK {
		t.Fatalf("AllowRepo(second) failed: %v", r.Value)
	}
	// Idempotent — re-allowing must not duplicate.
	r := AllowRepo("openai/gpt-oss-20b")
	if !r.OK {
		t.Fatalf("AllowRepo(repeat) failed: %v", r.Value)
	}
	allowed := r.Value.([]string)
	if len(allowed) != 2 || allowed[0] != "mlx-community/gemma-3-1b-it-4bit" || allowed[1] != "openai/gpt-oss-20b" {
		t.Fatalf("allowlist = %v, want both repos exactly once", allowed)
	}

	// The file is the engine's exact shape: {"repos": [...]}.
	data := core.ReadFile(allowedModelsPath())
	if !data.OK {
		t.Fatal("allowed-models.json not written")
	}
	var onDisk allowedModelsFile
	if r := core.JSONUnmarshal(data.Value.([]byte), &onDisk); !r.OK || len(onDisk.Repos) != 2 {
		t.Fatalf("on-disk allowlist = %v (parse ok=%t), want 2 repos under the engine's key", onDisk.Repos, r.OK)
	}
}

func TestAllowRepo_PreservesExistingEngineFile_Good(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	path := allowedModelsPath()
	if r := core.MkdirAll(core.PathDir(path), 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	seed := `{"repos":["lthn/LEM-Gemma3-1B","openai/gpt-oss-20b"]}`
	if r := core.WriteFile(path, []byte(seed), 0o600); !r.OK {
		t.Fatalf("seed: %v", r.Value)
	}

	r := AllowRepo("mlx-community/gemma-3-1b-it-4bit")
	if !r.OK {
		t.Fatalf("AllowRepo over real engine file failed: %v", r.Value)
	}
	repos := r.Value.([]string)
	if len(repos) != 3 || repos[0] != "lthn/LEM-Gemma3-1B" || repos[2] != "mlx-community/gemma-3-1b-it-4bit" {
		t.Fatalf("repos = %v, want existing entries preserved + new appended", repos)
	}
}

func TestAllowRepo_EmptyRepo_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	if r := AllowRepo("  "); r.OK {
		t.Fatal("AllowRepo(blank) succeeded, want refusal")
	}
}

func TestAllowRepo_CorruptFile_Ugly(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	path := allowedModelsPath()
	if r := core.MkdirAll(core.PathDir(path), 0o755); !r.OK {
		t.Fatalf("mkdir: %v", r.Value)
	}
	if r := core.WriteFile(path, []byte(`not json at all`), 0o600); !r.OK {
		t.Fatalf("seed corrupt file: %v", r.Value)
	}
	// A corrupt allowlist must refuse loudly, never silently overwrite the
	// operator's file.
	if r := AllowRepo("mlx-community/gemma-3-1b-it-4bit"); r.OK {
		t.Fatal("AllowRepo over corrupt file succeeded, want refusal")
	}
}

func TestAdminCalls_RequireRunningEngine_Bad(t *testing.T) {
	t.Setenv("HOME", t.TempDir())
	svc := &Service{}
	if r := svc.DownloadModel(RuntimeMLX, "org/repo", "main"); r.OK {
		t.Fatal("DownloadModel with no running engine succeeded, want refusal")
	}
	if r := svc.DownloadJobStatus(RuntimeMLX, "job-1"); r.OK {
		t.Fatal("DownloadJobStatus with no running engine succeeded, want refusal")
	}
}
