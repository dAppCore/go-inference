// SPDX-Licence-Identifier: EUPL-1.2

package serving

import (
	"testing"
	"time"

	core "dappco.re/go"
)

// writeConfig writes JSON to a temp file and returns its path.
func writeConfig(t *testing.T, json string) string {
	t.Helper()
	p := core.PathJoin(t.TempDir(), "models.json")
	if r := core.WriteFile(p, []byte(json), 0o644); !r.OK {
		t.Fatalf("write config: %v", r.Value)
	}
	return p
}

// TestLoadModelsConfig_Good proves a full config parses into specs + options,
// including aliases, pinning, a memory ceiling, an idle TTL, and a named profile.
func TestLoadModelsConfig_Good(t *testing.T) {
	path := writeConfig(t, `{
      "memory_ceiling_bytes": 1073741824,
      "idle_ttl": "10m",
      "sweep_interval": "30s",
      "models": [
        {"id": "qwen3", "path": "/m/qwen3", "aliases": ["qwen"],
         "profiles": {"creative": {"temperature": 0.9, "max_tokens": 512}}},
        {"id": "bge", "path": "/m/bge", "pinned": true, "est_bytes": 200}
      ]
    }`)

	specs, opts, err := LoadModelsConfig(path)
	if err != nil {
		t.Fatalf("LoadModelsConfig: %v", err)
	}
	if len(specs) != 2 {
		t.Fatalf("specs len = %d, want 2", len(specs))
	}
	if opts.MemoryCeiling != 1073741824 || opts.IdleTTL != 10*time.Minute || opts.SweepInterval != 30*time.Second {
		t.Fatalf("opts = %+v, want ceiling 1GiB / ttl 10m / sweep 30s", opts)
	}
	if specs[0].ID != "qwen3" || len(specs[0].Aliases) != 1 || specs[0].Aliases[0] != "qwen" {
		t.Fatalf("qwen3 spec = %+v, want id qwen3 + alias qwen", specs[0])
	}
	prof, ok := specs[0].Profiles["creative"]
	if !ok || prof.Temperature == nil || *prof.Temperature != 0.9 || prof.MaxTokens == nil || *prof.MaxTokens != 512 {
		t.Fatalf("creative profile = %+v, want temperature 0.9 + max_tokens 512", prof)
	}
	if !specs[1].Pinned || specs[1].EstBytes != 200 {
		t.Fatalf("bge spec = %+v, want pinned + est_bytes 200", specs[1])
	}
}

// TestLoadModelsConfig_EmptyDurations_Good proves omitted durations parse to 0
// (the resolver's "off/auto" value), not an error.
func TestLoadModelsConfig_EmptyDurations_Good(t *testing.T) {
	path := writeConfig(t, `{"models": [{"id": "a", "path": "/m/a"}]}`)
	_, opts, err := LoadModelsConfig(path)
	if err != nil {
		t.Fatalf("LoadModelsConfig: %v", err)
	}
	if opts.IdleTTL != 0 || opts.SweepInterval != 0 {
		t.Fatalf("empty durations = %+v, want 0/0", opts)
	}
}

// TestLoadModelsConfig_NoModels_Bad proves a config with no models is refused —
// a malformed multi-model config must fail at boot.
func TestLoadModelsConfig_NoModels_Bad(t *testing.T) {
	path := writeConfig(t, `{"memory_ceiling_bytes": 100}`)
	if _, _, err := LoadModelsConfig(path); err == nil {
		t.Fatal("a config with no models should error")
	}
}

// TestLoadModelsConfig_ModelNoPath_Bad proves a model entry missing its path is
// refused.
func TestLoadModelsConfig_ModelNoPath_Bad(t *testing.T) {
	path := writeConfig(t, `{"models": [{"id": "a"}]}`)
	if _, _, err := LoadModelsConfig(path); err == nil {
		t.Fatal("a model with no path should error")
	}
}

// TestLoadModelsConfig_BadDuration_Bad proves an unparseable idle_ttl is refused.
func TestLoadModelsConfig_BadDuration_Bad(t *testing.T) {
	path := writeConfig(t, `{"idle_ttl": "not-a-duration", "models": [{"id": "a", "path": "/m/a"}]}`)
	if _, _, err := LoadModelsConfig(path); err == nil {
		t.Fatal("an unparseable idle_ttl should error")
	}
}

// TestLoadModelsConfig_MissingFile_Bad proves a missing config file errors rather
// than serving an empty registry.
func TestLoadModelsConfig_MissingFile_Bad(t *testing.T) {
	if _, _, err := LoadModelsConfig("/no/such/models/config.json"); err == nil {
		t.Fatal("a missing config file should error")
	}
}
