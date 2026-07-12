// SPDX-Licence-Identifier: EUPL-1.2

// serve_multimodel_config.go loads the declarative multi-model config a serve
// reads from a --models-config JSON file. Keeping the parse in the library (not
// cmd/lem) means the model-list + profile + budget business logic is tested here
// and cmd stays a thin flag-to-library shim.

package serving

import (
	"time"

	core "dappco.re/go"
)

// ModelsConfig is the on-disk shape of a --models-config file: the residency
// budget plus the list of models, each with aliases, pinning, and named
// generation profiles.
//
//	{
//	  "memory_ceiling_bytes": 17179869184,
//	  "idle_ttl": "10m",
//	  "models": [
//	    {"id": "qwen3", "path": "/abs/qwen3", "aliases": ["qwen"],
//	     "profiles": {"creative": {"temperature": 0.9, "top_p": 0.95, "max_tokens": 512}}},
//	    {"id": "bge", "path": "/abs/bge-small", "pinned": true}
//	  ]
//	}
type ModelsConfig struct {
	MemoryCeilingBytes uint64            `json:"memory_ceiling_bytes,omitempty"`
	IdleTTL            string            `json:"idle_ttl,omitempty"`       // Go duration string, e.g. "10m"; empty = never
	SweepInterval      string            `json:"sweep_interval,omitempty"` // Go duration string; empty = auto
	Models             []ModelSpecConfig `json:"models"`
}

// ModelSpecConfig is one model's declarative entry in a ModelsConfig.
type ModelSpecConfig struct {
	ID       string                   `json:"id,omitempty"`        // canonical id; empty derives from the path basename
	Path     string                   `json:"path"`                // on-disk model directory
	Aliases  []string                 `json:"aliases,omitempty"`   // alternate request names
	Pinned   bool                     `json:"pinned,omitempty"`    // exempt from eviction
	EstBytes uint64                   `json:"est_bytes,omitempty"` // residency-cost override; 0 = measure the pack
	Profiles map[string]ProfileConfig `json:"profiles,omitempty"`  // named generation presets → id:profile
}

// LoadModelsConfig reads and parses a --models-config file into the resolver's
// spec list plus its budget options. Durations are parsed from Go strings; an
// unparseable duration or a model with no path is an error (a malformed
// multi-model config must fail at boot, not serve a half-built registry).
func LoadModelsConfig(path string) ([]ModelSpec, MultiModelOptions, error) {
	read := core.ReadFile(path)
	if !read.OK {
		return nil, MultiModelOptions{}, core.E("serving.LoadModelsConfig", core.Sprintf("read %q", path), read.Err())
	}
	raw, ok := read.Value.([]byte)
	if !ok {
		return nil, MultiModelOptions{}, core.E("serving.LoadModelsConfig", "config bytes unavailable", nil)
	}
	var cfg ModelsConfig
	if r := core.JSONUnmarshal(raw, &cfg); !r.OK {
		return nil, MultiModelOptions{}, core.E("serving.LoadModelsConfig", core.Sprintf("parse %q", path), r.Err())
	}
	if len(cfg.Models) == 0 {
		return nil, MultiModelOptions{}, core.E("serving.LoadModelsConfig", core.Sprintf("%q declares no models", path), nil)
	}

	idleTTL, err := parseConfigDuration(cfg.IdleTTL)
	if err != nil {
		return nil, MultiModelOptions{}, core.E("serving.LoadModelsConfig", "idle_ttl", err)
	}
	sweep, err := parseConfigDuration(cfg.SweepInterval)
	if err != nil {
		return nil, MultiModelOptions{}, core.E("serving.LoadModelsConfig", "sweep_interval", err)
	}

	specs := make([]ModelSpec, 0, len(cfg.Models))
	for i, m := range cfg.Models {
		if core.Trim(m.Path) == "" {
			return nil, MultiModelOptions{}, core.E("serving.LoadModelsConfig", core.Sprintf("model[%d] has no path", i), nil)
		}
		specs = append(specs, ModelSpec{
			ID:       m.ID,
			Path:     m.Path,
			Aliases:  m.Aliases,
			Pinned:   m.Pinned,
			EstBytes: m.EstBytes,
			Profiles: m.Profiles,
		})
	}
	opts := MultiModelOptions{
		MemoryCeiling: cfg.MemoryCeilingBytes,
		IdleTTL:       idleTTL,
		SweepInterval: sweep,
	}
	return specs, opts, nil
}

// parseConfigDuration parses a Go duration string; empty yields 0 (the "unset"
// value the resolver reads as "off / auto").
func parseConfigDuration(s string) (time.Duration, error) {
	if core.Trim(s) == "" {
		return 0, nil
	}
	r := core.ParseDuration(s)
	if !r.OK {
		return 0, r.Err()
	}
	d, ok := r.Value.(time.Duration)
	if !ok {
		return 0, core.E("serving.parseConfigDuration", core.Sprintf("duration %q not a time.Duration", s), nil)
	}
	return d, nil
}
