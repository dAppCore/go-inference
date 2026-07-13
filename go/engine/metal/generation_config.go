// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
	"dappco.re/go/inference/engine"
)

// generation_config.go reads the checkpoint's declared generation defaults —
// the stop set and, below, the sampling intent. gemma4 declares eos_token_id
// [1 <eos>, 106 <turn|>, 50 <|tool_response>]; the engine's derived defaults
// (tokenizer <eos> + the turn-close marker) cover the first two, and the
// declared set adds the rest (stopping the model before it hallucinates a
// tool's output). Mirrors the HIP lane, which has read generation_config.json
// since its first serve.

// generationConfigStops parses generation_config.json bytes and returns the
// declared eos_token_id set. The field ships as either a single integer or an
// array of integers; anything unparseable returns nil (the engine's derived
// defaults still apply).
func generationConfigStops(data []byte) []int32 {
	var cfg struct {
		EOSTokenID any `json:"eos_token_id"`
	}
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return nil
	}
	switch v := cfg.EOSTokenID.(type) {
	case float64:
		return []int32{int32(v)}
	case []any:
		ids := make([]int32, 0, len(v))
		for _, e := range v {
			if f, ok := e.(float64); ok {
				ids = append(ids, int32(f))
			}
		}
		if len(ids) == 0 {
			return nil
		}
		return ids
	}
	return nil
}

// loadGenerationConfigStops reads dir/generation_config.json and returns its
// declared stop ids, or nil when the file is absent or carries none — the
// same soft-optional convention as processor_config.json on the vision path.
func loadGenerationConfigStops(dir string) []int32 {
	read := core.ReadFile(core.PathJoin(dir, "generation_config.json"))
	if !read.OK {
		return nil
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return nil
	}
	return generationConfigStops(data)
}

// DeclaredStopTokens reports the checkpoint's generation_config stop set
// (engine.StopTokenDeclarer) — empty when the checkpoint declares none.
func (m *NativeTokenModel) DeclaredStopTokens() []int32 {
	if m == nil {
		return nil
	}
	return m.declaredStops
}

// generationConfigSampling is the on-the-wire shape of generation_config.json's
// sampling block — the do_sample/temperature/top_p/top_k/min_p/suppress_tokens
// siblings of eos_token_id above. Pointer fields recover "the file declared
// this key" (json.Unmarshal only allocates a target when the key is present),
// distinct from a request's separate unset-vs-zero problem (see
// engine.SamplingDefaultsDeclarer). Unlike eos_token_id, none of these fields
// are documented or observed (across the cached mlx-community gemma4/Qwen3.5
// snapshots) to ship in a scalar-or-array dual form, so plain typed fields
// suffice — no any-typed switch needed. min_p ships in the Qwen3.5 configs
// (as 0.0); it carries the same zero-ambiguity as top_p/top_k, so a declared
// 0.0 folds onto an unset request as a no-op (min-p disabled either way).
type generationConfigSampling struct {
	DoSample       *bool    `json:"do_sample"`
	Temperature    *float32 `json:"temperature"`
	TopP           *float32 `json:"top_p"`
	TopK           *int     `json:"top_k"`
	MinP           *float32 `json:"min_p"`
	SuppressTokens []int32  `json:"suppress_tokens"`
}

// generationConfigSamplingDefaults parses generation_config.json bytes for the
// checkpoint's declared sampling intent. A field absent from the file, or the
// whole file unparseable, comes back as that field's zero value (nil pointer /
// nil slice) — "the file said nothing", never "the file said zero".
func generationConfigSamplingDefaults(data []byte) engine.SamplingDefaults {
	var cfg generationConfigSampling
	if r := core.JSONUnmarshal(data, &cfg); !r.OK {
		return engine.SamplingDefaults{}
	}
	return engine.SamplingDefaults{
		DoSample:       cfg.DoSample,
		Temperature:    cfg.Temperature,
		TopP:           cfg.TopP,
		TopK:           cfg.TopK,
		MinP:           cfg.MinP,
		SuppressTokens: cfg.SuppressTokens,
	}
}

// loadGenerationConfigSamplingDefaults reads dir/generation_config.json and
// returns its declared sampling defaults, or the zero value when the file is
// absent — the same soft-optional convention as loadGenerationConfigStops.
func loadGenerationConfigSamplingDefaults(dir string) engine.SamplingDefaults {
	read := core.ReadFile(core.PathJoin(dir, "generation_config.json"))
	if !read.OK {
		return engine.SamplingDefaults{}
	}
	data, ok := read.Value.([]byte)
	if !ok {
		return engine.SamplingDefaults{}
	}
	return generationConfigSamplingDefaults(data)
}

// DeclaredSamplingDefaults reports the checkpoint's generation_config sampling
// intent (engine.SamplingDefaultsDeclarer) — the zero value when the
// checkpoint declares none.
func (m *NativeTokenModel) DeclaredSamplingDefaults() engine.SamplingDefaults {
	if m == nil {
		return engine.SamplingDefaults{}
	}
	return m.declaredSampling
}
