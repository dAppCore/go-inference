// SPDX-Licence-Identifier: EUPL-1.2

//go:build darwin && arm64

package native

import (
	core "dappco.re/go"
)

// generation_config.go reads the checkpoint's declared generation defaults —
// today the stop set. gemma4 declares eos_token_id [1 <eos>, 106 <turn|>,
// 50 <|tool_response>]; the engine's derived defaults (tokenizer <eos> + the
// turn-close marker) cover the first two, and the declared set adds the rest
// (stopping the model before it hallucinates a tool's output). Mirrors the
// HIP lane, which has read generation_config.json since its first serve.

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
