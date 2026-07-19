// SPDX-Licence-Identifier: EUPL-1.2

package dotsocr

import core "dappco.re/go"

// GenerationConfig is the architecture-relevant subset of a DOTS-OCR generation_config.json —
// just the stop-token set GreedyDecode needs (ocr.go). eos_token_id ships as a JSON array on the
// real checkpoint ([151643, 151673] — <|endoftext|> and <|endofassistant|>) but the HF convention
// also allows a bare single integer on other checkpoints, so this parses either shape rather than
// assuming the array form.
type GenerationConfig struct {
	EOSTokenIDs []int32
}

type generationConfigJSON struct {
	EOSTokenID any `json:"eos_token_id"`
}

// ParseGenerationConfig parses a DOTS-OCR generation_config.json.
func ParseGenerationConfig(data []byte) (*GenerationConfig, error) {
	var raw generationConfigJSON
	if r := core.JSONUnmarshal(data, &raw); !r.OK {
		return nil, core.NewError("dotsocr.ParseGenerationConfig: generation_config.json parse failed")
	}
	var ids []int32
	switch v := raw.EOSTokenID.(type) {
	case float64:
		ids = []int32{int32(v)}
	case []any:
		for _, e := range v {
			if f, ok := e.(float64); ok {
				ids = append(ids, int32(f))
			}
		}
	}
	return &GenerationConfig{EOSTokenIDs: ids}, nil
}

// EOSSet returns EOSTokenIDs as a lookup set — the shape GreedyDecode's stop check wants.
func (g *GenerationConfig) EOSSet() map[int32]bool {
	set := make(map[int32]bool, len(g.EOSTokenIDs))
	for _, id := range g.EOSTokenIDs {
		set[id] = true
	}
	return set
}
