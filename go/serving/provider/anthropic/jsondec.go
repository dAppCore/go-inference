// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON-decoding for the Anthropic Messages wire types.
// Fires at HTTP request-entry per Messages call — the encoding/json
// reflect path costs 26-107 allocs for the canonical 1/5/20-turn
// shapes (encoder state machine, per-field reflect.Value boxing,
// per-string allocation, per-pointer-field heap allocation).
//
// The single-pass walker per type lands at ~6-10 allocs for typical
// shapes — predominantly the per-string clones the wire contract
// already requires. Slice fields are pre-sized when the array length
// is cheap to count; pointer fields skip the per-field heap escape
// by stack-allocating the indirected value and taking address.
//
// Each UnmarshalJSON returns errors via the package-local
// resultError shape (matches the encoding/json contract — wrapped
// for the caller's `core.JSONUnmarshal*` Result) so existing tests
// continue to receive a single error.

package anthropic

import (
	core "dappco.re/go"
	"dappco.re/go/inference/jsonenc"
)

// UnmarshalJSON walks the MessageRequest wire shape in a single pass.
// Wire-compatible with json.Unmarshal across every branch:
//   - model, system, messages, max_tokens, temperature, top_p, min_p,
//     top_k, stream, stop_sequences — dispatched by exact key
//     byte-compare.
//   - Unknown keys SkipJSONValue past — matches encoding/json's
//     default decoder behaviour (silent ignore unless DisallowUnknownFields
//     is set, which this package does not).
//   - Pointer fields (Temperature, TopP, MinP, TopK) point at heap copies
//     of the parsed value only when the field is present and not
//     null — same as the reflect path.
//   - StopSequences via jsonenc.ParseJSONStringList (string or
//     array of strings, plus null).
//
// Allocations come from:
//   - One per parsed string (model/system/role/content text). Same
//     floor encoding/json pays.
//   - One per non-empty Messages slice (pre-sized via prescanning the
//     array length).
//   - One per non-empty Content slice within each Message.
//   - One per non-nil pointer field (Temperature, TopP, MinP, TopK).
func (r *MessageRequest) UnmarshalJSON(data []byte) error {
	*r = MessageRequest{}
	i, err := jsonenc.MatchObjectStart(data, 0)
	if err != nil {
		return err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		i, err = r.unmarshalField(data, i, key)
		if err != nil {
			return err
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return nil
		}
		return jsonenc.ErrInvalidJSON
	}
}

// unmarshalField dispatches one MessageRequest field by key. Returns
// the index one past the consumed value (which may itself be an
// object or array).
func (r *MessageRequest) unmarshalField(data []byte, i int, key []byte) (int, error) {
	switch string(key) {
	case "model":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Model = s
		return next, nil
	case "system":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.System = s
		return next, nil
	case "messages":
		msgs, next, err := parseMessageArray(data, i)
		if err != nil {
			return next, err
		}
		r.Messages = msgs
		return next, nil
	case "tools":
		// Tools carry a nested schema (array → object → input_schema → properties
		// map) that is rare (only agentic requests) and off the hot chat path, so
		// rather than hand-roll the whole tree we capture its span and reflect-
		// decode just this field. The hand-rolled fast path still owns model /
		// messages / sampling — the per-request cost that actually multiplies.
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		end, err := jsonenc.SkipJSONValue(data, i)
		if err != nil {
			return end, err
		}
		if res := core.JSONUnmarshal(data[i:end], &r.Tools); !res.OK {
			return end, res.Err()
		}
		return end, nil
	case "tool_choice":
		// tool_choice is always a small object ({"type":".."} or {"type":"tool",
		// "name":".."}) on the same cold agentic-request path as tools — reflect-
		// decodes the captured span rather than hand-rolling the branch here.
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		end, err := jsonenc.SkipJSONValue(data, i)
		if err != nil {
			return end, err
		}
		var choice ToolChoice
		if res := core.JSONUnmarshal(data[i:end], &choice); !res.OK {
			return end, res.Err()
		}
		r.ToolChoice = &choice
		return end, nil
	case "max_tokens":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.MaxTokens = int(n)
		return next, nil
	case "temperature":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		v, next, err := jsonenc.ParseJSONFloat32(data, i)
		if err != nil {
			return next, err
		}
		r.Temperature = &v
		return next, nil
	case "top_p":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		v, next, err := jsonenc.ParseJSONFloat32(data, i)
		if err != nil {
			return next, err
		}
		r.TopP = &v
		return next, nil
	case "min_p":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		v, next, err := jsonenc.ParseJSONFloat32(data, i)
		if err != nil {
			return next, err
		}
		r.MinP = &v
		return next, nil
	case "top_k":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		v, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		k := int(v)
		r.TopK = &k
		return next, nil
	case "stream":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		v, next, err := jsonenc.ParseJSONBool(data, i)
		if err != nil {
			return next, err
		}
		r.Stream = v
		return next, nil
	case "stop_sequences":
		next, err := jsonenc.SkipJSONValue(data, i)
		if err != nil {
			return next, err
		}
		stops, err := jsonenc.ParseJSONStringList(data[i:next])
		if err != nil {
			return next, err
		}
		r.StopSequences = stops
		return next, nil
	}
	return jsonenc.SkipJSONValue(data, i)
}

// UnmarshalJSON walks the MessageResponse wire shape in a single pass.
// Same dispatch pattern as MessageRequest; covers every field the
// hand-rolled AppendMessageResponse emits.
func (r *MessageResponse) UnmarshalJSON(data []byte) error {
	*r = MessageResponse{}
	i, err := jsonenc.MatchObjectStart(data, 0)
	if err != nil {
		return err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		i, err = r.unmarshalField(data, i, key)
		if err != nil {
			return err
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return nil
		}
		return jsonenc.ErrInvalidJSON
	}
}

// unmarshalField dispatches one MessageResponse field by key.
func (r *MessageResponse) unmarshalField(data []byte, i int, key []byte) (int, error) {
	switch string(key) {
	case "id":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.ID = s
		return next, nil
	case "type":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Type = s
		return next, nil
	case "role":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Role = s
		return next, nil
	case "model":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Model = s
		return next, nil
	case "content":
		blocks, next, err := parseContentBlockArray(data, i)
		if err != nil {
			return next, err
		}
		r.Content = blocks
		return next, nil
	case "stop_reason":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.StopReason = s
		return next, nil
	case "stop_sequence":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.StopSequence = s
		return next, nil
	case "usage":
		usage, next, err := parseUsage(data, i)
		if err != nil {
			return next, err
		}
		r.Usage = usage
		return next, nil
	}
	return jsonenc.SkipJSONValue(data, i)
}

// parseMessageArray walks a JSON array of Message objects at data[i].
// Uses append-grow rather than a CountJSONArrayElements prescan: the
// prescan walks the whole array via SkipJSONValue twice (once to
// count, once to parse) and costs more than the append-double cascade
// it would have saved (single-turn 4.1 µs vs 2.6 µs without).
func parseMessageArray(data []byte, i int) ([]Message, int, error) {
	if jsonenc.IsJSONNull(data, i) {
		return nil, i + 4, nil
	}
	i, err := jsonenc.MatchArrayStart(data, i)
	if err != nil {
		return nil, i, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == ']' {
		return nil, i + 1, nil
	}
	var out []Message
	for {
		msg, next, err := parseMessage(data, i)
		if err != nil {
			return nil, next, err
		}
		out = append(out, msg)
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) {
			return nil, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i = jsonenc.SkipJSONWhitespace(data, i+1)
			continue
		}
		if data[i] == ']' {
			return out, i + 1, nil
		}
		return nil, i, jsonenc.ErrInvalidJSON
	}
}

// parseMessage walks a single Message object at data[i].
func parseMessage(data []byte, i int) (Message, int, error) {
	var msg Message
	i, err := jsonenc.MatchObjectStart(data, i)
	if err != nil {
		return msg, i, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return msg, i + 1, nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return msg, i, jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return msg, next, err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return msg, i, jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		switch string(key) {
		case "role":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return msg, vnext, verr
			}
			msg.Role = s
			i = vnext
		case "content":
			blocks, vnext, verr := parseContentBlockArray(data, i)
			if verr != nil {
				return msg, vnext, verr
			}
			msg.Content = blocks
			i = vnext
		default:
			vnext, verr := jsonenc.SkipJSONValue(data, i)
			if verr != nil {
				return msg, vnext, verr
			}
			i = vnext
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return msg, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return msg, i + 1, nil
		}
		return msg, i, jsonenc.ErrInvalidJSON
	}
}

// parseContentBlockArray walks a JSON array of ContentBlock objects.
// append-grow path — content arrays typically carry 1-3 blocks per
// turn, well under the first-grow threshold.
func parseContentBlockArray(data []byte, i int) ([]ContentBlock, int, error) {
	if jsonenc.IsJSONNull(data, i) {
		return nil, i + 4, nil
	}
	i, err := jsonenc.MatchArrayStart(data, i)
	if err != nil {
		return nil, i, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == ']' {
		return nil, i + 1, nil
	}
	var out []ContentBlock
	for {
		block, next, err := parseContentBlock(data, i)
		if err != nil {
			return nil, next, err
		}
		out = append(out, block)
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) {
			return nil, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i = jsonenc.SkipJSONWhitespace(data, i+1)
			continue
		}
		if data[i] == ']' {
			return out, i + 1, nil
		}
		return nil, i, jsonenc.ErrInvalidJSON
	}
}

// parseToolResultContent reads a tool_result "content" value — a bare string or
// an array of content blocks (Anthropic accepts both). An array renders through
// blockText; a null/other value yields empty content. The result is the tool
// output the <|tool_response> render feeds back to the model.
func parseToolResultContent(data []byte, i int) (string, int, error) {
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i >= len(data) {
		return "", i, jsonenc.ErrInvalidJSON
	}
	switch data[i] {
	case '"':
		return jsonenc.ParseJSONString(data, i)
	case '[':
		blocks, next, err := parseContentBlockArray(data, i)
		if err != nil {
			return "", next, err
		}
		return blockText(blocks), next, nil
	default:
		next, err := jsonenc.SkipJSONValue(data, i)
		return "", next, err
	}
}

// parseContentBlock walks a single ContentBlock object at data[i].
func parseContentBlock(data []byte, i int) (ContentBlock, int, error) {
	var block ContentBlock
	i, err := jsonenc.MatchObjectStart(data, i)
	if err != nil {
		return block, i, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return block, i + 1, nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return block, i, jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return block, next, err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return block, i, jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		switch string(key) {
		case "type":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return block, vnext, verr
			}
			block.Type = s
			i = vnext
		case "text":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return block, vnext, verr
			}
			block.Text = s
			i = vnext
		case "tool_use_id":
			// tool_result -> the id of the tool_use it answers.
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return block, vnext, verr
			}
			block.ToolUseID = s
			i = vnext
		case "content":
			// tool_result content: a bare string, or an array of text blocks
			// (Anthropic allows both). Either way it lands in Text as the tool
			// output the <|tool_response> render feeds back to the model.
			s, vnext, verr := parseToolResultContent(data, i)
			if verr != nil {
				return block, vnext, verr
			}
			block.Text = s
			i = vnext
		case "name":
			// tool_use -> the called function's name.
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return block, vnext, verr
			}
			block.Name = s
			i = vnext
		case "input":
			// tool_use -> the call's arguments object. Decoded (id/name/input) so a
			// stateless client replaying full history can re-render the prior call.
			start := i
			vnext, verr := jsonenc.SkipJSONValue(data, i)
			if verr != nil {
				return block, vnext, verr
			}
			m := map[string]any{}
			if res := core.JSONUnmarshal(data[start:vnext], &m); res.OK {
				block.Input = m
			}
			i = vnext
		default:
			vnext, verr := jsonenc.SkipJSONValue(data, i)
			if verr != nil {
				return block, vnext, verr
			}
			i = vnext
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return block, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return block, i + 1, nil
		}
		return block, i, jsonenc.ErrInvalidJSON
	}
}

// parseUsage walks a Usage object at data[i].
func parseUsage(data []byte, i int) (Usage, int, error) {
	var u Usage
	i, err := jsonenc.MatchObjectStart(data, i)
	if err != nil {
		return u, i, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return u, i + 1, nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return u, i, jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return u, next, err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return u, i, jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		switch string(key) {
		case "input_tokens":
			n, vnext, verr := jsonenc.ParseJSONInt(data, i)
			if verr != nil {
				return u, vnext, verr
			}
			u.InputTokens = int(n)
			i = vnext
		case "output_tokens":
			n, vnext, verr := jsonenc.ParseJSONInt(data, i)
			if verr != nil {
				return u, vnext, verr
			}
			u.OutputTokens = int(n)
			i = vnext
		default:
			vnext, verr := jsonenc.SkipJSONValue(data, i)
			if verr != nil {
				return u, vnext, verr
			}
			i = vnext
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return u, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return u, i + 1, nil
		}
		return u, i, jsonenc.ErrInvalidJSON
	}
}
