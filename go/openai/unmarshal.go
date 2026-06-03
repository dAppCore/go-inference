// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON-decoding for the OpenAI wire types. Fires at
// HTTP request-entry per chat-completion / responses / services
// call — the encoding/json reflect path costs 22-65 allocs on the
// canonical 1/5/20-turn chat shapes.
//
// The single-pass walker per type lands at ~7-13 allocs for typical
// shapes — predominantly the per-string clones the wire contract
// already requires. Pointer fields (Temperature/TopP/TopK/MaxTokens)
// take address of stack-allocated locals only when the field is
// present and not null.
//
// All decoders SkipJSONValue past unknown fields (matches the
// stdlib default — DisallowUnknownFields is not configured on the
// adapter).

package openai

import (
	"dappco.re/go/inference/jsonenc"
)

// UnmarshalJSON walks the ChatCompletionRequest wire shape in a
// single pass. Replaces the encoding/json reflect path; saves the
// per-field reflect.Value boxing and the per-pointer-field heap
// escape.
func (r *ChatCompletionRequest) UnmarshalJSON(data []byte) error {
	*r = ChatCompletionRequest{}
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

// unmarshalField dispatches one ChatCompletionRequest field by key.
func (r *ChatCompletionRequest) unmarshalField(data []byte, i int, key []byte) (int, error) {
	switch string(key) {
	case "model":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Model = s
		return next, nil
	case "messages":
		msgs, next, err := parseChatMessageArray(data, i)
		if err != nil {
			return next, err
		}
		r.Messages = msgs
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
	case "max_tokens":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		v, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		k := int(v)
		r.MaxTokens = &k
		return next, nil
	case "reasoning_effort":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.ReasoningEffort = s
		return next, nil
	case "chat_template_kwargs":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		kw, next, err := parseChatTemplateKwargs(data, i)
		if err != nil {
			return next, err
		}
		r.ChatTemplateKwargs = kw
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
	case "stop":
		next, err := jsonenc.SkipJSONValue(data, i)
		if err != nil {
			return next, err
		}
		stops, err := jsonenc.ParseJSONStringList(data[i:next])
		if err != nil {
			return next, err
		}
		r.Stop = stops
		return next, nil
	case "user":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.User = s
		return next, nil
	}
	return jsonenc.SkipJSONValue(data, i)
}

// parseChatMessageArray walks a JSON array of ChatMessage objects.
func parseChatMessageArray(data []byte, i int) ([]ChatMessage, int, error) {
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
	var out []ChatMessage
	for {
		msg, next, err := parseChatMessage(data, i)
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

// parseChatMessage walks a single ChatMessage object at data[i].
func parseChatMessage(data []byte, i int) (ChatMessage, int, error) {
	var msg ChatMessage
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
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return msg, vnext, verr
			}
			msg.Content = s
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

// UnmarshalJSON walks the ResponseRequest wire shape in a single pass.
// Same dispatch shape as ChatCompletionRequest with the Responses
// field-name set (input / instructions / max_output_tokens).
func (r *ResponseRequest) UnmarshalJSON(data []byte) error {
	*r = ResponseRequest{}
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

// parseChatTemplateKwargs walks a chat_template_kwargs object, capturing the
// fields the runtime acts on (enable_thinking) and skipping the rest — mirrors
// the single-pass object walk in UnmarshalJSON.
func parseChatTemplateKwargs(data []byte, i int) (*ChatTemplateKwargs, int, error) {
	i, err := jsonenc.MatchObjectStart(data, i)
	if err != nil {
		return nil, i, err
	}
	kw := &ChatTemplateKwargs{}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return kw, i + 1, nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return nil, i, jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return nil, next, err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return nil, i, jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		if string(key) == "enable_thinking" {
			if jsonenc.IsJSONNull(data, i) {
				i += 4
			} else {
				v, n, err := jsonenc.ParseJSONBool(data, i)
				if err != nil {
					return nil, n, err
				}
				kw.EnableThinking = &v
				i = n
			}
		} else {
			n, err := jsonenc.SkipJSONValue(data, i)
			if err != nil {
				return nil, n, err
			}
			i = n
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return nil, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return kw, i + 1, nil
		}
		return nil, i, jsonenc.ErrInvalidJSON
	}
}

func (r *ResponseRequest) unmarshalField(data []byte, i int, key []byte) (int, error) {
	switch string(key) {
	case "model":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Model = s
		return next, nil
	case "input":
		msgs, next, err := parseResponseInputMessageArray(data, i)
		if err != nil {
			return next, err
		}
		r.Input = msgs
		return next, nil
	case "instructions":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Instructions = s
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
	case "max_output_tokens":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		v, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		k := int(v)
		r.MaxOutputTokens = &k
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
	case "stop":
		next, err := jsonenc.SkipJSONValue(data, i)
		if err != nil {
			return next, err
		}
		stops, err := jsonenc.ParseJSONStringList(data[i:next])
		if err != nil {
			return next, err
		}
		r.Stop = stops
		return next, nil
	case "user":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.User = s
		return next, nil
	}
	return jsonenc.SkipJSONValue(data, i)
}

// parseResponseInputMessageArray walks a JSON array of
// ResponseInputMessage objects.
func parseResponseInputMessageArray(data []byte, i int) ([]ResponseInputMessage, int, error) {
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
	var out []ResponseInputMessage
	for {
		msg, next, err := parseResponseInputMessage(data, i)
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

// parseResponseInputMessage walks one ResponseInputMessage at data[i].
func parseResponseInputMessage(data []byte, i int) (ResponseInputMessage, int, error) {
	var msg ResponseInputMessage
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
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return msg, vnext, verr
			}
			msg.Content = s
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
