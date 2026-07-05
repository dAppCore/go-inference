// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON-decoding for the Ollama wire types. Fires at
// HTTP request-entry per chat/generate call — encoding/json's
// reflect path costs 12-55 allocs on the canonical chat-shape
// turns; the single-pass walker lands at ~7-12 allocs.
//
// Same single-pass byte-walker shape as anthropic/openai. Each
// type's UnmarshalJSON dispatches by exact key byte-compare;
// unknown fields SkipJSONValue past silently (matches stdlib
// default — DisallowUnknownFields is not configured).

package ollama

import (
	"dappco.re/go/inference/jsonenc"
)

// UnmarshalJSON walks the ChatRequest wire shape in a single pass.
func (r *ChatRequest) UnmarshalJSON(data []byte) error {
	*r = ChatRequest{}
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

func (r *ChatRequest) unmarshalField(data []byte, i int, key []byte) (int, error) {
	switch string(key) {
	case "model":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Model = s
		return next, nil
	case "messages":
		msgs, next, err := parseMessageArray(data, i)
		if err != nil {
			return next, err
		}
		r.Messages = msgs
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
	case "options":
		opts, next, err := parseOptions(data, i)
		if err != nil {
			return next, err
		}
		r.Options = opts
		return next, nil
	}
	return jsonenc.SkipJSONValue(data, i)
}

// UnmarshalJSON walks the GenerateRequest wire shape.
func (r *GenerateRequest) UnmarshalJSON(data []byte) error {
	*r = GenerateRequest{}
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

func (r *GenerateRequest) unmarshalField(data []byte, i int, key []byte) (int, error) {
	switch string(key) {
	case "model":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Model = s
		return next, nil
	case "prompt":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Prompt = s
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
	case "options":
		opts, next, err := parseOptions(data, i)
		if err != nil {
			return next, err
		}
		r.Options = opts
		return next, nil
	}
	return jsonenc.SkipJSONValue(data, i)
}

// parseMessageArray walks a JSON array of Message objects.
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

// parseMessage walks a single Message object.
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

// parseOptions walks an Options object.
func parseOptions(data []byte, i int) (Options, int, error) {
	var opts Options
	if jsonenc.IsJSONNull(data, i) {
		return opts, i + 4, nil
	}
	i, err := jsonenc.MatchObjectStart(data, i)
	if err != nil {
		return opts, i, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return opts, i + 1, nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return opts, i, jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return opts, next, err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return opts, i, jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		switch string(key) {
		case "temperature":
			v, vnext, verr := jsonenc.ParseJSONFloat32(data, i)
			if verr != nil {
				return opts, vnext, verr
			}
			opts.Temperature = v
			i = vnext
		case "top_k":
			n, vnext, verr := jsonenc.ParseJSONInt(data, i)
			if verr != nil {
				return opts, vnext, verr
			}
			opts.TopK = int(n)
			i = vnext
		case "top_p":
			v, vnext, verr := jsonenc.ParseJSONFloat32(data, i)
			if verr != nil {
				return opts, vnext, verr
			}
			opts.TopP = v
			i = vnext
		case "min_p":
			v, vnext, verr := jsonenc.ParseJSONFloat32(data, i)
			if verr != nil {
				return opts, vnext, verr
			}
			opts.MinP = v
			i = vnext
		case "num_predict":
			n, vnext, verr := jsonenc.ParseJSONInt(data, i)
			if verr != nil {
				return opts, vnext, verr
			}
			opts.NumPredict = int(n)
			i = vnext
		default:
			vnext, verr := jsonenc.SkipJSONValue(data, i)
			if verr != nil {
				return opts, vnext, verr
			}
			i = vnext
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return opts, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return opts, i + 1, nil
		}
		return opts, i, jsonenc.ErrInvalidJSON
	}
}

// UnmarshalJSON walks the ChatResponse wire shape.
func (r *ChatResponse) UnmarshalJSON(data []byte) error {
	*r = ChatResponse{}
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

func (r *ChatResponse) unmarshalField(data []byte, i int, key []byte) (int, error) {
	switch string(key) {
	case "model":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Model = s
		return next, nil
	case "message":
		msg, next, err := parseMessage(data, i)
		if err != nil {
			return next, err
		}
		r.Message = msg
		return next, nil
	case "done":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		v, next, err := jsonenc.ParseJSONBool(data, i)
		if err != nil {
			return next, err
		}
		r.Done = v
		return next, nil
	case "prompt_eval_count":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.PromptEvalCount = int(n)
		return next, nil
	case "eval_count":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.EvalCount = int(n)
		return next, nil
	case "total_duration":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.TotalDuration = n
		return next, nil
	case "load_duration":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.LoadDuration = n
		return next, nil
	case "prompt_eval_duration":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.PromptEvalDuration = n
		return next, nil
	case "eval_duration":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.EvalDuration = n
		return next, nil
	}
	return jsonenc.SkipJSONValue(data, i)
}

// UnmarshalJSON walks the TagsResponse wire shape — the /api/tags
// list-models response from a client perspective.
func (r *TagsResponse) UnmarshalJSON(data []byte) error {
	*r = TagsResponse{}
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
		switch string(key) {
		case "models":
			tags, vnext, verr := parseModelTagArray(data, i)
			if verr != nil {
				return verr
			}
			r.Models = tags
			i = vnext
		default:
			vnext, verr := jsonenc.SkipJSONValue(data, i)
			if verr != nil {
				return verr
			}
			i = vnext
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

// parseModelTagArray walks a JSON array of ModelTag objects.
func parseModelTagArray(data []byte, i int) ([]ModelTag, int, error) {
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
	var out []ModelTag
	for {
		tag, next, err := parseModelTag(data, i)
		if err != nil {
			return nil, next, err
		}
		out = append(out, tag)
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

// parseModelTag walks a single ModelTag object.
func parseModelTag(data []byte, i int) (ModelTag, int, error) {
	var tag ModelTag
	i, err := jsonenc.MatchObjectStart(data, i)
	if err != nil {
		return tag, i, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return tag, i + 1, nil
	}
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return tag, i, jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONStringRaw(data, i)
		if err != nil {
			return tag, next, err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return tag, i, jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		switch string(key) {
		case "name":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return tag, vnext, verr
			}
			tag.Name = s
			i = vnext
		case "model":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return tag, vnext, verr
			}
			tag.Model = s
			i = vnext
		case "modified_at":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return tag, vnext, verr
			}
			tag.ModifiedAt = s
			i = vnext
		case "size":
			n, vnext, verr := jsonenc.ParseJSONInt(data, i)
			if verr != nil {
				return tag, vnext, verr
			}
			tag.Size = n
			i = vnext
		default:
			vnext, verr := jsonenc.SkipJSONValue(data, i)
			if verr != nil {
				return tag, vnext, verr
			}
			i = vnext
		}
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) {
			return tag, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return tag, i + 1, nil
		}
		return tag, i, jsonenc.ErrInvalidJSON
	}
}

// UnmarshalJSON walks the GenerateResponse wire shape.
func (r *GenerateResponse) UnmarshalJSON(data []byte) error {
	*r = GenerateResponse{}
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

func (r *GenerateResponse) unmarshalField(data []byte, i int, key []byte) (int, error) {
	switch string(key) {
	case "model":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Model = s
		return next, nil
	case "response":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Response = s
		return next, nil
	case "done":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		v, next, err := jsonenc.ParseJSONBool(data, i)
		if err != nil {
			return next, err
		}
		r.Done = v
		return next, nil
	case "prompt_eval_count":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.PromptEvalCount = int(n)
		return next, nil
	case "eval_count":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.EvalCount = int(n)
		return next, nil
	case "total_duration":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.TotalDuration = n
		return next, nil
	case "load_duration":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.LoadDuration = n
		return next, nil
	case "prompt_eval_duration":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.PromptEvalDuration = n
		return next, nil
	case "eval_duration":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.EvalDuration = n
		return next, nil
	}
	return jsonenc.SkipJSONValue(data, i)
}
