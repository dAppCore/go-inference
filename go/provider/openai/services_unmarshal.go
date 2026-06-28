// SPDX-Licence-Identifier: EUPL-1.2

// Hand-rolled JSON-decoding for the openai services types
// (EmbeddingRequest, RerankRequest, CacheWarmRequest,
// CacheClearRequest, CancelRequest). Same single-pass byte-walker
// shape as openai/unmarshal.go.

package openai

import (
	"dappco.re/go/inference/jsonenc"
)

// UnmarshalJSON walks the EmbeddingRequest wire shape.
func (r *EmbeddingRequest) UnmarshalJSON(data []byte) error {
	*r = EmbeddingRequest{}
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

func (r *EmbeddingRequest) unmarshalField(data []byte, i int, key []byte) (int, error) {
	switch string(key) {
	case "model":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Model = s
		return next, nil
	case "input":
		// EmbeddingInput is []string with its own UnmarshalJSON;
		// call ParseJSONStringList directly to skip the nested
		// dispatch path.
		next, err := jsonenc.SkipJSONValue(data, i)
		if err != nil {
			return next, err
		}
		values, err := jsonenc.ParseJSONStringList(data[i:next])
		if err != nil {
			return next, err
		}
		r.Input = values
		return next, nil
	case "encoding_format":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.EncodingFormat = s
		return next, nil
	case "dimensions":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		k := int(n)
		r.Dimensions = &k
		return next, nil
	case "user":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.User = s
		return next, nil
	case "normalize":
		if jsonenc.IsJSONNull(data, i) {
			return i + 4, nil
		}
		v, next, err := jsonenc.ParseJSONBool(data, i)
		if err != nil {
			return next, err
		}
		r.Normalize = v
		return next, nil
	}
	return jsonenc.SkipJSONValue(data, i)
}

// UnmarshalJSON walks the RerankRequest wire shape.
func (r *RerankRequest) UnmarshalJSON(data []byte) error {
	*r = RerankRequest{}
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

func (r *RerankRequest) unmarshalField(data []byte, i int, key []byte) (int, error) {
	switch string(key) {
	case "model":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Model = s
		return next, nil
	case "query":
		s, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return next, err
		}
		r.Query = s
		return next, nil
	case "documents":
		next, err := jsonenc.SkipJSONValue(data, i)
		if err != nil {
			return next, err
		}
		docs, err := jsonenc.ParseJSONStringList(data[i:next])
		if err != nil {
			return next, err
		}
		r.Documents = docs
		return next, nil
	case "top_n":
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return next, err
		}
		r.TopN = int(n)
		return next, nil
	}
	return jsonenc.SkipJSONValue(data, i)
}

// UnmarshalJSON walks the CancelRequest wire shape.
func (r *CancelRequest) UnmarshalJSON(data []byte) error {
	*r = CancelRequest{}
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
		case "model":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return verr
			}
			r.Model = s
			i = vnext
		case "id":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return verr
			}
			r.ID = s
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

// UnmarshalJSON walks the CacheClearRequest wire shape. Labels
// (map[string]string) parsed via parseStringMap.
func (r *CacheClearRequest) UnmarshalJSON(data []byte) error {
	*r = CacheClearRequest{}
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
		case "model":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return verr
			}
			r.Model = s
			i = vnext
		case "labels":
			labels, vnext, verr := parseStringMap(data, i)
			if verr != nil {
				return verr
			}
			r.Labels = labels
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

// UnmarshalJSON walks the CacheWarmRequest wire shape. Tokens
// ([]int32) parsed via parseInt32Array; Labels via parseStringMap.
func (r *CacheWarmRequest) UnmarshalJSON(data []byte) error {
	*r = CacheWarmRequest{}
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
		case "model":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return verr
			}
			r.Model = s
			i = vnext
		case "prompt":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return verr
			}
			r.Prompt = s
			i = vnext
		case "tokens":
			toks, vnext, verr := parseInt32Array(data, i)
			if verr != nil {
				return verr
			}
			r.Tokens = toks
			i = vnext
		case "mode":
			s, vnext, verr := jsonenc.ParseJSONString(data, i)
			if verr != nil {
				return verr
			}
			r.Mode = s
			i = vnext
		case "labels":
			labels, vnext, verr := parseStringMap(data, i)
			if verr != nil {
				return verr
			}
			r.Labels = labels
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

// parseStringMap walks a JSON object with string keys + string
// values and returns a map[string]string. Used for the Labels
// fields on CacheWarm / CacheClear requests.
func parseStringMap(data []byte, i int) (map[string]string, int, error) {
	if jsonenc.IsJSONNull(data, i) {
		return nil, i + 4, nil
	}
	i, err := jsonenc.MatchObjectStart(data, i)
	if err != nil {
		return nil, i, err
	}
	i = jsonenc.SkipJSONWhitespace(data, i)
	if i < len(data) && data[i] == '}' {
		return nil, i + 1, nil
	}
	out := make(map[string]string)
	for {
		i = jsonenc.SkipJSONWhitespace(data, i)
		if i >= len(data) || data[i] != '"' {
			return nil, i, jsonenc.ErrInvalidJSON
		}
		key, next, err := jsonenc.ParseJSONString(data, i)
		if err != nil {
			return nil, next, err
		}
		i = jsonenc.SkipJSONWhitespace(data, next)
		if i >= len(data) || data[i] != ':' {
			return nil, i, jsonenc.ErrInvalidJSON
		}
		i = jsonenc.SkipJSONWhitespace(data, i+1)
		val, vnext, verr := jsonenc.ParseJSONString(data, i)
		if verr != nil {
			return nil, vnext, verr
		}
		out[key] = val
		i = jsonenc.SkipJSONWhitespace(data, vnext)
		if i >= len(data) {
			return nil, i, jsonenc.ErrInvalidJSON
		}
		if data[i] == ',' {
			i++
			continue
		}
		if data[i] == '}' {
			return out, i + 1, nil
		}
		return nil, i, jsonenc.ErrInvalidJSON
	}
}

// parseInt32Array walks a JSON array of integers and returns the
// parsed slice. Used for the Tokens field on CacheWarmRequest.
func parseInt32Array(data []byte, i int) ([]int32, int, error) {
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
	// Pre-size from an exact element count (alloc-free scan) so the
	// per-token append never re-grows the backing array.
	out := make([]int32, 0, jsonenc.CountJSONArrayElements(data, i))
	for {
		n, next, err := jsonenc.ParseJSONInt(data, i)
		if err != nil {
			return nil, next, err
		}
		out = append(out, int32(n))
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
